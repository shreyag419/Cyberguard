
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import config

# Step 1: load and merge the 8 CSV files
def normalise_label(s: pd.Series) -> pd.Series:
    """Fix labels: strip spaces and replace the garbled dash in 'Web Attack - ...'."""
    s = s.astype(str).str.replace(r"[^\x20-\x7E]", "-", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def load_raw() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {config.RAW_DIR}")

    parts = []
    for f in files:
        # latin-1 because the original files contain a non-UTF-8 dash in some labels
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()

        # pandas renames the duplicated 'Fwd Header Length' column to '...Length.1'
        dups = [c for c in df.columns
                if c.rsplit(".", 1)[-1].isdigit() and c.rsplit(".", 1)[0] in df.columns]
        df = df.drop(columns=dups)

        df[config.LABEL_COL] = normalise_label(df[config.LABEL_COL])
        feats = [c for c in df.columns if c != config.LABEL_COL]
        df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
        df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).astype("float32")
        before = len(df)
        df = df.dropna()
        print(f"  {f.name}: {before:,} rows, {before - len(df):,} dropped (inf/NaN)")
        parts.append(df)

    cols = set(parts[0].columns)
    for p in parts:
        if set(p.columns) != cols:
            raise ValueError("The CSV files do not all have the same columns.")
    return pd.concat(parts, ignore_index=True)

# Step 2: clean

def clean(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates()
    print(f"  duplicates removed: {n - len(df):,}")

    feats = [c for c in df.columns if c != config.LABEL_COL]
    constant = [c for c in feats if df[c].min() == df[c].max()]
    if constant:
        print(f"  constant columns removed ({len(constant)}): {constant}")
        df = df.drop(columns=constant)

    missing = [c for c in config.FUZZY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Fuzzy columns missing from the data: {missing}")
    return df

# Step 3: split (unseen attacks are kept completely apart)
def _cap_per_class(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    parts = []
    for _, g in frame.groupby(config.LABEL_COL):
        parts.append(g.sample(min(len(g), n), random_state=config.RANDOM_SEED))
    return pd.concat(parts) if parts else frame.iloc[0:0]


def make_splits(df: pd.DataFrame):
    seed = config.RANDOM_SEED
    labels = df[config.LABEL_COL]

    absent = set(config.UNSEEN_ATTACKS) - set(labels.unique())
    if absent:
        raise ValueError(f"Unseen attack names not found in data: {absent}\n"
                         f"Labels available: {sorted(labels.unique())}")

    benign = df[labels == config.BENIGN_LABEL]
    attacks = df[labels != config.BENIGN_LABEL]
    unseen = attacks[attacks[config.LABEL_COL].isin(config.UNSEEN_ATTACKS)]
    known = attacks[~attacks[config.LABEL_COL].isin(config.UNSEEN_ATTACKS)]

    # benign: sample, then train / val / test
    benign = benign.sample(min(config.BENIGN_SAMPLE, len(benign)), random_state=seed)
    tr, va, te = config.BENIGN_SPLIT
    b_train, b_rest = train_test_split(benign, train_size=tr, random_state=seed)
    b_val, b_test = train_test_split(b_rest, train_size=va / (va + te), random_state=seed)

    # known attacks: cap each type, then split each type between val and test
    known = _cap_per_class(known, config.MAX_PER_ATTACK)
    k_val, k_test = [], []
    for name, g in known.groupby(config.LABEL_COL):
        if len(g) < 2:
            print(f"  warning: only {len(g)} row of '{name}', put in validation only")
            k_val.append(g)
            continue
        a, b = train_test_split(g, test_size=1 - config.KNOWN_ATTACK_VAL_FRACTION,
                                random_state=seed)
        k_val.append(a)
        k_test.append(b)

    def shuffled(parts):
        return pd.concat(parts).sample(frac=1.0, random_state=seed)

    splits = {
        "train": b_train,
        "val": shuffled([b_val] + k_val),
        "test": shuffled([b_test] + k_test),
        "unseen": _cap_per_class(unseen, config.MAX_PER_ATTACK),
    }
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits: dict) -> None:
    """Stop with an error if anything leaks between splits."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = splits[a].index.intersection(splits[b].index)
            assert len(overlap) == 0, f"Rows shared between {a} and {b}"
    assert (splits["train"][config.LABEL_COL] == config.BENIGN_LABEL).all(), "train must be benign only"
    for name in ("train", "val", "test"):
        bad = splits[name][config.LABEL_COL].isin(config.UNSEEN_ATTACKS).any()
        assert not bad, f"Unseen attack found in {name}"
    assert splits["unseen"][config.LABEL_COL].isin(config.UNSEEN_ATTACKS).all()
    print("  leakage checks passed")

# Step 4: feature selection, log transform, scaling (fitted on benign train only)

def _log(frame: pd.DataFrame) -> np.ndarray:
    """log1p squashes heavy-tailed values; negatives (bad data / -1 flags) become 0."""
    return np.log1p(frame.clip(lower=0).to_numpy(dtype="float64"))


def select_features(train: pd.DataFrame) -> list:
    feats = [c for c in train.columns if c != config.LABEL_COL]
    protected = list(config.FUZZY_COLS)
    ordered = protected + [c for c in feats if c not in protected]

    sample = train[ordered].sample(min(len(train), config.CORR_SAMPLE),
                                   random_state=config.RANDOM_SEED)
    corr = pd.DataFrame(_log(sample), columns=ordered).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [c for c in ordered
            if c not in protected and (upper[c] > config.CORR_THRESHOLD).any()]
    print(f"  dropped {len(drop)} highly correlated features, kept {len(feats) - len(drop)}")
    return [c for c in feats if c not in drop]


def build_output(df: pd.DataFrame, features: list, scaler: MinMaxScaler) -> pd.DataFrame:
    scaled = scaler.transform(_log(df[features])).astype("float32")
    out = pd.DataFrame(scaled, columns=features)
    for c in config.FUZZY_COLS:                      # raw values for the fuzzy engine
        out["raw_" + c] = df[c].clip(lower=0).to_numpy()
    out[config.LABEL_COL] = df[config.LABEL_COL].to_numpy()
    out["is_attack"] = (df[config.LABEL_COL] != config.BENIGN_LABEL).astype("int8").to_numpy()
    return out

def main() -> None:
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading CSV files")
    df = load_raw()
    print(f"  merged: {len(df):,} rows")

    print("[2/5] Cleaning")
    df = clean(df)
    print("  label counts after cleaning:")
    print(df[config.LABEL_COL].value_counts().to_string())

    print("[3/5] Splitting")
    splits = make_splits(df)

    print("[4/5] Selecting features and scaling")
    features = select_features(splits["train"])
    scaler = MinMaxScaler().fit(_log(splits["train"][features]))

    print("[5/5] Saving")
    summary = []
    for name, part in splits.items():
        out = build_output(part, features, scaler)
        out.to_parquet(config.PROC_DIR / f"{name}.parquet", index=False)
        for label, n in out[config.LABEL_COL].value_counts().items():
            summary.append({"split": name, "label": label, "rows": n})
    joblib.dump(scaler, config.MODEL_DIR / "scaler.joblib")
    (config.PROC_DIR / "features.json").write_text(json.dumps(features, indent=2))
    summary = pd.DataFrame(summary)
    summary.to_csv(config.PROC_DIR / "split_summary.csv", index=False)

    print("\nRows per split:")
    print(summary.groupby("split")["rows"].sum().to_string())
    print(f"\nFeatures used: {len(features)}")
    print(f"Files saved in {config.PROC_DIR}")


if __name__ == "__main__":
    main()