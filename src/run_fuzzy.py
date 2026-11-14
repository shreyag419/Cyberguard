
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import metrics
from fuzzy_engine import (FuzzyRiskEngine, RISK_NAMES, categorize, describe_rules,
                          plot_membership_functions)

C_NORMAL, C_KNOWN, C_UNSEEN = "#0072B2", "#D55E00", "#009E73"
CAT_COLOURS = ["#56B4E9", "#E8C100", "#E69F00", "#D55E00"]


def load_scores():
    names = ["val", "test", "unseen"]
    paths = [config.PROC_DIR / f"scores_{n}.parquet" for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Run  python src/train_ann.py  first "
                                "(or ask Member 1 for the scores_*.parquet files).")
    return {n: pd.read_parquet(p) for n, p in zip(names, paths)}


def add_risk(df, engine):
    df = df.copy()
    df["risk_fuzzy_alone"] = engine.assess_frame(df, use_anomaly=False)
    df["risk_ann_fuzzy"] = engine.assess_frame(df, use_anomaly=True)
    return df


def comparison_table(eval_sets):
    rows = []
    for set_name, d in eval_sets.items():
        configs = {
            "ANN alone": (d.anomaly_score, d.anomaly_score >= config.ANN_ALERT_THRESHOLD),
            "Fuzzy alone": (d.risk_fuzzy_alone, d.risk_fuzzy_alone > config.FUZZY_FLAG_RISK),
            "ANN + Fuzzy (untuned)": (d.risk_ann_fuzzy, d.risk_ann_fuzzy > config.FUZZY_FLAG_RISK),
        }
        for name, (score, flagged) in configs.items():
            rows.append({"eval_set": set_name, "configuration": name,
                         **metrics.binary_metrics(d.is_attack, flagged, score)})
    return pd.DataFrame(rows)


def category_table(sets):
    rows = []
    for set_name, d in sets.items():
        cats = pd.Series(categorize(d.risk_ann_fuzzy), index=d.index)
        for label, idx in d.groupby(config.LABEL_COL).groups.items():
            share = cats.loc[idx].value_counts(normalize=True) * 100
            rows.append({"group": set_name, "label": label, "rows": len(idx),
                         **{c: round(float(share.get(c, 0.0)), 2) for c in RISK_NAMES}})
    return pd.DataFrame(rows)


def plot_risk_hist(test, unseen):
    groups = [("Normal (test)", test[test.is_attack == 0].risk_ann_fuzzy, C_NORMAL),
              ("Known attacks (test)", test[test.is_attack == 1].risk_ann_fuzzy, C_KNOWN),
              ("Unseen attacks", unseen.risk_ann_fuzzy, C_UNSEEN)]
    bins = np.linspace(0, 100, 51)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals, col in groups:
        ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.3, color=col)
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.8, color=col, label=name)
    for edge in config.FUZZY_CATEGORY_EDGES:
        ax.axvline(edge, color="gray", ls=":", lw=1)
    ax.set_xlabel("Fuzzy risk score (ANN + Fuzzy, untuned)"); ax.set_ylabel("Density")
    ax.set_title("Risk score distribution", fontsize=12, loc="left")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "fuzzy_risk_histogram.png", dpi=150); plt.close(fig)


def plot_categories(cat):
    d = cat.copy()
    d["name"] = np.where(d.group.str.startswith("unseen"), d.label + " (unseen)", d.label)
    d = d.sort_values("Normal", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 2))
    left = np.zeros(len(d))
    for c, col in zip(RISK_NAMES, CAT_COLOURS):
        ax.barh(d["name"], d[c], left=left, color=col, label=c)
        left += d[c].to_numpy()
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("% of flows"); ax.set_title("Risk category by traffic type (ANN + Fuzzy, untuned)",
                                              fontsize=12, loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "fuzzy_category_by_label.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Loading anomaly scores from Phase 2")
    scores = load_scores()

    print("[2/4] Running the fuzzy engine (default membership shapes)")
    engine = FuzzyRiskEngine()
    t0 = time.perf_counter()
    data = {n: add_risk(d, engine) for n, d in scores.items()}
    n_rows = sum(len(d) for d in data.values())
    ms = (time.perf_counter() - t0) * 1000 / n_rows * 1000 / 2     # two assessments per row
    print(f"  speed: {ms:.1f} ms per 1000 flows per assessment")

    print("[3/4] Comparing ANN alone, Fuzzy alone and ANN + Fuzzy")
    test, unseen = data["test"], data["unseen"]
    eval_sets = {"test (known attacks)": test,
                 "zero-day (unseen attacks)": pd.concat([test[test.is_attack == 0], unseen])}
    comp = comparison_table(eval_sets)
    comp.to_csv(config.REPORT_DIR / "fuzzy_comparison.csv", index=False)
    cat = category_table({"test": test, "unseen": unseen})
    cat.to_csv(config.REPORT_DIR / "fuzzy_category_by_label.csv", index=False)

    print("[4/4] Drawing graphs and writing the rule list")
    plot_membership_functions({"default": engine}, config.REPORT_DIR / "fuzzy_membership_functions.png")
    plot_risk_hist(test, unseen)
    plot_categories(cat)
    (config.REPORT_DIR / "fuzzy_rules.txt").write_text(
        "ANN + Fuzzy rules (27)\n" + "\n".join(describe_rules(True)) +
        "\n\nFuzzy-alone rules (9)\n" + "\n".join(describe_rules(False)))

    pd.set_option("display.width", 200)
    cols = ["configuration", "precision", "detection_rate", "f1", "false_positive_rate", "false_negative_rate", "auc"]
    for set_name in eval_sets:
        print(f"\n=== {set_name} ===")
        print(comp[comp.eval_set == set_name][cols].round(4).to_string(index=False))
    print("\nShare of each traffic type per risk category (%), ANN + Fuzzy:")
    print(cat.drop(columns="group").to_string(index=False))
    print(f"\nGraphs and tables saved in {config.REPORT_DIR}")


if __name__ == "__main__":
    main()

