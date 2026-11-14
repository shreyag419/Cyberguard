
import json
import time

import matplotlib
matplotlib.use("Agg")                     
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, roc_auc_score, roc_curve

import config
from ann_model import AnomalyScorer

# colour-blind-safe palette
C_NORMAL, C_KNOWN, C_UNSEEN = "#0072B2", "#D55E00", "#009E73"
SPLITS = ["train", "val", "test", "unseen"]


def load_splits():
    needed = [config.PROC_DIR / f"{s}.parquet" for s in SPLITS] + [config.PROC_DIR / "features.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Run  python src/data_prep.py  first.")
    features = json.loads((config.PROC_DIR / "features.json").read_text())
    return features, {s: pd.read_parquet(config.PROC_DIR / f"{s}.parquet") for s in SPLITS}


def save_scores(name, df, scorer):
    """Save scores next to the raw fuzzy inputs and labels."""
    keep = ["raw_" + c for c in config.FUZZY_COLS] + [config.LABEL_COL, "is_attack"]
    out = df[keep].copy()
    err = scorer.errors(df[scorer.features].to_numpy())
    out["recon_error"] = err.astype("float32")
    out["anomaly_score"] = (err / (err + scorer.ref_error)).astype("float32")
    out.to_parquet(config.PROC_DIR / f"scores_{name}.parquet", index=False)
    return out


# --------------------------------------------------------------------------
# Metrics (all computed from the real scores, nothing hard-coded)
# --------------------------------------------------------------------------
def detection_table(frame, split_name, thr):
    rows = []
    for label, g in frame[frame.is_attack == 1].groupby(config.LABEL_COL):
        hit = int((g.anomaly_score >= thr).sum())
        rows.append({"split": split_name, "label": label, "rows": len(g),
                     "detected": hit, "detection_rate": hit / len(g)})
    return pd.DataFrame(rows)


def compute_metrics(scores, scorer, ms_per_1000):
    thr = config.ANN_ALERT_THRESHOLD
    test, unseen = scores["test"], scores["unseen"]
    benign_test = test[test.is_attack == 0]

    pred = (test.anomaly_score >= thr).astype(int)
    y = test.is_attack
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())

    zd = pd.concat([benign_test, unseen])            # zero-day test = normal test traffic + unseen attacks
    zd_hit = int((unseen.anomaly_score >= thr).sum())

    summary = {
        "epochs_trained": len(scorer.history["val"]),
        "best_epoch": scorer.history["best_epoch"],
        "best_val_mse": min(scorer.history["val"]),
        "alert_threshold": thr,
        "test_auc": roc_auc_score(y, test.anomaly_score),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_detection_rate_recall": tp / max(tp + fn, 1),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_false_positive_rate": fp / max(fp + tn, 1),
        "test_false_negative_rate": fn / max(fn + tp, 1),
        "unseen_auc": roc_auc_score(zd.is_attack, zd.anomaly_score),
        "unseen_detection_rate": zd_hit / max(len(unseen), 1),
        "ms_per_1000_flows": ms_per_1000,
    }
    per_attack = pd.concat([detection_table(test, "test (known)", thr),
                            detection_table(unseen, "unseen (zero-day)", thr)])
    return summary, per_attack


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------
def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_training(scorer):
    h = scorer.history
    ep = np.arange(1, len(h["val"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, h["train"], color=C_NORMAL, lw=2, label="training (normal traffic)")
    ax.plot(ep, h["val"], color=C_KNOWN, lw=2, label="validation (normal traffic)")
    ax.axvline(h["best_epoch"], color="gray", ls="--", lw=1)
    ax.text(h["best_epoch"], ax.get_ylim()[1], " best epoch", va="top", fontsize=9, color="gray")
    ax.set_yscale("log"); ax.legend(frameon=False)
    _style(ax, "Epoch", "Reconstruction error (MSE, log scale)", "Autoencoder learning curve")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "ann_training_curve.png", dpi=150); plt.close(fig)


def plot_error_hist(scores, scorer):
    sets = [("Normal (test)", scores["test"][scores["test"].is_attack == 0], C_NORMAL),
            ("Known attacks (test)", scores["test"][scores["test"].is_attack == 1], C_KNOWN),
            ("Unseen attacks", scores["unseen"], C_UNSEEN)]
    logs = [np.log10(s.recon_error.to_numpy() + 1e-12) for _, s, _ in sets]
    bins = np.linspace(min(l.min() for l in logs), max(l.max() for l in logs), 80)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (name, _, col), l in zip(sets, logs):
        ax.hist(l, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=col)
        ax.hist(l, bins=bins, density=True, histtype="step", lw=1.8, color=col, label=name)
    ax.axvline(np.log10(scorer.ref_error), color="black", ls="--", lw=1)
    ax.text(np.log10(scorer.ref_error), ax.get_ylim()[1], " score = 0.5", va="top", fontsize=9)
    ax.legend(frameon=False)
    _style(ax, "log10 reconstruction error", "Density", "Attacks are rebuilt worse than normal traffic")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "ann_error_histogram.png", dpi=150); plt.close(fig)


def plot_roc(scores):
    test, unseen = scores["test"], scores["unseen"]
    zd = pd.concat([test[test.is_attack == 0], unseen])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, d, col in [("Known attacks (test)", test, C_KNOWN), ("Unseen attacks (zero-day)", zd, C_UNSEEN)]:
        fpr, tpr, _ = roc_curve(d.is_attack, d.anomaly_score)
        ax.plot(fpr, tpr, color=col, lw=2.2,
                label=f"{name}  AUC={roc_auc_score(d.is_attack, d.anomaly_score):.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "False positive rate", "Detection rate", "ROC curve (anomaly score only)")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "ann_roc.png", dpi=150); plt.close(fig)


def plot_detection(per_attack, fpr):
    d = per_attack.sort_values("detection_rate")
    cols = [C_UNSEEN if s.startswith("unseen") else C_KNOWN for s in d.split]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.8))
    ax.barh(d.label, d.detection_rate * 100, color=cols)
    for y, v in enumerate(d.detection_rate * 100):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 112)
    _style(ax, f"% of flows flagged (score >= {config.ANN_ALERT_THRESHOLD})", "",
           f"Detection by attack type  (false alarms on normal traffic: {fpr * 100:.1f}%)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_KNOWN), plt.Rectangle((0, 0), 1, 1, color=C_UNSEEN)],
              labels=["Known attack", "Unseen (zero-day)"], frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "ann_detection_by_attack.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading Phase 1 data")
    features, data = load_splits()
    val = data["val"]
    print(f"  {len(features)} features | train {len(data['train']):,} normal rows | "
          f"validation normal rows {int((val.is_attack == 0).sum()):,}")

    print("[2/4] Training the autoencoder (normal traffic only)")
    scorer = AnomalyScorer.fit(features, data["train"][features].to_numpy(),
                               val.loc[val.is_attack == 0, features].to_numpy())
    scorer.save()
    print(f"  best epoch {scorer.history['best_epoch']}, reference error {scorer.ref_error:.6f}")

    print("[3/4] Scoring every split")
    scores = {}
    for name in ["val", "test", "unseen"]:
        t0 = time.perf_counter()
        scores[name] = save_scores(name, data[name], scorer)
        if name == "test":
            ms = (time.perf_counter() - t0) * 1000 / len(data[name]) * 1000
    print(f"  scoring speed: {ms:.1f} ms per 1000 flows")

    print("[4/4] Evaluating and drawing graphs")
    summary, per_attack = compute_metrics(scores, scorer, ms)
    pd.Series(summary).rename("value").to_csv(config.REPORT_DIR / "ann_summary.csv", index_label="metric")
    per_attack.to_csv(config.REPORT_DIR / "ann_per_attack.csv", index=False)
    plot_training(scorer); plot_error_hist(scores, scorer)
    plot_roc(scores); plot_detection(per_attack, summary["test_false_positive_rate"])

    print("\n=== ANN-only results (score >= %.2f counts as flagged) ===" % config.ANN_ALERT_THRESHOLD)
    for k, v in summary.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")
    print("\nDetection by attack type:")
    print(per_attack.assign(detection_rate=lambda d: (d.detection_rate * 100).round(1)).to_string(index=False))
    print(f"\nModel saved in {config.MODEL_DIR}; graphs and tables in {config.REPORT_DIR}")


if __name__ == "__main__":
    main()