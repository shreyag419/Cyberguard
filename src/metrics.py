"""
metrics.py - shared evaluation helpers (used by run_fuzzy.py now, and by the
GA fitness function and the final comparison later).

"flagged" means the system raised an alert. An attack that is flagged is a
true positive; a normal flow that is flagged is a false positive.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config


def binary_metrics(y_true, flagged, score=None) -> dict:
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flagged).astype(bool)
    tp = int((f & (y == 1)).sum())
    fp = int((f & (y == 0)).sum())
    tn = int((~f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0       # also called detection rate
    out = {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": precision,
        "detection_rate": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }
    if score is not None and len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, score)
    return out


def per_attack_detection(frame: pd.DataFrame, flagged, split_name: str) -> pd.DataFrame:
    """Detection rate for each attack type in `frame` (needs Label and is_attack columns)."""
    d = frame.assign(_flagged=np.asarray(flagged).astype(bool))
    rows = []
    for label, g in d[d.is_attack == 1].groupby(config.LABEL_COL):
        hit = int(g["_flagged"].sum())
        rows.append({"split": split_name, "label": label, "rows": len(g),
                     "detected": hit, "detection_rate": hit / len(g)})
    return pd.DataFrame(rows)