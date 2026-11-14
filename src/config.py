"""
config.py - every setting for Phase 1 lives here.
Change a value here instead of hunting through the code.
"""
from pathlib import Path

# ---------- Folders ----------
ROOT = Path(__file__).resolve().parent.parent      # the cyberguard/ folder
RAW_DIR = ROOT / "data" / "raw"                    # the 8 original CSVs
PROC_DIR = ROOT / "data" / "processed"             # cleaned + scaled splits
MODEL_DIR = ROOT / "models"                        # scaler + autoencoder
REPORT_DIR = ROOT / "reports"                      # graphs and metric tables (safe to commit)

# ---------- Reproducibility ----------
RANDOM_SEED = 42

# ---------- Labels ----------
LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"

# Attack types held out as "zero-day" test data. The autoencoder and the GA
# must NEVER see these. Names must match the Label values exactly
# (data_prep.py prints all label names if one is not found).
UNSEEN_ATTACKS = ["PortScan", "DoS Hulk"]

# ---------- Sampling (keeps training fast on a laptop) ----------
BENIGN_SAMPLE = 300_000      # max benign rows kept
MAX_PER_ATTACK = 10_000      # max rows kept per attack type

# ---------- Splits ----------
BENIGN_SPLIT = (0.70, 0.15, 0.15)   # benign -> train / validation / test
KNOWN_ATTACK_VAL_FRACTION = 0.5     # known attacks -> half validation, half test

# ---------- Features ----------
# Raw (unscaled) copies of these are saved for the fuzzy engine, and they are
# never removed by feature selection.
#   packet rate               -> Flow Packets/s
#   failed-connection proxy   -> SYN Flag Count, RST Flag Count
FUZZY_COLS = ["Flow Packets/s", "SYN Flag Count", "RST Flag Count"]

CORR_THRESHOLD = 0.95        # drop a feature if |correlation| with a kept one is above this
CORR_SAMPLE = 100_000        # rows used to compute correlations (speed)

# ---------- Phase 2: autoencoder ----------
ANN_HIDDEN = (32, 16, 8, 16, 32)   # mirror shape; the middle layer (8) is the "bottleneck"
ANN_BATCH_SIZE = 256
ANN_LEARNING_RATE = 1e-3
ANN_MAX_EPOCHS = 60                # upper limit; training stops earlier if validation stops improving
ANN_PATIENCE = 6                   # stop after this many epochs without improvement

# Anomaly score = error / (error + ref_error), where ref_error is this percentile
# of the error on NORMAL validation traffic. A score of 0.5 therefore means
# "as strange as the strangest 1% of normal traffic".
ANOMALY_REF_PERCENTILE = 99
ANN_ALERT_THRESHOLD = 0.5          # score at or above this counts as "flagged" in ANN-only metrics

# ---------- Phase 2/3: fuzzy risk engine (Member 2) ----------
FUZZY_CATEGORY_EDGES = (25, 50, 75)   # risk 0-25 Normal, 26-50 Suspicious, 51-75 High Risk, 76-100 Critical
FUZZY_FLAG_RISK = 25                  # risk above this (anything but Normal) is flagged as an alert