
from pathlib import Path

# ---------- Folders ----------
ROOT = Path(__file__).resolve().parent.parent     
RAW_DIR = ROOT / "data" / "raw"                   
PROC_DIR = ROOT / "data" / "processed"            
MODEL_DIR = ROOT / "models"                       

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