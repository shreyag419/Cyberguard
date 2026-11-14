"""
ann_model.py - the autoencoder and the anomaly score.

An autoencoder is a neural network trained to copy its input to its output
through a narrow middle layer. Trained only on NORMAL traffic, it gets good at
rebuilding normal flows and bad at rebuilding unusual ones. The rebuild error
(reconstruction error) is therefore a measure of "how strange is this flow?".

Typical use (this is also how the fuzzy engine / pipeline will call it):

    scorer = AnomalyScorer.load()
    scores = scorer.score_frame(df)     # numpy array, one 0-1 score per row
"""
import copy
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

import config

MODEL_FILE = "autoencoder.joblib"


def _row_errors(model, X: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error for every row."""
    return np.mean((X - model.predict(X)) ** 2, axis=1)


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, verbose: bool = True):
    """Train one epoch at a time, watching normal validation data.
    Returns the best model (lowest validation error) and the loss history."""
    model = MLPRegressor(
        hidden_layer_sizes=config.ANN_HIDDEN,
        activation="relu",
        solver="adam",
        batch_size=config.ANN_BATCH_SIZE,
        learning_rate_init=config.ANN_LEARNING_RATE,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )
    history = {"train": [], "val": []}
    best_model, best_val, wait = None, np.inf, 0

    for epoch in range(1, config.ANN_MAX_EPOCHS + 1):
        model.partial_fit(X_train, X_train)        # target = input: that is what makes it an autoencoder
        tr = float(_row_errors(model, X_train).mean())
        va = float(_row_errors(model, X_val).mean())
        history["train"].append(tr)
        history["val"].append(va)
        improved = va < best_val - 1e-8
        if improved:
            best_val, best_model, wait = va, copy.deepcopy(model), 0
        else:
            wait += 1
        if verbose:
            print(f"  epoch {epoch:3d}  train_mse={tr:.6f}  val_mse={va:.6f}{'  *' if improved else ''}")
        if wait >= config.ANN_PATIENCE:
            print(f"  no improvement for {config.ANN_PATIENCE} epochs, stopping")
            break

    history["best_epoch"] = int(np.argmin(history["val"])) + 1
    return best_model, history


class AnomalyScorer:
    """Wraps the trained autoencoder + the rule that turns error into a 0-1 score."""

    def __init__(self, features, model, ref_error, history=None):
        self.features = list(features)
        self.model = model
        self.ref_error = max(float(ref_error), 1e-12)
        self.history = history

    # ---- training ----
    @classmethod
    def fit(cls, features, X_train, X_val_benign, verbose=True):
        X_train = np.asarray(X_train, dtype="float64")
        X_val = np.asarray(X_val_benign, dtype="float64")
        model, history = train_autoencoder(X_train, X_val, verbose)
        # Calibrate on NORMAL validation traffic only (no attack information used)
        ref = np.percentile(_row_errors(model, X_val), config.ANOMALY_REF_PERCENTILE)
        return cls(features, model, ref, history)

    # ---- scoring ----
    def errors(self, X) -> np.ndarray:
        return _row_errors(self.model, np.asarray(X, dtype="float64"))

    def score(self, X) -> np.ndarray:
        e = self.errors(X)
        return e / (e + self.ref_error)

    def score_frame(self, df) -> np.ndarray:
        return self.score(df[self.features].to_numpy())

    # ---- saving ----
    def save(self, path=None):
        path = path or (config.MODEL_DIR / MODEL_FILE)
        joblib.dump({"features": self.features, "model": self.model,
                     "ref_error": self.ref_error, "history": self.history}, path)

    @classmethod
    def load(cls, path=None):
        d = joblib.load(path or (config.MODEL_DIR / MODEL_FILE))
        return cls(d["features"], d["model"], d["ref_error"], d.get("history"))