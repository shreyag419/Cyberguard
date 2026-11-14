
import numpy as np

import config

SET_NAMES = ["Low", "Medium", "High"]
RISK_NAMES = ["Normal", "Suspicious", "High Risk", "Critical"]
INPUT_NAMES = ["anomaly", "packet_rate", "conn_flags"]
INPUT_LABELS = {
    "anomaly": "Anomaly score (ANN)",
    "packet_rate": "Packet rate: log10(1 + packets/s)",
    "conn_flags": "SYN + RST flag count",
}

# range each input can take (also the GA search bounds)
UNIVERSE = {"anomaly": (0.0, 1.0), "packet_rate": (0.0, 7.0), "conn_flags": (0.0, 4.0)}

# hand-chosen starting shapes (before the GA improves them)
DEFAULT_PARAMS = {
    "anomaly": (0.30, 0.45, 0.65, 0.85),
    "packet_rate": (2.0, 3.0, 4.0, 5.0),     # about 100, 1,000, 10,000, 100,000 packets/s
    "conn_flags": (0.0, 0.5, 1.5, 2.0),      # 0 flags = Low, 1 = Medium, 2+ = High
}

# ---- output side (fixed, not tuned): four overlapping sets on 0-100 ----
RISK_GRID = np.linspace(0, 100, 101)
OUTPUT_SETS = np.vstack([
    np.interp(RISK_GRID, [12.5, 37.5], [1, 0]),              # Normal
    np.interp(RISK_GRID, [12.5, 37.5, 62.5], [0, 1, 0]),     # Suspicious
    np.interp(RISK_GRID, [37.5, 62.5, 87.5], [0, 1, 0]),     # High Risk
    np.interp(RISK_GRID, [62.5, 87.5], [0, 1]),              # Critical
])


# --------------------------------------------------------------------------
# Rule base (expert-written, fixed). 27 rules with the anomaly score, 9 without.
# --------------------------------------------------------------------------
def _risk_index(s: int, three_inputs: bool) -> int:
    """Turn a 'danger points' total into a risk level (0 Normal ... 3 Critical)."""
    if three_inputs:                    # s = 3*anomaly + packet_rate + conn_flags, each level 0/1/2
        return 0 if s <= 1 else 1 if s <= 4 else 2 if s <= 6 else 3
    return 0 if s <= 1 else s - 1       # s = packet_rate + conn_flags


def build_rules(use_anomaly: bool = True):
    """List of (anomaly_level, rate_level, flags_level, risk_level); levels 0=Low 1=Medium 2=High."""
    rules = []
    for a in (range(3) if use_anomaly else [None]):
        for p in range(3):
            for f in range(3):
                s = (3 * a if use_anomaly else 0) + p + f
                rules.append((a, p, f, _risk_index(s, use_anomaly)))
    return rules


RULES_FULL = build_rules(True)      # ANN + Fuzzy
RULES_ALONE = build_rules(False)    # Fuzzy alone (no anomaly score available)


def describe_rules(use_anomaly: bool = True):
    """The rules as readable sentences (for the report and viva)."""
    out = []
    for a, p, f, k in (RULES_FULL if use_anomaly else RULES_ALONE):
        parts = ([f"anomaly is {SET_NAMES[a]}"] if a is not None else []) + \
                [f"packet rate is {SET_NAMES[p]}", f"SYN+RST flags are {SET_NAMES[f]}"]
        out.append(f"IF {' AND '.join(parts)} THEN risk is {RISK_NAMES[k]}")
    return out


def categorize(risk) -> np.ndarray:
    """Risk score -> category name using config.FUZZY_CATEGORY_EDGES."""
    idx = np.digitize(np.asarray(risk, dtype=float), config.FUZZY_CATEGORY_EDGES, right=True)
    return np.array(RISK_NAMES)[idx]


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class FuzzyRiskEngine:
    def __init__(self, params=None):
        params = params or DEFAULT_PARAMS
        # sorting repairs any ordering mistakes a GA mutation might create
        self.params = {n: tuple(sorted(float(x) for x in params[n])) for n in INPUT_NAMES}

    # ---- GA interface ----
    def to_vector(self) -> np.ndarray:
        return np.array([v for n in INPUT_NAMES for v in self.params[n]])

    @classmethod
    def from_vector(cls, vec):
        vec = np.asarray(vec, dtype=float)
        return cls({n: vec[4 * i:4 * i + 4] for i, n in enumerate(INPUT_NAMES)})

    @staticmethod
    def bounds():
        lo = np.array([UNIVERSE[n][0] for n in INPUT_NAMES for _ in range(4)])
        hi = np.array([UNIVERSE[n][1] for n in INPUT_NAMES for _ in range(4)])
        return lo, hi

    # ---- step 1: fuzzification ----
    def memberships(self, name: str, x) -> np.ndarray:
        """Degrees of Low / Medium / High for each value in x -> array (N, 3), rows sum to 1."""
        p1, p2, p3, p4 = self.params[name]
        x = np.asarray(x, dtype=float)
        low = np.clip((p2 - x) / max(p2 - p1, 1e-9), 0, 1)
        high = np.clip((x - p3) / max(p4 - p3, 1e-9), 0, 1)
        medium = np.clip(1 - low - high, 0, 1)
        return np.column_stack([low, medium, high])

    # ---- steps 2-4: rules, aggregation, defuzzification ----
    def _assess_chunk(self, pr, cf, an):
        mu_p = self.memberships("packet_rate", pr)
        mu_f = self.memberships("conn_flags", cf)
        mu_a = None if an is None else self.memberships("anomaly", an)

        # strength of each risk level = strongest rule pointing at it (AND = min, combine = max)
        strength = np.zeros((len(pr), len(RISK_NAMES)))
        for a, p, f, k in (RULES_ALONE if mu_a is None else RULES_FULL):
            s = np.minimum(mu_p[:, p], mu_f[:, f])
            if a is not None:
                s = np.minimum(s, mu_a[:, a])
            strength[:, k] = np.maximum(strength[:, k], s)

        # clip each output set at its strength, merge them, take the centre of mass
        merged = np.zeros((len(pr), len(RISK_GRID)))
        for k in range(len(RISK_NAMES)):
            np.maximum(merged, np.minimum(strength[:, k:k + 1], OUTPUT_SETS[k][None, :]), out=merged)
        return merged @ RISK_GRID / np.maximum(merged.sum(axis=1), 1e-12)

    def assess(self, packets_per_s, syn, rst, anomaly=None, chunk: int = 50_000) -> np.ndarray:
        """Risk score 0-100 for each flow. Leave anomaly=None for the fuzzy-alone configuration."""
        pr = np.log10(1 + np.clip(np.asarray(packets_per_s, dtype=float), 0, None))
        cf = np.clip(np.asarray(syn, dtype=float) + np.asarray(rst, dtype=float), *UNIVERSE["conn_flags"])
        an = None if anomaly is None else np.asarray(anomaly, dtype=float)
        risk = np.empty(len(pr))
        for i in range(0, len(pr), chunk):
            sl = slice(i, i + chunk)
            risk[sl] = self._assess_chunk(pr[sl], cf[sl], None if an is None else an[sl])
        return risk

    def assess_frame(self, df, use_anomaly: bool = True) -> np.ndarray:
        """Convenience: df is a scores_*.parquet table from train_ann.py."""
        return self.assess(df["raw_Flow Packets/s"], df["raw_SYN Flag Count"], df["raw_RST Flag Count"],
                           df["anomaly_score"] if use_anomaly else None)


# --------------------------------------------------------------------------
# Graph helper (also used later to compare default vs GA-tuned shapes)
# --------------------------------------------------------------------------
def plot_membership_functions(engines: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#0072B2", "#E69F00", "#D55E00"]
    styles = ["-", "--", ":"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, name in zip(axes[:3], INPUT_NAMES):
        lo, hi = UNIVERSE[name]
        x = np.linspace(lo, hi, 600)
        for e, (ename, eng) in enumerate(engines.items()):
            mu = eng.memberships(name, x)
            for j, sname in enumerate(SET_NAMES):
                label = sname if len(engines) == 1 else f"{sname} ({ename})"
                ax.plot(x, mu[:, j], color=colours[j], ls=styles[e], lw=2, label=label)
        ax.set_title(INPUT_LABELS[name], fontsize=10, loc="left")
        ax.set_ylim(-0.03, 1.05); ax.legend(frameon=False, fontsize=8)
    ocol = ["#56B4E9", "#E8C100", "#E69F00", "#D55E00"]
    for k, rname in enumerate(RISK_NAMES):
        axes[3].plot(RISK_GRID, OUTPUT_SETS[k], color=ocol[k], lw=2, label=rname)
    for edge in config.FUZZY_CATEGORY_EDGES:
        axes[3].axvline(edge, color="gray", ls=":", lw=1)
    axes[3].set_title("Risk output (0-100)", fontsize=10, loc="left")
    axes[3].legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
        ax.set_ylabel("Membership degree" if ax is axes[0] else "")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)