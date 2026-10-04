"""Case-level binary metrics. Same operating point and bootstrap as fig2."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

BINARY_THRESHOLD = 0.5
BOOTSTRAP_N = 100
BOOTSTRAP_SEED = 42
METRIC_NAMES = (
    "auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
)


def confusion_at_threshold(y_true, y_score, threshold=BINARY_THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / len(y_true) if len(y_true) else np.nan
    f1 = (
        2 * ppv * sens / (ppv + sens)
        if ppv == ppv and sens == sens and (ppv + sens) > 0
        else np.nan
    )
    bacc = (
        (sens + spec) / 2 if sens == sens and spec == spec else np.nan
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": bacc,
    }


def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def compute_roc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)
    return {
        "auc": float(auc(fpr, tpr)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }


def _finite(x):
    return x == x


def bootstrap_metric_ci(
    y_true,
    y_score,
    threshold=BINARY_THRESHOLD,
    n_boot=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    point = {"auc": compute_auc(y_true, y_score)}
    conf = confusion_at_threshold(y_true, y_score, threshold)
    for k in METRIC_NAMES:
        if k != "auc":
            point[k] = conf[k]

    n = len(y_true)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in METRIC_NAMES}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n) if n else np.array([], dtype=int)
        y_b, s_b = y_true[idx], y_score[idx]
        samples["auc"].append(compute_auc(y_b, s_b))
        c = confusion_at_threshold(y_b, s_b, threshold)
        for k in METRIC_NAMES:
            if k == "auc":
                continue
            if _finite(c[k]):
                samples[k].append(c[k])

    out = {}
    for k in METRIC_NAMES:
        pv = point[k]
        vals = [v for v in samples[k] if _finite(v)]
        out[k] = {
            "point": None if not _finite(pv) else float(pv),
            "ci_lower": float(np.percentile(vals, 2.5)) if vals else None,
            "ci_upper": float(np.percentile(vals, 97.5)) if vals else None,
        }
    return out
