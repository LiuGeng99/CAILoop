#!/usr/bin/env python3
"""Pluggable image-level labeling strategies for active learning.

Add new strategies by registering a function in STRATEGIES.
Each strategy receives inference items and returns labeled decisions.

Inference item fields (minimum):
  - rel_path, pred, confidence, true_class, stage
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence


StrategyFn = Callable[..., List[dict]]


def _base_row(it: dict) -> dict:
    return {
        "rel_path": it["rel_path"],
        "path": it.get("path"),
        "stage": it.get("stage"),
        "true_class": it["true_class"],
        "pred": int(it["pred"]),
        "confidence": float(it["confidence"]),
        "case_key": it.get("case_key"),
        "class_name": it.get("class_name"),
    }


def lowest_confidence_frac(
    items: Sequence[dict],
    frac: float = 0.1,
    locked_true: Sequence[str] | None = None,
    **_kwargs,
) -> List[dict]:
    """Mark lowest-`frac` confidence images (among unlocked) as true labels.

    Locked paths (already expert-labeled) always keep source=true and true_class.
    Remaining unlocked images: bottom frac -> true, others -> pseudo(pred).
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")

    locked = set(locked_true or [])
    out: List[dict] = []
    unlocked: List[dict] = []

    for it in items:
        base = _base_row(it)
        if base["rel_path"] in locked:
            base["source"] = "true"
            base["label"] = int(it["true_class"])
            out.append(base)
        else:
            unlocked.append(base)

    unlocked_sorted = sorted(unlocked, key=lambda x: (x["confidence"], x["rel_path"]))
    n_true = int(round(len(unlocked_sorted) * frac)) if unlocked_sorted else 0
    if frac > 0 and unlocked_sorted and n_true == 0:
        n_true = 1
    if frac == 0:
        n_true = 0
    if frac == 1:
        n_true = len(unlocked_sorted)

    true_set = {x["rel_path"] for x in unlocked_sorted[:n_true]}
    for it in unlocked:
        if it["rel_path"] in true_set:
            it["source"] = "true"
            it["label"] = int(it["true_class"])
        else:
            it["source"] = "pseudo"
            it["label"] = int(it["pred"])
        out.append(it)

    return out


def _should_query(pred: int, conf: float, tmap: Dict[int, dict]) -> bool:
    """True if per-pred threshold would send this image to the expert."""
    info = tmap.get(int(pred))
    if info is None:
        return True
    if info["all_query"] or info["T"] is None:
        return True
    return float(conf) < float(info["T"])


def _load_threshold_map(thresholds: Any) -> Dict[int, dict]:
    """Accept thresholds_v1 dict, or a flat {pred: T} mapping."""
    if thresholds is None:
        raise ValueError("per_pred_precision_threshold requires thresholds=...")

    if isinstance(thresholds, dict) and "thresholds" in thresholds:
        raw = thresholds["thresholds"]
        out = {}
        for k, v in raw.items():
            p = int(k)
            if isinstance(v, dict):
                out[p] = {
                    "T": None if v.get("T_all_query") else float(v["T"]),
                    "all_query": bool(v.get("T_all_query", False)),
                }
            else:
                out[p] = {"T": float(v), "all_query": False}
        return out

    # flat map pred -> T
    out = {}
    for k, v in thresholds.items():
        out[int(k)] = {"T": float(v), "all_query": False}
    return out


def per_pred_precision_threshold(
    items: Sequence[dict],
    thresholds: Any = None,
    locked_true: Sequence[str] | None = None,
    **_kwargs,
) -> List[dict]:
    """Use per-pred confidence thresholds from calibrate_thresholds.py.

    Rule:
      if locked -> true
      elif all_query[pred] or conf < T[pred] -> true
      else -> pseudo(pred)
    """
    tmap = _load_threshold_map(thresholds)
    locked = set(locked_true or [])
    out: List[dict] = []

    for it in items:
        row = _base_row(it)
        pred = row["pred"]
        if row["rel_path"] in locked:
            row["source"] = "true"
            row["label"] = int(it["true_class"])
            out.append(row)
            continue

        info = tmap.get(pred)
        if info is None:
            T_used = None
        elif info["all_query"] or info["T"] is None:
            T_used = None
        else:
            T_used = info["T"]
        use_true = _should_query(pred, row["confidence"], tmap)

        if use_true:
            row["source"] = "true"
            row["label"] = int(it["true_class"])
        else:
            row["source"] = "pseudo"
            row["label"] = pred
        row["threshold"] = T_used
        out.append(row)

    return out


def per_pred_precision_threshold_no_pseudo(
    items: Sequence[dict],
    thresholds: Any = None,
    locked_true: Sequence[str] | None = None,
    **kwargs,
) -> List[dict]:
    """Same query gate as per_pred_precision_threshold; rest is unlabeled (not trained)."""
    out = per_pred_precision_threshold(
        items, thresholds=thresholds, locked_true=locked_true, **kwargs
    )
    for row in out:
        if row.get("source") == "pseudo":
            row["source"] = "unlabeled"
    return out


def random_matched_threshold(
    items: Sequence[dict],
    thresholds: Any = None,
    locked_true: Sequence[str] | None = None,
    seed: int = 0,
    **_kwargs,
) -> List[dict]:
    """Query a random unlocked subset whose size matches the threshold gate.

    Locked expert labels are kept. Among unlocked images, n equals the number
    that per_pred_precision_threshold would query; those n get true labels,
    the rest get pseudo(pred).
    """
    tmap = _load_threshold_map(thresholds)
    locked = set(locked_true or [])
    rng = random.Random(int(seed))

    out: List[dict] = []
    unlocked: List[dict] = []

    for it in items:
        row = _base_row(it)
        if row["rel_path"] in locked:
            row["source"] = "true"
            row["label"] = int(it["true_class"])
            out.append(row)
        else:
            unlocked.append(row)

    n_query = sum(
        1 for row in unlocked if _should_query(row["pred"], row["confidence"], tmap)
    )
    n_query = min(n_query, len(unlocked))

    order = list(range(len(unlocked)))
    order.sort(key=lambda i: unlocked[i]["rel_path"])
    rng.shuffle(order)
    query_idx = set(order[:n_query])

    for i, row in enumerate(unlocked):
        if i in query_idx:
            row["source"] = "true"
            row["label"] = int(row["true_class"])
        else:
            row["source"] = "pseudo"
            row["label"] = int(row["pred"])
        out.append(row)

    return out


def _split_locked(items: Sequence[dict], locked_true: Sequence[str] | None):
    locked = set(locked_true or [])
    kept: List[dict] = []
    unlocked: List[dict] = []
    for it in items:
        row = _base_row(it)
        if row["rel_path"] in locked:
            row["source"] = "true"
            row["label"] = int(it["true_class"])
            kept.append(row)
        else:
            unlocked.append(row)
    return kept, unlocked


def _random_true_on_unlocked(
    unlocked: List[dict], n_query: int, seed: int
) -> None:
    n_query = int(n_query)
    if n_query < 0:
        raise ValueError(f"n_query must be >= 0, got {n_query}")
    n_query = min(n_query, len(unlocked))
    rng = random.Random(int(seed))
    order = list(range(len(unlocked)))
    order.sort(key=lambda i: unlocked[i]["rel_path"])
    rng.shuffle(order)
    query_idx = set(order[:n_query])
    for i, row in enumerate(unlocked):
        if i in query_idx:
            row["source"] = "true"
            row["label"] = int(row["true_class"])
        else:
            row["source"] = "pseudo"
            row["label"] = int(row["pred"])


def random_matched_n(
    items: Sequence[dict],
    n_query: int = 0,
    locked_true: Sequence[str] | None = None,
    seed: int = 0,
    **_kwargs,
) -> List[dict]:
    """Query a random unlocked subset of size n_query; rest get pseudo(pred).

    n_query is an explicit budget (e.g. AL's new expert count that round),
    not derived from this model's confidence gate.
    """
    kept, unlocked = _split_locked(items, locked_true)
    _random_true_on_unlocked(unlocked, n_query, seed)
    return kept + unlocked


def all_pseudo(
    items: Sequence[dict],
    locked_true: Sequence[str] | None = None,
    **_kwargs,
) -> List[dict]:
    """No new expert labels: every unlocked image is pseudo(pred)."""
    kept, unlocked = _split_locked(items, locked_true)
    for row in unlocked:
        row["source"] = "pseudo"
        row["label"] = int(row["pred"])
    return kept + unlocked


STRATEGIES: Dict[str, StrategyFn] = {
    "lowest_confidence_frac": lowest_confidence_frac,
    "per_pred_precision_threshold": per_pred_precision_threshold,
    "per_pred_precision_threshold_no_pseudo": per_pred_precision_threshold_no_pseudo,
    "random_matched_threshold": random_matched_threshold,
    "random_matched_n": random_matched_n,
    "all_pseudo": all_pseudo,
}


def get_strategy(name: str) -> StrategyFn:
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]

