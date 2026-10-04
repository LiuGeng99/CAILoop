#!/usr/bin/env python3
"""Calibrate per-pred confidence thresholds from a labeled inference JSON.

Rule (v1):
  For each predicted class c, find the smallest T_c such that
      P(correct | pred=c, conf >= T_c) >= tau
  Small-n protection: if n_c < min_n, T_c = max(T_c_fit, min_T_small)
  If unreachable at any T, mark all_query (T = +inf).
  If class never predicted on calibration set, T = 1.0 (conservative).

Example:
  python calibrate_thresholds.py \\
    --infer outputs/infer_val.json \\
    --tau 0.99 --min_n 20 --min_T_small 0.99 \\
    --out outputs/thresholds_val_tau99.json
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.labels_io import load_json, save_json


ALL_QUERY_T = float("inf")


def _fit_threshold(
    pairs: Sequence[Tuple[float, bool]],
    tau: float,
) -> Tuple[float, int, Optional[float], str]:
    """Return (T, keep, prec, note)."""
    if not pairs:
        return 1.0, 0, None, "no_pred"

    confs = [c for c, _ in pairs]
    oks = [ok for _, ok in pairs]
    uniq = sorted(set(confs))

    candidates = []
    for T in uniq:
        kept = [(c, ok) for c, ok in zip(confs, oks) if c >= T]
        if not kept:
            continue
        prec = sum(1 for _, ok in kept if ok) / len(kept)
        if prec >= tau:
            candidates.append((T, len(kept), prec))

    if candidates:
        # smallest T among those meeting tau (maximize keep)
        T, keep, prec = min(candidates, key=lambda x: (x[0], -x[1]))
        return float(T), int(keep), float(prec), "ok"

    # Cannot reach tau even at highest conf
    return ALL_QUERY_T, 0, None, "unreachable_all_query"


def calibrate(
    items: List[dict],
    *,
    num_classes: int = 30,
    tau: float = 0.99,
    min_n: int = 20,
    min_T_small: float = 0.99,
) -> dict:
    by_pred: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    id2name: Dict[int, str] = {}
    for it in items:
        p = int(it["pred"])
        by_pred[p].append((float(it["confidence"]), int(it["pred"]) == int(it["true_class"])))
        id2name[int(it["true_class"])] = it.get("class_name") or f"{it['true_class']:03d}"
        id2name.setdefault(p, it.get("class_name") or f"{p:03d}")

    thresholds = {}
    for p in range(num_classes):
        pairs = by_pred.get(p, [])
        n = len(pairs)
        name = id2name.get(p, f"{p:03d}")
        T_fit, keep, prec, note = _fit_threshold(pairs, tau)
        T = T_fit
        base_acc = (sum(1 for _, ok in pairs if ok) / n) if n else None

        if note == "ok" and n < min_n:
            T = max(T_fit, min_T_small)
            kept = [(c, ok) for c, ok in pairs if c >= T]
            keep = len(kept)
            prec = (sum(1 for _, ok in kept if ok) / keep) if keep else None
            note = "ok_minT"

        if note == "no_pred":
            T = 1.0
            keep = 0
            prec = None

        thresholds[str(p)] = {
            "pred": p,
            "class_name": name,
            "n_val": n,
            "base_acc": base_acc,
            "T": None if math.isinf(T) else float(T),
            "T_all_query": math.isinf(T),
            "T_fit": None if (T_fit is None or math.isinf(T_fit)) else float(T_fit),
            "keep": keep,
            "keep_frac": (keep / n) if n else 0.0,
            "query": n - keep,
            "achieved_prec": prec,
            "note": note,
        }

    # Apply back on calibration set for summary
    n_true = n_pseudo = pseudo_ok = 0
    for it in items:
        p = int(it["pred"])
        info = thresholds[str(p)]
        use_true = info["T_all_query"] or (float(it["confidence"]) < float(info["T"] or 1.0))
        # when T is None (all_query), always true
        if info["T"] is None:
            use_true = True
        if use_true:
            n_true += 1
        else:
            n_pseudo += 1
            if int(it["pred"]) == int(it["true_class"]):
                pseudo_ok += 1

    return {
        "format": "thresholds_v1",
        "tau": tau,
        "min_n": min_n,
        "min_T_small": min_T_small,
        "num_classes": num_classes,
        "calibration_summary": {
            "num_images": len(items),
            "num_true": n_true,
            "num_pseudo": n_pseudo,
            "true_frac": n_true / len(items) if items else 0.0,
            "pseudo_precision": (pseudo_ok / n_pseudo) if n_pseudo else None,
        },
        "thresholds": thresholds,
    }


def print_table(obj: dict) -> None:
    print(
        f"{'id':>3} {'class':<22} {'n':>5} {'base%':>6} {'T':>8} "
        f"{'keep%':>7} {'q':>4} {'prec':>6}  note"
    )
    print("-" * 85)
    for p in range(obj["num_classes"]):
        r = obj["thresholds"][str(p)]
        if r["n_val"] == 0:
            print(f"{p:3d} {r['class_name']:<22} {0:>5} {'—':>6} {1.0:8.4f} {'—':>7} {'—':>4} {'—':>6}  {r['note']}")
            continue
        T_str = "  +inf" if r["T_all_query"] else f"{r['T']:8.4f}"
        ba = f"{100 * r['base_acc']:.1f}" if r["base_acc"] is not None else "  nan"
        pr = f"{r['achieved_prec']:.3f}" if r["achieved_prec"] is not None else "  nan"
        print(
            f"{p:3d} {r['class_name']:<22} {r['n_val']:>5} {ba:>5}% "
            f"{T_str} {100 * r['keep_frac']:6.1f}% {r['query']:4d} {pr:>6}  {r['note']}"
        )
    s = obj["calibration_summary"]
    print("-" * 85)
    print(
        f"calib apply: query={s['num_true']}/{s['num_images']} "
        f"({100 * s['true_frac']:.1f}%) | "
        f"pseudo_prec={s['pseudo_precision']}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--infer", type=str, required=True, help="Labeled infer JSON (e.g. infer_val.json)")
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--min_n", type=int, default=20)
    p.add_argument("--min_T_small", type=float, default=0.99)
    p.add_argument("--num_classes", type=int, default=30)
    p.add_argument(
        "--out",
        type=str,
        default="active_learning/outputs/thresholds_val_tau99.json",
    )
    args = p.parse_args()

    infer = load_json(args.infer)
    obj = calibrate(
        infer["items"],
        num_classes=args.num_classes,
        tau=args.tau,
        min_n=args.min_n,
        min_T_small=args.min_T_small,
    )
    obj["infer_path"] = args.infer
    obj["dino_ckpt"] = infer.get("dino_ckpt")
    obj["cls_ckpt"] = infer.get("cls_ckpt")
    obj["image_root"] = infer.get("image_root")
    obj["pool_stages"] = infer.get("pool_stages")

    save_json(args.out, obj)
    print_table(obj)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
