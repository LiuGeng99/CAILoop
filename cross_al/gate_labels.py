#!/usr/bin/env python3
"""Apply frozen Qilu thresholds to a center infer JSON and write binary AL labels.

Expert = fig2 binary GT (000_Tumor=0, 001_NonTumor=1).
Pseudo = 30-class pred mapped with 0-6 -> 0, 7-29 -> 1.
Also writes a same-n random arm manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.labels_io import load_json, save_json
from active_learning.strategies import all_pseudo, per_pred_precision_threshold, random_matched_n
from cross_al.common import (
    COHORTS,
    OUT_DIR,
    SPLIT_SEED,
    THRESHOLDS_JSON,
    infer_json,
    labels_json,
    pred30_to_binary,
)


def _binarize(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        row = dict(it)
        pred30 = int(row["pred"])
        row["pred_30"] = pred30
        row["pred_binary"] = pred30_to_binary(pred30)
        if row["source"] == "pseudo":
            row["label"] = row["pred_binary"]
        else:
            row["label"] = int(row["true_class"])
        row["pseudo_correct"] = int(row["pred_binary"] == int(row["true_class"]))
        out.append(row)
    return out


def _summarize(items: list[dict]) -> dict:
    n = len(items)
    n_true = sum(1 for it in items if it["source"] == "true")
    n_pseudo = n - n_true
    n_pseudo_ok = sum(1 for it in items if it["source"] == "pseudo" and it["pseudo_correct"])
    n_cases = len({it.get("case_key") for it in items})
    return {
        "num_images": n,
        "num_cases": n_cases,
        "num_true": n_true,
        "num_pseudo": n_pseudo,
        "true_frac": (n_true / n) if n else 0.0,
        "pseudo_agree_binary": (n_pseudo_ok / n_pseudo) if n_pseudo else None,
    }


def gate_center(center: str, thresholds: dict, seed: int) -> dict:
    infer_path = infer_json(center, "train")
    if not infer_path.is_file():
        raise FileNotFoundError(f"Missing infer JSON (run GPU infer first): {infer_path}")

    infer = load_json(infer_path)
    al_items = _binarize(
        per_pred_precision_threshold(infer["items"], thresholds=thresholds)
    )
    n_query = sum(1 for it in al_items if it["source"] == "true")
    rand_items = _binarize(
        random_matched_n(infer["items"], n_query=n_query, seed=seed)
    )
    expert_items = _binarize(all_pseudo(infer["items"]))
    for it in expert_items:
        it["source"] = "true"
        it["label"] = int(it["true_class"])

    common = {
        "format": "cross_al_labels_v1",
        "center": center,
        "num_classes": 2,
        "labeler_num_classes": 30,
        "thresholds_path": str(THRESHOLDS_JSON),
        "infer_path": str(infer_path),
        "dino_ckpt": infer.get("dino_ckpt"),
        "cls_ckpt": infer.get("cls_ckpt"),
        "image_root": infer.get("image_root"),
    }
    manifests = {
        "al": {
            **common,
            "arm": "al",
            "strategy": "per_pred_precision_threshold",
            "summary": _summarize(al_items),
            "items": al_items,
        },
        "random": {
            **common,
            "arm": "random",
            "strategy": "random_matched_n",
            "strategy_kwargs": {"n_query": n_query, "seed": seed},
            "summary": _summarize(rand_items),
            "items": rand_items,
        },
        "all_expert": {
            **common,
            "arm": "all_expert",
            "strategy": "all_expert",
            "summary": _summarize(expert_items),
            "items": expert_items,
        },
    }
    for arm, man in manifests.items():
        out = labels_json(center, arm)
        save_json(out, man)
        print(
            f"  {arm}: {out} | true={man['summary']['num_true']} "
            f"pseudo={man['summary']['num_pseudo']} "
            f"true_frac={man['summary']['true_frac']:.3f} "
            f"pseudo_agree={man['summary']['pseudo_agree_binary']}"
        )
    return manifests["al"]["summary"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--centers", nargs="*", default=[c for c, _ in COHORTS])
    p.add_argument("--thresholds", type=str, default=str(THRESHOLDS_JSON))
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = p.parse_args()

    thresholds = load_json(args.thresholds)
    rows = []
    for center in args.centers:
        print(f"===== {center} =====")
        s = gate_center(center, thresholds, seed=args.seed)
        rows.append({"center": center, **s})

    table = pd.DataFrame(rows)
    out_csv = OUT_DIR / "labels" / "gate_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
