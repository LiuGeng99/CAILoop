#!/usr/bin/env python3
"""Apply a labeling strategy on inference results to build AL training labels.

Examples:
  # v1 recommended: per-pred thresholds calibrated on val
  python build_labels.py \\
    --infer outputs/infer_s6.json \\
    --strategy per_pred_precision_threshold \\
    --thresholds outputs/thresholds_val_tau99.json \\
    --out outputs/al_labels_s6_tau99.json

  # legacy: global lowest-confidence fraction
  python build_labels.py \\
    --infer outputs/infer_s6.json \\
    --strategy lowest_confidence_frac --frac 0.1 \\
    --out outputs/al_labels_s6.json

  # refresh pseudo but keep previous expert labels locked
  python build_labels.py \\
    --infer outputs/infer_s6_round2.json \\
    --strategy per_pred_precision_threshold \\
    --thresholds outputs/thresholds_val_tau99.json \\
    --lock_true_from outputs/al_labels_s6_tau99.json \\
    --out outputs/al_labels_s6_round2.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.labels_io import (
    build_label_manifest,
    load_json,
    locked_true_paths,
    save_json,
    summarize_labels,
)
from active_learning.strategies import get_strategy


def main():
    p = argparse.ArgumentParser(description="Build AL labels from inference + strategy")
    p.add_argument("--infer", type=str, required=True, help="infer_v1 JSON from infer_pseudo.py")
    p.add_argument(
        "--strategy",
        type=str,
        default="per_pred_precision_threshold",
        help=(
            "lowest_confidence_frac | per_pred_precision_threshold | "
            "per_pred_precision_threshold_no_pseudo | random_matched_threshold | "
            "random_matched_n | all_pseudo"
        ),
    )
    p.add_argument(
        "--frac",
        type=float,
        default=0.1,
        help="For lowest_confidence_frac only",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for random_matched_threshold / random_matched_n",
    )
    p.add_argument(
        "--n_query",
        type=int,
        default=None,
        help="For random_matched_n: number of new expert labels this round",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="thresholds_v1 JSON from calibrate_thresholds.py",
    )
    p.add_argument(
        "--lock_true_from",
        type=str,
        default=None,
        help="Previous al_labels JSON; paths with source=true stay expert-labeled",
    )
    p.add_argument(
        "--out",
        type=str,
        default="active_learning/outputs/al_labels.json",
    )
    args = p.parse_args()

    infer = load_json(args.infer)
    prev = load_json(args.lock_true_from) if args.lock_true_from else None
    locked = locked_true_paths(prev)

    strategy = get_strategy(args.strategy)
    strategy_kwargs = {}
    call_kwargs = {"locked_true": locked}

    needs_thresholds = args.strategy in {
        "per_pred_precision_threshold",
        "per_pred_precision_threshold_no_pseudo",
        "random_matched_threshold",
    }
    if args.strategy == "lowest_confidence_frac":
        strategy_kwargs["frac"] = args.frac
        call_kwargs["frac"] = args.frac
    elif args.strategy == "random_matched_n":
        if args.n_query is None:
            raise ValueError("--n_query is required for random_matched_n")
        strategy_kwargs["n_query"] = args.n_query
        strategy_kwargs["seed"] = args.seed
        call_kwargs["n_query"] = args.n_query
        call_kwargs["seed"] = args.seed
    elif args.strategy == "all_pseudo":
        pass
    elif needs_thresholds:
        if not args.thresholds:
            raise ValueError(f"--thresholds is required for {args.strategy}")
        thresholds_obj = load_json(args.thresholds)
        strategy_kwargs["thresholds_path"] = args.thresholds
        strategy_kwargs["tau"] = thresholds_obj.get("tau")
        strategy_kwargs["min_n"] = thresholds_obj.get("min_n")
        strategy_kwargs["min_T_small"] = thresholds_obj.get("min_T_small")
        call_kwargs["thresholds"] = thresholds_obj
        if args.strategy == "random_matched_threshold":
            strategy_kwargs["seed"] = args.seed
            call_kwargs["seed"] = args.seed
    else:
        if args.thresholds:
            call_kwargs["thresholds"] = load_json(args.thresholds)
            strategy_kwargs["thresholds_path"] = args.thresholds
        strategy_kwargs["frac"] = args.frac
        call_kwargs["frac"] = args.frac

    labeled_items = strategy(infer["items"], **call_kwargs)

    pool_stages = infer.get("pool_stages", [])
    manifest = build_label_manifest(
        labeled_items,
        pool_stages=pool_stages if pool_stages is not None else [],
        strategy_name=args.strategy,
        strategy_kwargs=strategy_kwargs,
        infer_path=args.infer,
        ckpt_dino=infer.get("dino_ckpt"),
        ckpt_cls=infer.get("cls_ckpt"),
        prev_labels_path=args.lock_true_from,
    )
    if infer.get("image_root"):
        manifest["image_root"] = infer["image_root"]
    manifest["summary"] = summarize_labels(labeled_items)
    save_json(args.out, manifest)

    s = manifest["summary"]
    print(
        f"Wrote {args.out} | images={s['num_images']} | "
        f"true={s['num_true']} pseudo={s['num_pseudo']} "
        f"unlabeled={s.get('num_unlabeled', 0)} | "
        f"true_frac={s['true_frac']:.3f} | "
        f"pseudo_agree={s['pseudo_agree_with_true']}"
    )


if __name__ == "__main__":
    main()
