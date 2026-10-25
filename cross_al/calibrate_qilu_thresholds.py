#!/usr/bin/env python3
"""Freeze per-pred thresholds from the v2_0204 Qilu val inference.

Uses the existing infer JSON (no GPU). Same rule as active_learning:
smallest T_c with P(correct | pred=c, conf>=T_c) >= tau.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.calibrate_thresholds import calibrate, print_table
from active_learning.labels_io import load_json, save_json
from cross_al.common import (
    INFER_VAL_V2_0204,
    MIN_N,
    MIN_T_SMALL,
    NUM_CLASSES_30,
    TAU,
    THRESHOLDS_JSON,
    V2_0204_CLS,
    V2_0204_DINO,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--infer", type=str, default=str(INFER_VAL_V2_0204))
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--min_n", type=int, default=MIN_N)
    p.add_argument("--min_T_small", type=float, default=MIN_T_SMALL)
    p.add_argument("--out", type=str, default=str(THRESHOLDS_JSON))
    args = p.parse_args()

    infer = load_json(args.infer)
    dino = infer.get("dino_ckpt")
    cls = infer.get("cls_ckpt")
    if dino and Path(dino).resolve() != V2_0204_DINO.resolve():
        print(f"WARNING: infer dino_ckpt is {dino}, expected {V2_0204_DINO}")
    if cls and Path(cls).resolve() != V2_0204_CLS.resolve():
        print(f"WARNING: infer cls_ckpt is {cls}, expected {V2_0204_CLS}")

    obj = calibrate(
        infer["items"],
        num_classes=NUM_CLASSES_30,
        tau=args.tau,
        min_n=args.min_n,
        min_T_small=args.min_T_small,
    )
    obj["infer_path"] = args.infer
    obj["dino_ckpt"] = infer.get("dino_ckpt")
    obj["cls_ckpt"] = infer.get("cls_ckpt")
    obj["image_root"] = infer.get("image_root")
    obj["frozen"] = True
    obj["note"] = (
        "Calibrated once on Qilu val_new with dinov2_exp_v2_0204. "
        "Do not recalibrate on external centers."
    )
    save_json(args.out, obj)
    print_table(obj)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
