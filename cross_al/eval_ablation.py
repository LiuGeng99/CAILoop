#!/usr/bin/env python3
"""Eval a yantai ablation run at pre-declared checkpoints. Does not write official eval/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_al.common import OUT_DIR, run_dir
from cross_al.eval_case import aggregate_cases, infer_binary, load_test_items


def spec_at_sens(y, s, target=0.90):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    ok = np.where(tpr >= target)[0]
    if ok.size == 0:
        return float("nan")
    return float(1.0 - fpr[ok[np.argmin(fpr[ok])]])


def resolve_ckpt(rd: Path, name: str) -> Path:
    if name == "last":
        return rd / "last.pth"
    if name.endswith(".pth"):
        p = rd / "checkpoints" / name
        return p if p.is_file() else rd / name
    p = rd / "checkpoints" / f"{name}.pth"
    if p.is_file():
        return p
    return rd / f"{name}.pth"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--center", default="yantai")
    p.add_argument("--arm", default="al")
    p.add_argument("--run_name", required=True)
    p.add_argument("--ckpts", nargs="+", default=["ft_ep010", "last"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    rd = run_dir(args.center, args.arm, args.run_name)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df = load_test_items(args.center)
    out_dir = OUT_DIR / "eval_sweep" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    for name in args.ckpts:
        ckpt = resolve_ckpt(rd, name)
        if not ckpt.is_file():
            print(f"MISSING {ckpt}", flush=True)
            continue
        p_mal = infer_binary(
            df["path"].tolist(), ckpt, device, args.batch_size, args.num_workers
        )
        cases = aggregate_cases(df, p_mal)
        y = cases["true_mal"].to_numpy()
        s = cases["p_mal"].to_numpy()
        row = {
            "center": args.center,
            "arm": args.arm,
            "run_name": args.run_name,
            "ckpt": name,
            "ckpt_path": str(ckpt),
            "n_cases": int(len(cases)),
            "auc": float(roc_auc_score(y, s)),
            "sp90": spec_at_sens(y, s, 0.90),
            "sp95": spec_at_sens(y, s, 0.95),
        }
        cases.to_csv(out_dir / f"{args.center}_{args.run_name}_{name}.csv", index=False)
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"{args.center}/{args.run_name}/{name}  "
            f"AUC={row['auc']:.4f} Sp90={row['sp90']:.4f} Sp95={row['sp95']:.4f}",
            flush=True,
        )
    print("EVAL_DONE", args.run_name)


if __name__ == "__main__":
    main()
