#!/usr/bin/env python3
"""Image-level malignant/non-malignant eval on a folder split (default: test_holdout).

Malignant class ids: 0..6 (000_*_Ca ... 006_*_Ca), same as eval_binary_cases.py.

For each checkpoint directory containing dino_best.pth + cls_best.pth:
  1) run inference on --image_root
  2) save inference_results.csv (30-class probs + labels)
  3) print / save metrics_binary.json (30-class acc + binary metrics)

Example:
  python active_learning/eval_binary_holdout.py \\
    --ckpt_path checkpoints/al_scheme1_recal_scratch_holdoutES/upto_s10 \\
    --out_dir active_learning/outputs/binary_eval/al_s10

  # metrics only from an existing CSV:
  python active_learning/eval_binary_holdout.py \\
    --csv path/to/inference_results.csv --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.common import iter_folder_images, resolve

from cailoop.runtime import dinov2_backbone

MAL_IDS = list(range(7))  # 0..6


class PathListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["path"]).convert("RGB")
        return self.transform(img), idx


def build_model(dino_ckpt, cls_ckpt, num_classes, device):
    dino = dinov2_backbone()(pretrained=False)
    dino.load_state_dict(torch.load(dino_ckpt, map_location="cpu"))
    clf = nn.Linear(768, num_classes)
    clf.load_state_dict(torch.load(cls_ckpt, map_location="cpu"))
    dino.to(device).eval()
    clf.to(device).eval()
    return dino, clf


@torch.no_grad()
def infer_folder(image_root, dino_ckpt, cls_ckpt, num_classes, batch_size, num_workers, device):
    items = iter_folder_images(image_root)
    if not items:
        raise RuntimeError(f"No images under {image_root}")

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(device)
    dino, clf = build_model(dino_ckpt, cls_ckpt, num_classes, device)
    tfm = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        PathListDataset(items, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            logits = clf(dino(imgs))
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for j, idx in enumerate(idxs.tolist()):
            it = items[idx]
            true_c = int(it["true_class"])
            pred_c = int(preds[j])
            p = probs[j]
            row = {
                "image_path": it["path"],
                "rel_path": it.get("rel_path"),
                "case_key": it.get("case_key"),
                "true_label": true_c,
                "pred_label": pred_c,
                "confidence": float(p[pred_c]),
                "true_malignant": int(true_c in MAL_IDS),
                "pred_malignant": int(pred_c in MAL_IDS),
                "prob_malignant": float(p[MAL_IDS].sum()),
            }
            for c in range(num_classes):
                row[f"prob_{c}"] = float(p[c])
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_df(df: pd.DataFrame) -> dict:
    y30_t = df["true_label"].to_numpy()
    y30_p = df["pred_label"].to_numpy()
    yb_t = df["true_malignant"].to_numpy()
    yb_p = df["pred_malignant"].to_numpy()
    score = df["prob_malignant"].to_numpy()

    tp = int(((yb_p == 1) & (yb_t == 1)).sum())
    fp = int(((yb_p == 1) & (yb_t == 0)).sum())
    fn = int(((yb_p == 0) & (yb_t == 1)).sum())
    tn = int(((yb_p == 0) & (yb_t == 0)).sum())

    def _safe(n, d):
        return float(n / d) if d else None

    out = {
        "n_images": int(len(df)),
        "n_malignant": int((yb_t == 1).sum()),
        "n_non_malignant": int((yb_t == 0).sum()),
        "acc_30": float((y30_p == y30_t).mean()),
        "acc_binary": float((yb_p == yb_t).mean()),
        "sensitivity_malignant": _safe(tp, tp + fn),
        "specificity_non_malignant": _safe(tn, tn + fp),
        "precision_malignant": _safe(tp, tp + fp),
        "f1_malignant": None,
        "fn_rate_malignant": _safe(fn, tp + fn),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    sens = out["sensitivity_malignant"]
    prec = out["precision_malignant"]
    if sens is not None and prec is not None and (sens + prec) > 0:
        out["f1_malignant"] = float(2 * prec * sens / (prec + sens))

    # optional AUROC if both classes present
    if out["n_malignant"] > 0 and out["n_non_malignant"] > 0:
        try:
            from sklearn.metrics import roc_auc_score

            out["auroc_malignant"] = float(roc_auc_score(yb_t, score))
        except Exception:
            out["auroc_malignant"] = None
    else:
        out["auroc_malignant"] = None

    # simple case-level (any-malignant)
    if "case_key" in df.columns and df["case_key"].notna().any():
        g = df.dropna(subset=["case_key"]).groupby("case_key")
        case_true = g["true_malignant"].max().to_numpy()
        case_pred = g["pred_malignant"].max().to_numpy()
        out["n_cases"] = int(len(case_true))
        out["acc_binary_case_any"] = float((case_pred == case_true).mean())
        ctp = int(((case_pred == 1) & (case_true == 1)).sum())
        cfn = int(((case_pred == 0) & (case_true == 1)).sum())
        out["sensitivity_malignant_case_any"] = _safe(ctp, ctp + cfn)
    return out


def main():
    p = argparse.ArgumentParser(description="Holdout binary (mal/non-mal) eval for a ckpt")
    p.add_argument("--ckpt_path", type=str, default=None, help="Dir with dino_best.pth + cls_best.pth")
    p.add_argument("--csv", type=str, default=None, help="Existing inference_results.csv (skip infer)")
    p.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Folder split for inference (default: test_holdout)",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--tag", type=str, default="", help="Optional name stored in metrics json")
    args = p.parse_args()
    if args.image_root is None:
        args.image_root = resolve("holdout_root")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        df = pd.read_csv(args.csv)
        if "true_malignant" not in df.columns:
            df["true_malignant"] = df["true_label"].astype(int).isin(MAL_IDS).astype(int)
        if "pred_malignant" not in df.columns:
            df["pred_malignant"] = df["pred_label"].astype(int).isin(MAL_IDS).astype(int)
        if "prob_malignant" not in df.columns:
            cols = [f"prob_{i}" for i in MAL_IDS if f"prob_{i}" in df.columns]
            if not cols:
                raise ValueError("CSV needs prob_0..prob_6 or prob_malignant")
            df["prob_malignant"] = df[cols].sum(axis=1)
        csv_path = out_dir / "inference_results.csv"
        if Path(args.csv).resolve() != csv_path.resolve():
            df.to_csv(csv_path, index=False)
    else:
        if not args.ckpt_path:
            raise ValueError("Provide --ckpt_path or --csv")
        ckpt = Path(args.ckpt_path)
        dino_ckpt = ckpt / "dino_best.pth"
        cls_ckpt = ckpt / "cls_best.pth"
        if not dino_ckpt.exists() or not cls_ckpt.exists():
            raise FileNotFoundError(f"Missing {dino_ckpt} or {cls_ckpt}")
        print(f"=> Infer on {args.image_root}")
        print(f"=> ckpt {ckpt}")
        df = infer_folder(
            args.image_root,
            str(dino_ckpt),
            str(cls_ckpt),
            args.num_classes,
            args.batch_size,
            args.num_workers,
            args.device,
        )
        csv_path = out_dir / "inference_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"=> wrote {csv_path}")

    metrics = metrics_from_df(df)
    metrics["tag"] = args.tag or out_dir.name
    metrics["ckpt_path"] = args.ckpt_path
    metrics["image_root"] = args.image_root
    metrics["csv"] = str(out_dir / "inference_results.csv")

    metrics_path = out_dir / "metrics_binary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=> Metrics:")
    for k in [
        "n_images",
        "acc_30",
        "acc_binary",
        "sensitivity_malignant",
        "specificity_non_malignant",
        "precision_malignant",
        "f1_malignant",
        "fn_rate_malignant",
        "auroc_malignant",
        "acc_binary_case_any",
        "sensitivity_malignant_case_any",
    ]:
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
    print(f"=> wrote {metrics_path}")


if __name__ == "__main__":
    main()
