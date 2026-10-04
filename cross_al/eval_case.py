#!/usr/bin/env python3
"""Case-level heldout eval. Official number is last epoch, never used for selection.

Original: 30-class v2_0204, P(mal) = sum p_0..p_6
Adapted arms: Linear(768,2), P(mal) = softmax[:, 0]  (0=Tumor)
Aggregation: mean over images in a case, threshold 0.5, bootstrap by case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from active_learning.labels_io import load_json, save_json
from cross_al.common import (
    COHORTS,
    MAL_IDS_30,
    OUT_DIR,
    V2_0204_CLS,
    V2_0204_DINO,
    eval_dir,
    labels_json,
    run_dir,
    split_csv,
    split_image_root,
)
from cross_al.metrics import (
    BINARY_THRESHOLD,
    bootstrap_metric_ci,
    compute_auc,
    compute_roc,
    confusion_at_threshold,
)

from cailoop.runtime import dinov2_backbone


class PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def load_test_items(center: str) -> pd.DataFrame:
    df = pd.read_csv(split_csv(center))
    df = df[(df["split"] == "test") & df["readable"]].copy()
    if df.empty:
        raise RuntimeError(f"No readable test images for {center}")
    root = split_image_root(center, "test")
    df["path"] = [
        str(root / ("000_Tumor" if lab == "Tumor" else "001_NonTumor") / name)
        for lab, name in zip(df["image_label"], df["mapped_filename"])
    ]
    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test files missing, e.g. {missing[0]}")
    return df


def test_transform():
    return transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def infer_binary(paths, ckpt_path, device, batch_size, num_workers):
    pack = torch.load(ckpt_path, map_location="cpu")
    backbone = dinov2_backbone()(pretrained=False)
    backbone.load_state_dict(pack["backbone"])
    classifier = nn.Linear(768, 2)
    classifier.load_state_dict(pack["classifier"])
    backbone.to(device).eval()
    classifier.to(device).eval()
    loader = DataLoader(
        PathDataset(paths, test_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    p_mal = [None] * len(paths)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            prob = F.softmax(classifier(backbone(images)).float(), dim=1)
            for i, idx in enumerate(indices.tolist()):
                p_mal[idx] = float(prob[i, 0].item())
    return p_mal


def infer_original(paths, device, batch_size, num_workers):
    backbone = dinov2_backbone()(pretrained=False)
    backbone.load_state_dict(torch.load(V2_0204_DINO, map_location="cpu"))
    classifier = nn.Linear(768, 30)
    classifier.load_state_dict(torch.load(V2_0204_CLS, map_location="cpu"))
    backbone.to(device).eval()
    classifier.to(device).eval()
    loader = DataLoader(
        PathDataset(paths, test_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    p_mal = [None] * len(paths)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            prob = F.softmax(classifier(backbone(images)).float(), dim=1)
            mal = prob[:, MAL_IDS_30].sum(dim=1)
            for i, idx in enumerate(indices.tolist()):
                p_mal[idx] = float(mal[i].item())
    return p_mal


def pmal_from_infer_json(df: pd.DataFrame, infer_path: Path) -> list[float]:
    infer = load_json(infer_path)
    by_name = {}
    for it in infer["items"]:
        name = Path(it["rel_path"]).name
        if "probs" not in it:
            raise ValueError(f"{infer_path} has no probs; re-run infer with --save_probs")
        by_name[name] = float(sum(it["probs"][c] for c in MAL_IDS_30))
    out = []
    for name in df["mapped_filename"]:
        if name not in by_name:
            raise KeyError(f"{name} not in {infer_path}")
        out.append(by_name[name])
    return out


def aggregate_cases(df: pd.DataFrame, p_mal: list[float]) -> pd.DataFrame:
    df = df.copy()
    df["p_mal"] = p_mal
    rows = []
    for case_id, g in df.groupby("case_id", sort=True):
        score = float(g["p_mal"].mean())
        true_mal = 1 if g["case_label"].iloc[0] == "Tumor" else 0
        rows.append(
            {
                "case_id": case_id,
                "n_images": int(len(g)),
                "true_mal": true_mal,
                "p_mal": score,
                "pred_mal": int(score >= BINARY_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


def write_summary_row(row: dict):
    path = OUT_DIR / "eval" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.is_file():
        old = pd.read_csv(path)
        key = ["center", "arm", "ckpt"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


def eval_one(center: str, arm: str, ckpt_name: str, args):
    df = load_test_items(center)
    if arm == "original":
        if args.from_infer:
            p_mal = pmal_from_infer_json(df, Path(args.from_infer))
            ckpt_used = args.from_infer
        else:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            p_mal = infer_original(df["path"].tolist(), device, args.batch_size, args.num_workers)
            ckpt_used = f"{V2_0204_DINO}|{V2_0204_CLS}"
    else:
        ckpt_path = Path(args.ckpt) if args.ckpt else (run_dir(center, arm) / "last.pth")
        if ckpt_name != "last" and not args.ckpt:
            ckpt_path = run_dir(center, arm) / "checkpoints" / f"{ckpt_name}.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        p_mal = infer_binary(df["path"].tolist(), ckpt_path, device, args.batch_size, args.num_workers)
        ckpt_used = str(ckpt_path)

    img_df = df[["case_id", "mapped_filename", "image_label", "case_label", "path"]].copy()
    img_df["p_mal"] = p_mal
    cases = aggregate_cases(df, p_mal)
    y_true = cases["true_mal"].to_numpy()
    y_score = cases["p_mal"].to_numpy()
    conf = confusion_at_threshold(y_true, y_score)
    boot = bootstrap_metric_ci(y_true, y_score)
    roc = compute_roc(y_true, y_score)

    lab_sum = {}
    lab_path = labels_json(center, arm) if arm != "original" else None
    if lab_path and lab_path.is_file():
        lab_sum = load_json(lab_path).get("summary", {})

    out = eval_dir(center, arm)
    out.mkdir(parents=True, exist_ok=True)
    img_df.to_csv(out / "image_predictions.csv", index=False)
    cases.to_csv(out / "case_predictions.csv", index=False)
    report = {
        "center": center,
        "arm": arm,
        "ckpt": ckpt_name,
        "ckpt_path": ckpt_used,
        "aggregation": "mean",
        "threshold": BINARY_THRESHOLD,
        "n_images": int(len(img_df)),
        "n_cases": int(len(cases)),
        "n_tumor_cases": int((cases["true_mal"] == 1).sum()),
        "n_nontumor_cases": int((cases["true_mal"] == 0).sum()),
        "label_summary": lab_sum,
        "confusion": conf,
        "auc": compute_auc(y_true, y_score),
        "metrics_bootstrap_ci": boot,
        "roc": roc,
        "note": "Official metric is last-epoch. Do not pick best on heldout.",
    }
    save_json(out / "metrics.json", report)

    summary_path = write_summary_row(
        {
            "center": center,
            "arm": arm,
            "ckpt": ckpt_name,
            "n_cases": report["n_cases"],
            "n_tumor": report["n_tumor_cases"],
            "n_nontumor": report["n_nontumor_cases"],
            "n_images": report["n_images"],
            "expert_n": lab_sum.get("num_true"),
            "expert_frac": lab_sum.get("true_frac"),
            "auc": boot["auc"]["point"],
            "auc_lo": boot["auc"]["ci_lower"],
            "auc_hi": boot["auc"]["ci_upper"],
            "acc": boot["accuracy"]["point"],
            "sens": boot["sensitivity"]["point"],
            "spec": boot["specificity"]["point"],
        }
    )
    print(
        f"{center}/{arm}/{ckpt_name}  cases={report['n_cases']} "
        f"AUC={boot['auc']['point']:.3f} "
        f"[{boot['auc']['ci_lower']:.3f},{boot['auc']['ci_upper']:.3f}] "
        f"sens={boot['sensitivity']['point']:.3f} spec={boot['specificity']['point']:.3f}"
    )
    print(f"Wrote {out / 'metrics.json'} and {summary_path}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--center", required=True)
    p.add_argument("--arm", required=True, choices=["original", "al", "random", "all_expert"])
    p.add_argument("--ckpt", type=str, default=None, help="Override ckpt path (adapted arms)")
    p.add_argument("--ckpt_name", type=str, default="last")
    p.add_argument("--from_infer", type=str, default=None, help="Original: reuse infer JSON with probs")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    eval_one(args.center, args.arm, args.ckpt_name, args)


if __name__ == "__main__":
    main()
