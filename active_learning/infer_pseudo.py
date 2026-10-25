#!/usr/bin/env python3
"""Run model inference on stages or a class-folder split (val/holdout).

Output JSON (infer_v1):
  items[]: rel_path, path, stage, true_class, pred, confidence, probs (optional)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.common import (
    DEFAULTS,
    iter_folder_images,
    iter_stage_images,
    parse_stages,
    resolve,
)
from active_learning.labels_io import save_json

from cailoop.runtime import dinov2_backbone


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
def run_infer(args):
    if args.image_root:
        items = iter_folder_images(args.image_root)
        pool_desc = {"image_root": args.image_root}
        stages = None
    else:
        if not args.stages:
            raise ValueError("Provide --stages or --image_root")
        stages = parse_stages(args.stages)
        items = iter_stage_images(stages, args.stages_manifest, args.train_root)
        pool_desc = {"pool_stages": stages}

    if not items:
        raise RuntimeError("No images found for inference")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={args.device} but CUDA is unavailable. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}. "
            "If this srun only has 1 GPU, use CUDA_VISIBLE_DEVICES=0 (or unset)."
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Infer device={device} | visible_gpus={torch.cuda.device_count()}")
    dino, clf = build_model(args.dino_ckpt, args.cls_ckpt, args.num_classes, device)

    tfm = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        PathListDataset(items, tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds, confs = [None] * len(items), [None] * len(items)
    probs_store = [None] * len(items) if args.save_probs else None

    correct = 0
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            logits = clf(dino(images))
        prob = F.softmax(logits.float(), dim=1)
        conf, pred = prob.max(dim=1)
        for i, idx in enumerate(indices.tolist()):
            preds[idx] = int(pred[i].item())
            confs[idx] = float(conf[i].item())
            if args.save_probs:
                probs_store[idx] = [float(x) for x in prob[i].cpu().tolist()]
            if preds[idx] == items[idx]["true_class"]:
                correct += 1

    out_items = []
    for i, it in enumerate(items):
        row = {
            "rel_path": it["rel_path"],
            "path": it["path"],
            "stage": it["stage"],
            "true_class": it["true_class"],
            "case_key": it["case_key"],
            "class_name": it["class_name"],
            "pred": preds[i],
            "confidence": confs[i],
        }
        if args.save_probs:
            row["probs"] = probs_store[i]
        out_items.append(row)

    result = {
        "format": "infer_v1",
        "score": "max_softmax",
        **pool_desc,
        "dino_ckpt": args.dino_ckpt,
        "cls_ckpt": args.cls_ckpt,
        "num_classes": args.num_classes,
        "num_images": len(out_items),
        "top1_acc_vs_true": correct / len(out_items),
        "items": out_items,
    }
    save_json(args.out, result)
    print(
        f"Wrote {args.out} | images={len(out_items)} | "
        f"top1_vs_true={result['top1_acc_vs_true']:.4f}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Infer on stages (--stages) or a folder split (--image_root)"
    )
    p.add_argument("--stages", type=str, default=None, help="e.g. 6 or 6-7")
    p.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Class-folder root, e.g. val_new or test_holdout",
    )
    p.add_argument("--dino_ckpt", type=str, required=True)
    p.add_argument("--cls_ckpt", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        default="active_learning/outputs/infer.json",
    )
    p.add_argument("--train_root", type=str, default=None)
    p.add_argument("--stages_manifest", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=DEFAULTS["num_classes"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_probs", action="store_true", help="Store full softmax vector")
    args = p.parse_args()
    if args.train_root is None:
        args.train_root = resolve("train_root")
    if args.stages_manifest is None:
        args.stages_manifest = resolve("stages_manifest")
    run_infer(args)


if __name__ == "__main__":
    main()
