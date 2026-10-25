#!/usr/bin/env python3
"""Binary adaptation from a frozen v2_0204 backbone + new Linear(768, 2).

Protocol:
  warmup:  freeze backbone, train head (default 2 epochs)
  finetune: unfreeze both (default 50 epochs)
  official ckpt: last finetune epoch (not selected on heldout)
  extras: warmup last + every 5 finetune epochs

Do not run this on the shared GPU unless the user launches it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.labels_io import load_json
from cross_al.common import V2_0204_DINO, labels_json, run_dir

from cailoop.runtime import dinov2_backbone


class StrongNoFlipTrivialAugment:
    """Same no-flip trivial augment as the old external finetune.py."""

    def __init__(self, num_bins=31):
        self.num_bins = num_bins

    def __call__(self, img):
        magnitude = random.randint(0, self.num_bins)
        m = magnitude / float(self.num_bins)
        sign = random.choice([-1, 1])
        ops = [
            lambda im: im,
            lambda im: TF.autocontrast(im),
            lambda im: TF.equalize(im),
            lambda im: TF.invert(im),
            lambda im: TF.posterize(im, bits=max(1, 8 - int(m * 4))),
            lambda im: TF.solarize(im, threshold=255.0 - m * 255.0),
            lambda im: TF.adjust_brightness(im, brightness_factor=1.0 + m * sign * 0.9),
            lambda im: TF.adjust_contrast(im, contrast_factor=1.0 + m * sign * 0.9),
            lambda im: TF.adjust_sharpness(im, sharpness_factor=1.0 + m * sign * 0.9),
            lambda im: TF.affine(im, angle=0.0, translate=[0, 0], scale=1.0, shear=[m * 16.0 * sign, 0.0]),
            lambda im: TF.affine(im, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, m * 16.0 * sign]),
            lambda im: TF.affine(im, angle=0.0, translate=[int(m * 30 * sign), 0], scale=1.0, shear=[0.0, 0.0]),
            lambda im: TF.affine(im, angle=0.0, translate=[0, int(m * 30 * sign)], scale=1.0, shear=[0.0, 0.0]),
        ]
        return random.choice(ops)(img)


class LabelManifestDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path = Path(it["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(it["label"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, backbone, classifier, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **meta,
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
        },
        path,
    )


def run_epoch(backbone, classifier, loader, optimizer, criterion, device, desc):
    backbone.train() if any(p.requires_grad for p in backbone.parameters()) else backbone.eval()
    classifier.train()
    total_loss = 0.0
    correct = 0
    n = 0
    pbar = tqdm(loader, desc=desc, ncols=100, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(backbone(images))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        n += int(labels.size(0))
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return {"loss": total_loss / n if n else 0.0, "acc": correct / n if n else 0.0, "n": n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--center", required=True)
    p.add_argument("--arm", required=True, choices=["al", "random", "all_expert"])
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--ft_epochs", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_classifier", type=float, default=1e-4)
    p.add_argument(
        "--freeze_backbone",
        choices=["never", "warmup2", "always"],
        default="warmup2",
        help="never=unfreeze from epoch 1; warmup2=freeze during warmup; always=linear probe",
    )
    p.add_argument(
        "--expert_only",
        action="store_true",
        help="Train only on gate-queried expert labels (source==true).",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Write to runs/{center}/{arm}__{run_name} instead of runs/{center}/{arm}.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dino_ckpt", type=str, default=str(V2_0204_DINO))
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume finetune from a pack (last.pth / ft_epXXX.pth). "
        "--ft_epochs is the target epoch, not extra epochs.",
    )
    args = p.parse_args()

    labels_path = labels_json(args.center, args.arm)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels: {labels_path}")
    man = load_json(labels_path)
    items = man["items"]
    if args.expert_only:
        items = [it for it in items if it.get("source") == "true"]
    if not items:
        raise RuntimeError(f"Empty label manifest: {labels_path}")

    if args.freeze_backbone == "never":
        args.warmup_epochs = 0
    elif args.freeze_backbone == "warmup2" and args.warmup_epochs <= 0:
        args.warmup_epochs = 2

    out = run_dir(args.center, args.arm, args.run_name)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "labels": str(labels_path), "summary": man.get("summary")}, indent=2) + "\n")

    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"{args.center}/{args.arm} run={args.run_name or args.arm} "
        f"freeze={args.freeze_backbone} device={device} "
        f"n={len(items)} expert_only={args.expert_only}"
    )

    start_ft = 1
    resume_meta = {}
    backbone = dinov2_backbone()(pretrained=False)
    classifier = nn.Linear(768, 2)
    if args.resume:
        pack = torch.load(args.resume, map_location="cpu")
        backbone.load_state_dict(pack["backbone"])
        classifier.load_state_dict(pack["classifier"])
        resume_meta = {k: pack[k] for k in pack if k not in ("backbone", "classifier")}
        if pack.get("stage") == "finetune":
            start_ft = int(pack.get("epoch", 0)) + 1
        print(f"Resume {args.resume} stage={pack.get('stage')} epoch={pack.get('epoch')} -> ft {start_ft}..{args.ft_epochs}")
        if start_ft > args.ft_epochs:
            raise ValueError(f"Already at epoch {start_ft - 1}, --ft_epochs={args.ft_epochs}")
    else:
        backbone.load_state_dict(torch.load(args.dino_ckpt, map_location="cpu"))
        nn.init.zeros_(classifier.bias)
    backbone.to(device)
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()

    train_tfm = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            StrongNoFlipTrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        LabelManifestDataset(items, train_tfm),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    log_path = out / "train_log.jsonl"
    log_f = open(log_path, "a" if args.resume else "w", encoding="utf-8")

    def log_row(row: dict):
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(
            f"{row['stage']} ep{row['epoch']:02d} | "
            f"loss={row['loss']:.4f} acc={100 * row['acc']:.1f}%"
        )

    if not args.resume and args.warmup_epochs > 0:
        print(">> warmup (backbone frozen)")
        for p_ in backbone.parameters():
            p_.requires_grad = False
        opt = optim.AdamW(classifier.parameters(), lr=args.lr_classifier)
        for epoch in range(1, args.warmup_epochs + 1):
            stats = run_epoch(
                backbone, classifier, loader, opt, criterion, device, f"warmup {epoch}"
            )
            log_row({"stage": "warmup", "epoch": epoch, **stats})
        save_ckpt(
            ckpt_dir / f"warmup_ep{args.warmup_epochs:02d}.pth",
            backbone,
            classifier,
            {"center": args.center, "arm": args.arm, "stage": "warmup", "epoch": args.warmup_epochs},
        )

    if args.freeze_backbone == "always":
        print(">> finetune (backbone remains frozen)")
        for p_ in backbone.parameters():
            p_.requires_grad = False
        opt = optim.AdamW(classifier.parameters(), lr=args.lr_classifier)
    else:
        print(">> finetune (backbone unfrozen)")
        for p_ in backbone.parameters():
            p_.requires_grad = True
        opt = optim.AdamW(
            [
                {"params": backbone.parameters(), "lr": args.lr_backbone},
                {"params": classifier.parameters(), "lr": args.lr_classifier},
            ]
        )
    for epoch in range(start_ft, args.ft_epochs + 1):
        stats = run_epoch(
            backbone, classifier, loader, opt, criterion, device, f"ft {epoch}"
        )
        log_row({"stage": "finetune", "epoch": epoch, **stats})
        if epoch % args.ckpt_every == 0 or epoch == args.ft_epochs:
            save_ckpt(
                ckpt_dir / f"ft_ep{epoch:03d}.pth",
                backbone,
                classifier,
                {"center": args.center, "arm": args.arm, "stage": "finetune", "epoch": epoch},
            )

    last_path = out / "last.pth"
    save_ckpt(
        last_path,
        backbone,
        classifier,
        {"center": args.center, "arm": args.arm, "stage": "finetune", "epoch": args.ft_epochs, "official": True},
    )
    torch.save(backbone.state_dict(), out / "last_backbone.pth")
    torch.save(classifier.state_dict(), out / "last_classifier.pth")
    log_f.close()
    print(f"Wrote official last ckpt: {last_path}")
    print("Heldout was not used. Eval separately with eval_case.py")


if __name__ == "__main__":
    main()
