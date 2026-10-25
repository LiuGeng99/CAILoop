#!/usr/bin/env python3
"""DINOv2 DDP training for staged / active-learning laryngo data.

Train set = images from --true_stages (ground-truth)
          + optional --al_labels (true/pseudo mix from build_labels.py).

Model selection / early-stopping on val_new (EMA).
Optional holdout eval is logged but does not affect checkpointing.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.common import DEFAULTS, parse_stages, resolve
from active_learning.dataset import (
    build_test_dataset,
    build_train_dataset,
    describe_train_mix,
)

from cailoop.runtime import dinov2_backbone


def _set_random(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate_model(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _, inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = model(inputs)
                outputs = classifier(features)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    metrics = torch.tensor([correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item() / metrics[1].item()


def train(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    _set_random(args.seed + local_rank)

    true_stages = parse_stages(args.true_stages) if args.true_stages.strip() else []

    if local_rank == 0:
        os.makedirs(args.ckpt_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(f"{args.ckpt_path}/train.log"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
        logging.info(f"Arguments: {vars(args)}")
        mix = describe_train_mix(
            true_stages,
            al_labels_path=args.al_labels,
            stages_manifest=args.stages_manifest,
            train_root=args.train_root,
        )
        logging.info(f"Train mix: {mix}")

    train_dataset = build_train_dataset(
        true_stages=true_stages,
        al_labels_path=args.al_labels,
        stages_manifest=args.stages_manifest,
        train_root=args.train_root,
    )
    val_dataset = build_test_dataset(test_root=args.val_root)
    holdout_loader = None
    if args.holdout_root:
        holdout_dataset = build_test_dataset(test_root=args.holdout_root)
    else:
        holdout_dataset = None

    if local_rank == 0:
        msg = (
            f"Train images: {len(train_dataset)} | "
            f"Val images: {len(val_dataset)}"
        )
        if holdout_dataset is not None:
            msg += f" | Holdout images: {len(holdout_dataset)}"
        logging.info(msg)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if holdout_dataset is not None:
        holdout_sampler = DistributedSampler(holdout_dataset, shuffle=False)
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=args.batch_size,
            sampler=holdout_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    dino_model = dinov2_backbone()(pretrained=False)
    dino_model.load_state_dict(torch.load(args.backbone_ckpt, map_location="cpu"))
    classifier = nn.Linear(768, args.num_classes)

    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        classifier.load_state_dict(torch.load(args.cls_ckpt, map_location="cpu"))
        if local_rank == 0:
            logging.info(f"Loaded classifier from {args.cls_ckpt}")

    dino_model.to(device)
    classifier.to(device)

    dino_model = DDP(dino_model, device_ids=[local_rank], find_unused_parameters=True)
    classifier = DDP(classifier, device_ids=[local_rank])

    ema_dino = EMA(dino_model.module, beta=0.9999, update_after_step=100, update_every=10)
    ema_cls = EMA(classifier.module, beta=0.9999, update_after_step=100, update_every=10)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    if args.cls_epochs > 0:
        if local_rank == 0:
            logging.info("Starting Phase 1: Linear Probing...")

        cls_optimizer = optim.AdamW(
            classifier.parameters(), lr=args.cls_lr, weight_decay=args.weight_decay
        )
        dino_model.eval()

        for epoch in range(args.cls_epochs):
            train_sampler.set_epoch(epoch)
            classifier.train()
            running_loss, n_steps = 0.0, 0

            for _, inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        features = dino_model.module(inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs, targets)

                cls_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(cls_optimizer)
                scaler.update()

                running_loss += loss.item()
                n_steps += 1

            if local_rank == 0:
                logging.info(
                    f"Linear Probe epoch {epoch+1}/{args.cls_epochs} | "
                    f"loss={running_loss / max(n_steps, 1):.4f}"
                )

    if local_rank == 0:
        logging.info("Starting Phase 2: Joint Fine-tuning...")

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, dino_model.parameters()),
                "lr": args.dino_lr,
            },
            {"params": classifier.parameters(), "lr": args.cls_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_acc = -1.0
    best_epoch = 0
    stop_flag = torch.zeros(1, device=device)
    metric_name = args.early_stop_metric  # "val" or "holdout"
    if metric_name not in ("val", "holdout"):
        raise ValueError("--early_stop_metric must be 'val' or 'holdout'")
    if metric_name == "holdout" and holdout_loader is None:
        raise ValueError("--early_stop_metric=holdout requires --holdout_root")

    if local_rank == 0:
        logging.info(
            f"Early-stop metric={metric_name} | min_epochs={args.min_epochs} | "
            f"patience={args.patience} | ft_epochs={args.ft_epochs}"
        )

    for epoch in range(args.ft_epochs):
        train_sampler.set_epoch(epoch)
        dino_model.train()
        classifier.train()
        running_loss, n_steps = 0.0, 0

        for _, inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = dino_model(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema_dino.update()
            ema_cls.update()

            running_loss += loss.item()
            n_steps += 1

        if local_rank == 0:
            logging.info(
                f"Epoch {epoch+1}/{args.ft_epochs} | "
                f"train_loss={running_loss / max(n_steps, 1):.4f}"
            )

        do_eval = (epoch + 1) % args.eval_freq == 0 or epoch == args.ft_epochs - 1
        if do_eval:
            val_acc = evaluate_model(
                ema_dino.ema_model, ema_cls.ema_model, val_loader, device
            )
            holdout_acc = None
            if holdout_loader is not None:
                holdout_acc = evaluate_model(
                    ema_dino.ema_model, ema_cls.ema_model, holdout_loader, device
                )

            if local_rank == 0:
                score = holdout_acc if metric_name == "holdout" else val_acc
                improved = score > best_acc
                if improved:
                    best_acc = score
                    best_epoch = epoch + 1
                    torch.save(
                        ema_dino.ema_model.state_dict(),
                        f"{args.ckpt_path}/dino_best.pth",
                    )
                    torch.save(
                        ema_cls.ema_model.state_dict(),
                        f"{args.ckpt_path}/cls_best.pth",
                    )
                line = (
                    f"Epoch {epoch+1} | val_acc={val_acc:.4f}"
                )
                if holdout_acc is not None:
                    line += f" | holdout_acc={holdout_acc:.4f}"
                line += (
                    f" | best_{metric_name}={best_acc:.4f}@{best_epoch}"
                    f"{' *' if improved else ''}"
                )
                logging.info(line)

                reached_min = (epoch + 1) >= args.min_epochs
                no_improve = (
                    args.patience > 0
                    and best_epoch > 0
                    and (epoch + 1 - best_epoch) >= args.patience
                )
                if reached_min and no_improve:
                    logging.info(
                        f"Early stop: no {metric_name} improvement for "
                        f"{args.patience} epochs after min_epochs={args.min_epochs} "
                        f"(best@{best_epoch}={best_acc:.4f})"
                    )
                    stop_flag.fill_(1)

            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

    if local_rank == 0:
        logging.info(
            f"Training done. Best EMA {metric_name}_acc: {best_acc:.4f} @ epoch {best_epoch}"
        )

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINOv2 DDP train with true stages + optional AL labels"
    )

    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        required=True,
        help="DINOv2 ViT-B/14 register pretrain weights",
    )
    parser.add_argument("--cls_ckpt", type=str, default=None)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/stages",
    )

    parser.add_argument(
        "--true_stages",
        type=str,
        default="1-5",
        help="Stages with ground-truth labels, e.g. '1-5'. Empty string allowed if only al_labels.",
    )
    parser.add_argument(
        "--al_labels",
        type=str,
        default=None,
        help="AL label manifest from build_labels.py (true+pseudo). Cumulative with true_stages.",
    )
    parser.add_argument("--stages_manifest", type=str, default=None)
    parser.add_argument("--train_root", type=str, default=None)
    parser.add_argument(
        "--val_root",
        type=str,
        default=None,
        help="Validation set for model selection / early stopping",
    )
    parser.add_argument(
        "--holdout_root",
        type=str,
        default=None,
        help="Holdout/test set. Used for logging; also for early stop if --early_stop_metric=holdout.",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default=None,
        help="Deprecated alias; if set without --val_root override, ignored when val_root is set.",
    )

    parser.add_argument("--num_classes", type=int, default=DEFAULTS["num_classes"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_freq", type=int, default=5)

    parser.add_argument("--cls_epochs", type=int, default=0)
    parser.add_argument("--cls_lr", type=float, default=1e-3)
    parser.add_argument("--ft_epochs", type=int, default=100)
    parser.add_argument("--dino_lr", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stop if selected metric has no improvement for this many epochs (0 disables)",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=50,
        help="Do not early-stop before this many FT epochs",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="holdout",
        choices=["val", "holdout"],
        help="Metric used to save best ckpt and trigger early stopping",
    )

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.true_stages is None:
        args.true_stages = ""

    # An empty --holdout_root disables the holdout set.
    disable_holdout = args.holdout_root is not None and args.holdout_root.strip() == ""
    if args.stages_manifest is None:
        args.stages_manifest = resolve("stages_manifest")
    if args.train_root is None:
        args.train_root = resolve("train_root")
    if args.val_root is None:
        args.val_root = resolve("val_root")
    if disable_holdout:
        args.holdout_root = None
    elif args.holdout_root is None:
        args.holdout_root = resolve("holdout_root")

    train(args)
