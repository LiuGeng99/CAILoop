#!/usr/bin/env python3
"""Encode an active-learning manifest into the semiparametric memory bank.

New images are written to working memory, then exported and moved to
long-term memory. Already stored paths are left unchanged. The default
codec, used only when the bank is created, is bmshj2018 factorized prior
quality 8.

Examples:
  python active_learning/consolidate_memory.py \\
    --bank checkpoints/smc_memory \\
    --al_labels outputs/al_labels_s1.json \\
    --train_root "$CAILOOP_DATA_ROOT/v5/train_new"

  # Encode only. Export later with --commit_only.
  python active_learning/consolidate_memory.py \\
    --bank checkpoints/smc_memory \\
    --al_labels outputs/al_labels_s1.json \\
    --train_root "$CAILOOP_DATA_ROOT/v5/train_new" \\
    --no_commit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.labels_io import load_json
from active_learning.smc_memory import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LAMBDA,
    DEFAULT_LR,
    DEFAULT_QUALITY,
    MemoryBank,
    records_from_al_labels,
)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate AL images into semiparametric memory. "
        "The new bank uses the quality-8 bmshj2018 factorized prior."
    )
    parser.add_argument("--bank", type=str, required=True, help="Directory of the memory bank.")
    parser.add_argument("--al_labels", type=str, default=None, help="Label manifest to encode.")
    parser.add_argument("--train_root", type=str, default=None, help="Root used to resolve relative rel_path values.")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="Codec quality used when creating a bank (1-8).")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--latent_lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--decoder_lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lmbda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: cuda when available.")
    parser.add_argument("--no_commit", action="store_true", help="Keep new cues in working memory and skip PNG export.")
    parser.add_argument("--commit_only", action="store_true", help="Export existing working memory without encoding new images.")
    args = parser.parse_args()

    if args.no_commit and args.commit_only:
        parser.error("Use only one of --no_commit and --commit_only.")
    if not args.commit_only and not args.al_labels:
        parser.error("--al_labels is required unless --commit_only is set.")

    bank = MemoryBank.open(args.bank, quality=args.quality, image_size=args.image_size)
    print(
        f"bank={bank.root} architecture={bank.meta['architecture']} "
        f"quality={bank.quality} stored={len(bank.items)}"
    )

    added = 0
    skipped = 0
    if not args.commit_only:
        manifest = load_json(args.al_labels)
        records, skipped = records_from_al_labels(manifest, args.train_root, bank.known_sources())
        added = bank.fit_new(
            records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_lr=args.latent_lr,
            decoder_lr=args.decoder_lr,
            lmbda=args.lmbda,
            num_workers=args.num_workers,
            clip_max_norm=args.clip_max_norm,
            seed=args.seed,
            device=args.device,
        )

    committed = 0
    if not args.no_commit:
        committed = bank.commit(batch_size=args.batch_size, device=args.device)

    working = sum(1 for item in bank.items if item["split"] == "working")
    long_term = sum(1 for item in bank.items if item["split"] == "long_term")
    print(
        f"added={added} skipped_existing={skipped} exported={committed} "
        f"working={working} long_term={long_term}"
    )


if __name__ == "__main__":
    main()
