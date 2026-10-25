#!/usr/bin/env python3
"""Build an ``--al_labels`` manifest that replays long-term memory.

Recalled images are absolute paths, so ``train_ddp.py`` can read them with
``--true_stages ""`` and no change to the training loop. The current stage,
when given, is appended as the original files.

Example:
  python active_learning/build_replay_manifest.py \\
    --bank checkpoints/smc_memory \\
    --al_labels outputs/al_labels_s2.json \\
    --train_root "$CAILOOP_DATA_ROOT/v5/train_new" \\
    --out outputs/replay_s2.json

  python active_learning/train_ddp.py \\
    --true_stages "" \\
    --al_labels outputs/replay_s2.json \\
    --backbone_ckpt /path/to/dinov2_vitb14_reg4_pretrain.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.labels_io import build_label_manifest, load_json, save_json
from active_learning.smc_memory import MemoryBank, replay_items


def main():
    parser = argparse.ArgumentParser(
        description="Write a training manifest of recalled memory images plus the current raw stage."
    )
    parser.add_argument("--bank", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--al_labels", type=str, default=None, help="Current stage manifest. Omit to replay memory only.")
    parser.add_argument("--train_root", type=str, default=None, help="Root for relative paths in --al_labels.")
    args = parser.parse_args()

    bank = MemoryBank.load(args.bank)
    current = load_json(args.al_labels) if args.al_labels else None
    items, skipped = replay_items(bank, current, args.train_root)
    if not items:
        parser.error("Replay manifest is empty.")

    stages = sorted({int(item["stage"]) for item in items if int(item["stage"]) >= 0})
    num_classes = 30
    if current is not None and current.get("num_classes"):
        num_classes = int(current["num_classes"])
    manifest = build_label_manifest(
        items,
        pool_stages=stages,
        strategy_name="smc_replay",
        strategy_kwargs={"bank": str(Path(args.bank).resolve())},
    )
    manifest["num_classes"] = num_classes
    save_json(args.out, manifest)

    recalled = sum(1 for item in items if item.get("replay") == "smc")
    raw = sum(1 for item in items if item.get("replay") == "raw")
    print(
        f"wrote {args.out} recalled={recalled} raw={raw} "
        f"skipped_already_stored={skipped} total={len(items)}"
    )


if __name__ == "__main__":
    main()
