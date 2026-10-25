#!/usr/bin/env python3
"""Shared helpers for active learning on train_new stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Paths are relative to CAILOOP_DATA_ROOT. Clinical images are not in this repo.
DEFAULTS = {
    "train_root": "v5/train_new",
    "test_root": "v5/test_new",
    "val_root": "v5/val_new",
    "holdout_root": "v5/test_holdout",
    "stages_manifest": "v5/train_new/stages_v1.json",
    "num_classes": 30,
}


def resolve(key: str) -> str | int:
    value = DEFAULTS[key]
    if not isinstance(value, str):
        return value
    root = os.environ.get("CAILOOP_DATA_ROOT")
    if not root:
        raise FileNotFoundError(
            "Set CAILOOP_DATA_ROOT to the directory that contains v5/. "
            "Clinical images are not included in this repository."
        )
    return str(Path(root) / value)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_stages(stages_str: str) -> List[int]:
    stages: List[int] = []
    for part in stages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            stages.extend(range(int(a), int(b) + 1))
        else:
            stages.append(int(part))
    stages = sorted(set(stages))
    if not stages:
        raise ValueError(f"Empty stages from: {stages_str}")
    return stages


def class_to_index(class_name: str) -> int:
    return int(class_name.split("_", 1)[0])


def load_stages_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_stage_images(
    stages: Sequence[int],
    stages_manifest: str | Path | None = None,
    train_root: str | Path | None = None,
) -> List[dict]:
    if stages_manifest is None:
        stages_manifest = resolve("stages_manifest")
    if train_root is None:
        train_root = resolve("train_root")
    """Return image-level records for the given stages.

    Each item: path (absolute), rel_path, stage, true_class, case_key.
    """
    manifest = load_stages_manifest(stages_manifest)
    case_map = {c["case_key"]: c for c in manifest["cases"]}
    train_root = Path(train_root)

    items: List[dict] = []
    for stage in stages:
        for case_key in manifest["by_stage"][str(stage)]:
            case = case_map[case_key]
            true_class = class_to_index(case["class"])
            for rel in case["images"]:
                items.append(
                    {
                        "path": str(train_root / rel),
                        "rel_path": rel,
                        "stage": int(stage),
                        "true_class": true_class,
                        "case_key": case_key,
                        "class_name": case["class"],
                    }
                )
    return items


def iter_folder_images(image_root: str | Path) -> List[dict]:
    """Scan a class-folder dataset (e.g. val_new / test_holdout).

    Each item: path, rel_path, stage=None, true_class, case_key, class_name.
    """
    import re

    name_re = re.compile(
        r"^(?P<cls>.+?)_(?P<kind>case|single)_(?P<cid>\d+)_image_",
        re.IGNORECASE,
    )
    root = Path(image_root)
    items: List[dict] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        true_class = class_to_index(cls_dir.name)
        for f in sorted(cls_dir.iterdir()):
            if not f.is_file() and not f.is_symlink():
                continue
            if f.suffix.lower() not in IMG_EXTS:
                continue
            m = name_re.match(f.name)
            case_key = (
                f"{m.group('cls')}_{m.group('kind')}_{m.group('cid')}"
                if m
                else f.stem
            )
            rel = f"{cls_dir.name}/{f.name}"
            items.append(
                {
                    "path": str(root / rel),
                    "rel_path": rel,
                    "stage": None,
                    "true_class": true_class,
                    "case_key": case_key,
                    "class_name": cls_dir.name,
                }
            )
    return items


def rel_key(path_or_rel: str, train_root: str | Path | None = None) -> str:
    if train_root is None:
        train_root = resolve("train_root")
    """Normalize to relative path under train_root when possible."""
    p = Path(path_or_rel)
    root = Path(train_root).resolve()
    try:
        return str(p.resolve().relative_to(root))
    except Exception:
        # already relative, or outside root
        s = str(path_or_rel).replace("\\", "/")
        root_s = str(root).replace("\\", "/")
        if s.startswith(root_s):
            return s[len(root_s) :].lstrip("/")
        return s
