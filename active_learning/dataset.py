#!/usr/bin/env python3
"""Build training image/label lists for active learning."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torchvision import transforms

from utils.data_manager import DummyDataset
from utils.datasets import AllowEmptyImageFolder, split_images_labels
from utils.transforms import StrongTrivialAugment

from .common import resolve, iter_stage_images
from .labels_io import load_json


def _train_transform():
    return transforms.Compose(
        [
            transforms.Resize([224, 224]),
            StrongTrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _test_transform():
    return transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_true_stage_xy(
    true_stages: Sequence[int],
    stages_manifest: str | None = None,
    train_root: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if stages_manifest is None:
        stages_manifest = resolve("stages_manifest")
    if train_root is None:
        train_root = resolve("train_root")
    items = iter_stage_images(true_stages, stages_manifest, train_root)
    paths = np.array([it["path"] for it in items], dtype=object)
    labels = np.array([it["true_class"] for it in items], dtype=np.int64)
    return paths, labels


def collect_al_label_xy(
    al_labels_path: str,
    train_root: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_root is None:
        train_root = resolve("train_root")
    manifest = load_json(al_labels_path)
    root = Path(train_root)
    paths, labels = [], []
    for it in manifest["items"]:
        if it.get("source") == "unlabeled":
            continue
        rel = it["rel_path"]
        paths.append(str(root / rel))
        labels.append(int(it["label"]))
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)


def build_train_dataset(
    true_stages: Sequence[int],
    al_labels_path: Optional[str] = None,
    stages_manifest: str | None = None,
    train_root: str | None = None,
) -> DummyDataset:
    if stages_manifest is None:
        stages_manifest = resolve("stages_manifest")
    if train_root is None:
        train_root = resolve("train_root")
    """Cumulative train set = true_stages (GT) + optional AL label manifest."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    if true_stages:
        x, y = collect_true_stage_xy(true_stages, stages_manifest, train_root)
        xs.append(x)
        ys.append(y)

    if al_labels_path:
        x, y = collect_al_label_xy(al_labels_path, train_root)
        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError("Empty training set: provide true_stages and/or al_labels")

    data = np.concatenate(xs)
    targets = np.concatenate(ys)
    return DummyDataset(data, targets, _train_transform(), use_path=True, aug=1)


def build_test_dataset(test_root: str | None = None) -> DummyDataset:
    if test_root is None:
        test_root = resolve("test_root")
    test_dset = AllowEmptyImageFolder(test_root)
    data, targets = split_images_labels(test_dset.imgs)
    return DummyDataset(data, targets, _test_transform(), use_path=True, aug=1)


def describe_train_mix(
    true_stages: Sequence[int],
    al_labels_path: Optional[str] = None,
    stages_manifest: str | None = None,
    train_root: str | None = None,
) -> dict:
    if stages_manifest is None:
        stages_manifest = resolve("stages_manifest")
    if train_root is None:
        train_root = resolve("train_root")
    info = {"true_stages": list(true_stages), "true_images": 0, "al_images": 0, "al_true": 0, "al_pseudo": 0}
    if true_stages:
        info["true_images"] = len(iter_stage_images(true_stages, stages_manifest, train_root))
    if al_labels_path:
        man = load_json(al_labels_path)
        items = man["items"]
        info["al_true"] = sum(1 for it in items if it["source"] == "true")
        info["al_pseudo"] = sum(1 for it in items if it["source"] == "pseudo")
        info["al_unlabeled"] = sum(1 for it in items if it["source"] == "unlabeled")
        info["al_images"] = info["al_true"] + info["al_pseudo"]
        info["al_summary"] = man.get("summary")
    info["total_images"] = info["true_images"] + info["al_images"]
    return info
