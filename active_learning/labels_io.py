#!/usr/bin/env python3
"""Read / write active-learning label manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_labels(items: List[dict]) -> Dict[str, Any]:
    src = Counter(it["source"] for it in items)
    n = len(items)
    n_true = src.get("true", 0)
    n_pseudo = src.get("pseudo", 0)
    n_unlabeled = src.get("unlabeled", 0)
    # oracle simulation accuracy of pseudo
    pseudo_correct = sum(
        1 for it in items if it["source"] == "pseudo" and it["label"] == it["true_class"]
    )
    by_stage: Dict[str, Dict[str, int]] = {}
    for it in items:
        st = str(it["stage"])
        by_stage.setdefault(st, {"true": 0, "pseudo": 0})
        by_stage[st][it["source"]] = by_stage[st].get(it["source"], 0) + 1

    return {
        "num_images": n,
        "num_true": n_true,
        "num_pseudo": n_pseudo,
        "num_unlabeled": n_unlabeled,
        "true_frac": (n_true / n) if n else 0.0,
        "pseudo_agree_with_true": (pseudo_correct / n_pseudo) if n_pseudo else None,
        "by_stage": by_stage,
    }


def locked_true_paths(prev_labels: Optional[dict]) -> List[str]:
    if not prev_labels:
        return []
    return [
        it["rel_path"]
        for it in prev_labels.get("items", [])
        if it.get("source") == "true"
    ]


def build_label_manifest(
    items: List[dict],
    *,
    pool_stages: List[int],
    strategy_name: str,
    strategy_kwargs: dict,
    infer_path: str | None = None,
    ckpt_dino: str | None = None,
    ckpt_cls: str | None = None,
    prev_labels_path: str | None = None,
) -> dict:
    summary = summarize_labels(items)
    return {
        "format": "al_labels_v1",
        "num_classes": 30,
        "pool_stages": pool_stages,
        "strategy": {"name": strategy_name, **strategy_kwargs},
        "infer_path": infer_path,
        "ckpt_dino": ckpt_dino,
        "ckpt_cls": ckpt_cls,
        "prev_labels_path": prev_labels_path,
        "summary": summary,
        "items": items,
    }
