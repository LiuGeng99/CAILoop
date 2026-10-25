#!/usr/bin/env python3
"""Semiparametric memory for CAILoop continual learning.

Historical images are stored as quantized latent cues and recalled by the
codec decoder. New stage images stay on disk. The recalled pictures are
written once and then consumed by the existing ``--al_labels`` loader.

The default codec is the CompressAI bmshj2018 factorized prior at quality 8.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

META_NAME = "meta.json"
INDEX_NAME = "index.json"
CODEC_NAME = "codec.pth"
CUES_NAME = "cues.pt"
IMAGE_DIR = "images"

DEFAULT_QUALITY = 8
DEFAULT_IMAGE_SIZE = 256
DEFAULT_LAMBDA = 2e-3
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3


def resolve_image(rel_path: str, train_root: str | None) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    if not train_root:
        raise FileNotFoundError(f"Relative path {rel_path!r} needs a train root.")
    return Path(train_root) / path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _plain(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


class _PixelDataset(Dataset):
    """Images in the [0, 1] range expected by the compression codec."""

    def __init__(self, paths: Sequence[str], image_size: int):
        self.paths = list(paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), idx


def _rate_distortion(x_hat, target, likelihoods, lmbda: float):
    count, _, height, width = target.shape
    bpp = torch.log(likelihoods).sum() / (-math.log(2) * count * height * width)
    mse = torch.mean((x_hat - target) ** 2)
    loss = lmbda * (255**2) * mse + bpp
    return loss, bpp.detach(), mse.detach()


def _save_png(image: torch.Tensor, path: Path) -> None:
    array = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _load_factorized_prior(quality: int, metric: str, state_dict=None):
    try:
        from compressai.models.google import FactorizedPrior
        from compressai.zoo import bmshj2018_factorized
    except ImportError as exc:
        raise ImportError(
            "Semiparametric memory needs the compressai package. "
            "Install it with: pip install -r requirements-train.txt"
        ) from exc
    if state_dict is None:
        return bmshj2018_factorized(
            quality=quality, metric=metric, pretrained=True, progress=True
        )
    model = FactorizedPrior.from_state_dict(state_dict)
    model.load_state_dict(state_dict)
    return model


class MemoryBank:
    """On-disk working memory and long-term memory for one continual run."""

    def __init__(self, root: Path, meta: dict, items: list, cues: torch.Tensor | None):
        self.root = Path(root)
        self.meta = meta
        self.items = items
        self.cues = cues

    @property
    def image_size(self) -> int:
        return int(self.meta["image_size"])

    @property
    def quality(self) -> int:
        return int(self.meta["quality"])

    @classmethod
    def open(
        cls,
        root: str | Path,
        quality: int = DEFAULT_QUALITY,
        image_size: int = DEFAULT_IMAGE_SIZE,
        metric: str = "mse",
    ) -> "MemoryBank":
        root = Path(root)
        meta_path = root / META_NAME
        if meta_path.exists():
            return cls.load(root)
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be a multiple of 16, got {image_size}")
        if not 1 <= quality <= 8:
            raise ValueError(f"quality must be in 1..8, got {quality}")
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "architecture": "bmshj2018-factorized",
            "quality": quality,
            "metric": metric,
            "image_size": image_size,
            "pretrained": "bmshj2018-factorized-prior-q8" if quality == 8 else f"bmshj2018-factorized-prior-q{quality}",
        }
        bank = cls(root, meta, [], None)
        bank.save()
        return bank

    @classmethod
    def load(cls, root: str | Path) -> "MemoryBank":
        root = Path(root)
        meta = _read_json(root / META_NAME)
        index = _read_json(root / INDEX_NAME)
        cues_path = root / CUES_NAME
        cues = torch.load(cues_path, map_location="cpu", weights_only=True) if cues_path.exists() else None
        items = index["items"]
        if cues is not None and len(items) != cues.shape[0]:
            raise RuntimeError(
                f"Memory index has {len(items)} rows but cues have {cues.shape[0]}."
            )
        return cls(root, meta, items, cues)

    def save(self) -> None:
        _write_json(self.root / META_NAME, self.meta)
        _write_json(self.root / INDEX_NAME, {"items": self.items})
        if self.cues is not None:
            _atomic_torch_save(self.cues, self.root / CUES_NAME)

    def known_sources(self) -> set[str]:
        return {item["source_path"] for item in self.items}

    def load_codec(self, device: torch.device) -> nn.Module:
        codec_path = self.root / CODEC_NAME
        if codec_path.exists():
            state = torch.load(codec_path, map_location="cpu", weights_only=True)
            model = _load_factorized_prior(self.quality, self.meta["metric"], state)
        else:
            model = _load_factorized_prior(self.quality, self.meta["metric"])
            _atomic_torch_save(model.state_dict(), codec_path)
        model.to(device)
        for parameter in model.g_a.parameters():
            parameter.requires_grad = False
        return model

    def save_codec(self, model: nn.Module) -> None:
        _atomic_torch_save({key: value.detach().cpu() for key, value in model.state_dict().items()}, self.root / CODEC_NAME)

    def fit_new(
        self,
        records: Sequence[dict],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        latent_lr: float = DEFAULT_LR,
        decoder_lr: float = DEFAULT_LR,
        lmbda: float = DEFAULT_LAMBDA,
        num_workers: int = 4,
        clip_max_norm: float = 1.0,
        seed: int = 0,
        device: str | None = None,
    ) -> int:
        """Encode new images into working memory. Returns how many were added."""
        if not records:
            return 0
        torch.manual_seed(seed)
        device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        paths = [record["source_path"] for record in records]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing image: {missing[0]}")

        model = self.load_codec(device_t)
        dataset = _PixelDataset(paths, self.image_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device_t.type == "cuda",
        )
        initial = []
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="Encode cues", ncols=80):
                initial.append(model.g_a(images.to(device_t)).cpu())
        new_latents = nn.Parameter(torch.cat(initial, 0).to(device_t))

        net_params = [new_latents]
        aux_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith(".quantiles"):
                aux_params.append(parameter)
            else:
                net_params.append(parameter)
        param_groups = [{"params": [new_latents], "lr": latent_lr}]
        if len(net_params) > 1:
            param_groups.append({"params": net_params[1:], "lr": decoder_lr})
        optimizer = torch.optim.Adam(param_groups)
        aux_optimizer = torch.optim.Adam(aux_params, lr=decoder_lr) if aux_params else None

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device_t.type == "cuda",
        )
        model.train()
        model.g_a.eval()
        for epoch in range(epochs):
            totals = {"loss": 0.0, "bpp": 0.0, "mse": 0.0, "n": 0}
            for images, indexes in tqdm(train_loader, desc=f"Memory {epoch + 1}/{epochs}", ncols=80):
                images = images.to(device_t)
                cues = new_latents[indexes.to(device_t)]
                y_hat, likelihoods = model.entropy_bottleneck(cues)
                x_hat = model.g_s(y_hat).clamp(0, 1)
                loss, bpp, mse = _rate_distortion(x_hat, images, likelihoods, lmbda)
                optimizer.zero_grad(set_to_none=True)
                if aux_optimizer is not None:
                    aux_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(net_params, clip_max_norm)
                optimizer.step()
                if aux_optimizer is not None:
                    aux_loss = model.aux_loss()
                    aux_loss.backward()
                    aux_optimizer.step()
                batch_n = images.shape[0]
                totals["loss"] += loss.item() * batch_n
                totals["bpp"] += float(bpp) * batch_n
                totals["mse"] += float(mse) * batch_n
                totals["n"] += batch_n
            denom = max(totals["n"], 1)
            print(
                f"memory epoch {epoch + 1}/{epochs} "
                f"loss={totals['loss'] / denom:.4f} "
                f"bpp={totals['bpp'] / denom:.4f} "
                f"mse={totals['mse'] / denom:.6f}"
            )

        added = new_latents.detach().cpu()
        if self.cues is None:
            self.cues = added
        else:
            if added.shape[1:] != self.cues.shape[1:]:
                raise RuntimeError(
                    f"New cues have shape {tuple(added.shape[1:])}, "
                    f"bank cues have shape {tuple(self.cues.shape[1:])}."
                )
            self.cues = torch.cat([self.cues, added], 0)
        for record in records:
            self.items.append(
                {
                    "source_path": record["source_path"],
                    "origin_rel_path": record["origin_rel_path"],
                    "label": int(record["label"]),
                    "source": record["source"],
                    "true_class": int(record["true_class"]),
                    "stage": int(record["stage"]),
                    "case_key": record.get("case_key"),
                    "class_name": record.get("class_name"),
                    "pred": _plain(record.get("pred")),
                    "confidence": _plain(record.get("confidence")),
                    "split": "working",
                    "replay_image": None,
                }
            )
        self.save_codec(model)
        self.save()
        return len(records)

    def commit(self, *, batch_size: int = DEFAULT_BATCH_SIZE, device: str | None = None) -> int:
        """Export working-memory cues and move those rows into long-term memory."""
        pending = [i for i, item in enumerate(self.items) if item["split"] == "working"]
        if not pending:
            return 0
        if self.cues is None:
            raise RuntimeError("Working memory has index rows but no cues.")
        device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = self.load_codec(device_t)
        model.eval()
        image_dir = self.root / IMAGE_DIR
        exported = 0
        with torch.no_grad():
            for start in tqdm(range(0, len(pending), batch_size), desc="Export memory", ncols=80):
                batch_ids = pending[start : start + batch_size]
                y_hat, _ = model.entropy_bottleneck(self.cues[batch_ids].to(device_t))
                decoded = model.g_s(y_hat).clamp(0, 1).cpu()
                for local, index in enumerate(batch_ids):
                    path = image_dir / f"{index:06d}.png"
                    _save_png(decoded[local], path)
                    self.items[index]["replay_image"] = str(path.resolve())
                    self.items[index]["split"] = "long_term"
                    exported += 1
        self.save()
        return exported


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


def records_from_al_labels(
    manifest: dict,
    train_root: str | None,
    known_sources: Iterable[str],
) -> tuple[list[dict], int]:
    """Training rows not already stored. Unlabeled rows are left out."""
    known = set(known_sources)
    records = []
    skipped = 0
    for item in manifest["items"]:
        if item.get("source") == "unlabeled":
            continue
        if "label" not in item:
            raise KeyError("AL item is missing label.")
        source_path = str(resolve_image(item["rel_path"], train_root).resolve())
        if source_path in known:
            skipped += 1
            continue
        true_class = _as_int(item.get("true_class"), int(item["label"]))
        stage = _as_int(item.get("stage"), -1)
        records.append(
            {
                "source_path": source_path,
                "origin_rel_path": item["rel_path"],
                "label": int(item["label"]),
                "source": item.get("source", "true"),
                "true_class": true_class,
                "stage": stage,
                "case_key": item.get("case_key"),
                "class_name": item.get("class_name"),
                "pred": item.get("pred"),
                "confidence": item.get("confidence"),
            }
        )
        known.add(source_path)
    return records, skipped


def replay_items(bank: MemoryBank, current_manifest: dict | None, train_root: str | None) -> tuple[list[dict], int]:
    """Long-term recalls followed by the current stage's raw images."""
    items = []
    for stored in bank.items:
        if stored["split"] != "long_term":
            continue
        if not stored.get("replay_image"):
            raise RuntimeError(
                f"Long-term cue {stored['source_path']} has no exported image. "
                "Run consolidate_memory.py without --no_commit first."
            )
        recalled = {
            "rel_path": stored["replay_image"],
            "label": stored["label"],
            "source": stored["source"],
            "true_class": stored["true_class"],
            "stage": stored["stage"],
            "replay": "smc",
            "origin_rel_path": stored["origin_rel_path"],
            "source_path": stored["source_path"],
        }
        for key in ("case_key", "class_name", "pred", "confidence"):
            if stored.get(key) is not None:
                recalled[key] = stored[key]
        items.append(recalled)

    stored_sources = {item["source_path"] for item in bank.items if item["split"] == "long_term"}
    skipped = 0
    if current_manifest is not None:
        for item in current_manifest["items"]:
            if item.get("source") == "unlabeled":
                continue
            source_path = str(resolve_image(item["rel_path"], train_root).resolve())
            if source_path in stored_sources:
                skipped += 1
                continue
            current = dict(item)
            current["origin_rel_path"] = item["rel_path"]
            current["rel_path"] = source_path
            current["source_path"] = source_path
            current["replay"] = "raw"
            current["source"] = item.get("source", "true")
            current["label"] = int(item["label"])
            current["true_class"] = _as_int(item.get("true_class"), int(item["label"]))
            current["stage"] = _as_int(item.get("stage"), -1)
            items.append(current)
    return items, skipped
