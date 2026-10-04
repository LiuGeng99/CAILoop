"""Shared paths and constants for multi-center active-learning adaptation."""

from __future__ import annotations

import os
from pathlib import Path

CROSS_AL_ROOT = Path(__file__).resolve().parent
OUT_DIR = CROSS_AL_ROOT / "outputs"


def _data_root() -> Path:
    raw = os.environ.get("CAILOOP_DATA_ROOT")
    if not raw:
        raise FileNotFoundError(
            "Set CAILOOP_DATA_ROOT. Clinical images are not included in this repository."
        )
    return Path(raw)


def external_data_root() -> Path:
    return _data_root() / "external_v2"


def internal_case_root() -> Path:
    return _data_root() / "v5"

COHORTS = [
    ("dezhou", "Cohort 1"),
    ("qianfoshan", "Cohort 2"),
    ("qingdao", "Cohort 3"),
    ("weihai", "Cohort 4"),
    ("yantai", "Cohort 5"),
    ("linyi", "Cohort 6"),
]

# fig2 forest-plot case counts (test_case_*); Mal = classes 0-6
FIG2_CASE_COUNTS = {
    "dezhou": {"n": 229, "tumor": 31, "nontumor": 198},
    "qianfoshan": {"n": 721, "tumor": 104, "nontumor": 617},
    "qingdao": {"n": 762, "tumor": 199, "nontumor": 563},
    "weihai": {"n": 892, "tumor": 130, "nontumor": 762},
    "yantai": {"n": 889, "tumor": 494, "nontumor": 395},
    "linyi": {"n": 931, "tumor": 273, "nontumor": 658},
}

SPLIT_NAME = "dataset_split_0.2_al"
TRAIN_RATIO = 0.2
SPLIT_SEED = 42

# Weights are supplied by the user. They are not included in this repository.
V2_0204_DIR = Path(os.environ.get("CAILOOP_CKPT", "checkpoints/dinov2_exp_v2_0204"))
V2_0204_DINO = V2_0204_DIR / "dino_best.pth"
V2_0204_CLS = V2_0204_DIR / "cls_best.pth"
INFER_VAL_V2_0204 = V2_0204_DIR / "infer_val_new.json"
THRESHOLDS_JSON = OUT_DIR / "thresholds_v2_0204_val_tau99.json"

MAL_IDS_30 = list(range(7))  # 0-6 malignant in the 30-class taxonomy
NUM_CLASSES_30 = 30
TAU = 0.99
MIN_N = 20
MIN_T_SMALL = 0.99

CLASS_DIRS = {
    "Tumor": "000_Tumor",
    "NonTumor": "001_NonTumor",
}


# Local path remaps are not shipped. Add pairs here only in a private checkout.
FIG2_PATH_REWRITES = ()


def test_case_root(center: str) -> Path:
    return internal_case_root() / f"test_case_{center}"


def split_root(center: str) -> Path:
    return external_data_root() / center / SPLIT_NAME


def split_image_root(center: str, split: str) -> Path:
    return split_root(center) / split


def infer_json(center: str, split: str) -> Path:
    return OUT_DIR / "infer" / f"{center}_{split}.json"


def labels_json(center: str, arm: str = "al") -> Path:
    return OUT_DIR / "labels" / f"{center}_{arm}.json"


def split_csv(center: str) -> Path:
    return OUT_DIR / "splits" / f"{center}.csv"


def run_dir(center: str, arm: str, run_name: str | None = None) -> Path:
    if run_name:
        return OUT_DIR / "runs" / center / f"{arm}__{run_name}"
    return OUT_DIR / "runs" / center / arm


def eval_dir(center: str, arm: str) -> Path:
    return OUT_DIR / "eval" / center / arm


def pred30_to_binary(pred: int) -> int:
    """0 = Tumor / malignant, 1 = NonTumor. Matches ImageFolder 000_/001_."""
    return 0 if int(pred) in MAL_IDS_30 else 1
