#!/usr/bin/env python3
"""Case-level 20/80 split aligned to the external cohorts.

Universe = $CAILOOP_DATA_ROOT/v5/test_case_{center}. A case is
{class}_{case|single}_{id}. Binary label is malignant iff the 30-class
id is 0-6.

Writes ImageFolder trees under:
  $CAILOOP_DATA_ROOT/external_v2/{center}/dataset_split_0.2_al/{train,test}/
and mappings to cross_al/outputs/splits/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_al.common import (
    CLASS_DIRS,
    COHORTS,
    external_data_root,
    FIG2_CASE_COUNTS,
    FIG2_PATH_REWRITES,
    MAL_IDS_30,
    OUT_DIR,
    SPLIT_SEED,
    TRAIN_RATIO,
    split_root,
    test_case_root,
)

NAME_RE = re.compile(
    r"^(?P<cls>.+?)_(?P<kind>case|single)_(?P<cid>\d+)_image_",
    re.IGNORECASE,
)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _n_train(n: int, ratio: float) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 0
    k = int(round(n * ratio))
    return max(1, min(n - 1, k))


def _assign_splits(case_rows: list[tuple[str, str]], seed: int, ratio: float) -> dict[str, str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for case_id, case_label in case_rows:
        by_label[case_label].append(case_id)

    assigned = {}
    for label in ("Tumor", "NonTumor"):
        ids = sorted(by_label.get(label, []))
        rng = random.Random(seed)
        rng.shuffle(ids)
        k = _n_train(len(ids), ratio)
        for i, cid in enumerate(ids):
            assigned[cid] = "train" if i < k else "test"
    return assigned


def _rewrite_moved(target: str) -> str | None:
    for old, new in FIG2_PATH_REWRITES:
        if target.startswith(old):
            cand = new + target[len(old) :]
            if os.path.isfile(cand):
                return cand
    return None


def _resolve_src(path: Path) -> tuple[str | None, bool]:
    """Return (readable_path, path_moved). Labels stay fig2; only the file may move."""
    try:
        if path.exists():
            src = str(path.resolve()) if path.is_symlink() else str(path)
            return src, False
    except OSError:
        pass
    if not path.is_symlink():
        return None, False
    try:
        target = os.readlink(path)
    except OSError:
        return None, False
    if os.path.isfile(target):
        return target, False
    rewritten = _rewrite_moved(target)
    if rewritten:
        return rewritten, True
    return None, False


def inventory_center(center: str) -> pd.DataFrame:
    root = test_case_root(center)
    if not root.is_dir():
        raise FileNotFoundError(root)

    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_id = int(cls_dir.name.split("_", 1)[0])
        binary = "Tumor" if class_id in MAL_IDS_30 else "NonTumor"
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            if not f.is_file() and not f.is_symlink():
                continue
            m = NAME_RE.match(f.name)
            if not m:
                raise ValueError(f"Unparseable fig2 filename: {f}")
            case_id = f"{cls_dir.name}_{m.group('kind')}_{m.group('cid')}"
            src, path_moved = _resolve_src(f)
            rows.append(
                {
                    "case_id": case_id,
                    "class_id": class_id,
                    "class_name": cls_dir.name,
                    "kind": m.group("kind"),
                    "cid": m.group("cid"),
                    "rel_path": f"{cls_dir.name}/{f.name}",
                    "test_case_path": str(f),
                    "full_original_path": src,
                    "image_label": binary,
                    "case_label": binary,
                    "mapped_filename": f.name,
                    "readable": src is not None,
                    "path_moved": path_moved,
                }
            )
    if not rows:
        raise RuntimeError(f"{center}: empty test_case folder")
    return pd.DataFrame(rows)


def _assert_fig2(center: str, df: pd.DataFrame) -> None:
    exp = FIG2_CASE_COUNTS[center]
    cases = df.drop_duplicates("case_id")
    n = int(len(cases))
    n_t = int((cases["case_label"] == "Tumor").sum())
    n_n = n - n_t
    if (n, n_t, n_n) != (exp["n"], exp["tumor"], exp["nontumor"]):
        raise RuntimeError(
            f"{center} case counts {n} ({n_t}/{n_n}) != fig2 "
            f"{exp['n']} ({exp['tumor']}/{exp['nontumor']})"
        )


def split_center(center: str, seed: int, ratio: float, force: bool) -> dict:
    df = inventory_center(center)
    _assert_fig2(center, df)

    case_label = df.drop_duplicates("case_id").set_index("case_id")["case_label"].to_dict()
    split_of = _assign_splits(sorted(case_label.items()), seed=seed, ratio=ratio)

    dest_root = split_root(center)
    if dest_root.exists():
        if not force:
            raise FileExistsError(f"{dest_root} exists (use --force to replace)")
        shutil.rmtree(dest_root)
    for split in ("train", "test"):
        for folder in CLASS_DIRS.values():
            (dest_root / split / folder).mkdir(parents=True, exist_ok=True)

    records = []
    n_linked = 0
    for rec in df.to_dict(orient="records"):
        split = split_of[rec["case_id"]]
        rec["split"] = split
        if rec["readable"]:
            folder = CLASS_DIRS[rec["image_label"]]
            link_path = dest_root / split / folder / rec["mapped_filename"]
            os.symlink(rec["full_original_path"], link_path)
            n_linked += 1
        records.append(rec)

    mapping = pd.DataFrame(records)
    mapping_path = dest_root / "split_mapping_record.csv"
    mapping.to_csv(mapping_path, index=False)

    exp_dir = OUT_DIR / "splits"
    exp_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(exp_dir / f"{center}.csv", index=False)
    unread = mapping[~mapping["readable"]]
    miss_path = exp_dir / f"{center}_unreadable.csv"
    if miss_path.exists():
        miss_path.unlink()
    if len(unread):
        unread.to_csv(miss_path, index=False)
    moved = mapping[mapping["path_moved"]] if "path_moved" in mapping.columns else mapping.iloc[0:0]
    moved_path = exp_dir / f"{center}_path_moved.csv"
    if moved_path.exists():
        moved_path.unlink()
    if len(moved):
        moved.to_csv(moved_path, index=False)
    old_miss = exp_dir / f"{center}_missing.csv"
    if old_miss.exists():
        old_miss.unlink()

    def _n_img(split: str, label: str | None = None) -> int:
        sub = mapping[(mapping["split"] == split) & mapping["readable"]]
        if label is not None:
            sub = sub[sub["image_label"] == label]
        return int(len(sub))

    def _n_cases(split: str, label: str | None = None, readable_only: bool = False) -> int:
        sub = mapping[mapping["split"] == split]
        if readable_only:
            sub = sub[sub["readable"]]
        if label is not None:
            sub = sub[sub["case_label"] == label]
        return int(sub["case_id"].nunique())

    unusable_cases = sorted(
        set(mapping.loc[~mapping["readable"], "case_id"])
        - set(mapping.loc[mapping["readable"], "case_id"])
    )
    fig = FIG2_CASE_COUNTS[center]
    summary = {
        "center": center,
        "source": str(test_case_root(center)),
        "seed": seed,
        "train_ratio": ratio,
        "fig2_cases": fig["n"],
        "fig2_tumor": fig["tumor"],
        "fig2_nontumor": fig["nontumor"],
        "unreadable_images": int((~mapping["readable"]).sum()),
        "path_moved_images": int(mapping["path_moved"].sum()) if "path_moved" in mapping.columns else 0,
        "unusable_cases": unusable_cases,
        "images": {
            "train": _n_img("train"),
            "test": _n_img("test"),
            "train_tumor": _n_img("train", "Tumor"),
            "train_nontumor": _n_img("train", "NonTumor"),
            "test_tumor": _n_img("test", "Tumor"),
            "test_nontumor": _n_img("test", "NonTumor"),
        },
        "cases": {
            "train": _n_cases("train"),
            "test": _n_cases("test"),
            "train_tumor": _n_cases("train", "Tumor"),
            "train_nontumor": _n_cases("train", "NonTumor"),
            "test_tumor": _n_cases("test", "Tumor"),
            "test_nontumor": _n_cases("test", "NonTumor"),
        },
        "usable_cases": {
            "train": _n_cases("train", readable_only=True),
            "test": _n_cases("test", readable_only=True),
        },
        "n_linked": n_linked,
        "mapping_csv": str(mapping_path),
    }
    with open(dest_root / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--centers", nargs="*", default=[c for c, _ in COHORTS])
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    unknown = set(args.centers) - {c for c, _ in COHORTS}
    if unknown:
        raise ValueError(f"Unknown centers: {sorted(unknown)}")

    summaries = []
    for center in args.centers:
        print(f"===== {center} =====")
        summary = split_center(center, seed=args.seed, ratio=args.train_ratio, force=args.force)
        summaries.append(summary)
        img, case = summary["images"], summary["cases"]
        fig = FIG2_CASE_COUNTS[center]
        print(
            f"  fig2 {fig['n']} ({fig['tumor']}/{fig['nontumor']}) | "
            f"train {case['train']} cases / {img['train']} img "
            f"(T {case['train_tumor']}/{img['train_tumor']}, "
            f"N {case['train_nontumor']}/{img['train_nontumor']}) | "
            f"test {case['test']} cases / {img['test']} img "
            f"(T {case['test_tumor']}/{img['test_tumor']}, "
            f"N {case['test_nontumor']}/{img['test_nontumor']})"
        )
        if summary.get("path_moved_images"):
            print(f"  path_moved images (fig2 label kept): {summary['path_moved_images']}")
        if summary["unusable_cases"]:
            print(f"  unusable (files gone): {summary['unusable_cases']}")

    rows = []
    for s in summaries:
        row = {
            "center": s["center"],
            "fig2_n": s["fig2_cases"],
            "fig2_tumor": s["fig2_tumor"],
            "fig2_nontumor": s["fig2_nontumor"],
            "split_n": s["cases"]["train"] + s["cases"]["test"],
            "unreadable_images": s["unreadable_images"],
            "path_moved_images": s.get("path_moved_images", 0),
            "unusable_cases": len(s["unusable_cases"]),
        }
        for k, v in s["images"].items():
            row[f"img_{k}"] = v
        for k, v in s["cases"].items():
            row[f"case_{k}"] = v
        rows.append(row)
    table = pd.DataFrame(rows)
    out_csv = OUT_DIR / "splits" / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    tot_fig = table["fig2_n"].sum()
    tot_split = table["split_n"].sum()
    tot_t = table["fig2_tumor"].sum()
    tot_n = table["fig2_nontumor"].sum()
    print(f"\nWrote {out_csv}")
    print(f"Pooled cases {tot_split} (fig2 {tot_fig}; Mal {tot_t} / Non-mal {tot_n})")
    print(f"ImageFolders under {external_data_root()}/<center>/{split_root('dezhou').name}/")
    if tot_split != 4424 or tot_t != 1231 or tot_n != 3193:
        raise SystemExit("Pooled totals do not match fig2 4424 (1231/3193)")


if __name__ == "__main__":
    main()
