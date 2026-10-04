"""
Case-folder inference for the 30-class DINOv2 model.

Input layout (not ImageFolder):
  <cases_root>/case_<id>/*.png

Output CSV columns are designed for later joins:
  case_id, image_path, pred_label, prob_0 ... prob_29

Downstream (you provide later):
  1) case_id -> text_label
  2) text_label -> benign/malignant
  Then merge on case_id (and optionally map pred_label via class-name table).
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cailoop.runtime import dinov2_backbone


TEST_TRSF = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def collect_image_paths(cases_root: str):
    root = Path(cases_root)
    if not root.is_dir():
        raise FileNotFoundError(f"cases_root not found: {cases_root}")

    paths = sorted(root.glob("case_*/*.png"))
    if not paths:
        # fallback for nested / mixed extensions
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    records = []
    for p in paths:
        case_id = p.parent.name  # e.g. case_20fb70c540518d0f8087
        records.append((case_id, str(p)))
    return records


class CaseImageDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform or TEST_TRSF

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        case_id, path = self.records[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, case_id, path


def do_inference(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    records = collect_image_paths(args.cases_root)
    print(f"=> Found {len(records)} images under {args.cases_root}")
    if len(records) == 0:
        raise RuntimeError("No images found.")

    n_cases = len({r[0] for r in records})
    print(f"=> Across {n_cases} cases")

    dataset = CaseImageDataset(records, transform=TEST_TRSF)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("=> Building model...")
    dino_model = dinov2_backbone()(pretrained=False)
    classifier = nn.Linear(768, args.num_classes)

    print(f"=> Loading weights from {args.ckpt_path}...")
    dino_model.load_state_dict(
        torch.load(os.path.join(args.ckpt_path, "dino_best.pth"), map_location="cpu")
    )
    classifier.load_state_dict(
        torch.load(os.path.join(args.ckpt_path, "cls_best.pth"), map_location="cpu")
    )

    dino_model.to(device)
    classifier.to(device)
    dino_model.eval()
    classifier.eval()

    all_case_ids = []
    all_paths = []
    all_preds = []
    all_probs = []

    print("=> Starting inference...")
    with torch.no_grad():
        for images, case_ids, paths in tqdm(loader):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = dino_model(images)
                logits = classifier(features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_case_ids.extend(list(case_ids))
            all_paths.extend(list(paths))
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.float().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame(
        {
            "case_id": all_case_ids,
            "image_path": all_paths,
            "pred_label": all_preds,
        }
    )
    for i in range(args.num_classes):
        results_df[f"prob_{i}"] = all_probs[:, i]

    csv_path = os.path.join(args.output_dir, "inference_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"=> Inference results saved to: {csv_path}")
    print(f"=> Rows: {len(results_df)}, cases: {results_df['case_id'].nunique()}")
    print(f"=> pred_label value counts:\n{results_df['pred_label'].value_counts().sort_index()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 case-folder inference (no labels)")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Directory containing dino_best.pth and cls_best.pth",
    )
    parser.add_argument(
        "--cases_root",
        type=str,
        required=True,
        help="Root directory with case_* subfolders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/infer_cases",
        help="Directory to save inference_results.csv",
    )
    parser.add_argument("--num_classes", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    do_inference(args)
