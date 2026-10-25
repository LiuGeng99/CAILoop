import argparse
import os
import pickle
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_manager import DataManager

# Configure DINOv2 path
from cailoop.rotation import aggregate_rotation_logits, build_classifier
from cailoop.runtime import dinov2_backbone

# Filename: 000_Epi_Ca_case_0000_image_0000.png
_CASE_RE = re.compile(
    r"^(?P<class_id>\d+)_.+?_(?P<case_type>case|single)_(?P<case_id>\d+)_image_(?P<image_id>\d+)\.png$",
    re.IGNORECASE,
)


def _parse_case_fields(image_path: str) -> dict:
    filename = os.path.basename(image_path)
    match = _CASE_RE.match(filename)
    if not match:
        return {
            "case_key": None,
            "case_type": None,
            "case_id": None,
            "image_id": None,
        }
    class_id = int(match.group("class_id"))
    case_type = match.group("case_type").lower()
    case_id = int(match.group("case_id"))
    return {
        "case_key": f"{class_id}_{case_type}_{case_id}",
        "case_type": case_type,
        "case_id": case_id,
        "image_id": int(match.group("image_id")),
    }


def do_inference(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    # 1. Prepare Data (shuffle=False is mandatory for inference)
    data_manager = DataManager(
        dataset_name=args.dataset_name,
        shuffle=False,
        seed=args.seed,
        init_cls=args.num_classes,
        increment=1,
    )

    test_dataset = data_manager.get_dataset(np.arange(args.num_classes), source="test", mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"=> Test images: {len(test_dataset)}")

    # 2. Build Models
    print("=> Building model...")
    dino_model = dinov2_backbone()(pretrained=False)

    # 3. Load Checkpoints
    print(f"=> Loading weights from {args.ckpt_path}...")
    dino_model.load_state_dict(torch.load(os.path.join(args.ckpt_path, "dino_best.pth"), map_location="cpu"))
    cls_state = torch.load(os.path.join(args.ckpt_path, "cls_best.pth"), map_location="cpu")
    classifier = build_classifier(cls_state, args.num_classes)

    dino_model.to(device)
    classifier.to(device)

    dino_model.eval()
    classifier.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_paths = []
    pkl_results = []

    print("=> Starting inference...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            idxs, inputs, targets, img_paths = batch_data
            inputs = inputs.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = dino_model(inputs)
                logits = classifier(features)
            logits = aggregate_rotation_logits(logits, args.num_classes)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            probs_np = probs.float().cpu().numpy()
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            idxs_np = idxs.cpu().numpy() if torch.is_tensor(idxs) else np.asarray(idxs)

            all_targets.append(targets_np)
            all_preds.append(preds_np)
            all_probs.append(probs_np)
            all_paths.extend(img_paths)

            for idx, target, path, prob in zip(idxs_np, targets_np, img_paths, probs_np):
                pkl_results.append(
                    {
                        "idx": int(idx),
                        "label": int(target),
                        "path": str(path),
                        "probabilities": prob.tolist(),
                    }
                )

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    os.makedirs(args.output_dir, exist_ok=True)

    case_fields = [_parse_case_fields(p) for p in all_paths]
    results_df = pd.DataFrame(
        {
            "image_path": all_paths,
            "case_key": [f["case_key"] for f in case_fields],
            "case_type": [f["case_type"] for f in case_fields],
            "case_id": [f["case_id"] for f in case_fields],
            "image_id": [f["image_id"] for f in case_fields],
            "true_label": all_targets,
            "pred_label": all_preds,
        }
    )
    for i in range(args.num_classes):
        results_df[f"prob_{i}"] = all_probs[:, i]

    csv_path = os.path.join(args.output_dir, "inference_results.csv")
    results_df.to_csv(csv_path, index=False)

    pkl_path = os.path.join(args.output_dir, "inference_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_results, f)

    acc = (all_preds == all_targets).mean()
    n_cases = results_df["case_key"].nunique(dropna=True)
    print(f"=> Inference CSV saved to: {csv_path}")
    print(f"=> Inference PKL saved to: {pkl_path}")
    print(f"=> Images={len(results_df)}  cases≈{n_cases}  image-acc={acc:.4f}")
    print("=> Inference Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 Inference Script")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Directory containing dino_best.pth and cls_best.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/infer_split",
        help="Directory to save inference_results.csv and inference_results.pkl",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="laryngo_holdout",
        help="Dataset key: laryngo_holdout | laryngo_val_new | laryngo_prospective | ...",
    )
    parser.add_argument("--num_classes", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    do_inference(args)
