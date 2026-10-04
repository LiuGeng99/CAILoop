# CAILoop

Case-level decisions for the laryngoscopy model in *Clinicians Close the Loop Between Diagnosis and Learning in Laryngoscopic AI*.

This repository contains the published case-level decision rules, plus the training and inference source. It does not contain clinical images or trained weights. The reviewer demo below runs on a CPU from a synthetic JSON file. Training and inference need a separate environment, a local DINOv2 checkout, and data you supply.

The rules are:

- P(malignant) for an image is the sum of class probabilities 0–6.
- Internal and external case scores are the mean of those image scores. The prospective case score is the mean of the three highest image scores. A case is called malignant when the score is at least 0.5.
- The fine-grained case class is the class predicted for the single image with the highest maximum softmax entry.
- A confidence shown to a reader is remapped as `1 - (1 - p) ** (1 / 3)`. Continual-learning gates use the untransformed softmax confidence. The full algorithm is described in the Methods section of the manuscript.

## System requirements

- Operating system: Linux, macOS, or Windows. Tested on Linux 5.15.
- Python 3.10 or newer. Tested with Python 3.13 and numpy 1.26.4.
- No non-standard hardware. The demo runs on a CPU.
- Training and folder inference need an NVIDIA GPU. The manuscript training run used four NVIDIA A100 GPUs.

## Installation

From this directory:

```bash
pip install -r requirements.txt
```

On a normal desktop this takes under 2 minutes when numpy is not already installed, and a few seconds when it is.

## Run the demo

```bash
PYTHONPATH=. python demo/run_demo.py
```

Expected output:

```text
case_id            cohort        score   call        fine_grained          display_conf
internal_demo      internal       0.5025  malignant     Epi_Mal      img 3   0.5519
prospective_demo   prospective    0.4000  non-malignant NC_Norm      img 0   0.6316
```

Expected run time on a normal desktop: under 5 seconds, including interpreter startup.

`internal_demo` uses the mean of four image scores, 0.10, 0.20, 0.80 and 0.91, which is 0.5025 and is called malignant. Its highest softmax entry is 0.91 on image 3, class `Epi_Mal`. `prospective_demo` uses the mean of the three highest image scores, 0.70, 0.40 and 0.10, which is 0.4000 and is called non-malignant.

Check the same numbers with:

```bash
python -m pytest tests/test_cailoop.py
```

## Run on your data

Replace `demo/demo_cases.json` with a file of the same shape. Each case needs `case_id`, `cohort`, `aggregation` (`mean` or `top3_mean`), and `images`. Each image needs `image_id` and `probabilities`, a list of 30 non-negative class probabilities. Then run:

```bash
PYTHONPATH=. python demo/run_demo.py
```

Use `mean` for an internal or external case and `top3_mean` for a prospective case. In Python:

```python
from cailoop import case_malignant_score, display_confidence, fine_grained_prediction

score = case_malignant_score(probabilities, aggregation="mean")
image_index, class_id, confidence = fine_grained_prediction(probabilities)
shown = display_confidence(confidence)
```

`probabilities` is an array of shape `(n_images, 30)`.

## Training and inference

These scripts are not part of the CPU demo. Install them separately:

```bash
pip install -r requirements-train.txt
git clone https://github.com/facebookresearch/dinov2
export DINOV2_PATH="$PWD/dinov2"
export CAILOOP_DATA_ROOT=/path/to/images
```

`CAILOOP_DATA_ROOT` is the directory that holds the image folders. The scripts expect this layout, and the images themselves are not in this repository:

```text
v5/train
v5/test
v5/train_new
v5/val_new
v5/test_new
v5/test_holdout
v5/train_new/stages_v1.json
prospective/dataset
diag/train
diag/test
cil/train
cil/test
loc/train
loc/val
multi/weihai
external_v2/<center>/
```

`scripts/train_diag.py` is the diagnostic fine-tune. Its defaults follow the Methods: 30 classes, AdamW, weight decay 1e-4, EMA beta 0.9999, learning rate 1e-5, batch size 128 per process, the backbone frozen for 5 epochs and then fine-tuned for 95 epochs. The backbone is loaded from a DINOv2 ViT-B/14 register checkpoint that you pass in. Weights are written under `checkpoints/diag`.

```bash
torchrun --nproc_per_node=4 scripts/train_diag.py \
  --backbone_ckpt /path/to/dinov2_vitb14_reg4_pretrain.pth
```

Folder-split inference and case-folder inference:

```bash
python scripts/infer_split.py --ckpt_path checkpoints/diag
python scripts/infer_cases.py \
  --ckpt_path checkpoints/diag \
  --cases_root /path/to/cases
```

`active_learning/` is the stage-wise continual-learning loop. `cross_al/` is the cross-centre adaptation loop. Set `CAILOOP_CKPT` to a directory that contains `dino_best.pth` and `cls_best.pth` before running the cross-centre scripts. Their outputs go under `active_learning/outputs/` and `cross_al/outputs/`.

On a machine that already has the CUDA build of PyTorch, the extra install is a few minutes. A fresh CUDA wheel download takes longer. A full four-GPU training run is on the order of the original experiment, not a desktop demo.

## Reproducing the manuscript

Reproducing the published cohort metrics requires the clinical images and the trained DINOv2 weights. Those files are not in this repository. The decision rules above are the ones used for the reported case-level binary and fine-grained results.

## License

Apache License 2.0. See `LICENSE`.
