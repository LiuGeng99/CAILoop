# CAILoop

Case-level decisions and training code for the laryngoscopy model in *Clinicians Close the Loop Between Diagnosis and Learning in Laryngoscopic AI*.

Clinical images are not in this repository. The demo below runs on a CPU from a synthetic file and does not load a checkpoint.

## Decision rules

- P(malignant) is the sum of class probabilities 0–6.
- Internal and external cases use the mean image score. Prospective cases use the mean of the three highest scores. A case is malignant at 0.5 or above.
- The fine-grained case class is taken from the image with the highest maximum softmax entry.
- Confidence shown to a reader is `1 - (1 - p) ** (1 / 3)`. Continual learning uses the untransformed softmax confidence.

## Demo

```bash
pip install -r requirements.txt
PYTHONPATH=. python demo/run_demo.py
python -m pytest tests/test_cailoop.py
```

`demo/demo_cases.json` is the input shape. Set `aggregation` to `mean` or `top3_mean`.

## Training

Training and inference need a GPU, PyTorch, and a local DINOv2 checkout. Install `requirements-train.txt`, set `DINOV2_PATH` and `CAILOOP_DATA_ROOT`, and supply your own images and backbone checkpoint.

- `scripts/train_diag.py` fine-tunes the diagnostic model.
- `scripts/infer_split.py` and `scripts/infer_cases.py` run inference.
- `active_learning/` is the continual-learning loop.
- `cross_al/` adapts the model to another centre.

## License

MIT License. See `LICENSE`.
