"""Case-level decisions from image-level class probabilities.

Binary malignant versus non-malignant score:
  internal and external cohorts use the mean of image P(malignant);
  the prospective cohort uses the mean of the three highest image scores.
P(malignant) is the sum of probabilities for classes 0–6.
A case is called malignant when the score is at least 0.5.

Fine-grained diagnosis uses one image per case: the image whose maximum
softmax entry is largest. That image's predicted class is the case class.
"""

from __future__ import annotations

import numpy as np

from cailoop.taxonomy import MALIGNANT_THRESHOLD, N_CLASSES, N_MALIGNANT


def image_malignant_probability(probabilities) -> np.ndarray:
    """Sum classes 0–6 for each image. `probabilities` is (n_images, 30)."""
    values = _as_probability_matrix(probabilities)
    return values[:, :N_MALIGNANT].sum(axis=1)


def case_malignant_score(probabilities, aggregation: str = "mean") -> float:
    """Aggregate image P(malignant) into one case score."""
    scores = image_malignant_probability(probabilities)
    if scores.size == 0:
        raise ValueError("a case must contain at least one image")
    if aggregation == "mean":
        return float(scores.mean())
    if aggregation == "top3_mean":
        k = min(3, scores.size)
        return float(np.sort(scores)[-k:].mean())
    raise ValueError("aggregation must be 'mean' or 'top3_mean'")


def call_malignant(score: float, threshold: float = MALIGNANT_THRESHOLD) -> bool:
    return bool(score >= threshold)


def fine_grained_prediction(probabilities) -> tuple[int, int, float]:
    """Return (image_index, class_id, confidence) for the max-confidence image."""
    values = _as_probability_matrix(probabilities)
    confidence = values.max(axis=1)
    image_index = int(np.argmax(confidence))
    class_id = int(np.argmax(values[image_index]))
    return image_index, class_id, float(confidence[image_index])


def _as_probability_matrix(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_CLASSES:
        raise ValueError(f"probabilities must have shape (n_images, {N_CLASSES})")
    if np.any(values < 0):
        raise ValueError("probabilities must be non-negative")
    return values
