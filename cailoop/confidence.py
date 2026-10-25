"""Display-only remapping of a softmax confidence.

This is the transform used before a confidence is shown to a reader.
It is not a calibrated probability, and continual-learning gates use the
untransformed softmax confidence.
"""

from __future__ import annotations

import numpy as np

DISPLAY_SENSITIVITY = 3.0


def display_confidence(probability, sensitivity: float = DISPLAY_SENSITIVITY):
    """Map a probability p to 1 - (1 - p) ** (1 / S)."""
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    values = np.asarray(probability, dtype=np.float64)
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("probability must be between 0 and 1")
    remapped = 1.0 - np.power(1.0 - values, 1.0 / sensitivity)
    if remapped.shape == ():
        return float(remapped)
    return remapped
