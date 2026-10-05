"""Four-way rotation head for the diagnostic trainer.

Each training image is shown at 0, 90, 180 and 270 degrees. The classifier
has ``4 * num_classes`` outputs: slot ``k * num_classes + c`` is class ``c``
at rotation ``k``. Evaluation averages those four logit blocks back to
``num_classes``. The image is not rotated again at evaluation time.
"""

from __future__ import annotations

import torch


def rotate_batch(inputs, targets, num_orig_classes: int):
    """Expand one batch into four rotations and remap the labels.

    ``inputs`` is ``[B, C, H, W]`` and ``targets`` is ``[B]``. The returned
    tensors are ``[4B, C, H, W]`` and ``[4B]``.
    """
    x_0 = inputs
    x_90 = torch.rot90(inputs, k=1, dims=[-2, -1])
    x_180 = torch.rot90(inputs, k=2, dims=[-2, -1])
    x_270 = torch.rot90(inputs, k=3, dims=[-2, -1])
    rotated_inputs = torch.cat([x_0, x_90, x_180, x_270], dim=0)

    t_0 = targets
    t_90 = targets + num_orig_classes
    t_180 = targets + 2 * num_orig_classes
    t_270 = targets + 3 * num_orig_classes
    rotated_targets = torch.cat([t_0, t_90, t_180, t_270], dim=0)
    return rotated_inputs, rotated_targets


def aggregate_rotation_logits(outputs, num_orig_classes: int):
    """Average a four-way head back to ``num_orig_classes`` logits.

    A head that is already ``num_orig_classes`` wide is returned unchanged,
    so a checkpoint trained without the rotation head still runs.
    """
    n_out = outputs.size(1)
    if n_out == num_orig_classes:
        return outputs
    if n_out != 4 * num_orig_classes:
        raise ValueError(
            f"classifier has {n_out} outputs; expected {num_orig_classes} "
            f"or {4 * num_orig_classes}"
        )
    return outputs.view(outputs.size(0), 4, num_orig_classes).mean(dim=1)


def build_classifier(state, num_orig_classes: int):
    """Build the linear head stored in ``state``.

    Accepts either a plain ``num_orig_classes`` head or the four-way head.
    """
    import torch.nn as nn

    n_out = int(state["weight"].shape[0])
    if n_out not in (num_orig_classes, 4 * num_orig_classes):
        raise ValueError(
            f"checkpoint head has {n_out} outputs; expected {num_orig_classes} "
            f"or {4 * num_orig_classes}"
        )
    layer = nn.Linear(int(state["weight"].shape[1]), n_out)
    layer.load_state_dict(state)
    return layer


def expand_classifier_state(state, num_orig_classes: int):
    """Repeat a ``num_orig_classes`` head four times to initialise the rotation head."""
    weight = state["weight"]
    n_out = int(weight.shape[0])
    if n_out == 4 * num_orig_classes:
        return state
    if n_out != num_orig_classes:
        raise ValueError(
            f"checkpoint head has {n_out} outputs; expected {num_orig_classes} "
            f"or {4 * num_orig_classes}"
        )
    return {
        "weight": weight.repeat(4, 1),
        "bias": state["bias"].repeat(4),
    }
