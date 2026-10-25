"""Locate the DINOv2 source tree at runtime.

The backbone implementation is the public DINOv2 repository. It is not
vendored here. Clone it and point DINOV2_PATH at that checkout before
training or inference.
"""

from __future__ import annotations

import os
import sys


DINOV2_URL = "https://github.com/facebookresearch/dinov2"


def dinov2_backbone():
    path = os.environ.get("DINOV2_PATH")
    if not path:
        raise SystemExit(
            "Set DINOV2_PATH to a local clone of " + DINOV2_URL
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    from dinov2.hub.backbones import dinov2_vitb14_reg

    return dinov2_vitb14_reg
