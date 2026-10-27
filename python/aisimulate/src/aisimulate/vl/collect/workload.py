# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible calibration images; generating and encoding them is not frontend time."""

from __future__ import annotations

import base64
import io


def generate_images(height: int, width: int, count: int, encoding: str, *, seed: int = 1729) -> list[bytes]:
    """RGB gradients with per-image noise so every image has a distinct identity."""
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[:height, :width]
    base = np.stack(
        (xx * 191 / max(width, 1), yy * 191 / max(height, 1), (xx + yy) * 127 / max(width + height, 1)), axis=-1
    )
    images = []
    for index in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        rgb = np.clip(base + rng.integers(0, 48, size=(height, width, 3)), 0, 255).astype(np.uint8)
        output = io.BytesIO()
        if encoding == "jpeg":
            Image.fromarray(rgb).save(output, format="JPEG", quality=85, subsampling=2)
        else:
            Image.fromarray(rgb).save(output, format="PNG", compress_level=6)
        images.append(output.getvalue())
    return images


def image_data_url(encoded: bytes, encoding: str) -> str:
    return f"data:image/{encoding};base64," + base64.b64encode(encoded).decode("ascii")
