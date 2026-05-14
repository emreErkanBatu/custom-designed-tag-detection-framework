# -*- coding: utf-8 -*-
"""Background photometric profiling utilities."""

# background_profile.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PatchProfile:
    """Photometric and quality profile of an image or ROI."""
    mean_L: float
    std_L: float
    mean_a: float
    mean_b: float
    noise: float
    lap_var: float
    mean_bgr: Tuple[float, float, float]  # (B, G, R)


def _noise_score(gray: np.ndarray) -> float:
    """Simple noise estimate based on the residual of gray - GaussianBlur(gray)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(resid))


def compute_patch_profile(bgr: np.ndarray,
                          roi: Optional[Tuple[int, int, int, int]] = None) -> PatchProfile:
    """
    bgr: BGR image.
    roi: (x0, y0, x1, y1), the crop region in scene coordinates.
         If None, the profile is computed from the full image.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image received.")

    if roi is not None:
        x0, y0, x1, y1 = roi
        patch = bgr[y0:y1, x0:x1]
    else:
        patch = bgr

    if patch is None or patch.size == 0:
        raise ValueError("Empty ROI received; the tag may extend outside the scene.")

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    mean_L = float(np.mean(L))
    std_L  = float(np.std(L))
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    noise = _noise_score(gray)
    lap_var = float(lap.var())

    mean_bgr = (
        float(np.mean(patch[:, :, 0])),
        float(np.mean(patch[:, :, 1])),
        float(np.mean(patch[:, :, 2])),
    )

    return PatchProfile(
        mean_L=mean_L, std_L=std_L,
        mean_a=mean_a, mean_b=mean_b,
        noise=noise, lap_var=lap_var,
        mean_bgr=mean_bgr
    )
