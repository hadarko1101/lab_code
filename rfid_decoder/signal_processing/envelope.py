from __future__ import annotations

import numpy as np


def envelope_detect(signal: np.ndarray, *, smooth_samples: int = 25) -> np.ndarray:
    rectified = np.abs(np.asarray(signal, dtype=float))
    if smooth_samples <= 1:
        return rectified
    kernel = np.ones(smooth_samples, dtype=float) / smooth_samples
    return np.convolve(rectified, kernel, mode="same")
