from __future__ import annotations

import numpy as np


def slice_signal(signal: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return np.array([], dtype=np.uint8)
    if threshold is None:
        low = float(np.percentile(data, 20))
        high = float(np.percentile(data, 80))
        threshold = (low + high) / 2.0
    return (data >= threshold).astype(np.uint8)
