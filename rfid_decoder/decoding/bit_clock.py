from __future__ import annotations

import numpy as np


def estimate_samples_per_bit(digital: np.ndarray, *, min_run: int = 2) -> int:
    """Estimate bit width from median transition spacing."""
    data = np.asarray(digital, dtype=np.uint8)
    if data.size < 2:
        raise ValueError("Need more samples to estimate bit clock")

    transitions = np.flatnonzero(np.diff(data) != 0) + 1
    if transitions.size < 2:
        raise ValueError("No transitions found for bit clock recovery")

    spacings = np.diff(transitions)
    spacings = spacings[spacings >= min_run]
    if spacings.size == 0:
        raise ValueError("Transitions are too close to estimate bit clock")

    half_bit = int(round(float(np.median(spacings))))
    return max(2, half_bit * 2)


def sample_bits(digital: np.ndarray, samples_per_bit: int) -> list[int]:
    data = np.asarray(digital, dtype=np.uint8)
    if samples_per_bit <= 0:
        raise ValueError("samples_per_bit must be positive")

    bits: list[int] = []
    for start in range(0, data.size - samples_per_bit + 1, samples_per_bit):
        window = data[start : start + samples_per_bit]
        bits.append(int(np.mean(window) >= 0.5))
    return bits
