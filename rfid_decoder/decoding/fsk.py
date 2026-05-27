from __future__ import annotations

import numpy as np


def decode_fsk(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    bit_rate_hz: float = 2_000,
    low_cycles_per_bit: int = 8,
    high_cycles_per_bit: int = 10,
) -> list[int]:
    """Decode simple FSK by counting zero crossings per bit cell."""
    data = np.asarray(signal, dtype=float)
    samples_per_bit = max(1, int(round(sample_rate_hz / bit_rate_hz)))
    expected_mid = (low_cycles_per_bit + high_cycles_per_bit) / 2.0

    bits: list[int] = []
    for start in range(0, data.size - samples_per_bit + 1, samples_per_bit):
        window = data[start : start + samples_per_bit]
        centered = window - float(np.mean(window))
        crossings = int(np.count_nonzero(np.diff(np.signbit(centered))))
        cycles = crossings / 2.0
        bits.append(1 if cycles >= expected_mid else 0)
    return bits
