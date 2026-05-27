from __future__ import annotations

import numpy as np

from rfid_decoder.decoding.bit_clock import estimate_samples_per_bit


def decode_manchester(
    digital: np.ndarray,
    sample_rate_hz: float,
    *,
    samples_per_bit: int | None = None,
) -> list[int]:
    """Decode Manchester symbols from a sliced signal.

    Convention used here:
    - low-to-high transition inside a bit cell is 1
    - high-to-low transition inside a bit cell is 0
    """
    del sample_rate_hz
    data = np.asarray(digital, dtype=np.uint8)
    if samples_per_bit is None:
        samples_per_bit = estimate_samples_per_bit(data)

    half = samples_per_bit // 2
    if half < 1:
        raise ValueError("samples_per_bit is too small")

    bits: list[int] = []
    for start in range(0, data.size - samples_per_bit + 1, samples_per_bit):
        left = data[start : start + half]
        right = data[start + half : start + samples_per_bit]
        first = int(np.mean(left) >= 0.5)
        second = int(np.mean(right) >= 0.5)

        if first == 0 and second == 1:
            bits.append(1)
        elif first == 1 and second == 0:
            bits.append(0)
        else:
            continue
    return bits
