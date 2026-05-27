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
    return bits


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
