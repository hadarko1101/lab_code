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


def decode_hid_prox_from_edges(
    digital: np.ndarray,
    sample_rate_hz: float,
    *,
    min_pulse_us: float = 20,
) -> tuple[list[int], dict]:
    """Recover raw HID-like bits by classifying pulse widths from envelope edges.

    Short and long edge intervals are separated with a two-cluster threshold. The
    returned bits are the classified pulse-width stream, which is then suitable
    for repeated 35-bit frame search.
    """
    data = np.asarray(digital, dtype=np.uint8)
    if data.size < 2:
        return [], {"reason": "not enough samples"}

    edges = np.flatnonzero(np.diff(data) != 0) + 1
    if edges.size < 3:
        return [], {"reason": "not enough edges"}

    widths_samples = np.diff(edges)
    min_samples = max(1, int(round(sample_rate_hz * min_pulse_us / 1_000_000)))
    widths_samples = widths_samples[widths_samples >= min_samples]
    if widths_samples.size < 2:
        return [], {"reason": "not enough valid pulse widths"}

    short_center, long_center = _two_cluster_centers(widths_samples.astype(float))
    threshold = (short_center + long_center) / 2.0
    bits = [1 if width >= threshold else 0 for width in widths_samples]

    return bits, {
        "edge_count": int(edges.size),
        "pulse_count": int(widths_samples.size),
        "short_pulse_us": short_center / sample_rate_hz * 1_000_000,
        "long_pulse_us": long_center / sample_rate_hz * 1_000_000,
        "pulse_threshold_us": threshold / sample_rate_hz * 1_000_000,
    }


def _two_cluster_centers(values: np.ndarray) -> tuple[float, float]:
    low = float(np.percentile(values, 25))
    high = float(np.percentile(values, 75))

    for _ in range(12):
        threshold = (low + high) / 2.0
        low_values = values[values <= threshold]
        high_values = values[values > threshold]
        if low_values.size == 0 or high_values.size == 0:
            break
        low = float(np.mean(low_values))
        high = float(np.mean(high_values))

    return (low, high) if low <= high else (high, low)
