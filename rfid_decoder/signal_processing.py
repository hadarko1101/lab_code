from __future__ import annotations

import numpy as np


CARRIER_HZ = 125_000


def remove_dc(signal: np.ndarray) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return data.copy()
    return data - float(np.mean(data))


def bandpass_filter(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    center_hz: float = CARRIER_HZ,
    bandwidth_hz: float = 30_000,
) -> np.ndarray:
    """Keep only the carrier band around 125 kHz."""
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return data.copy()

    freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(data)

    half_width = bandwidth_hz / 2.0
    keep = (freqs >= center_hz - half_width) & (freqs <= center_hz + half_width)
    spectrum[~keep] = 0

    return np.fft.irfft(spectrum, n=data.size)


def carrier_envelope(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    carrier_hz: float = CARRIER_HZ,
    cycles: float = 4.0,
) -> np.ndarray:
    """Estimate carrier amplitude with local RMS."""
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return data.copy()

    samples_per_cycle = sample_rate_hz / carrier_hz
    window = max(3, int(round(samples_per_cycle * cycles)))
    return np.sqrt(moving_average(data * data, window))


def smooth_envelope(
    envelope: np.ndarray,
    sample_rate_hz: float,
    *,
    smoothing_us: float = 100.0,
) -> np.ndarray:
    """Smooth the envelope until carrier ripple is gone."""
    window = max(1, int(round(sample_rate_hz * smoothing_us / 1_000_000)))
    return moving_average(envelope, window)


def normalize_0_to_1(signal: np.ndarray) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return data.copy()

    low = float(np.percentile(data, 5))
    high = float(np.percentile(data, 95))
    if high <= low:
        return np.zeros_like(data)

    return np.clip((data - low) / (high - low), 0.0, 1.0)


def hysteresis_threshold(
    signal: np.ndarray,
    *,
    low_threshold: float = 0.40,
    high_threshold: float = 0.60,
) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return np.array([], dtype=np.uint8)

    state = 1 if data[0] >= 0.5 else 0
    digital = np.zeros(data.size, dtype=np.uint8)

    for index, value in enumerate(data):
        if state == 0 and value >= high_threshold:
            state = 1
        elif state == 1 and value <= low_threshold:
            state = 0
        digital[index] = state

    return digital


def process_rfid_signal(
    raw_voltage: np.ndarray,
    sample_rate_hz: float,
    *,
    carrier_hz: float = CARRIER_HZ,
) -> dict[str, np.ndarray]:
    """Simple RFID signal chain.

    raw Pico voltage
    -> remove DC
    -> isolate 125 kHz carrier
    -> estimate carrier amplitude envelope
    -> smooth envelope
    -> normalize to 0..1
    -> threshold with hysteresis
    -> output digital 0/1 signal
    """
    dc_removed = remove_dc(raw_voltage)
    carrier = bandpass_filter(dc_removed, sample_rate_hz, center_hz=carrier_hz)
    envelope = carrier_envelope(carrier, sample_rate_hz, carrier_hz=carrier_hz)
    smoothed = smooth_envelope(envelope, sample_rate_hz)
    normalized = normalize_0_to_1(smoothed)
    digital = hysteresis_threshold(normalized)

    return {
        "dc_removed": dc_removed,
        "carrier": carrier,
        "envelope": envelope,
        "smoothed": smoothed,
        "normalized": normalized,
        "digital": digital,
    }


def slice_signal(signal: np.ndarray) -> np.ndarray:
    normalized = normalize_0_to_1(signal)
    return hysteresis_threshold(normalized)


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.size == 0 or window <= 1:
        return data.copy()

    window = min(int(window), data.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(data, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")
