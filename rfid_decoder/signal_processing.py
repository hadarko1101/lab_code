from __future__ import annotations

import numpy as np


def bandpass_filter(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    center_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Simple FFT bandpass suitable for offline captures."""
    data = np.asarray(signal, dtype=float)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(data)
    half_width = bandwidth_hz / 2.0
    mask = (freqs >= center_hz - half_width) & (freqs <= center_hz + half_width)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=data.size)


def lowpass_filter(signal: np.ndarray, sample_rate_hz: float, *, cutoff_hz: float) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(data)
    spectrum[freqs > cutoff_hz] = 0
    return np.fft.irfft(spectrum, n=data.size)


def envelope_detect(signal: np.ndarray, *, smooth_samples: int = 25) -> np.ndarray:
    rectified = np.abs(np.asarray(signal, dtype=float))
    if smooth_samples <= 1:
        return rectified
    kernel = np.ones(smooth_samples, dtype=float) / smooth_samples
    return np.convolve(rectified, kernel, mode="same")


def slice_signal(signal: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.size == 0:
        return np.array([], dtype=np.uint8)
    if threshold is None:
        low = float(np.percentile(data, 20))
        high = float(np.percentile(data, 80))
        threshold = (low + high) / 2.0
    return (data >= threshold).astype(np.uint8)
