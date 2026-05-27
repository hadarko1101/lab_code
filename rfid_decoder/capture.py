from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CaptureSamples:
    voltage: np.ndarray
    sample_rate_hz: float
    time_seconds: np.ndarray | None = None


def load_csv_capture(path: str | Path, *, sample_rate_hz: float | None = None) -> CaptureSamples:
    """Load CSV samples.

    Supported formats:
    - two columns: time, voltage
    - one column: voltage, with sample_rate_hz supplied
    Header rows are allowed. Time units in headers like "Time (ms)" are converted
    to seconds before sample-rate inference.
    """
    rows: list[list[float]] = []
    header: list[str] | None = None
    with Path(path).open("r", newline="") as file:
        reader = csv.reader(file)
        for raw_row in reader:
            if not raw_row:
                continue
            try:
                rows.append([float(cell.strip()) for cell in raw_row[:2] if cell.strip()])
            except ValueError:
                if header is None:
                    header = raw_row
                continue

    if not rows:
        raise ValueError(f"No numeric samples found in {path}")

    widths = {len(row) for row in rows}
    if widths == {1}:
        if sample_rate_hz is None:
            raise ValueError("Voltage-only CSV needs sample_rate_hz")
        voltage = np.array([row[0] for row in rows], dtype=float)
        return CaptureSamples(voltage=voltage, sample_rate_hz=float(sample_rate_hz))

    if not widths.issubset({1, 2}):
        raise ValueError("CSV must contain one or two numeric columns")

    paired_rows = [row for row in rows if len(row) == 2]
    time_scale = _time_scale_from_header(header)
    time_seconds = np.array([row[0] * time_scale for row in paired_rows], dtype=float)
    voltage = np.array([row[1] for row in paired_rows], dtype=float)
    inferred_rate = _infer_sample_rate(time_seconds)
    return CaptureSamples(voltage=voltage, sample_rate_hz=inferred_rate, time_seconds=time_seconds)


def capture_from_picoscope(*args, **kwargs):
    """PicoScope capture placeholder."""
    raise NotImplementedError("PicoScope capture is not implemented yet")


def _infer_sample_rate(time_seconds: np.ndarray) -> float:
    if time_seconds.size < 2:
        raise ValueError("Need at least two timestamps to infer sample rate")
    deltas = np.diff(time_seconds)
    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        raise ValueError("Timestamps must be strictly increasing")
    return 1.0 / median_delta


def _time_scale_from_header(header: list[str] | None) -> float:
    if not header:
        return 1.0

    time_label = header[0].strip().lower()
    if "(ms)" in time_label or "millisecond" in time_label:
        return 1e-3
    if "(us)" in time_label or "(µs)" in time_label or "microsecond" in time_label:
        return 1e-6
    if "(ns)" in time_label or "nanosecond" in time_label:
        return 1e-9
    return 1.0
