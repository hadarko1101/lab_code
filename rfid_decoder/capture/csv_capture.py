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
    - two columns: time_seconds, voltage
    - one column: voltage, with sample_rate_hz supplied
    Header rows are allowed if they contain non-numeric text.
    """
    rows: list[list[float]] = []
    with Path(path).open("r", newline="") as file:
        reader = csv.reader(file)
        for raw_row in reader:
            if not raw_row:
                continue
            try:
                rows.append([float(cell.strip()) for cell in raw_row[:2] if cell.strip()])
            except ValueError:
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
    time_seconds = np.array([row[0] for row in paired_rows], dtype=float)
    voltage = np.array([row[1] for row in paired_rows], dtype=float)
    inferred_rate = _infer_sample_rate(time_seconds)
    return CaptureSamples(voltage=voltage, sample_rate_hz=inferred_rate, time_seconds=time_seconds)


def _infer_sample_rate(time_seconds: np.ndarray) -> float:
    if time_seconds.size < 2:
        raise ValueError("Need at least two timestamps to infer sample rate")
    deltas = np.diff(time_seconds)
    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        raise ValueError("Timestamps must be strictly increasing")
    return 1.0 / median_delta
