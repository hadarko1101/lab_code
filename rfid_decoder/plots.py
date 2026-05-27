from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_signal(time_seconds: np.ndarray, signal: np.ndarray, *, title: str = "RFID signal") -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time_seconds, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_stages(stages: dict[str, np.ndarray], sample_rate_hz: float) -> None:
    import matplotlib.pyplot as plt

    count = len(stages)
    fig, axes = plt.subplots(count, 1, sharex=True)
    if count == 1:
        axes = [axes]

    for axis, (name, signal) in zip(axes, stages.items()):
        time_seconds = np.arange(len(signal)) / sample_rate_hz
        axis.plot(time_seconds, signal)
        axis.set_title(name)
        axis.grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()


def save_signal_plot(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
    ylabel: str = "Amplitude",
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(time_seconds, signal, linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
