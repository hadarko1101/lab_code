from __future__ import annotations

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
