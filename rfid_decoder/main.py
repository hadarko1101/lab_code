from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from capture import load_csv_capture
    from decoding import decode_fsk, decode_manchester
    from output import print_result, result_to_json
    from plots import save_signal_plot
    from protocols import autodetect_protocol
    from signal_processing import bandpass_filter, envelope_detect, lowpass_filter, slice_signal
except ImportError:
    from rfid_decoder.capture import load_csv_capture
    from rfid_decoder.decoding import decode_fsk, decode_manchester
    from rfid_decoder.output import print_result, result_to_json
    from rfid_decoder.plots import save_signal_plot
    from rfid_decoder.protocols import autodetect_protocol
    from rfid_decoder.signal_processing import (
        bandpass_filter,
        envelope_detect,
        lowpass_filter,
        slice_signal,
    )

DEFAULT_SAMPLE_RATE_HZ = 1_000_000
DATA_RATE = 64


def decode_capture(
    csv_path: str,
    *,
    modulation: str = "manchester",
    sample_rate_hz: float | None = DEFAULT_SAMPLE_RATE_HZ,
    samples_per_bit: int | None = None,
) -> dict:
    samples = load_csv_capture(csv_path, sample_rate_hz=sample_rate_hz)

    filtered = bandpass_filter(
        samples.voltage,
        samples.sample_rate_hz,
        center_hz=125_000,
        bandwidth_hz=40_000,
    )
    envelope = envelope_detect(filtered)
    baseband = lowpass_filter(envelope, samples.sample_rate_hz, cutoff_hz=5_000)
    sliced = slice_signal(baseband)
    plot_paths = save_debug_plots(csv_path, samples, baseband)

    if modulation == "manchester":
        samples_per_bit = samples_per_bit or int(samples.sample_rate_hz / (125_000 / DATA_RATE))
        bits = decode_manchester(sliced, samples.sample_rate_hz, samples_per_bit=samples_per_bit)
    elif modulation == "fsk":
        bits = decode_fsk(sliced, samples.sample_rate_hz)
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    protocol_result = autodetect_protocol(bits)
    return {
        "source": csv_path,
        "sample_rate_hz": samples.sample_rate_hz,
        "modulation": modulation,
        "bit_count": len(bits),
        "bits": "".join(str(bit) for bit in bits),
        "protocol": protocol_result.protocol,
        "card_id": protocol_result.card_id,
        "details": protocol_result.details,
        "plots": plot_paths,
    }


def save_debug_plots(csv_path: str, samples, envelope: np.ndarray) -> dict[str, str]:
    results_dir = Path("results")
    source_name = Path(csv_path).stem
    time_seconds = samples.time_seconds
    if time_seconds is None:
        time_seconds = np.arange(samples.voltage.size) / samples.sample_rate_hz

    original_path = results_dir / f"{source_name}_original.png"
    envelope_path = results_dir / f"{source_name}_envelope.png"

    save_signal_plot(
        time_seconds,
        samples.voltage,
        original_path,
        title="Original RFID Capture",
        ylabel="Voltage (V)",
    )
    save_signal_plot(
        time_seconds,
        envelope,
        envelope_path,
        title="RFID Envelope",
        ylabel="Envelope amplitude",
    )

    return {"original": str(original_path), "envelope": str(envelope_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a 125 kHz RFID capture.")
    parser.add_argument("csv", help="CSV file with time/voltage or voltage-only samples")
    parser.add_argument("--modulation", choices=("manchester", "fsk"), default="manchester")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="Required for voltage-only CSV files",
    )
    parser.add_argument(
        "--samples-per-bit",
        type=int,
        default=None,
        help="Override bit width for Manchester decoding",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of console text")
    return parser


ALLOWED_IDS = []



def main() -> None:
    args = build_parser().parse_args()
    result = decode_capture(
        args.csv,
        modulation=args.modulation,
        sample_rate_hz=args.sample_rate,
        samples_per_bit=args.samples_per_bit,
    )

    if args.json:
        print(result_to_json(result))
    else:
        print_result(result)

    if result["card_id"] not in ALLOWED_IDS:
        print("BAD CARD DETECTED!")
    else:
        print("LETS GOOO")

if __name__ == "__main__":
    main()
