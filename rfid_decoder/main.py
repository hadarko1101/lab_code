from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from capture import load_csv_capture
    from decoding import decode_fsk, decode_hid_prox_from_edges, decode_manchester
    from output import print_result, result_to_json
    from plots import save_signal_plot
    from protocols import ProtocolResult, autodetect_protocol, decode_repeated_wiegand35, decode_wiegand35
    from signal_processing import process_rfid_signal
except ImportError:
    from rfid_decoder.capture import load_csv_capture
    from rfid_decoder.decoding import decode_fsk, decode_hid_prox_from_edges, decode_manchester
    from rfid_decoder.output import print_result, result_to_json
    from rfid_decoder.plots import save_signal_plot
    from rfid_decoder.protocols import (
        ProtocolResult,
        autodetect_protocol,
        decode_repeated_wiegand35,
        decode_wiegand35,
    )
    from rfid_decoder.signal_processing import process_rfid_signal

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
    signals = process_rfid_signal(samples.voltage, samples.sample_rate_hz)
    plot_paths = save_debug_plots(
        csv_path,
        samples,
        signals,
    )
    decoder_details = {}

    if modulation == "manchester":
        samples_per_bit = samples_per_bit or int(samples.sample_rate_hz / (125_000 / DATA_RATE))
        bits = decode_manchester(
            signals["digital"],
            samples.sample_rate_hz,
            samples_per_bit=samples_per_bit,
        )
    elif modulation == "fsk":
        bits = decode_fsk(signals["carrier"], samples.sample_rate_hz)
    elif modulation == "hid":
        bits, decoder_details = decode_hid_prox_from_edges(
            signals["digital"],
            samples.sample_rate_hz,
        )
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    if modulation == "hid":
        wiegand = decode_repeated_wiegand35(bits)
        if wiegand is None:
            candidate = decode_wiegand35(bits)
            details = {
                "reason": "No repeated 35-bit frame found",
                **decoder_details,
            }
            if candidate is not None:
                details.update(
                    {
                        "candidate_facility_code": candidate.facility_code,
                        "candidate_card_number": candidate.card_number,
                        "candidate_card_id": candidate.card_id,
                        "candidate_raw_bits": candidate.raw_bits,
                        "candidate_repeat_count": candidate.repeat_count,
                    }
                )
            protocol_result = ProtocolResult(protocol=None, card_id=None, details=details)
        else:
            protocol_result = ProtocolResult(
                protocol="HID Prox / Wiegand35",
                card_id=wiegand.card_id,
                details={
                    "facility_code": wiegand.facility_code,
                    "card_number": wiegand.card_number,
                    "raw_bits": wiegand.raw_bits,
                    "repeat_count": wiegand.repeat_count,
                    **decoder_details,
                },
            )
    else:
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


def save_debug_plots(
    csv_path: str,
    samples,
    signals: dict[str, np.ndarray],
) -> dict[str, str]:
    results_dir = Path("results")
    source_name = Path(csv_path).stem
    time_seconds = samples.time_seconds
    if time_seconds is None:
        time_seconds = np.arange(samples.voltage.size) / samples.sample_rate_hz

    envelope_path = results_dir / f"{source_name}_envelope.png"
    digital_path = results_dir / f"{source_name}_digital.png"

    save_signal_plot(
        time_seconds,
        signals["smoothed"],
        envelope_path,
        title="Envelope",
        ylabel="Envelope amplitude",
        trim_fraction=0.02,
    )
    save_signal_plot(
        time_seconds,
        signals["digital"],
        digital_path,
        title="Digital 0/1 Signal",
        ylabel="0/1",
        trim_fraction=0.02,
    )

    return {
        "envelope": str(envelope_path),
        "digital": str(digital_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a 125 kHz RFID capture.")
    parser.add_argument("csv", help="CSV file with time/voltage or voltage-only samples")
    parser.add_argument("--modulation", choices=("manchester", "fsk", "hid"), default="manchester")
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
