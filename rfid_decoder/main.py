from __future__ import annotations

import argparse

from rfid_decoder.capture.csv_capture import load_csv_capture
from rfid_decoder.decoding.fsk import decode_fsk
from rfid_decoder.decoding.manchester import decode_manchester
from rfid_decoder.output.console import print_result
from rfid_decoder.output.json_output import result_to_json
from rfid_decoder.protocols.autodetect import autodetect_protocol
from rfid_decoder.signal_processing.envelope import envelope_detect
from rfid_decoder.signal_processing.filters import bandpass_filter, lowpass_filter
from rfid_decoder.signal_processing.slicer import slice_signal


def decode_capture(
    csv_path: str,
    *,
    modulation: str = "manchester",
    sample_rate_hz: float | None = None,
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
    baseband = lowpass_filter(envelope, samples.sample_rate_hz, cutoff_hz=10_000)
    sliced = slice_signal(baseband)

    if modulation == "manchester":
        bits = decode_manchester(sliced, samples.sample_rate_hz, samples_per_bit=samples_per_bit)
    elif modulation == "fsk":
        bits = decode_fsk(filtered, samples.sample_rate_hz)
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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a 125 kHz RFID capture.")
    parser.add_argument("csv", help="CSV file with time/voltage or voltage-only samples")
    parser.add_argument("--modulation", choices=("manchester", "fsk"), default="manchester")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
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


if __name__ == "__main__":
    main()
