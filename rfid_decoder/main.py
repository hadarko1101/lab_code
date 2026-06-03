import os
import pandas as pd
import matplotlib.pyplot as plt
from SignalProcessor import SignalProcessor
from BitProcessor import BitProcessor

def locate_csv_file(filename="rfid_capture_data.csv"):
    if os.path.exists(filename): return filename
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path): return parent_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_check = os.path.join(os.path.dirname(script_dir), filename)
    if os.path.exists(absolute_check): return absolute_check
    return filename

def load_rfid_data(csv_path):
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    time_col = next((col for col in ['Time (ms)', 'Time_ms', 'time_ms', 'Time'] if col in df.columns), df.columns[0])
    volt_col = next((col for col in ['Voltage (V)', 'Voltage_V', 'voltage_v', 'Voltage'] if col in df.columns), df.columns[1])
    return df[time_col].values, df[volt_col].values

<<<<<<< HEAD
def main():
    print("="*80)
    print("RFID DSP & Raw Baseband Extractor")
    print("="*80)
    
    csv_path = locate_csv_file()
    if not os.path.exists(csv_path):
        print("[ERROR] Could not find 'rfid_capture_data.csv'.")
        return
        
    time_ms, voltages = load_rfid_data(csv_path)
=======
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
>>>>>>> cb886f3100263974baeaabe043f78363eb98ba52

    sig_proc = SignalProcessor(first_cutoff_hz=15000.0, second_cutoff_hz=5000.0)
    bit_proc = BitProcessor(threshold_volts=0.0)

    print("Running DSP Pipeline...")
    env1, env1_centered, final_envelope = sig_proc.process(time_ms, voltages)

<<<<<<< HEAD
    print("Running Digital Thresholding...")
    digital_square_wave = bit_proc.apply_threshold(final_envelope)

    # We use 2525 Hz based on your previous manual math (~0.396 ms per bit). 
    # If it fails, set this to None to let the auto-estimator try.
    print("Extracting Raw Bits...")
    raw_bits = bit_proc.extract_raw_baseband(time_ms, digital_square_wave, force_baud_rate=None)

    print("\n--- PATTERN SEARCH ---")
    # Looking for common payload lengths (including Manchester-doubled lengths)
    payload = bit_proc.find_payload(raw_bits, expected_lengths=[256, 192, 128, 96, 64, 44])
    
    print(f"\nExtracted Cyclic Data:\n{payload}\n")
    print("="*80)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(time_ms, final_envelope, color="#d62728", label="0V Centered Envelope", linewidth=1.5)
    
    max_env = max(final_envelope)
    scaled_digital = [d * max_env for d in digital_square_wave]
    plt.plot(time_ms, scaled_digital, color="blue", label="Digital Bitstream", linewidth=1.0, alpha=0.7)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Analog Envelope to Digital Extraction")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
=======
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
>>>>>>> cb886f3100263974baeaabe043f78363eb98ba52

if __name__ == "__main__":
    main()