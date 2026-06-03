from scipy.integrate import quadpack
import os
import pandas as pd
import matplotlib.pyplot as plt
from SignalProcessor import SignalProcessor
from BitProcessor import BitProcessor

def locate_csv_file(filename="data_with_fink_card2.csv"):
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

def main():
    print("="*80)
    print("RFID DSP & Error-Tolerant Repeater Pipeline")
    print("="*80)
    
    csv_path = locate_csv_file()
    if not os.path.exists(csv_path):
        print("[ERROR] Could not find 'rfid_capture_data.csv'.")
        return
        
    time_ms, voltages = load_rfid_data(csv_path)

    print("Trimming edges to remove partial packets...")
    start_time_limit = time_ms[0] + 10.0
    end_time_limit = time_ms[-1] - 10.0
    valid_indices = (time_ms >= start_time_limit) & (time_ms <= end_time_limit)
    time_ms = time_ms[valid_indices]
    voltages = voltages[valid_indices]

    sig_proc = SignalProcessor(first_cutoff_hz=15000.0, second_cutoff_hz=5000.0)
    bit_proc = BitProcessor(threshold_volts=0.0)

    print("Running DSP Pipeline...")
    env1, env1_centered, final_envelope = sig_proc.process(time_ms, voltages)

    print("Applying Digital Threshold...")
    digital_square_wave = bit_proc.apply_threshold(final_envelope)

    # Convert wave to discrete 1T states
    quantized_str = bit_proc.get_quantized_states(time_ms, digital_square_wave)

    print(len(quantized_str))

    print("\n--- STAGE 1: SIGNAL EXTRACTION ---")
    # Extract raw data windows following the sync marker (Allow up to 2 phase errors)
    raw_instances = bit_proc.decode_payload(quantized_str, sync_marker="000111", num_bits=45, max_phase_errors=2)
    print(f"[+] Extracted {len(raw_instances)} raw candidates following sync markers.")

    print("\n--- STAGE 2: CROSS-PACKET REPETITION ANALYSIS ---")
    # Compare candidates and count matches allowing up to 2 Hamming bit errors
    final_payload, total_matches = bit_proc.analyze_repetitions(raw_instances, max_hamming_errors=2)
    
    print("\n" + "="*80)
    print("FINAL DECODER REPORT")
    print("="*80)
    if final_payload:
        print(f"Decoded Tag Payload : {final_payload}")
        print(f"Total Valid Matches : {total_matches} occurrences found in capture file.")
    else:
        print("[ERROR] No repeating pattern could be validated under current error thresholds.")
    print("="*80)

if __name__ == "__main__":
    main()