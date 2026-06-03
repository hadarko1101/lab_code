import os
import pandas as pd
import matplotlib.pyplot as plt
from SignalProcessor import SignalProcessor
from BitProcessor import BitProcessor

def locate_csv_file(filename="rfid_capture_data.csv"):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_check = os.path.join(os.path.dirname(script_dir), filename)
    if os.path.exists(absolute_check):
        return absolute_check
    return filename

def load_rfid_data(csv_path):
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    time_col = next((col for col in ['Time (ms)', 'Time_ms', 'time_ms', 'Time'] if col in df.columns), df.columns[0])
    volt_col = next((col for col in ['Voltage (V)', 'Voltage_V', 'voltage_v', 'Voltage'] if col in df.columns), df.columns[1])
    print(f"[OK] Identified columns: '{time_col}' and '{volt_col}'")
    return df[time_col].values, df[volt_col].values

def main():
    print("="*80)
    print("RFID DSP & Decoder Pipeline")
    print("="*80)
    
    # 1. Load Data
    csv_path = locate_csv_file()
    if not os.path.exists(csv_path):
        print("[ERROR] Could not find 'rfid_capture_data.csv'.")
        return
    time_ms, voltages = load_rfid_data(csv_path)

    # 2. Instantiate Processors
    # You can tweak these frequencies and baud rates based on your specific RFID protocol
    sig_proc = SignalProcessor(first_cutoff_hz=15000.0, second_cutoff_hz=5000.0)
    bit_proc = BitProcessor(threshold_volts=0.0)

    # 3. Run Signal Processing Pipeline
    print("Running DSP Pipeline...")
    env1, env1_centered, final_envelope = sig_proc.process(time_ms, voltages)

    # 4. Run Bit Processing Pipeline
    print("Running Digital Thresholding...")
    digital_square_wave = bit_proc.apply_threshold(final_envelope)

    print("Decoding Manchester Bitstream (Assuming 4000 baud)...")
    decoded_data = bit_proc.decode_manchester(time_ms, digital_square_wave, baud_rate_hz=4000)

    # 5. Output Results
    print("\n--- DECODED PAYLOAD ---")
    # Print the first 64 bits as a sample
    print(decoded_data[:64])
    print(f"Total bits extracted: {len(decoded_data)}")
    print("-----------------------\n")

    # Optional: Plot the thresholding step to verify it lines up with the envelope
    plt.figure(figsize=(14, 6))
    plt.plot(time_ms, final_envelope, color="#d62728", label="0V Centered Envelope", linewidth=1.5)
    
    # Scale the square wave slightly so it overlays nicely on the analog plot
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

if __name__ == "__main__":
    main()