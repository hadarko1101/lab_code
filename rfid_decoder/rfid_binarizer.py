import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def locate_csv_file(filename="rfid_capture_data.csv"):
    """
    Locates the CSV file by checking the current directory, the parent directory,
    and standard relative paths to ensure robust execution regardless of how
    the script is called.
    """
    # Check current directory
    if os.path.exists(filename):
        return filename
        
    # Check parent directory (e.g., if running from inside rfid_decoder/)
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
        
    # Check workspace root absolute paths if possible
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_check = os.path.join(os.path.dirname(script_dir), filename)
    if os.path.exists(absolute_check):
        return absolute_check
        
    return filename

def load_rfid_data(csv_path):
    """
    Reads the RFID capture data from the CSV file.
    Supports both 'Time (ms)'/'Voltage (V)' and 'Time_ms'/'Voltage_V' formats.
    """
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # Resolve time column name (handle spaces or underscores)
    time_col = None
    for col in ['Time (ms)', 'Time_ms', 'time_ms', 'Time']:
        if col in df.columns:
            time_col = col
            break
    if not time_col:
        time_col = df.columns[0]
        
    # Resolve voltage column name
    volt_col = None
    for col in ['Voltage (V)', 'Voltage_V', 'voltage_v', 'Voltage']:
        if col in df.columns:
            volt_col = col
            break
    if not volt_col:
        volt_col = df.columns[1]
        
    print(f"[OK] Identified columns: '{time_col}' and '{volt_col}'")
    return df[time_col].values, df[volt_col].values

def apply_envelope_detection(time_ms, voltages, cutoff_hz=15000.0):
    """
    Applies full-wave rectification and zero-phase Butterworth low-pass filtering.
    """
    # Calculate sample interval and sample rate Fs
    dt_seconds = (time_ms[1] - time_ms[0]) / 1000.0
    fs = 1.0 / dt_seconds
    
    # Full-wave rectification
    rectified_volts = np.abs(voltages)
    
    # Low-Pass Filter Design
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    # Apply forward-backward filtering to avoid group delay
    envelope = filtfilt(b, a, rectified_volts)
    return envelope

def plot_and_save_bitstream(time_ms, envelope, threshold_v, digital_wave, output_folder="results"):
    """
    Plots the analog envelope vs threshold and the resulting digital bitstream,
    saving the high-resolution visualization to the results directory.
    """
    # Ensure target output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[OK] Created output directory: '{output_folder}'")
        
    print("Generating digital bitstream graph...")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top Plot: The Analog Envelope with the Threshold Line
    ax1.plot(time_ms, envelope, color="#d62728", label="15 kHz Analog Envelope", linewidth=1.5)
    ax1.axhline(threshold_v, color="black", linestyle="--", label=f"Threshold ({threshold_v:.3f} V)")
    ax1.set_title("Step 1: Analog Envelope vs Dynamic Threshold", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper right")

    # Bottom Plot: The Digital Square Wave
    # We use a step plot to make the digital transitions perfectly vertical
    ax2.step(time_ms, digital_wave, color="#2ca02c", label="Digital Square Wave (Bits)", linewidth=2.0, where='post')
    ax2.set_title("Step 2: Extracted Digital Bitstream", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Logic Level")
    ax2.set_ylim(-0.2, 1.2) # Set Y limits slightly outside 0 and 1 for visual clarity
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="upper right")

    # Zoom in to a 10ms window so the individual bits are clearly visible
    # We start a few milliseconds in to skip any initial filter noise
    zoom_start = min(2.0, np.max(time_ms) * 0.1)
    zoom_end = min(zoom_start + 10.0, np.max(time_ms))
    ax1.set_xlim(zoom_start, zoom_end) 

    plt.tight_layout()

    # Save visualization to disk
    output_path = os.path.join(output_folder, "digital_bitstream_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Successfully saved bitstream graph to: '{output_path}'")
    
    # Display plot
    plt.show()

def extract_and_plot_digital_bits(csv_filename="rfid_capture_data.csv", output_folder="results"):
    print("="*80)
    print("RFID Digital Binarization: Analog Envelope to Square Wave")
    print("="*80)
    
    # 1. Locate the CSV
    csv_path = locate_csv_file(csv_filename)
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find '{csv_filename}'. Please ensure it is in the same directory or the parent directory.")
        return
        
    # 2. Read the CSV
    try:
        time_ms, voltages = load_rfid_data(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load data from {csv_path}: {e}")
        return

    # 3. DSP Pipeline: Envelope Detection
    envelope = apply_envelope_detection(time_ms, voltages)

    # 4. DSP Pipeline: Binarization (Dynamic Threshold)
    print("Calculating dynamic threshold and extracting bits...")
    threshold_v = np.mean(envelope)
    print(f"[OK] Dynamic Threshold set at: {threshold_v:.4f} V")
    
    # Extract binary waves (1 or 0)
    digital_wave = (envelope > threshold_v).astype(int)

    # 5. Resolve output folder relative to script location for predictability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_output_folder = os.path.join(script_dir, output_folder)

    # 6. Plotting and Saving Results
    plot_and_save_bitstream(time_ms, envelope, threshold_v, digital_wave, resolved_output_folder)

if __name__ == "__main__":
    extract_and_plot_digital_bits()
