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
    Applies a digital envelope detection DSP pipeline:
    1. Full-wave rectification (absolute value)
    2. Zero-phase 4th-order low-pass Butterworth filter to extract the envelope.
    """
    # Calculate sample interval and sample rate Fs
    # time_ms is in milliseconds, convert to seconds to compute Fs in Hz
    dt_seconds = (time_ms[1] - time_ms[0]) / 1000.0
    fs = 1.0 / dt_seconds
    print(f"[OK] Calculated Sampling Rate: {fs/1e6:.2f} MSa/s")
    
    print(f"Applying digital envelope detection (Low-Pass Cutoff: {cutoff_hz/1e3:.1f} kHz)...")
    
    # Step A: Rectify the AC signal (Full-wave rectification)
    rectified_volts = np.abs(voltages)
    
    # Step B: Low-Pass Filter design
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    
    # Create 4th-order Butterworth low-pass filter
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    # Apply forward & backward digital filtering to eliminate phase delay
    envelope = filtfilt(b, a, rectified_volts)
    
    return rectified_volts, envelope

def plot_and_save_results(time_ms, rectified_volts, envelope, output_folder="results"):
    """
    Plots the rectified high-frequency carrier and the demodulated low-pass envelope,
    saving the high-resolution output visualization to the results folder.
    """
    # Ensure target output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[OK] Created output directory: '{output_folder}'")
        
    print("Generating and saving the graph...")
    plt.figure(figsize=(14, 6))

    # Plot the rectified carrier in the background (light blue, very thin)
    plt.plot(time_ms, rectified_volts, color="#aec7e8", label="Rectified 125 kHz Carrier", linewidth=0.2)
    
    # Plot the smoothed envelope on top (bold red, thicker line)
    plt.plot(time_ms, envelope, color="#d62728", label="15 kHz Demodulated Envelope", linewidth=1.5)

    # Style plot to resemble an oscilloscope output
    plt.title("RFID Load Modulation: Demodulated Envelope", fontsize=14, fontweight='bold')
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    
    # Dynamically zoom the Y-axis slightly around the envelope boundaries to make modulation drops distinct
    v_min, v_max = np.min(envelope), np.max(envelope)
    y_margin = (v_max - v_min) * 0.20
    plt.ylim(max(0, v_min - y_margin), v_max + y_margin)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save visualization to disk
    output_path = os.path.join(output_folder, "envelope_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Successfully saved envelope graph to: '{output_path}'")
    
    # Display plot
    plt.show()

def detect_and_plot_envelope(csv_filename="rfid_capture_data.csv", output_folder="results"):
    print("="*80)
    print("RFID Envelope Detection & Digital Filtering")
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
    rectified_volts, envelope = apply_envelope_detection(time_ms, voltages)

    # 4. Resolve output folder relative to script location for predictability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_output_folder = os.path.join(script_dir, output_folder)

    # 5. Plotting and Saving Results
    plot_and_save_results(time_ms, rectified_volts, envelope, resolved_output_folder)

if __name__ == "__main__":
    detect_and_plot_envelope()
