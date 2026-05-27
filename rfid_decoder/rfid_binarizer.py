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
    """
    Reads the RFID capture data from the CSV file.
    Supports both 'Time (ms)'/'Voltage (V)' and 'Time_ms'/'Voltage_V' formats.
    """
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    time_col = None
    for col in ['Time (ms)', 'Time_ms', 'time_ms', 'Time']:
        if col in df.columns:
            time_col = col
            break
    if not time_col:
        time_col = df.columns[0]
        
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
    dt_seconds = (time_ms[1] - time_ms[0]) / 1000.0
    fs = 1.0 / dt_seconds
    print(f"[OK] Calculated Sampling Rate: {fs/1e6:.2f} MSa/s")
    
    print(f"Applying digital envelope detection (Low-Pass Cutoff: {cutoff_hz/1e3:.1f} kHz)...")
    
    rectified_volts = np.abs(voltages)
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    envelope = filtfilt(b, a, rectified_volts)
    
    return rectified_volts, envelope

def binarize_signal(time_ms, envelope):
    """
    Converts the analog envelope into a strict binary signal (0s and 1s).
    Uses the median of the envelope as a robust dynamic threshold.
    """
    print("Binarizing the envelope signal...")
    threshold = np.median(envelope)
    
    # 1 if above threshold, 0 if below
    binary_signal = (envelope > threshold).astype(int)
    
    return binary_signal, threshold

def decode_manchester(time_ms, binary_signal):
    """
    Decodes a binary array into raw bits and Manchester bits.
    Assumes standard Manchester where a transition in the middle of the bit period 
    defines the data.
    """
    print("Decoding Manchester bits...")
    
    # Find all transition edges (where the signal changes state)
    edges = np.diff(binary_signal)
    edge_indices = np.where(edges != 0)[0]
    edge_times = time_ms[edge_indices]
    
    if len(edge_times) < 2:
        print("[ERROR] Not enough edges found to decode.")
        return [], [], 0, 0

    # Calculate time between consecutive edges (delta t)
    dt = np.diff(edge_times)
    
    # In Manchester encoding, edge distances are either ~T/2 or ~T.
    # We estimate T/2 (half a bit period) by finding the median of the shorter intervals.
    # Filter out microscopic noise spikes first
    valid_dt = dt[dt > 0.05] 
    
    # Sort and take the median of the lower 30% of intervals to confidently hit T/2
    sorted_dt = np.sort(valid_dt)
    half_t = np.median(sorted_dt[:max(1, len(sorted_dt)//3)])
    full_t = half_t * 2
    
    print(f"[OK] Estimated Half-Bit Period (T/2): {half_t:.3f} ms")
    print(f"[OK] Estimated Full-Bit Period (T):   {full_t:.3f} ms")
    
    raw_bits = []
    manchester_bits = []
    
    # Standard Manchester Decoding Logic:
    # We synchronize to the first edge. To safely read the bit value, 
    # we sample the binary signal at 75% of the bit period (after the mid-bit transition).
    
    current_time = edge_times[0]
    end_time = time_ms[-1]
    
    start_time = current_time
    
    # Start sampling
    while current_time + full_t < end_time:
        # Sample point: 75% into the current bit period
        sample_time = current_time + (0.75 * full_t)
        
        # Find the index in time_ms closest to our sample_time
        sample_idx = np.searchsorted(time_ms, sample_time)
        
        if sample_idx >= len(binary_signal):
            break
            
        bit_val = binary_signal[sample_idx]
        raw_bits.append(bit_val)
        
        # In typical RFID Manchester (IEEE 802.3 standard):
        # High-to-Low transition (sampling a 0 in the second half) = '0'
        # Low-to-High transition (sampling a 1 in the second half) = '1'
        # (Note: If your specific RFID tag uses Thomas encoding, flip this logic)
        manchester_bits.append(bit_val) 
        
        # Step forward by one full bit period
        current_time += full_t

    print(f"[OK] Decoded {len(manchester_bits)} bits.")
    return raw_bits, manchester_bits, start_time, full_t

def plot_and_save_results(time_ms, rectified_volts, envelope, output_folder="results"):
    """
    Plots the demodulated low-pass envelope,
    saving the high-resolution output visualization to the results folder.
    """
    plt.rcParams['agg.path.chunksize'] = 10000

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[OK] Created output directory: '{output_folder}'")
        
    print("Generating and saving the envelope graph...")
    plt.figure(figsize=(14, 6))

    plt.plot(time_ms, envelope, color="#d62728", label="15 kHz Demodulated Envelope", linewidth=1.5)

    plt.title("RFID Load Modulation: Demodulated Envelope", fontsize=14, fontweight='bold')
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    
    v_min, v_max = np.min(envelope), np.max(envelope)
    y_margin = (v_max - v_min) * 0.20
    plt.ylim(max(0, v_min - y_margin), v_max + y_margin)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = os.path.join(output_folder, "envelope_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Successfully saved envelope graph to: '{output_path}'")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_encodings(time_ms, binary_signal, man_bits, start_time, full_t, output_folder="results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("Generating and saving the encodings graph...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    
    # 1. Manchester Encoding (Binary Signal)
    ax1.step(time_ms, binary_signal, color="#1f77b4", linewidth=2.0, where='post')
    ax1.set_title("Manchester Encoding (Binary Signal)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Logic Level")
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Zoom in to a window for better visibility (first 10ms after start_time)
    zoom_start = start_time
    zoom_end = zoom_start + 10.0
    if zoom_end > time_ms[-1]:
        zoom_end = time_ms[-1]
    ax1.set_xlim(zoom_start, zoom_end)
    
    # 2. Bit Encoding (Decoded Bits)
    # Create a time axis for the bits
    bit_times = [start_time + i * full_t for i in range(len(man_bits) + 1)]
    # Duplicate the last bit value for the step plot to extend to the end
    plot_bits = man_bits + [man_bits[-1]] if len(man_bits) > 0 else []
    
    if len(plot_bits) > 0:
        ax2.step(bit_times, plot_bits, color="#ff7f0e", linewidth=2.0, where='post')
        
    ax2.set_title("Bit Encoding (Decoded Bits)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Logic Level")
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(zoom_start, zoom_end)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, "encodings_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Successfully saved encodings graph to: '{output_path}'")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

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

    # ---------------------------------------------------------
    # NEW LOGIC: Binarize and Decode
    # ---------------------------------------------------------
    # 4. Binarization
    binary_signal, threshold = binarize_signal(time_ms, envelope)
    
    # 5. Manchester Decoding
    raw_bits, man_bits, start_time, full_t = decode_manchester(time_ms, binary_signal)
    
    # Print the first 64 bits to the console
    if man_bits:
        bit_string = "".join(str(b) for b in man_bits)
        print("\n--- DECODED MANCHESTER DATA ---")
        print(f"Data Stream (First 64 bits): {bit_string[:64]}")
        print("-------------------------------\n")
    # ---------------------------------------------------------

    # 6. Resolve output folder relative to script location for predictability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_output_folder = os.path.join(script_dir, output_folder)

    # 7. Plotting and Saving Results
    plot_and_save_results(time_ms, rectified_volts, envelope, resolved_output_folder)
    plot_encodings(time_ms, binary_signal, man_bits, start_time, full_t, resolved_output_folder)

if __name__ == "__main__":
    detect_and_plot_envelope()
