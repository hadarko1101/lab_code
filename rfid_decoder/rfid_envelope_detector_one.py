import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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

def apply_envelope_detection(time_ms, voltages, first_cutoff_hz=15000.0, second_cutoff_hz=5000.0):
    """
    Applies a two-stage digital envelope detection pipeline.
    """
    dt_seconds = (time_ms[1] - time_ms[0]) / 1000.0
    fs = 1.0 / dt_seconds
    nyquist = 0.5 * fs
    print(f"[OK] Calculated Sampling Rate: {fs/1e6:.2f} MSa/s")
    
    # ---------------------------------------------------------
    # STAGE 1: First Envelope (Remove main carrier)
    # ---------------------------------------------------------
    print(f"Applying Stage 1: Rectification & LPF ({first_cutoff_hz/1e3:.1f} kHz)...")
    rectified_volts = np.abs(voltages)
    b1, a1 = butter(4, first_cutoff_hz / nyquist, btype='low', analog=False)
    env1 = filtfilt(b1, a1, rectified_volts)
    
    # ---------------------------------------------------------
    # STAGE 2: Move to 0 (AC Couple) & Second Envelope
    # ---------------------------------------------------------
    print(f"Applying Stage 2: DC Removal & Second LPF ({second_cutoff_hz/1e3:.1f} kHz)...")
    
    # Move it to 0 by subtracting the mean (removes DC offset)
    env1_centered = env1 - np.mean(env1)
    
    # Rectify the centered signal to catch the subcarrier/ripple envelope
    rectified_env1 = np.abs(env1_centered)
    
    # Apply a tighter second Low-Pass Filter to smooth out the spikes
    b2, a2 = butter(4, second_cutoff_hz / nyquist, btype='low', analog=False)
    raw_env2 = filtfilt(b2, a2, rectified_env1)
    
    # ---------------------------------------------------------
    # STAGE 3: Keep the final signal at 0
    # ---------------------------------------------------------
    # Rectifying in Stage 2 pushed the signal above 0 again. 
    # Subtracting the mean here forces the final envelope to stay centered at 0V.
    final_envelope = raw_env2 - np.mean(raw_env2)
    
    return env1, env1_centered, final_envelope

def plot_and_save_results(time_ms, env1, env1_centered, final_envelope, output_folder="results"):
    plt.rcParams['agg.path.chunksize'] = 10000

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("Generating and saving the graph...")
    
    # Create a 3-subplot figure to visualize the transformation
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: The original first envelope
    ax1.plot(time_ms, env1, color="gray", label="Stage 1: First Envelope (with DC)", linewidth=1.2)
    ax1.set_title("Step 1: First Envelope Detection")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper right")

    # Plot 2: Moved to 0
    ax2.plot(time_ms, env1_centered, color="blue", label="Centered at 0V", linewidth=1.2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title("Step 2: Signal Moved to 0 (DC Offset Removed)")
    ax2.set_ylabel("Voltage (V)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="upper right")

    # Plot 3: Final smoothed envelope (Kept at 0)
    ax3.plot(time_ms, final_envelope, color="#d62728", label="Stage 2: Final Smoothed Envelope (0V Centered)", linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1) # Added 0V line for reference
    ax3.set_title("Step 3: Second Envelope (Spikes Removed & Kept at 0V)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Voltage (V)")
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc="upper right")

    plt.tight_layout()

    output_path = os.path.join(output_folder, "two_stage_envelope_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Successfully saved envelope graph to: '{output_path}'")
    
    plt.show()

def detect_and_plot_envelope(csv_filename="rfid_capture_data.csv", output_folder="results"):
    print("="*80)
    print("RFID 2-Stage Envelope Detection & Digital Filtering")
    print("="*80)
    
    csv_path = locate_csv_file(csv_filename)
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find '{csv_filename}'.")
        return
        
    try:
        time_ms, voltages = load_rfid_data(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Run the updated DSP pipeline
    env1, env1_centered, final_envelope = apply_envelope_detection(time_ms, voltages)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_output_folder = os.path.join(script_dir, output_folder)

    # Plot the results
    plot_and_save_results(time_ms, env1, env1_centered, final_envelope, resolved_output_folder)

if __name__ == "__main__":
    detect_and_plot_envelope()