import ctypes
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PICOSDK_DLL_NAME = "ps2000a.dll"

# --- USER CONFIGURATION ---
SAMPLE_TIME_MS = 50.0  # Duration to sample in milliseconds


def _candidate_picosdk_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_hints = (
        os.environ.get("PICOSDK_DIR"),
        os.environ.get("PICO_SDK_DIR"),
        os.environ.get("PICOSCOPE_SDK_DIR"),
    )
    for hint in env_hints:
        if hint:
            candidates.append(Path(hint))

    program_files = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    relative_dirs = (
        "Pico Technology\\SDK\\lib",
        "Pico Technology\\SDK\\lib64",
        "Pico Technology\\SDK\\lib\\x64",
        "Pico Technology\\PicoScope 7 T&M Stable\\drivers",
        "Pico Technology\\PicoScope 7 Automotive\\drivers",
    )
    for base in program_files:
        if not base:
            continue
        for relative_dir in relative_dirs:
            candidates.append(Path(base) / relative_dir)

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = str(candidate)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)
    return unique_candidates


def _prepare_picosdk_environment() -> list[Path]:
    added_dirs: list[Path] = []
    for candidate_dir in _candidate_picosdk_dirs():
        dll_path = candidate_dir / PICOSDK_DLL_NAME
        if not dll_path.exists():
            continue
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(candidate_dir))
        path_value = os.environ.get("PATH", "")
        path_entries = path_value.split(os.pathsep) if path_value else []
        if str(candidate_dir) not in path_entries:
            os.environ["PATH"] = str(candidate_dir) + os.pathsep + path_value
        added_dirs.append(candidate_dir)
    return added_dirs


def _load_picosdk():
    added_dirs = _prepare_picosdk_environment()
    try:
        from picosdk.functions import assert_pico_ok
        from picosdk.ps2000a import ps2000a as ps
    except Exception as exc:
        searched_dirs = [str(path) for path in _candidate_picosdk_dirs()]
        if added_dirs:
            searched_note = "\n".join(f"  - {path}" for path in searched_dirs)
            extra = f"\nChecked these PicoSDK locations:\n{searched_note}"
        else:
            searched_note = "\n".join(f"  - {path}" for path in searched_dirs)
            extra = (
                "\nNo usable PicoSDK driver was found automatically."
                f"\nChecked these common locations:\n{searched_note}"
            )
        raise RuntimeError(
            "Could not load PicoSDK driver 'ps2000a.dll'. "
            "Install PicoScope 2000 Series drivers/SDK or set PICOSDK_DIR to the folder containing ps2000a.dll."
            f"{extra}"
        ) from exc
    return ps, assert_pico_ok

def run_simple_capture():
    ps, assert_pico_ok = _load_picosdk()

    print("="*80)
    print(f"PicoScope 2206B: RFID Capture ({SAMPLE_TIME_MS} ms)")
    print("="*80)

    print("Connecting to PicoScope 2206B...")
    status = {}
    chandle = ctypes.c_int16()
    status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)
    assert_pico_ok(status["openunit"])

    try:
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps2000aMaximumValue(chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])
        maxADC_value = maxADC.value

        # 1. Configure Channel A (±5V Range, AC Coupled)
        range_index = 8  # Index 8 is EXACTLY ±5V in ps2000a API
        actual_range_volts = 5.0
        
        status["setChA"] = ps.ps2000aSetChannel(
            chandle, 0, 1, 0, range_index, 0
        )
        assert_pico_ok(status["setChA"])
        print("[OK] Channel A configured (±5V Range, AC Coupling)")

        # 2. Turn on AWG (Boosted to 2V Peak-to-Peak to guarantee tag boots up)
        print("\n--- Initializing Signal Generator (AWG) ---")
        status["setSigGenBuiltIn"] = ps.ps2000aSetSigGenBuiltIn(
            chandle, 0, 2000000, 0, 125000, 125000, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        assert_pico_ok(status["setSigGenBuiltIn"])
        print("[OK] Signal Generator active: 125 kHz Sine Wave, 2V Peak-to-Peak")

        print("Allowing coil magnetic field to stabilize. Hold the card on the coil...")
        time.sleep(3)

        # 3. Configure Timebase for 2206B (Timebase 9 = 8.93 MS/s)
        timebase = 9
        
        # Get time interval first to calculate maxSamples dynamically
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        status["getTimebase2_init"] = ps.ps2000aGetTimebase2(
            chandle, timebase, 1, ctypes.byref(timeIntervalns), 0, ctypes.byref(returnedMaxSamples), 0
        )
        assert_pico_ok(status["getTimebase2_init"])
        
        # Calculate maxSamples dynamically based on SAMPLE_TIME_MS
        maxSamples = int((SAMPLE_TIME_MS * 1e6) / timeIntervalns.value)
        
        # Verify the timebase settings with the calculated maxSamples
        status["getTimebase2"] = ps.ps2000aGetTimebase2(
            chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), 0, ctypes.byref(returnedMaxSamples), 0
        )
        assert_pico_ok(status["getTimebase2"])

        # Calculate sample rate in MS/s (MegaSamples per second)
        sample_rate_msps = 1000.0 / timeIntervalns.value
        print(f"[OK] Timebase set to {timebase} (Sample Interval: {timeIntervalns.value:.2f} ns, Sample Rate: {sample_rate_msps:.4f} MS/s)")

        # 4. Set up Trigger (40 mV, correctly scaled to the 5V ADC range)
        trigger_adc = int((40.0 / (actual_range_volts * 1000)) * maxADC_value)
        status["setTrigger"] = ps.ps2000aSetSimpleTrigger(
            chandle, 1, 0, trigger_adc, 2, 0, 1000
        )
        assert_pico_ok(status["setTrigger"])

        # 5. Capture
        print(f"\nCapturing raw data block ({maxSamples} samples)...")
        status["runBlock"] = ps.ps2000aRunBlock(
            chandle, 0, maxSamples, timebase, 0, 0, 0, 0, 0
        )
        assert_pico_ok(status["runBlock"])

        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps2000aIsReady(chandle, ctypes.byref(ready))
            time.sleep(0.01)

        # 6. Retrieve Data
        bufferMaxA = (ctypes.c_int16 * maxSamples)()
        status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(
            chandle, 0, ctypes.byref(bufferMaxA), 0, maxSamples, 0, 0
        )
        assert_pico_ok(status["setDataBuffersA"])

        cmaxSamples = ctypes.c_int32(maxSamples)
        overflow = ctypes.c_int16()
        status["getValues"] = ps.ps2000aGetValues(
            chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow)
        )
        assert_pico_ok(status["getValues"])

        # 7. Convert to Volts (Math Bug Fixed)
        print("Retrieving values and converting to Volts...")
        raw_adc_data = np.ctypeslib.as_array(bufferMaxA)
        voltages = (raw_adc_data * actual_range_volts) / maxADC_value
        time_ms = (np.arange(0, len(voltages)) * timeIntervalns.value) / 1e6

        # Save to CSV file
        csv_filename = "data_with_fink2_card.csv"
        print(f"Saving captured data to {csv_filename}...")
        df = pd.DataFrame({"Time (ms)": time_ms, "Voltage (V)": voltages})
        df.to_csv(csv_filename, index=False)
        print(f"[OK] Successfully saved {len(df)} samples to {csv_filename}")

        # 8. Plot the waveform
        print("\nDisplaying captured raw voltage...")
        plt.figure(figsize=(14, 6))
        
        # Adapt zoom window to sample time
        zoom_start = min(1.0, SAMPLE_TIME_MS * 0.1)
        zoom_end = min(3.0, SAMPLE_TIME_MS)
        if zoom_end <= zoom_start:
            zoom_start = 0.0
            zoom_end = SAMPLE_TIME_MS
            
        zoom_mask = (time_ms >= zoom_start) & (time_ms <= zoom_end)
        zoom_time = time_ms[zoom_mask]
        zoom_voltages = voltages[zoom_mask]
        
        # Darker navy color (#0B2545) for better clarity and high contrast
        plt.plot(zoom_time, zoom_voltages, color="#0B2545", label="Voltage A (AC)", linewidth=0.1)
        
        v_min, v_max = np.min(zoom_voltages), np.max(zoom_voltages)
        y_margin = (v_max - v_min) * 0.10
        if y_margin == 0:
            y_margin = 0.1
        plt.ylim(v_min - y_margin, v_max + y_margin)
        plt.xlim(zoom_start, zoom_end) 
        
        plt.title("PicoScope 2206B: 125 kHz Carrier Wave Zoomed View", fontsize=14, fontweight='bold')
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.grid(True, linestyle='--')
        plt.legend(loc="upper right", fontsize=10)
        
        plt.tight_layout()
        plt.show()

    finally:
        print("\nStopping Signal Generator and closing scope...")
        try:
            ps.ps2000aSetSigGenOff(chandle)
        except:
            pass
            
        try:
            status["closeUnit"] = ps.ps2000aCloseUnit(chandle)
            assert_pico_ok(status["closeUnit"])
            print("[OK] Scope closed cleanly.")
        except Exception as e:
            print(f"Error closing scope: {e}")

if __name__ == "__main__":
    try:
        run_simple_capture()
    except RuntimeError as exc:
        print(exc)
