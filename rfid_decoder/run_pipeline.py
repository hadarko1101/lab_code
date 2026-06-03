import ctypes
import os
import sys
import time
from pathlib import Path
import numpy as np

# Import your custom DSP processors
from SignalProcessor import SignalProcessor
from BitProcessor import BitProcessor

PICOSDK_DLL_NAME = "ps2000a.dll"

# --- SYSTEM CONFIGURATION ---
SAMPLE_TIME_MS = 100.0  # 100ms gives us ~10 frames per second, plenty of time to catch payloads
EMPTY_LOOP_TIMEOUT = 5  # How many empty blocks before we reset the user to NONE
AWG_STABILIZE_SECONDS = 3.0

# --- USER DATABASE ---
# Replace these keys with the actual 45-bit Manchester payloads you decoded earlier!
USER_DATABASE = {
    "000000010100001110111101001100001110111101100": "FINK (gay)",
    "000000010110001110111101001100001011010011101": "LEVI (Sigma)",
}

def _candidate_picosdk_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_hints = (os.environ.get("PICOSDK_DIR"), os.environ.get("PICO_SDK_DIR"), os.environ.get("PICOSCOPE_SDK_DIR"))
    for hint in env_hints:
        if hint: candidates.append(Path(hint))
    program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    relative_dirs = (
        "Pico Technology\\SDK\\lib", "Pico Technology\\SDK\\lib64", "Pico Technology\\SDK\\lib\\x64",
        "Pico Technology\\PicoScope 7 T&M Stable\\drivers", "Pico Technology\\PicoScope 7 Automotive\\drivers",
    )
    for base in program_files:
        if not base: continue
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
        if not dll_path.exists(): continue
        if hasattr(os, "add_dll_directory"): os.add_dll_directory(str(candidate_dir))
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
        raise RuntimeError("Could not load PicoSDK driver 'ps2000a.dll'.") from exc
    return ps, assert_pico_ok

def main():
    # 1. Initialize DSP Processors
    sig_proc = SignalProcessor(first_cutoff_hz=15000.0, second_cutoff_hz=5000.0)
    bit_proc = BitProcessor(threshold_volts=0.0)
    
    # State tracking variables
    active_user = "NONE"
    empty_loops = 0

    # 2. Connect to PicoScope
    ps, assert_pico_ok = _load_picosdk()
    print("="*80)
    print(f"Live RFID Reader: PicoScope 2206B")
    print("Press CTRL+C to exit safely.")
    print("="*80)

    status = {}
    chandle = ctypes.c_int16()
    status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)
    assert_pico_ok(status["openunit"])

    try:
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps2000aMaximumValue(chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])
        maxADC_value = maxADC.value

        # Configure Channel A (±5V)
        status["setChA"] = ps.ps2000aSetChannel(chandle, 0, 1, 0, 8, 0)
        assert_pico_ok(status["setChA"])

        # Turn on AWG Carrier Wave (125 kHz)
        status["setSigGenBuiltIn"] = ps.ps2000aSetSigGenBuiltIn(
            chandle, 0, 2000000, 0, 125000, 125000, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        assert_pico_ok(status["setSigGenBuiltIn"])
        print(f"[*] Allowing coil field to stabilize for {AWG_STABILIZE_SECONDS:.1f}s...")
        time.sleep(AWG_STABILIZE_SECONDS)
        
        # Configure Timebase
        timebase = 9
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        status["getTimebase2"] = ps.ps2000aGetTimebase2(
            chandle, timebase, 1, ctypes.byref(timeIntervalns), 0, ctypes.byref(returnedMaxSamples), 0
        )
        assert_pico_ok(status["getTimebase2"])
        
        maxSamples = int((SAMPLE_TIME_MS * 1e6) / timeIntervalns.value)
        bufferMaxA = (ctypes.c_int16 * maxSamples)()
        cmaxSamples = ctypes.c_int32(maxSamples)
        overflow = ctypes.c_int16()

        status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(
            chandle, 0, ctypes.byref(bufferMaxA), 0, maxSamples, 0, 0
        )
        assert_pico_ok(status["setDataBuffersA"])

        print("\n[*] Hardware initialized. Beginning live capture loop...\n")

        # ---------------------------------------------------------
        # THE LIVE STREAMING LOOP
        # ---------------------------------------------------------
        while True:
            # A. Capture Block
            cmaxSamples.value = maxSamples
            status["runBlock"] = ps.ps2000aRunBlock(chandle, 0, maxSamples, timebase, 0, 0, 0, 0, 0)
            assert_pico_ok(status["runBlock"])

            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                ps.ps2000aIsReady(chandle, ctypes.byref(ready))
                time.sleep(0.001) # Ultra-short sleep to keep CPU usage low

            # B. Retrieve Data
            status["getValues"] = ps.ps2000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow))
            assert_pico_ok(status["getValues"])

            if cmaxSamples.value <= 0:
                empty_loops += 1
                continue
            
            # Convert hardware arrays to Python Numpy arrays
            raw_adc_data = np.ctypeslib.as_array(bufferMaxA)[:cmaxSamples.value]
            voltages = (raw_adc_data * 5.0) / maxADC_value
            time_ms = (np.arange(0, len(voltages)) * timeIntervalns.value) / 1e6

            # C. Signal Processing Pipeline
            sys.stdout.write(".") # Simple visual indicator that it's scanning
            sys.stdout.flush()

            _, _, final_envelope = sig_proc.process(time_ms, voltages)
            digital_square_wave = bit_proc.apply_threshold(final_envelope)

            # Check if there is even a signal present (avoid processing dead air)
            if np.max(digital_square_wave) == 0:
                empty_loops += 1
            else:
                # D. Bit Processing Pipeline
                # We silence ALL internal prints in BitProcessor for a clean UI
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    
                    # Moved this INSIDE the silencer block!
                    quantized_str = bit_proc.get_quantized_states(time_ms, digital_square_wave)
                    
                    raw_instances = bit_proc.decode_payload(quantized_str, sync_marker="000111", num_bits=45, max_phase_errors=2)
                    consensus_payload, total_matches = bit_proc.analyze_repetitions(raw_instances, max_hamming_errors=2)

                # E. Logic Gate Requirements
                if consensus_payload and total_matches >= 2:
                    # Success! We found at least 2 repeating identical sequences
                    new_user = USER_DATABASE.get(consensus_payload, f"UNKNOWN TAG ({consensus_payload})")
                    
                    if new_user != active_user:
                        print(f"\n\n[+] TAG DETECTED! Active User changed: {active_user} -> {new_user}")
                        active_user = new_user
                    
                    empty_loops = 0 # Reset timeout counter
                else:
                    # No valid payload found in this block
                    empty_loops += 1

            # F. Timeout logic
            if empty_loops >= EMPTY_LOOP_TIMEOUT and active_user != "NONE":
                print(f"\n[-] TAG LOST. Active User resetting: {active_user} -> NONE")
                active_user = "NONE"

    except KeyboardInterrupt:
        print("\n\n[!] CTRL+C detected. Gracefully shutting down...")
    
    finally:
        # ALWAYS turn off the AWG and close the scope, even if Python crashes
        try:
            ps.ps2000aSetSigGenOff(chandle)
        except: pass
        try:
            ps.ps2000aCloseUnit(chandle)
            print("[OK] PicoScope closed and released.")
        except: pass

if __name__ == "__main__":
    main()
