import numpy as np
from collections import Counter

class BitProcessor:
    def __init__(self, threshold_volts=0.0):
        self.threshold = threshold_volts

    def apply_threshold(self, envelope):
        """Converts the analog envelope into a digital square wave."""
        return (envelope > self.threshold).astype(int)

    def get_quantized_states(self, time_ms, bitstream):
        """
        Measures the width of every pulse, calculates the base 1T time, 
        and converts the wave into a string of discrete states (e.g., "1100011101").
        """
        print("[*] Quantizing analog pulses into discrete T-states...")
        edges = np.where(np.diff(bitstream) != 0)[0]
        if len(edges) < 2:
            return ""

        durations = []
        polarities = []
        
        for i in range(len(edges) - 1):
            idx_start = edges[i]
            idx_end = edges[i+1]
            durations.append(time_ms[idx_end] - time_ms[idx_start])
            polarities.append(bitstream[idx_start + 1])

        valid_dt = np.array(durations)[np.array(durations) > 0.05]
        base_t = np.percentile(valid_dt, 10) 
        print(f"[*] Calculated Base Pulse Width (1T): {base_t:.4f} ms")

        quantized_str = ""
        for dur, pol in zip(durations, polarities):
            t_count = int(round(dur / base_t))
            if t_count > 10: 
                t_count = 10 
                
            char = "1" if pol == 1 else "0"
            quantized_str += char * t_count

        return quantized_str

    def decode_payload(self, quantized_str, sync_marker="000111", num_bits=45, max_phase_errors=2):
        """
        Finds the sync violation and extracts the subsequent Manchester payload,
        tolerating up to a user-defined threshold of corrupted Manchester pairs.
        """
        print(f"[*] Searching for Sync Marker: '{sync_marker}' (allowing up to {max_phase_errors} phase errors)...")
        
        start_idx = 0
        all_captured_instances = []

        while True:
            idx = quantized_str.find(sync_marker, start_idx)
            if idx == -1: 
                break
            
            payload_start = idx + len(sync_marker)
            expected_chars = num_bits * 2
            
            if payload_start + expected_chars <= len(quantized_str):
                raw_manchester = quantized_str[payload_start : payload_start + expected_chars]
                
                decoded_bits = []
                phase_errors = 0
                
                # Analyze two characters at a time
                for i in range(0, len(raw_manchester), 2):
                    pair = raw_manchester[i:i+2]
                    if pair == "10":
                        decoded_bits.append(1)
                    elif pair == "01":
                        decoded_bits.append(0)
                    else:
                        # We hit a '00' or '11' phase violation due to noise
                        phase_errors += 1
                        # Make an educated guess using the first half of the symbol phase
                        decoded_bits.append(1 if pair[0] == "1" else 0)
                
                # If this window didn't exceed our maximum allowed structural noise
                if phase_errors <= max_phase_errors:
                    payload_str = "".join(map(str, decoded_bits))
                    all_captured_instances.append({
                        "index": idx,
                        "payload": payload_str,
                        "phase_errors": phase_errors
                    })
            
            start_idx = idx + 1
            
        return all_captured_instances

    def analyze_repetitions(self, captured_instances, max_hamming_errors=2):
        """
        Finds the consensus payload string and calculates how many times it repeats
        across the capture. Explicitly alerts the user if only a single payload is found.
        """
        if not captured_instances:
            return None, 0

        # Step 1: Identify the most frequent payload to act as our true reference string
        raw_strings = [inst["payload"] for inst in captured_instances]
        string_counts = Counter(raw_strings)
        consensus_payload = string_counts.most_common(1)[0][0]
        
        # ---------------------------------------------------------
        # NEW: Handle the Single Capture Edge Case
        # ---------------------------------------------------------
        if len(captured_instances) == 1:
            print(f"\n[!] ALERT: Exactly ONE payload sequence was found in the capture.")
            print(f"[-] Because it does not repeat, we cannot use Hamming distance to verify its integrity.")
            return consensus_payload, 1
        # ---------------------------------------------------------
        
        print(f"\n[+] Dominant Reference Payload Identified: {consensus_payload}")
        print(f"[*] Analyzing all captures against reference (Max allowed Hamming errors: {max_hamming_errors})...")
        
        valid_match_count = 0
        
        # Step 2: Compare every captured instance to the reference payload
        for inst in captured_instances:
            p_str = inst["payload"]
            
            # Calculate Hamming Distance (mismatched bits)
            hamming_dist = sum(b1 != b2 for b1, b2 in zip(consensus_payload, p_str))
            
            if hamming_dist <= max_hamming_errors:
                valid_match_count += 1
                print(f"    -> Match found at string index {inst['index']}: {p_str}")
                print(f"       [Metrics] Bit deviations: {hamming_dist} | Sync phase errors: {inst['phase_errors']}")
                
        return consensus_payload, valid_match_count