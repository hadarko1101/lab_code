import numpy as np

class BitProcessor:
    def __init__(self, threshold_volts=0.0):
        self.threshold = threshold_volts

    def apply_threshold(self, envelope):
        """Converts the analog envelope into a digital square wave."""
        return (envelope > self.threshold).astype(int)

    def estimate_baud_rate(self, time_ms, bitstream):
        """
        Calculates the auto-baud rate based on the shortest physical pulse widths.
        """
        edges = np.where(np.diff(bitstream) != 0)[0]
        if len(edges) < 2:
            return 2500.0 # Fallback
            
        edge_times = time_ms[edges]
        dt_ms = np.diff(edge_times)
        
        # Filter out tiny microscopic noise glitches (< 0.05 ms)
        valid_dt = dt_ms[dt_ms > 0.05]
        
        # Find the shortest common pulse width (representing 1 symbol width)
        # Using 10th percentile to ignore noise anomalies
        t_symbol_ms = np.percentile(valid_dt, 10)
        
        # Convert ms to seconds to get Baud (Hz)
        baud_rate = 1.0 / (t_symbol_ms / 1000.0)
        return baud_rate

    def extract_raw_baseband(self, time_ms, bitstream, force_baud_rate=None):
        """
        Extracts raw high/low states by sampling the square wave at the baud rate.
        No decoding (Manchester, etc.) is applied.
        """
        baud_rate = force_baud_rate if force_baud_rate else self.estimate_baud_rate(time_ms, bitstream)
        print(f"[*] Sampling raw baseband at {baud_rate:.1f} Hz")
        
        dt_seconds = (time_ms[1] - time_ms[0]) / 1000.0
        bit_duration_sec = 1.0 / baud_rate
        samples_per_bit = int(bit_duration_sec / dt_seconds)
        
        # Find the first edge to synchronize our sampling "clock"
        edges = np.where(np.diff(bitstream) != 0)[0]
        if len(edges) == 0:
            return []
            
        first_edge_idx = edges[0]
                
        # Start sampling exactly in the middle of the first pulse
        current_idx = first_edge_idx + (samples_per_bit // 2)
        raw_bits = []
        
        while current_idx < len(bitstream):
            raw_bits.append(bitstream[current_idx])
            current_idx += samples_per_bit
            
        return raw_bits

    def find_payload(self, bits, expected_lengths=[256, 192, 128, 96, 64, 44], max_errors=5):
        """
        Scans the extracted raw bits to find cyclic, repeating sequences,
        allowing for a small number of bit errors (noise) between the chunks.
        """
        bit_str = ''.join(map(str, bits))
        
        print(f"[*] Searching for repeating cyclic sequences (allowing up to {max_errors} bit errors)...")
        
        # Slide a window across the bitstream looking for adjacent repeating chunks
        for length in expected_lengths:
            for offset in range(len(bit_str) - (length * 2)):
                chunk1 = bit_str[offset : offset + length]
                chunk2 = bit_str[offset + length : offset + length * 2]
                
                # Calculate the Hamming distance (number of mismatched bits)
                errors = sum(b1 != b2 for b1, b2 in zip(chunk1, chunk2))
                
                # If the chunks match within our allowed error threshold
                if errors <= max_errors:
                    print(f"[+] SUCCESS: Found {length}-bit sequence at offset {offset} with {errors} noise errors!")
                    
                    # Optional: If you want to see exactly where the errors are, you can print them
                    if errors > 0:
                        print(f"    Chunk 1: {chunk1}")
                        print(f"    Chunk 2: {chunk2}")
                        
                    return chunk1
                    
        return "No repeating sequence found. Raw bits sample: " + bit_str[:150] + "..."