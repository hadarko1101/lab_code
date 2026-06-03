import numpy as np
from scipy.signal import butter, filtfilt

class SignalProcessor:
    def __init__(self, first_cutoff_hz=15000.0, second_cutoff_hz=5000.0):
        """
        Initializes the DSP pipeline with configurable cutoff frequencies.
        """
        self.first_cutoff_hz = first_cutoff_hz
        self.second_cutoff_hz = second_cutoff_hz

    def process(self, time_ms, voltages):
        """
        Executes the 2-stage digital envelope detection pipeline.
        Returns: env1, env1_centered, final_envelope
        """
        dt_seconds = (time_ms[1] - time_ms[0]) / 1000.0
        fs = 1.0 / dt_seconds
        nyquist = 0.5 * fs
        
        # ---------------------------------------------------------
        # STAGE 1: First Envelope (Remove main carrier)
        # ---------------------------------------------------------
        rectified_volts = np.abs(voltages)
        b1, a1 = butter(4, self.first_cutoff_hz / nyquist, btype='low', analog=False)
        env1 = filtfilt(b1, a1, rectified_volts)
        
        # ---------------------------------------------------------
        # STAGE 2: Move to 0 (AC Couple) & Second Envelope
        # ---------------------------------------------------------
        env1_centered = env1 - np.mean(env1)
        rectified_env1 = np.abs(env1_centered)
        
        b2, a2 = butter(4, self.second_cutoff_hz / nyquist, btype='low', analog=False)
        raw_env2 = filtfilt(b2, a2, rectified_env1)
        
        # ---------------------------------------------------------
        # STAGE 3: Keep the final signal at 0
        # ---------------------------------------------------------
        final_envelope = raw_env2 - np.mean(raw_env2)
        
        return env1, env1_centered, final_envelope