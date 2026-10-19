import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

def design_speech_bandpass_filter(
    lowcut=300.0,
    highcut=3400.0,
    sr=16000,
    order=4,
    btype="butter"
):
    """
    Designs a speech Band-Pass Filter.
    Standard telephony speech band: 300 Hz to 3400 Hz.
    Uses Second-Order Sections (SOS) for numerical stability.
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1:
        raise ValueError(f"Cutoffs must be within (0, {nyq}) Hz.")

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return sos

def apply_bandpass_filter(y, sos):
    """
    Applies the bandpass filter with zero-phase distortion using sosfiltfilt.
    Zero-phase filtering preserves the exact temporal envelope alignment,
    which is critical for STOI speech intelligibility evaluation.
    """
    return signal.sosfiltfilt(sos, y)

def plot_filter_frequency_response(sos, sr=16000, save_path=None):
    """
    Plots the magnitude and phase response of the designed Band-Pass Filter.
    """
    w, h = signal.sosfreqz(sos, worN=2048, fs=sr)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Magnitude response
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    ax1.plot(w, h_db, "b", lw=1.5, label="Magnitude (dB)")
    ax1.set_title("Speech Band-Pass Filter (300 Hz - 3400 Hz) Frequency Response")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Gain (dB)", color="b")
    ax1.set_ylim([-60, 5])
    ax1.set_xlim([0, sr // 2])
    ax1.grid(True, alpha=0.3)

    # Highlight passband
    ax1.axvspan(300, 3400, color="green", alpha=0.15, label="Telephony Passband (300-3400 Hz)")

    # Phase response
    ax2 = ax1.twinx()
    phase = np.unwrap(np.angle(h))
    ax2.plot(w, phase, "r--", lw=1.0, alpha=0.7, label="Phase (rad)")
    ax2.set_ylabel("Phase (radians)", color="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[filter_module] Saved filter response plot to {save_path}")
    else:
        plt.show()
