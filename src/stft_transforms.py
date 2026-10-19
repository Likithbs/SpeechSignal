import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
import pystoi

N_FFT = 512
HOP_LENGTH = 128
WINDOW = "hann"

def compute_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW):
    """
    Computes Short-Time Fourier Transform.
    Implementation details:
    - n_fft: 512 samples (32 ms at 16 kHz)
    - win_length: 512 samples
    - hop_length: 128 samples (8 ms, 75% overlap)
    - window: 'hann'
    - center: True (reflection padding at boundaries)
    - representation: one-sided spectrum with N/2 + 1 = 257 frequency bins
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    MG = np.abs(S)
    PH = np.angle(S)
    return S, MG, PH

def invert_stft(S, hop_length=HOP_LENGTH, length=None, window=WINDOW):
    """
    Inverts complex STFT matrix back to time-domain signal.
    Uses librosa.istft which applies normalized overlap-add (OLA) and inverse real FFT (irfft).
    """
    return librosa.istft(S, hop_length=hop_length, length=length, window=window, center=True)

def evaluate_stft_istft_baseline(y, sr=16000):
    """
    Baseline test: x -> STFT -> ISTFT -> x_hat.
    Evaluates MSE, Max Absolute Error, and Reconstruction SNR.
    """
    S, MG, PH = compute_stft(y)
    y_hat = invert_stft(S, length=len(y))
    
    mse = float(np.mean((y - y_hat) ** 2))
    max_err = float(np.max(np.abs(y - y_hat)))
    snr_db = float(10.0 * np.log10(np.mean(y ** 2) / np.maximum(mse, 1e-15)))
    stoi_score = float(pystoi.stoi(y, y_hat, sr, extended=False))
    corr = float(np.corrcoef(y, y_hat)[0, 1])

    return {
        "mse": mse,
        "max_err": max_err,
        "snr_db": snr_db,
        "stoi": stoi_score,
        "corr": corr,
        "y_hat": y_hat,
    }

def compute_unified_metrics(ref, target, sr=16000):
    """
    Computes a comprehensive metric suite:
    - MSE
    - Pearson Correlation
    - Time-Domain SNR (dB)
    - STOI
    """
    min_len = min(len(ref), len(target))
    r = ref[:min_len]
    t = target[:min_len]

    error = r - t
    mse = float(np.mean(error ** 2))
    
    # Correlation
    r_std = np.std(r)
    t_std = np.std(t)
    if r_std > 0 and t_std > 0:
        corr = float(np.corrcoef(r, t)[0, 1])
    else:
        corr = 0.0

    # SNR
    ref_pow = np.mean(r ** 2)
    err_pow = np.mean(error ** 2)
    if err_pow > 0:
        snr_db = float(10.0 * np.log10(ref_pow / err_pow))
    else:
        snr_db = float("inf")

    # STOI
    stoi_score = float(pystoi.stoi(r, t, sr, extended=False))

    return {
        "mse": mse,
        "corr": corr,
        "snr_db": snr_db,
        "stoi": stoi_score,
    }

def compute_sig1(MG, hop_length=HOP_LENGTH, length=None, window=WINDOW):
    """
    Computes sig1 using only magnitude (zero phase: PH = 0).
    S1 = MG * exp(j * 0) = MG
    """
    S1 = MG.astype(np.complex64)  # Zero phase implies exp(0) = 1 + 0j
    sig1 = invert_stft(S1, hop_length=hop_length, length=length, window=window)
    return sig1, S1

def generate_odd_symmetric_phase(shape, seed=42):
    """
    Generates a phase matrix PH1 for a one-sided spectrum (k = 0 ... N/2)
    with boundary values compatible with Hermitian symmetry:
    - DC bin (k=0) and Nyquist bin (k=N/2) set to 0.
    - Interior bins (k = 1 ... N/2 - 1) have random phases in [-pi, pi].
    When mirrored to negative frequencies in full N-point DFT, PH1[-k] = -PH1[k].
    """
    rng = np.random.RandomState(seed)
    n_freqs, n_frames = shape
    PH1 = rng.uniform(-np.pi, np.pi, size=shape).astype(np.float32)
    # Enforce real condition at DC and Nyquist
    PH1[0, :] = 0.0
    PH1[-1, :] = 0.0
    return PH1

def compute_sig2(MG, PH1=None, seed=42, hop_length=HOP_LENGTH, length=None, window=WINDOW):
    """
    Computes sig2 using MG and random odd-symmetric phase PH1.
    S2 = MG * exp(j * PH1)
    """
    if PH1 is None:
        PH1 = generate_odd_symmetric_phase(MG.shape, seed=seed)

    S2 = MG * np.exp(1j * PH1)
    sig2 = invert_stft(S2, hop_length=hop_length, length=length, window=window)
    return sig2, S2, PH1

def verify_real_valued_condition_dft(frame_mg, n_fft=N_FFT, seed=42):
    """
    Mathematical verification on full 2-sided N-point DFT:
    Tests the Hermitian symmetry condition:
        max_k |X[k] - X^*[N - k]|
    and evaluates the resulting time-domain imaginary component.
    """
    rng = np.random.RandomState(seed)
    
    # Construct full two-sided symmetric magnitude from one-sided frame_mg
    mag_full = np.zeros(n_fft, dtype=np.float32)
    mag_full[:n_fft // 2 + 1] = frame_mg
    mag_full[n_fft // 2 + 1:] = frame_mg[1:n_fft // 2][::-1]

    # Helper to calculate Hermitian symmetry error: max_k |X[k] - X^*[(-k) mod N]|
    def get_hermitian_error(X):
        X_target = np.conj(np.concatenate(([X[0]], X[1:][::-1])))
        return float(np.max(np.abs(X - X_target)))

    # 1. Arbitrary random phase without odd symmetry
    ph_arbitrary = rng.uniform(-np.pi, np.pi, size=n_fft)
    X_arbitrary = mag_full * np.exp(1j * ph_arbitrary)
    x_arbitrary = np.fft.ifft(X_arbitrary)
    max_imag_arbitrary = float(np.max(np.abs(np.imag(x_arbitrary))))
    herm_err_arbitrary = get_hermitian_error(X_arbitrary)

    # 2. Odd-symmetric random phase
    ph_odd = np.zeros(n_fft, dtype=np.float32)
    ph_odd[0] = 0.0  # DC
    ph_odd[n_fft // 2] = 0.0  # Nyquist
    pos_phase = rng.uniform(-np.pi, np.pi, size=n_fft // 2 - 1)
    ph_odd[1:n_fft // 2] = pos_phase
    ph_odd[n_fft // 2 + 1:] = -pos_phase[::-1]  # phi[N-k] = -phi[k]

    X_odd = mag_full * np.exp(1j * ph_odd)
    x_odd = np.fft.ifft(X_odd)
    max_imag_odd = float(np.max(np.abs(np.imag(x_odd))))
    herm_err_odd = get_hermitian_error(X_odd)

    return {
        "max_imag_arbitrary": max_imag_arbitrary,
        "herm_err_arbitrary": herm_err_arbitrary,
        "max_imag_odd": max_imag_odd,
        "herm_err_odd": herm_err_odd,
    }

def plot_magnitude_and_phase(MG, PH, sr, hop_length=HOP_LENGTH, title_prefix="Speech", save_path=None):
    """
    Separately plots Magnitude (in dB) and Phase (in radians).
    """
    n_freqs, n_frames = MG.shape
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2) / 1000.0  # in kHz

    MG_db = librosa.amplitude_to_db(MG, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    im0 = axes[0].pcolormesh(times, freqs, MG_db, shading="auto", cmap="magma")
    axes[0].set_ylabel("Frequency (kHz)")
    axes[0].set_title(f"{title_prefix} - Magnitude Spectrogram (dB)")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("Magnitude (dB)")

    im1 = axes[1].pcolormesh(times, freqs, PH, shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].set_title(f"{title_prefix} - Phase Spectrogram (radians)")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("Phase (rad)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[stft_transforms] Saved plot to {save_path}")
    else:
        plt.show()

def plot_reconstruction_comparison(orig, sig1, sig2, sr, save_path=None):
    """
    Plots time-domain waveforms and spectral differences of Original, sig1, and sig2.
    """
    time_axis = np.arange(len(orig)) / sr
    zoom_slice = slice(int(2.0 * sr), int(2.05 * sr))  # 50 ms zoom for fine structure
    zoom_time = time_axis[zoom_slice]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))

    # Full waveforms
    axes[0, 0].plot(time_axis, orig, color="black", lw=0.6)
    axes[0, 0].set_title("Original Speech (Full 10s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(time_axis, sig1, color="tab:blue", lw=0.6)
    axes[1, 0].set_title("sig1: Zero Phase Reconstruction (Full 10s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(time_axis, sig2, color="tab:red", lw=0.6)
    axes[2, 0].set_title("sig2: Random Odd Phase Reconstruction (Full 10s)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].grid(True, alpha=0.3)

    # 50ms Zooms showing fine temporal structure
    axes[0, 1].plot(zoom_time * 1000, orig[zoom_slice], color="black", lw=1.2)
    axes[0, 1].set_title("Original Speech (50 ms Zoom: Natural Glottal Pulse Coherence)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(zoom_time * 1000, sig1[zoom_slice], color="tab:blue", lw=1.2)
    axes[1, 1].set_title("sig1 (50 ms Zoom: Temporal Concentration Artifacts)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 1].plot(zoom_time * 1000, sig2[zoom_slice], color="tab:red", lw=1.2)
    axes[2, 1].set_title("sig2 (50 ms Zoom: Dispersed Phase Temporal Structure)")
    axes[2, 1].set_xlabel("Time (ms)")
    axes[2, 1].set_ylabel("Amplitude")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[stft_transforms] Saved comparison plot to {save_path}")
    else:
        plt.show()

def plot_same_magnitude_different_waveform(orig, sig1, sig2, sr, hop_length=HOP_LENGTH, save_path=None):
    """
    Demonstrates visually that:
    SAME MAGNITUDE != SAME WAVEFORM
    Top row: Magnitude spectrograms of Original, sig1, sig2 (identical)
    Bottom row: 50 ms time waveforms of Original, sig1, sig2 (vastly different)
    """
    _, MG_orig, _ = compute_stft(orig)
    _, MG_sig1, _ = compute_stft(sig1)
    _, MG_sig2, _ = compute_stft(sig2)

    MG_orig_db = librosa.amplitude_to_db(MG_orig, ref=np.max)
    MG_sig1_db = librosa.amplitude_to_db(MG_sig1, ref=np.max)
    MG_sig2_db = librosa.amplitude_to_db(MG_sig2, ref=np.max)

    n_freqs, n_frames = MG_orig.shape
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2) / 1000.0

    zoom_slice = slice(int(2.0 * sr), int(2.05 * sr))
    t_zoom = np.arange(len(orig))[zoom_slice] / sr * 1000.0  # ms

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))

    # Magnitude Spectrograms
    im0 = axes[0, 0].pcolormesh(times, freqs, MG_orig_db, shading="auto", cmap="magma")
    axes[0, 0].set_title("Original: |X(t, f)| (dB)")
    axes[0, 0].set_ylabel("Frequency (kHz)")

    im1 = axes[0, 1].pcolormesh(times, freqs, MG_sig1_db, shading="auto", cmap="magma")
    axes[0, 1].set_title("sig1 (Zero Phase): |X(t, f)| (dB)")

    im2 = axes[0, 2].pcolormesh(times, freqs, MG_sig2_db, shading="auto", cmap="magma")
    axes[0, 2].set_title("sig2 (Random Phase): |X(t, f)| (dB)")

    for ax in axes[0, :]:
        ax.set_ylim([0, 8])

    # Waveform Zooms
    axes[1, 0].plot(t_zoom, orig[zoom_slice], color="black", lw=1.2)
    axes[1, 0].set_title("Original Waveform (50 ms)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_zoom, sig1[zoom_slice], color="tab:blue", lw=1.2)
    axes[1, 1].set_title("sig1 Waveform (50 ms)")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(t_zoom, sig2[zoom_slice], color="tab:red", lw=1.2)
    axes[1, 2].set_title("sig2 Waveform (50 ms)")
    axes[1, 2].set_xlabel("Time (ms)")
    axes[1, 2].grid(True, alpha=0.3)

    # Set uniform y-limits for waveforms
    max_amp = max(np.max(np.abs(orig[zoom_slice])), np.max(np.abs(sig1[zoom_slice])), np.max(np.abs(sig2[zoom_slice]))) * 1.1
    for ax in axes[1, :]:
        ax.set_ylim([-max_amp, max_amp])

    fig.suptitle("Visual Proof: Identical Spectral Magnitude != Identical Waveform", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[stft_transforms] Saved same-magnitude plot to {save_path}")
    else:
        plt.show()

def plot_phase_spectrogram_comparison(PH_orig, PH_sig1, PH_sig2, sr, hop_length=HOP_LENGTH, save_path=None):
    """
    Side-by-side phase visualization comparing:
    - Original Phase
    - Zero Phase (sig1)
    - Random Odd-Symmetric Phase (sig2)
    """
    n_freqs, n_frames = PH_orig.shape
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2) / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)

    im0 = axes[0].pcolormesh(times, freqs, PH_orig, shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Original Phase: $\\angle X(t, f)$")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (kHz)")

    im1 = axes[1].pcolormesh(times, freqs, PH_sig1, shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("sig1 Phase (Zero Phase $\\equiv 0$)")
    axes[1].set_xlabel("Time (s)")

    im2 = axes[2].pcolormesh(times, freqs, PH_sig2, shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("sig2 Phase (Odd-Symmetric Random)")
    axes[2].set_xlabel("Time (s)")

    cbar = fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Phase (radians)")

    fig.suptitle("Phase Spectrogram Comparison: Original vs. Zero Phase vs. Random Phase", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[stft_transforms] Saved phase spectrogram comparison to {save_path}")
    else:
        plt.show()

