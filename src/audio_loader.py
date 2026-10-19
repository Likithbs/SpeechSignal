import os
import io
import urllib.request
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy import signal

TARGET_SR = 16000
TARGET_DURATION_SEC = 10.0
NOISE_URLS = {
    "white": "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92/white.wav",
    "machinegun": "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92/machinegun.wav",
    "babble": "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92/babble.wav",
}

def prepare_clean_speech(
    input_audio_path="VoiceRecord.mp3",
    output_wav_path="data/clean_voice.wav",
    sr=TARGET_SR,
    duration=TARGET_DURATION_SEC,
):
    """
    Loads input speech, trims/pads to specified duration, resamples to sr,
    normalizes peak to 0.95, and saves as clean_voice.wav.
    """
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input speech file not found: {input_audio_path}")

    y, orig_sr = librosa.load(input_audio_path, sr=sr, mono=True)
    target_samples = int(duration * sr)

    if len(y) >= target_samples:
        y_trimmed = y[:target_samples]
    else:
        y_trimmed = np.pad(y, (0, target_samples - len(y)))

    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_trimmed = y_trimmed / peak * 0.95

    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, y_trimmed, sr, subtype="PCM_16")
    print(f"[audio_loader] Prepared clean speech: {output_wav_path} ({len(y_trimmed)/sr:.2f}s, {sr} Hz)")
    return y_trimmed, sr

def download_and_prepare_noise(noise_name, output_dir="data/noises", sr=TARGET_SR, target_len=None):
    """
    Downloads NoiseX-92 noise if missing, resamples to sr, ensures length >= target_len.
    """
    if noise_name not in NOISE_URLS:
        raise ValueError(f"Unknown noise {noise_name}. Choose from {list(NOISE_URLS.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, f"{noise_name}.wav")

    if not os.path.exists(dest_path):
        url = NOISE_URLS[noise_name]
        print(f"[audio_loader] Downloading {noise_name} from {url}...")
        raw_data = urllib.request.urlopen(url).read()
        raw_wav, orig_sr = sf.read(io.BytesIO(raw_data))
        if raw_wav.ndim > 1:
            raw_wav = np.mean(raw_wav, axis=1)

        if orig_sr != sr:
            noise_resampled = librosa.resample(raw_wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
        else:
            noise_resampled = raw_wav.astype(np.float32)

        sf.write(dest_path, noise_resampled, sr, subtype="PCM_16")
        print(f"[audio_loader] Saved resampled noise to {dest_path}")
    else:
        noise_resampled, file_sr = sf.read(dest_path)
        if file_sr != sr:
            noise_resampled = librosa.resample(noise_resampled.astype(np.float32), orig_sr=file_sr, target_sr=sr)
            sf.write(dest_path, noise_resampled, sr, subtype="PCM_16")

    if target_len is not None:
        if len(noise_resampled) < target_len:
            n_repeats = int(np.ceil(target_len / len(noise_resampled)))
            noise_resampled = np.tile(noise_resampled, n_repeats)[:target_len]
        else:
            noise_resampled = noise_resampled[:target_len]

    return noise_resampled, sr

def mix_noise_at_snr(speech, noise, target_snr_db):
    """
    Mixes speech and noise signals at the target SNR in dB.
    Returns:
        noisy_speech: speech + scaled_noise
        scaled_noise: noise scaled to achieve target SNR
        actual_snr_db: measured SNR of the mixture
    """
    if len(noise) < len(speech):
        n_repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, n_repeats)[:len(speech)]
    elif len(noise) > len(speech):
        noise = noise[:len(speech)]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        raise ValueError("Noise power is 0, cannot calculate SNR.")
    if speech_power == 0:
        raise ValueError("Speech power is 0, cannot calculate SNR.")

    scale = np.sqrt(speech_power / (noise_power * (10 ** (target_snr_db / 10.0))))
    scaled_noise = scale * noise
    noisy_speech = speech + scaled_noise

    actual_snr = 10.0 * np.log10(speech_power / np.mean(scaled_noise ** 2))
    return noisy_speech, scaled_noise, actual_snr

def plot_noise_characteristics(noise_dict, sr=TARGET_SR, save_path=None):
    """
    Plots Waveform (2s), Spectrogram, and Power Spectral Density (PSD)
    for White, Machinegun, and Babble noises to analyze their temporal and spectral masking traits.
    """
    noises = list(noise_dict.keys())
    fig, axes = plt.subplots(3, len(noises), figsize=(15, 9))

    t_2s = np.arange(int(2.0 * sr)) / sr

    for col, name in enumerate(noises):
        data = noise_dict[name][:int(2.0 * sr)]
        full_data = noise_dict[name]

        # Row 0: Waveform (2s)
        axes[0, col].plot(t_2s, data, lw=0.8, color="#1f77b4")
        axes[0, col].set_title(f"{name.capitalize()} Waveform (2s)")
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("Amplitude")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_ylim([-1.1, 1.1])

        # Row 1: Spectrogram (2s)
        S = np.abs(librosa.stft(data, n_fft=512, hop_length=128))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        im = axes[1, col].imshow(
            S_db,
            origin="lower",
            aspect="auto",
            extent=[0, 2.0, 0, sr / 2000.0],
            cmap="magma",
        )
        axes[1, col].set_title(f"{name.capitalize()} Spectrogram")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_ylabel("Frequency (kHz)")

        # Row 2: Power Spectral Density (PSD)
        freqs, psd = signal.welch(full_data, fs=sr, nperseg=1024)
        psd_db = 10 * np.log10(np.maximum(psd, 1e-12))
        axes[2, col].plot(freqs / 1000.0, psd_db, color="#2ca02c", lw=1.5)
        axes[2, col].set_title(f"{name.capitalize()} PSD (Welch)")
        axes[2, col].set_xlabel("Frequency (kHz)")
        axes[2, col].set_ylabel("PSD (dB/Hz)")
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].set_xlim([0, sr / 2000.0])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[audio_loader] Saved noise characteristics plot to {save_path}")
    else:
        plt.show()

def plot_noise_psd_overlay(clean_speech, noise_dict, sr=TARGET_SR, save_path=None):
    """
    Plots an overlay of the Power Spectral Densities (PSD) of Clean Speech,
    White Noise, Machinegun Noise, and Babble Noise, with the BPF passband highlighted.
    """
    plt.figure(figsize=(10, 5))

    # Clean speech PSD
    f_speech, psd_speech = signal.welch(clean_speech, fs=sr, nperseg=1024)
    plt.plot(f_speech / 1000.0, 10 * np.log10(np.maximum(psd_speech, 1e-12)), color="black", lw=2, label="Clean Speech")

    # Noise PSDs
    colors = {"white": "#1f77b4", "machinegun": "#d62728", "babble": "#ff7f0e"}
    styles = {"white": "-", "machinegun": "--", "babble": "-."}

    for name in ["white", "machinegun", "babble"]:
        f_noise, psd_n = signal.welch(noise_dict[name], fs=sr, nperseg=1024)
        plt.plot(
            f_noise / 1000.0,
            10 * np.log10(np.maximum(psd_n, 1e-12)),
            color=colors[name],
            linestyle=styles[name],
            lw=1.8,
            label=f"{name.capitalize()} Noise",
        )

    # Highlight BPF Passband (300 Hz - 3400 Hz)
    plt.axvspan(0.3, 3.4, color="green", alpha=0.15, label="BPF Passband (0.3 - 3.4 kHz)")

    plt.title("Power Spectral Density (PSD) Comparison: Speech vs. Noise vs. BPF Passband", fontsize=12)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.xlim([0, sr / 2000.0])
    plt.ylim([-90, -10])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[audio_loader] Saved noise PSD overlay to {save_path}")
    else:
        plt.show()

