import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
import pystoi

from src.stft_transforms import (
    compute_stft,
    invert_stft,
    generate_odd_symmetric_phase,
    HOP_LENGTH,
    N_FFT,
    WINDOW,
)

def create_unintelligible_speech(
    clean_audio,
    sr=16000,
    seed=1337,
    output_path="outputs/audio/unintelligible_scrambled.wav"
):
    """
    Renders speech signal unintelligible using an odd yet random phase mask,
    while preserving the magnitude spectrum.
    """
    S, MG, PH_orig = compute_stft(clean_audio)
    PH_scramble = generate_odd_symmetric_phase(MG.shape, seed=seed)

    # Scrambled spectrum with scrambled odd phase
    S_scrambled = MG * np.exp(1j * PH_scramble)
    scrambled_audio = invert_stft(S_scrambled, length=len(clean_audio))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, scrambled_audio, sr, subtype="PCM_16")
        print(f"[phase_recovery] Created unintelligible audio at: {output_path}")

    return scrambled_audio, MG, PH_orig, PH_scramble

def recover_with_true_phase(MG, PH_orig, length=None, output_path="outputs/audio/recovered_keyed.wav", sr=16000):
    """
    Recovers the exact original audio using the true phase (or reversing the keyed phase).
    """
    S_rec = MG * np.exp(1j * PH_orig)
    rec_audio = invert_stft(S_rec, length=length)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, rec_audio, sr, subtype="PCM_16")
        print(f"[phase_recovery] Keyed recovered audio saved at: {output_path}")

    return rec_audio

def recover_with_griffin_lim(
    MG,
    n_iter=64,
    init_phase=None,
    length=None,
    output_path="outputs/audio/recovered_griffin_lim.wav",
    sr=16000,
):
    """
    Blind phase retrieval using Griffin-Lim Algorithm (GLA).
    Recovers an intelligible time-domain speech signal directly from STFT magnitude.
    """
    rec_audio = librosa.griffinlim(
        MG,
        n_iter=n_iter,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=WINDOW,
        length=length,
        init=None if init_phase is None else "random",
        random_state=42,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, rec_audio, sr, subtype="PCM_16")
        print(f"[phase_recovery] Griffin-Lim recovered audio saved at: {output_path}")

    return rec_audio

def griffin_lim_iteration_study(
    MG,
    ref_clean,
    iterations=(1, 4, 8, 16, 32, 64, 128),
    sr=16000,
):
    """
    Evaluates Griffin-Lim reconstruction across iteration counts:
    Reports STOI, Waveform MSE, and Spectral Convergence error.
    """
    results = []
    mg_norm = np.linalg.norm(MG, "fro")

    for n_iter in iterations:
        rec_audio = librosa.griffinlim(
            MG,
            n_iter=n_iter,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window=WINDOW,
            length=len(ref_clean),
            random_state=42,
        )
        # Compute STOI
        stoi_score = float(pystoi.stoi(ref_clean, rec_audio, sr, extended=False))
        # Compute Waveform MSE
        wf_mse = float(np.mean((ref_clean - rec_audio) ** 2))
        # Compute Spectral Convergence Error
        _, mg_rec, _ = compute_stft(rec_audio)
        min_frames = min(MG.shape[1], mg_rec.shape[1])
        spec_conv = float(np.linalg.norm(MG[:, :min_frames] - mg_rec[:, :min_frames], "fro") / mg_norm)

        results.append({
            "iterations": n_iter,
            "stoi": round(stoi_score, 4),
            "waveform_mse": round(wf_mse, 6),
            "spectral_conv": round(spec_conv, 4),
        })
        print(f"  [Griffin-Lim Study] iter={n_iter:3d} | STOI={stoi_score:.4f} | MSE={wf_mse:.6f} | SpecConv={spec_conv:.4f}")

    return results

def plot_griffin_lim_iterations(iter_results, save_path=None):
    """
    Plots STOI vs Iterations and Spectral Convergence Error vs Iterations.
    """
    iters = [r["iterations"] for r in iter_results]
    stois = [r["stoi"] for r in iter_results]
    spec_convs = [r["spectral_conv"] for r in iter_results]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = "tab:blue"
    ax1.set_xlabel("Griffin-Lim Iterations (log2 scale)")
    ax1.set_ylabel("STOI Score (Intelligibility)", color=color)
    ax1.plot(iters, stois, color=color, marker="o", lw=2, label="STOI")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(iters)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.05])

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Spectral Convergence Error", color=color)
    ax2.plot(iters, spec_convs, color=color, marker="s", lw=2, linestyle="--", label="Spectral Conv Error")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Griffin-Lim Reconstruction Quality vs. Iteration Count")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[phase_recovery] Saved Griffin-Lim iteration curve to {save_path}")
    else:
        plt.show()
