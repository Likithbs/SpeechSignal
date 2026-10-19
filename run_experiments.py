import os
import time
import numpy as np
import soundfile as sf
import pandas as pd

from src.audio_loader import (
    prepare_clean_speech,
    download_and_prepare_noise,
    mix_noise_at_snr,
    plot_noise_characteristics,
    plot_noise_psd_overlay,
    TARGET_SR,
    TARGET_DURATION_SEC,
)
from src.stft_transforms import (
    compute_stft,
    compute_sig1,
    compute_sig2,
    evaluate_stft_istft_baseline,
    compute_unified_metrics,
    plot_magnitude_and_phase,
    plot_phase_spectrogram_comparison,
    plot_reconstruction_comparison,
    plot_same_magnitude_different_waveform,
    verify_real_valued_condition_dft,
)
from src.phase_recovery import (
    create_unintelligible_speech,
    recover_with_true_phase,
    recover_with_griffin_lim,
    griffin_lim_iteration_study,
    plot_griffin_lim_iterations,
)
from src.filter_module import (
    design_speech_bandpass_filter,
    apply_bandpass_filter,
    plot_filter_frequency_response,
)
from src.stoi_benchmark import (
    compute_stoi_safe,
    plot_stoi_snr_curves,
    export_results_table,
)

def run_all_experiments():
    start_time = time.time()
    print("=" * 80)
    print("STARTING UPGRADED END-TO-END SPEECH SIGNAL PROCESSING PIPELINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # STAGE 1: Audio Ingestion, Preprocessing & Baseline
    # ---------------------------------------------------------
    print("\n--- STAGE 1: Preparing Clean Speech, Noise Datasets & STFT Baseline ---")
    clean_audio, sr = prepare_clean_speech(
        input_audio_path="VoiceRecord.mp3",
        output_wav_path="data/clean_voice.wav",
        sr=TARGET_SR,
        duration=TARGET_DURATION_SEC,
    )
    print(f"Clean speech ready: duration={len(clean_audio)/sr:.2f}s, sample_rate={sr} Hz")

    # STFT -> ISTFT Baseline Test
    baseline_metrics = evaluate_stft_istft_baseline(clean_audio, sr=sr)
    print("\nSTFT -> ISTFT Identity Baseline:")
    print(f"  - Waveform MSE:           {baseline_metrics['mse']:.6e}")
    print(f"  - Max Absolute Error:     {baseline_metrics['max_err']:.6e}")
    print(f"  - Reconstruction SNR:     {baseline_metrics['snr_db']:.2f} dB")
    print(f"  - Pearson Correlation:    {baseline_metrics['corr']:.6f}")
    print(f"  - Baseline STOI:          {baseline_metrics['stoi']:.4f}")

    # Prepare Noises
    noise_dict = {}
    for noise_name in ["white", "machinegun", "babble"]:
        print(f"Fetching/preparing NoiseX-92 '{noise_name}'...")
        noise_audio, _ = download_and_prepare_noise(
            noise_name, output_dir="data/noises", sr=sr, target_len=len(clean_audio)
        )
        noise_dict[noise_name] = noise_audio

    # Plot Noise Temporal & Spectral Characteristics
    plot_noise_characteristics(
        noise_dict, sr=sr, save_path="outputs/plots/noise_characteristics.png"
    )
    # Plot Noise PSD vs Speech vs BPF Passband
    plot_noise_psd_overlay(
        clean_audio, noise_dict, sr=sr, save_path="outputs/plots/noise_psd_bpf_overlay.png"
    )

    # ---------------------------------------------------------
    # STAGE 2: STFT, Phase Analysis, sig1, sig2 & Hermitian Test
    # ---------------------------------------------------------
    print("\n--- STAGE 2: STFT Analysis & Phase Manipulation ---")
    S_clean, MG_clean, PH_clean = compute_stft(clean_audio)
    print(f"STFT Matrix shape: {S_clean.shape} (Freq bins: {S_clean.shape[0]}, Time frames: {S_clean.shape[1]})")

    # Plot MG and PH separately
    plot_magnitude_and_phase(
        MG_clean,
        PH_clean,
        sr=sr,
        title_prefix="Clean Speech",
        save_path="outputs/plots/clean_stft_magnitude_phase.png",
    )

    # Compute sig1 (Zero Phase)
    sig1, S1 = compute_sig1(MG_clean, length=len(clean_audio))
    sf.write("outputs/audio/sig1_zero_phase.wav", sig1, sr, subtype="PCM_16")
    print("Generated and saved sig1 (Zero Phase) to outputs/audio/sig1_zero_phase.wav")

    # Compute sig2 (Random Odd Phase PH1)
    sig2, S2, PH1 = compute_sig2(MG_clean, seed=42, length=len(clean_audio))
    sf.write("outputs/audio/sig2_random_phase.wav", sig2, sr, subtype="PCM_16")
    print("Generated and saved sig2 (Random Odd Phase) to outputs/audio/sig2_random_phase.wav")

    # Plot Phase Spectrogram Comparison: Original vs Zero Phase vs Random Phase
    plot_phase_spectrogram_comparison(
        PH_clean,
        np.zeros_like(PH_clean),
        PH1,
        sr=sr,
        save_path="outputs/plots/phase_spectrogram_comparison.png",
    )

    # Quantitative Hermitian Symmetry Test on 2-Sided DFT
    herm_test = verify_real_valued_condition_dft(MG_clean[:, 50])
    print("\nHermitian Symmetry Real-Valued Test (Full 2-Sided DFT on Active Frame 50):")
    print(f"  - Arbitrary Random Phase: Hermitian Error max|X[k]-X^*[N-k]| = {herm_test['herm_err_arbitrary']:.6f}, max|Im(x)| = {herm_test['max_imag_arbitrary']:.6e}")
    print(f"  - Odd-Symmetric Phase:    Hermitian Error max|X[k]-X^*[N-k]| = {herm_test['herm_err_odd']:.6f}, max|Im(x)| = {herm_test['max_imag_odd']:.6e} (Strictly REAL)")

    # Plot Same Magnitude != Same Waveform
    plot_same_magnitude_different_waveform(
        clean_audio, sig1, sig2, sr=sr, save_path="outputs/plots/same_magnitude_different_waveform.png"
    )

    # Plot Comparative Waveforms
    plot_reconstruction_comparison(
        clean_audio, sig1, sig2, sr=sr, save_path="outputs/plots/waveform_comparison_orig_sig1_sig2.png"
    )

    # ---------------------------------------------------------
    # STAGE 3: Unintelligible Speech & Recovery Challenge
    # ---------------------------------------------------------
    print("\n--- STAGE 3: Unintelligible Speech Creation & Recovery ---")
    scrambled_audio, MG_scram, PH_orig, PH_scram = create_unintelligible_speech(
        clean_audio,
        sr=sr,
        seed=2026,
        output_path="outputs/audio/unintelligible_scrambled.wav",
    )

    # Recovery 1: Keyed Phase Descrambler
    rec_keyed = recover_with_true_phase(
        MG_scram, PH_orig, length=len(clean_audio), output_path="outputs/audio/recovered_keyed.wav", sr=sr
    )

    # Recovery 2: Griffin-Lim Phase Retrieval
    rec_gl = recover_with_griffin_lim(
        MG_scram, n_iter=64, length=len(clean_audio), output_path="outputs/audio/recovered_griffin_lim.wav", sr=sr
    )

    # Griffin-Lim Iteration Count Study
    print("\nRunning Griffin-Lim Iteration Count Study...")
    gl_study_results = griffin_lim_iteration_study(
        MG_scram, ref_clean=clean_audio, iterations=[1, 4, 8, 16, 32, 64, 128], sr=sr
    )
    plot_griffin_lim_iterations(gl_study_results, save_path="outputs/plots/griffin_lim_iterations.png")

    # Unified Metric Suite Comparison Table
    unified_records = []
    signals_to_compare = [
        ("ISTFT (Original STFT)", baseline_metrics["y_hat"]),
        ("sig1 (Zero Phase)", sig1),
        ("sig2 (Random Odd Phase)", sig2),
        ("Scrambled Speech", scrambled_audio),
        ("Griffin-Lim (64 iters)", rec_gl),
        ("Keyed Recovered", rec_keyed),
    ]

    for name, sig in signals_to_compare:
        m = compute_unified_metrics(clean_audio, sig, sr=sr)
        unified_records.append({
            "Signal": name,
            "Waveform MSE": f"{m['mse']:.6e}",
            "Pearson Corr": f"{m['corr']:.4f}",
            "SNR (dB)": f"{m['snr_db']:.2f}",
            "STOI": f"{m['stoi']:.4f}",
        })

    unified_df = pd.DataFrame(unified_records)
    print("\nUnified Metric Comparison (Waveform Similarity vs Intelligibility):")
    print(unified_df.to_string(index=False))
    unified_df.to_csv("outputs/tables/phase_manipulation_unified_metrics.csv", index=False)
    with open("outputs/tables/phase_manipulation_unified_metrics.md", "w", encoding="utf-8") as f:
        f.write("# Phase Manipulation Unified Metrics\n\n")
        f.write(unified_df.to_markdown(index=False))
        f.write("\n")

    # ---------------------------------------------------------
    # STAGE 4: Designing Band-Pass Filter (300 Hz - 3400 Hz)
    # ---------------------------------------------------------
    print("\n--- STAGE 4: Designing Band-Pass Filter (300 Hz - 3400 Hz) ---")
    bpf_sos = design_speech_bandpass_filter(lowcut=300.0, highcut=3400.0, sr=sr, order=4)
    plot_filter_frequency_response(bpf_sos, sr=sr, save_path="outputs/plots/bpf_frequency_response.png")
    clean_bpf = apply_bandpass_filter(clean_audio, bpf_sos)

    # ---------------------------------------------------------
    # STAGE 5: Noise Injection & Dual-Reference STOI Evaluation
    # ---------------------------------------------------------
    print("\n--- STAGE 5: Noise Injection & Dual-Reference STOI Evaluation ---")
    snr_levels = [-2.0, 0.0, 2.0, 5.0, 10.0]
    records = []

    for noise_name in ["white", "machinegun", "babble"]:
        noise_audio = noise_dict[noise_name]
        print(f"\nEvaluating Noise: {noise_name.upper()}")

        for snr in snr_levels:
            noisy_audio, scaled_noise, actual_snr = mix_noise_at_snr(clean_audio, noise_audio, snr)

            if snr in [0.0, 10.0]:
                sf.write(f"outputs/audio/noisy_{noise_name}_{int(snr)}dB.wav", noisy_audio, sr, subtype="PCM_16")

            # Raw Noisy STOI vs Full-band Clean
            stoi_raw = compute_stoi_safe(clean_audio, noisy_audio, sr)

            # Reconstructions on noisy speech:
            S_noisy, MG_noisy, PH_noisy = compute_stft(noisy_audio)
            noisy_sig1, _ = compute_sig1(MG_noisy, length=len(clean_audio))
            stoi_noisy_sig1 = compute_stoi_safe(clean_audio, noisy_sig1, sr)

            noisy_sig2, _, _ = compute_sig2(MG_noisy, seed=int(snr * 10 + 50), length=len(clean_audio))
            stoi_noisy_sig2 = compute_stoi_safe(clean_audio, noisy_sig2, sr)

            # Apply Zero-Phase Band-Pass Filter (BPF)
            bpf_noisy_audio = apply_bandpass_filter(noisy_audio, bpf_sos)

            if snr in [0.0, 10.0]:
                sf.write(f"outputs/audio/bpf_{noise_name}_{int(snr)}dB.wav", bpf_noisy_audio, sr, subtype="PCM_16")

            # Evaluation A: Practical comparison against Full-Band Clean
            stoi_bpf_eval_a = compute_stoi_safe(clean_audio, bpf_noisy_audio, sr)
            delta_stoi_a = stoi_bpf_eval_a - stoi_raw

            # Evaluation B: Controlled comparison against Bandpass-Matched Clean
            stoi_bpf_eval_b = compute_stoi_safe(clean_bpf, bpf_noisy_audio, sr)
            delta_stoi_b = stoi_bpf_eval_b - stoi_raw

            records.append({
                "noise": noise_name,
                "snr_db": snr,
                "measured_snr": round(actual_snr, 2),
                "stoi_raw": round(stoi_raw, 4),
                "stoi_bpf_eval_a": round(stoi_bpf_eval_a, 4),
                "delta_stoi_a": round(delta_stoi_a, 4),
                "stoi_bpf_eval_b": round(stoi_bpf_eval_b, 4),
                "delta_stoi_b": round(delta_stoi_b, 4),
                "stoi_sig1": round(stoi_noisy_sig1, 4),
                "stoi_sig2": round(stoi_noisy_sig2, 4),
            })
            print(f"  SNR={snr:+2.0f} dB | Raw: {stoi_raw:.4f} | BPF Eval A: {stoi_bpf_eval_a:.4f} (Delta A: {delta_stoi_a:+.4f}) | BPF Eval B: {stoi_bpf_eval_b:.4f} (Delta B: {delta_stoi_b:+.4f})")

    results_df = pd.DataFrame(records)
    export_results_table(results_df, csv_path="outputs/tables/stoi_results.csv", md_path="outputs/tables/stoi_results.md")

    # Plot STOI vs SNR Curves
    plot_stoi_snr_curves(results_df, save_path="outputs/plots/stoi_comparison_curves.png")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"UPGRADED PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    run_all_experiments()
