import numpy as np
import pystoi
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_stoi_safe(clean, deg, sr=16000):
    """
    Computes STOI between clean reference and degraded signal.
    Handles length matching and NaN safeguards.
    """
    min_len = min(len(clean), len(deg))
    clean_aligned = clean[:min_len]
    deg_aligned = deg[:min_len]
    
    score = pystoi.stoi(clean_aligned, deg_aligned, sr, extended=False)
    if np.isnan(score):
        return 0.0
    return float(score)

def plot_stoi_snr_curves(results_df, save_path=None):
    """
    Plots STOI vs SNR curves comparing:
    - Raw Noisy (vs Full-Band Clean)
    - BPF Filtered (Eval A: vs Full-Band Clean)
    - BPF Filtered (Eval B: vs Bandpass-Matched Clean)
    - sig1 (Zero Phase)
    - sig2 (Random Phase)
    across White, Machinegun, and Babble noises.
    """
    noises = results_df["noise"].unique()
    fig, axes = plt.subplots(1, len(noises), figsize=(5.5 * len(noises), 4.8), sharey=True)

    if len(noises) == 1:
        axes = [axes]

    for ax, noise_name in zip(axes, noises):
        sub_df = results_df[results_df["noise"] == noise_name].sort_values("snr_db")
        snrs = sub_df["snr_db"].values

        ax.plot(snrs, sub_df["stoi_raw"], label="Raw Noisy (vs Full Clean)", color="#1f77b4", marker="o", lw=2)
        ax.plot(snrs, sub_df["stoi_bpf_eval_a"], label="BPF Eval A (vs Full Clean)", color="#e377c2", marker="s", lw=1.8, linestyle="--")
        ax.plot(snrs, sub_df["stoi_bpf_eval_b"], label="BPF Eval B (vs BPF Clean)", color="#2ca02c", marker="D", lw=2, linestyle="-")
        ax.plot(snrs, sub_df["stoi_sig1"], label="sig1 (Zero Phase)", color="#ff7f0e", marker="^", lw=1.4, linestyle=":")
        ax.plot(snrs, sub_df["stoi_sig2"], label="sig2 (Random Phase)", color="#d62728", marker="x", lw=1.4, linestyle="-.")

        ax.set_title(f"Noise: {noise_name.capitalize()}")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(snrs)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

    axes[0].set_ylabel("STOI Score")
    axes[-1].legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[stoi_benchmark] Saved STOI curve plot to {save_path}")
    else:
        plt.show()

def export_results_table(results_df, csv_path="outputs/tables/stoi_results.csv", md_path="outputs/tables/stoi_results.md"):
    """
    Exports results table to CSV and Markdown.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# STOI Benchmark Results\n\n")
        try:
            f.write(results_df.to_markdown(index=False))
        except Exception:
            f.write(results_df.to_string(index=False))
        f.write("\n")
    print(f"[stoi_benchmark] Exported results to {csv_path} and {md_path}")
