# Speech Digital Signal Processing: STFT, Phase Analysis, Noise Robustness & STOI Benchmark

A comprehensive research-grade speech signal processing framework implemented in Python. This repository investigates the role of spectral magnitude and phase representations, mathematically and empirically verifies Hermitian symmetry conditions for real-valued signals, synthesizes unintelligible speech with blind and keyed phase recovery, benchmarks speech intelligibility under NoiseX-92 noise injection across multiple SNRs, and analyzes Band-Pass Filter (BPF) design decisions via dual-reference Short-Time Objective Intelligibility (STOI) evaluation.

---

## Table of Contents
1. [Repository Architecture](#repository-architecture)
2. [Quickstart & Installation](#quickstart--installation)
3. [Question-by-Question Solution Breakdown](#question-by-question-solution-breakdown)
   - [Q1: 10-Second Clean Speech Preparation](#q1-10-second-clean-speech-preparation)
   - [Q2: STFT & Magnitude/Phase Representation](#q2-stft--magnitudephase-representation)
   - [Q3: Zero-Phase Reconstruction (sig1) & Analysis](#q3-zero-phase-reconstruction-sig1--analysis)
   - [Q4: Random Phase Reconstruction (sig2), Real-Valued Proof & Pattern Counting](#q4-random-phase-reconstruction-sig2-real-valued-proof--pattern-counting)
   - [Q5: Unintelligible Speech Synthesis & Recovery (Griffin-Lim vs Keyed)](#q5-unintelligible-speech-synthesis--recovery-griffin-lim-vs-keyed)
   - [Q6 & Q7: NoiseX-92 Noise Injection & STOI Calculation](#q6--q7-noisex-92-noise-injection--stoi-calculation)
   - [Q8: Repeating STFT & Phase Operations Across All Noisy Signals](#q8-repeating-stft--phase-operations-across-all-noisy-signals)
   - [Q9: Band-Pass Filter (BPF) Design Decisions & Dual-Reference STOI Analysis](#q9-band-pass-filter-bpf-design-decisions--dual-reference-stoi-analysis)
4. [Master Experimental Results Table](#master-experimental-results-table)
5. [Key Theoretical Takeaways for Technical Interviews](#key-theoretical-takeaways-for-technical-interviews)
6. [Automated Verification & Unit Testing](#automated-verification--unit-testing)

---

## Repository Architecture

```
d:\IISC/
├── data/
│   ├── clean_voice.wav              # Standardized 10.0s clean speech (16 kHz, mono)
│   └── noises/                      # NoiseX-92 audio files (resampled to 16 kHz)
│       ├── white.wav
│       ├── machinegun.wav
│       └── babble.wav
├── outputs/
│   ├── audio/                       # 17 synthesized & processed audio files (.wav)
│   ├── plots/                       # Spectrograms, waveforms, filter response, STOI curves (.png)
│   └── tables/                      # STOI CSV and Markdown benchmark tables
├── src/
│   ├── __init__.py
│   ├── audio_loader.py              # Ingestion, resampling, RMS SNR mixing, PSD overlays
│   ├── stft_transforms.py           # STFT, ISTFT, sig1, sig2, Hermitian symmetry test
│   ├── phase_recovery.py            # Phase scrambler challenge, Griffin-Lim & keyed recovery
│   ├── filter_module.py             # 4th-order zero-phase Butterworth BPF (300-3400 Hz)
│   └── stoi_benchmark.py            # STOI dual-reference scoring engine & curve visualizer
├── run_experiments.py               # Main CLI orchestrator executing all stages
├── test_dsp_pipeline.py             # Automated unit test suite (7 passing tests)
├── ANALYSIS_REPORT.md               # Exhaustive academic report with full derivations
└── README.md                        # Complete project documentation & solution guide
```

---

## Quickstart & Installation

### 1. Requirements
- Python 3.10+
- Dependencies: `numpy`, `scipy`, `librosa`, `soundfile`, `pystoi`, `matplotlib`, `pandas`, `tabulate`

Install all dependencies via pip:
```bash
pip install numpy scipy librosa soundfile pystoi matplotlib pandas tabulate
```

### 2. Run Automated Unit Tests
To verify implementation correctness (Hermitian symmetry proof, STFT baseline, SNR scaling, filter stability):
```bash
python -m unittest test_dsp_pipeline.py
```

### 3. Run the Complete End-to-End Experiment
To process audio, generate all plots, compute STOI metrics, and export benchmark tables:
```bash
python run_experiments.py
```

---

## Question-by-Question Solution Breakdown

### Q1: 10-Second Clean Speech Preparation
> *Question: Use a voice recording wav file of about 10 seconds with minimal background noise.*

#### How It Is Solved:
- **Module:** [`src/audio_loader.py`](file:///d:/IISC/src/audio_loader.py) $\to$ `prepare_clean_speech()`
- **Input:** Loaded raw speech recording `VoiceRecord.mp3`.
- **Standardization:**
  1. Downmixed from multi-channel to mono.
  2. Resampled to $f_s = 16{,}000\text{ Hz}$ (the canonical sampling rate for speech intelligibility analysis and `pystoi`).
  3. Trimmed to exactly $10.0\text{ seconds}$ ($N_{\text{samples}} = 160{,}000$).
  4. Peak-normalized to $0.95$ amplitude to prevent clipping during subsequent overlap-add synthesis and filter convolution.
  5. Saved as standard 16-bit PCM WAV at [`data/clean_voice.wav`](file:///d:/IISC/data/clean_voice.wav).

---

### Q2: STFT & Magnitude/Phase Representation
> *Question: Take the STFT of the signal, separately plot the magnitude (MG) and phase (PH) of the signal.*

#### How It Is Solved:
- **Module:** [`src/stft_transforms.py`](file:///d:/IISC/src/stft_transforms.py) $\to$ `compute_stft()`, `plot_magnitude_and_phase()`, `plot_phase_spectrogram_comparison()`
- **Mathematical Formulation:**
  For the discrete-time speech signal $x[n]$ with analysis window $w[n]$:
  $$X(m, k) = \sum_{n=0}^{N-1} x[mR + n] \, w[n] \, e^{-j \frac{2\pi}{N} k n}$$
  - **Magnitude (MG):** $\text{MG}(m, k) = |X(m, k)| = \sqrt{\text{Re}\{X(m, k)\}^2 + \text{Im}\{X(m, k)\}^2}$
  - **Phase (PH):** $\text{PH}(m, k) = \angle X(m, k) = \text{atan2}(\text{Im}\{X(m, k)\}, \text{Re}\{X(m, k)\}) \in (-\pi, \pi]$
- **STFT Parameter Choices:**
  - FFT size: $N = 512$ ($32.0\text{ ms}$ window at $16\text{ kHz}$).
  - Hop size: $R = 128$ ($8.0\text{ ms}$, 75% overlap).
  - Window: Periodic Hann window, $w[n] = 0.5 - 0.5\cos(2\pi n / N)$.
  - Framing: Centered analysis (`center=True`) with reflection padding.
  - Frequency bins: $K = N/2 + 1 = 257$ one-sided bins ($0\text{ Hz}$ to $8\text{ kHz}$).
- **STFT $\to$ ISTFT Baseline Verification:**
  Before altering the phase, the analysis-synthesis pipeline was tested for Perfect Reconstruction (PR):
  $$\text{Waveform MSE} = 5.36 \times 10^{-17}, \quad \text{Max Absolute Error} = 1.19 \times 10^{-7}, \quad \text{Reconstruction SNR} = \mathbf{132.55\text{ dB}}$$
- **Generated Plots:**
  - [`outputs/plots/clean_stft_magnitude_phase.png`](file:///d:/IISC/outputs/plots/clean_stft_magnitude_phase.png): Separate magnitude (dB) and wrapped phase (radians) spectrograms.
  - [`outputs/plots/phase_spectrogram_comparison.png`](file:///d:/IISC/outputs/plots/phase_spectrogram_comparison.png): Side-by-side comparison of Original Phase vs. Zero Phase vs. Random Phase.

---

### Q3: Zero-Phase Reconstruction (sig1) & Analysis
> *Question: Using only the magnitude (consider zero phase) do inverse FT, call it sig1. What is the difference between the original and sig1? Give reasoning and analysis.*

#### How It Is Solved:
- **Module:** [`src/stft_transforms.py`](file:///d:/IISC/src/stft_transforms.py) $\to$ `compute_sig1()`
- **Synthesis:**
  $$X_1(m, k) = \text{MG}(m, k) \, e^{j \cdot 0} = \text{MG}(m, k) \in \mathbb{R}^+, \quad \text{sig}_1[n] = \text{ISTFT}\{X_1(m, k)\}$$
  Saved to [`outputs/audio/sig1_zero_phase.wav`](file:///d:/IISC/outputs/audio/sig1_zero_phase.wav).
- **Difference Analysis:**

| Dimension | Original Clean Speech | $\text{sig}_1$ (Zero Phase) |
| :--- | :--- | :--- |
| **Spectral Magnitude** | $|X(m, k)|$ | Identical $|X(m, k)|$ |
| **Phase Spectrum** | Natural $\angle X(m, k)$ | Identically $0\text{ rad}$ |
| **Waveform Structure** | Asymmetric glottal pulse shaping, natural dispersion | Even-symmetric frame impulses, strong time-localization |
| **Acoustic Timbre** | Natural human speech | Harsh, robotic, buzzy, metallic timbre |
| **Waveform MSE / SNR** | Reference ($0.0\text{ MSE}, \infty\text{ dB}$) | $\text{MSE} = 0.0187, \quad \text{SNR} = -0.16\text{ dB}$ |
| **STOI Score** | $1.0000$ | $\mathbf{0.8400}$ |

- **Physical & Acoustic Reasoning:**
  1. **Cosine Alignment:** Setting all spectral phases to zero aligns the cosine components, producing a waveform with a strong even-symmetry/time-localization characteristic that differs substantially from the original speech frame.
  2. **Frame-Rate Artifacts:** Zero-phase reconstruction removes the original phase relationships and can introduce strong temporal concentration and periodic artifacts related to the frame structure. In our reconstruction, these artifacts were perceived as a robotic/buzzy quality.
  3. **Preservation of Intelligibility:** Although naturalness is lost, intelligibility remains surprisingly high ($\text{STOI} = 0.8400$). The preserved spectral envelope and temporal energy patterns retain information relevant to speech intelligibility, even though the original fine phase relationships are disrupted.

---

### Q4: Random Phase Reconstruction (sig2), Real-Valued Proof & Pattern Counting
> *Question: Create an array (PH1) with size identical to that of PH and consider MG & PH1 to do inverse FT, call it sig2. Is it possible to get real valued sig2? What is the difference between original and sig2? What is the difference between sig1 and sig2? Is there a pattern in PH1 that ensures real valued sig2? How many such patterns are possible?*

#### How It Is Solved:
- **Module:** [`src/stft_transforms.py`](file:///d:/IISC/src/stft_transforms.py) $\to$ `compute_sig2()`, `generate_odd_symmetric_phase()`, `verify_real_valued_condition_dft()`
- **Can $\text{sig}_2$ Be Real-Valued?**  
  **Yes, absolutely.** A discrete-time signal $x[n]$ is real-valued if and only if its Fourier transform satisfies **Hermitian (conjugate) symmetry**.
- **What is the pattern in $\text{PH}_1$ that ensures a real-valued signal?**  
  For the full $N$-point FFT ($k = 0, \dots, N-1$):
  $$X[k] = X^*[N - k]$$
  In magnitude and phase polar representation ($X[k] = |X[k]| e^{j \phi[k]}$):
  1. **Even Magnitude:** $|X[k]| = |X[N - k]|$
  2. **Odd (Anti-Symmetric) Phase:**
     $$\phi[k] \equiv -\phi[N - k] \pmod{2\pi} \quad \text{for } 1 \le k \le \frac{N}{2} - 1$$
  3. **Boundary Bins (DC & Nyquist):**
     - DC ($k = 0$): $\phi[0] \in \{0, \pi\}$
     - Nyquist ($k = N/2$): $\phi[N/2] \in \{0, \pi\}$
- **Numerical Verification on Active Speech Frame 50 (Full 2-Sided 512-Point DFT):**

| Phase Configuration | Hermitian Error $\max_k |X[k] - X^*[N-k]|$ | Time-Domain Imaginary Error $\max |\text{Im}(x)|$ | Real-Valued? |
| :--- | :---: | :---: | :---: |
| **Arbitrary Random Phase** | $\mathbf{14.2224}$ | $\mathbf{1.4783 \times 10^{-1}}$ | **NO** (Complex) |
| **Odd-Symmetric Phase** | $\mathbf{0.000000}$ | $\mathbf{0.000000}$ | **YES** (Strictly Real) |

- **How Many Such Patterns Are Possible?**
  1. **Continuous Phase ($\phi \in (-\pi, \pi]$):** **Uncountably infinite ($2^{\aleph_0}$)**. For $N=512$, there are $N/2 - 1 = 255$ independent interior positive frequency bins. Choosing any phase values on those 255 bins and setting $\phi[N-k] = -\phi[k]$ and $\phi[0], \phi[N/2] \in \{0, \pi\}$ guarantees a purely real time signal.
  2. **Binary Phase ($\phi \in \{0, \pi\}$):** Since $\sin(0) = \sin(\pi) = 0$, odd symmetry $\phi[N-k] \equiv -\phi[k] \pmod{2\pi}$ is automatically satisfied for all bins!
     $$\text{Total binary patterns per frame} = 2 \times 2 \times 2^{N/2 - 1} = 2^{N/2 + 1} = 2^{257} \approx \mathbf{2.3158 \times 10^{77} \text{ patterns}}$$
  3. **Discrete $M$-Level Quantization (e.g., IEEE-754 64-bit float):** $4 \times M^{N/2 - 1}$ per frame, and $(4 \cdot M^{N/2 - 1})^T$ across $T$ STFT frames.
- **Difference: Original vs. $\text{sig}_2$:**
  - Randomizing phase does **not** mathematically remove the fundamental frequency $F_0$ or harmonic peaks from the magnitude spectrum ($|X(m, k)|$ still contains peaks at $F_0, 2F_0, 3F_0, \dots$).
  - However, random phase destroys the temporal phase coherence across harmonics responsible for natural periodic waveform structure, substantially reducing naturalness and degrading perceived pitch/voicing.
  - The reconstruction retains the spectral envelope but loses the original temporal phase coherence, producing a noise-like or unnatural speech percept (subjectively perceived as whispered or noise-vocoded speech, $\text{STOI} = 0.8653$).
- **Difference: $\text{sig}_1$ vs. $\text{sig}_2$:**
  - $\text{sig}_1$ (Zero Phase) is deterministic and localized, aligning cosine components to create frame-rate periodic buzz artifacts.
  - $\text{sig}_2$ (Random Odd Phase) is stochastic and diffuse, dispersing energy randomly across each frame without constructive peaking (smooth colored noise).
- **Visual Proof:** [`outputs/plots/same_magnitude_different_waveform.png`](file:///d:/IISC/outputs/plots/same_magnitude_different_waveform.png) demonstrates that Original, $\text{sig}_1$, and $\text{sig}_2$ share identical magnitude spectrograms despite possessing completely different time-domain waveforms.

---

### Q5: Unintelligible Speech Synthesis & Recovery (Griffin-Lim vs Keyed)
> *Question: Create unintelligible version (using odd yet random phase) of every speech signal and ask them to recover the original back.*

#### How It Is Solved:
- **Module:** [`src/phase_recovery.py`](file:///d:/IISC/src/phase_recovery.py) $\to$ `create_unintelligible_speech()`, `recover_with_true_phase()`, `recover_with_griffin_lim()`, `griffin_lim_iteration_study()`
- **Synthesis:** An odd yet pseudorandom phase mask $\phi_{\text{scramble}}(m, k)$ replaces the original phase:
  $$S_{\text{scrambled}}(m, k) = \text{MG}(m, k) \, e^{j \phi_{\text{scramble}}(m, k)}$$
  Saved to [`outputs/audio/unintelligible_scrambled.wav`](file:///d:/IISC/outputs/audio/unintelligible_scrambled.wav).
- **Two Distinct Recovery Mechanisms:**
  1. **Keyed Phase Descrambler (Exact Inversion):**
     Reverses the phase mask using the known key/seed:
     $$\text{STOI} = \mathbf{1.0000}, \quad \text{Reconstruction SNR} = \mathbf{+143.57\text{ dB}}, \quad \text{MSE} = 7.91 \times 10^{-17}$$
     Saved to [`outputs/audio/recovered_keyed.wav`](file:///d:/IISC/outputs/audio/recovered_keyed.wav).
  2. **Griffin-Lim Algorithm (Blind Iterative Phase Retrieval):**
     Estimates a consistent phase from magnitude alone by alternating between spectral magnitude projection and time-domain STFT projection across $N_{\text{iter}}$ iterations:
     $$\hat{x}^{(i+1)} = \text{ISTFT} \left( |Y| \cdot \frac{\text{STFT}(\hat{x}^{(i)})}{|\text{STFT}(\hat{x}^{(i)})|} \right)$$
     Saved to [`outputs/audio/recovered_griffin_lim.wav`](file:///d:/IISC/outputs/audio/recovered_griffin_lim.wav).
- **Griffin-Lim Iteration Convergence Study:**
  Evaluated across $1, 4, 8, 16, 32, 64, 128$ iterations (plotted in [`outputs/plots/griffin_lim_iterations.png`](file:///d:/IISC/outputs/plots/griffin_lim_iterations.png)):

| Iterations | STOI Score | Waveform MSE | Spectral Convergence Error | Perceptual Character |
| :---: | :---: | :---: | :---: | :--- |
| **1** | $0.9369$ | $0.033168$ | $0.3844$ | Coarse intelligibility, audible phase roughness |
| **4** | $0.9673$ | $0.035730$ | $0.2489$ | Marked clarity improvement |
| **8** | $0.9794$ | $0.037278$ | $0.1820$ | Formants stabilized, low artifacts |
| **16** | $0.9891$ | $0.037685$ | $0.1213$ | High speech intelligibility |
| **32** | $0.9936$ | $0.037419$ | $0.0753$ | Natural consonant transitions |
| **64** | $0.9964$ | $0.036375$ | $0.0473$ | Highly clean, natural speech |
| **128** | $\mathbf{0.9984}$ | $0.034173$ | $\mathbf{0.0307}$ | Asymptotically converged magnitude consistency |

- **Fundamental Scientific Insight:**
  Under the chosen STFT parameters and 64 iterations, Griffin-Lim produced a reconstruction with high STOI ($0.9964$), indicating that substantial speech intelligibility can be recovered from the magnitude representation alone.
  However, notice that Griffin-Lim achieves $\text{STOI} = 0.9964$ with a negative waveform SNR ($-3.68\text{ dB}$) and negative Pearson correlation ($-0.17$). This demonstrates that:
  $$\mathbf{\text{Waveform Similarity}} \ne \mathbf{\text{Perceptual Similarity}} \ne \mathbf{\text{Speech Intelligibility}}$$

---

### Q6 & Q7: NoiseX-92 Noise Injection & STOI Calculation
> *Question: Add white, machinegun, babble noises from NoiseX dataset to the original wav file at -2dB, 0dB, 2dB, 5dB and 10dB SNR. Calculate STOI scores in each scenario.*

#### How It Is Solved:
- **Module:** [`src/audio_loader.py`](file:///d:/IISC/src/audio_loader.py) $\to$ `download_and_prepare_noise()`, `mix_noise_at_snr()`, [`src/stoi_benchmark.py`](file:///d:/IISC/src/stoi_benchmark.py) $\to$ `compute_stoi_safe()`
- **Noise Acquisition:** Automatically downloaded `white.wav`, `machinegun.wav`, and `babble.wav` from GitHub NoiseX-92, resampled from $19.98\text{ kHz} \to 16.0\text{ kHz}$.
- **SNR Mixing Formulation:**
  For clean speech $s[n]$ and noise $w[n]$:
  $$P_s = \frac{1}{N} \sum_{n=0}^{N-1} s^2[n], \quad P_w = \frac{1}{N} \sum_{n=0}^{N-1} w^2[n]$$
  $$\alpha = \sqrt{\frac{P_s}{P_w \cdot 10^{\text{SNR}_{\text{target}} / 10}}}, \quad y[n] = s[n] + \alpha \cdot w[n]$$
- **Noise Characterization (Waveform, Spectrogram, Welch PSD):**
  Plotted in [`outputs/plots/noise_characteristics.png`](file:///d:/IISC/outputs/plots/noise_characteristics.png) and [`outputs/plots/noise_psd_bpf_overlay.png`](file:///d:/IISC/outputs/plots/noise_psd_bpf_overlay.png):
  - **Machinegun Noise:** Impulsive, bursty temporal structure with quiet intervals between rounds. Yields the **highest STOI** ($0.7866$ at $-2\text{ dB}$ to $0.9556$ at $10\text{ dB}$) because speech formants remain unmasked during inter-burst gaps.
  - **White Noise:** Flat PSD across $0 - 8\text{ kHz}$. Moderate degradation ($0.6603$ at $-2\text{ dB}$ to $0.8469$ at $10\text{ dB}$) due to stationary broadband masking across all critical bands.
  - **Babble Noise:** Speech-shaped PSD peaking in the $300 - 3000\text{ Hz}$ formant region. Deepest degradation ($0.5837$ at $-2\text{ dB}$ to $0.8635$ at $10\text{ dB}$) due to overlapping formant energy (energetic masking) and competing phonemes (informational masking).

---

### Q8: Repeating STFT & Phase Operations Across All Noisy Signals
> *Question: Repeat the above operations (2 to 5) for each noise.*

#### How It Is Solved:
- **Module:** [`run_experiments.py`](file:///d:/IISC/run_experiments.py) $\to$ Stage 5 loop
- For all 3 noises $\times$ 5 SNR levels ($15\text{ scenarios}$):
  1. STFT computed on noisy speech: $S_{\text{noisy}} = \text{STFT}(y)$.
  2. $\text{sig}_1$ (Zero Phase) synthesized from $|S_{\text{noisy}}|$ and evaluated via STOI against clean speech.
  3. $\text{sig}_2$ (Random Odd Phase) synthesized from $|S_{\text{noisy}}| e^{j\text{PH}_1}$ and evaluated via STOI against clean speech.
- **Key Observation:**
  As noise power increases (lower SNR), the gap between raw noisy STOI and zero/random phase STOI narrows. At $-2\text{ dB}$, external noise energy dominates the spectral magnitude, making magnitude distortion comparable in severity to phase distortion.

---

### Q9: Band-Pass Filter (BPF) Design Decisions & Dual-Reference STOI Analysis
> *Question: Apply a Band Pass Filter to the signal and compare the STOI score with each noise. Comment on the filter design decisions and STOI scores.*

#### How It Is Solved:
- **Module:** [`src/filter_module.py`](file:///d:/IISC/src/filter_module.py) $\to$ `design_speech_bandpass_filter()`, `apply_bandpass_filter()`, `plot_filter_frequency_response()`
- **Filter Specifications:**
  - **Passband:** $300\text{ Hz} - 3400\text{ Hz}$ (Standard ITU-T G.712 telephony bandwidth).
  - **Topology:** 4th-order Butterworth IIR filter implemented in Second-Order Sections (SOS).
  - **Phase Characteristic:** Bidirectional zero-phase filtering (`sosfiltfilt`).
- **Filter Design Decisions & Justifications:**
  1. **Passband Selection ($300 - 3400\text{ Hz}$):** Speech formants essential for intelligibility (F1: $300-900\text{ Hz}$, F2: $1000-2500\text{ Hz}$, F3: $2500-3400\text{ Hz}$) fall within this band.
  2. **Cutoff Attenuation:** The high-pass cutoff ($300\text{ Hz}$) attenuates low-frequency components such as DC offset, room rumble, and the fundamental components of mains interference, though higher harmonics of electrical interference can extend above 300 Hz. The low-pass cutoff ($3400\text{ Hz}$) attenuates high-frequency hiss.
  3. **Butterworth Maximally Flat Passband:** $0\text{ dB}$ passband ripple preserves the relative amplitudes of vowel formants without resonant coloration.
  4. **Second-Order Sections (SOS):** Avoids coefficient quantization errors and numerical instability.
  5. **Zero-Phase Filtering (`sosfiltfilt`):** Forward-backward filtering cancels phase distortion, preventing group-delay smearing of envelope peaks that would artificially degrade STOI. *(Caveat: Forward-backward filtering squares the magnitude response $|H(f)|^2$, doubling effective filter order to 8).*
- **Dual-Reference STOI Analysis (Why Metric Design Matters):**
  - **Evaluation A (Practical Comparison vs. Full-Band Clean):** $\text{STOI}(\text{clean}_{\text{full}}, \text{BPF}(\text{noisy}))$  
    *Result:* Shows an apparent penalty ($\Delta \text{STOI}_A \approx -0.05$ to $-0.07$). Because clean speech has bandwidth up to $8\text{ kHz}$, STOI penalizes the missing outer $1/3$-octave bands as correlation loss.
  - **Evaluation B (Controlled Bandwidth-Matched Comparison):** $\text{STOI}(\text{BPF}(\text{clean}), \text{BPF}(\text{noisy}))$  
    *Result:* When referenced against bandpass-matched clean speech, BPF produces **positive STOI improvements** for Machinegun noise at $5\text{ dB}$ and $10\text{ dB}$ ($\Delta \text{STOI}_B = +0.0023$ and $+0.0027$) and recovers baseline for White noise ($\Delta \text{STOI}_B = -0.0018$).
  - **Babble Inefficacy:** BPF cannot improve intelligibility for Babble noise under either evaluation because babble noise shares the exact same $300 - 3400\text{ Hz}$ passband as target speech. Linear stationary filtering cannot separate overlapping speech spectra.

---

## Master Experimental Results Table

All metric evaluations across the complete 15-scenario experimental matrix:

| Noise Type | Target SNR | Measured SNR | Raw Noisy STOI | BPF Eval A (vs Full Clean) | $\Delta \text{STOI}_A$ | BPF Eval B (vs BPF Clean) | $\Delta \text{STOI}_B$ | $\text{sig}_1$ (Zero PH) STOI | $\text{sig}_2$ (Random PH) STOI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **White** | $-2\text{ dB}$ | $-2.00\text{ dB}$ | $0.6603$ | $0.5956$ | $-0.0647$ | $0.6480$ | $-0.0123$ | $0.5599$ | $0.5635$ |
| **White** | $0\text{ dB}$ | $0.00\text{ dB}$ | $0.6901$ | $0.6246$ | $-0.0655$ | $0.6800$ | $-0.0101$ | $0.5832$ | $0.5837$ |
| **White** | $+2\text{ dB}$ | $+2.00\text{ dB}$ | $0.7208$ | $0.6549$ | $-0.0659$ | $0.7130$ | $-0.0078$ | $0.6078$ | $0.6098$ |
| **White** | $+5\text{ dB}$ | $+5.00\text{ dB}$ | $0.7687$ | $0.7022$ | $-0.0665$ | $0.7639$ | $-0.0048$ | $0.6469$ | $0.6529$ |
| **White** | $+10\text{ dB}$ | $+10.00\text{ dB}$ | $0.8469$ | $0.7789$ | $-0.0679$ | $0.8450$ | $-0.0018$ | $0.7122$ | $0.7198$ |
| **Machinegun** | $-2\text{ dB}$ | $-2.00\text{ dB}$ | $0.7866$ | $0.7407$ | $-0.0460$ | $0.7799$ | $-0.0068$ | $0.7188$ | $0.6934$ |
| **Machinegun** | $0\text{ dB}$ | $0.00\text{ dB}$ | $0.8259$ | $0.7759$ | $-0.0500$ | $0.8227$ | $-0.0032$ | $0.7458$ | $0.7270$ |
| **Machinegun** | $+2\text{ dB}$ | $+2.00\text{ dB}$ | $0.8615$ | $0.8075$ | $-0.0540$ | $0.8612$ | $-0.0003$ | $0.7680$ | $0.7541$ |
| **Machinegun** | $+5\text{ dB}$ | $+5.00\text{ dB}$ | $0.9062$ | $0.8467$ | $-0.0595$ | $0.9085$ | **+0.0023** | $0.7927$ | $0.7960$ |
| **Machinegun** | $+10\text{ dB}$ | $+10.00\text{ dB}$ | $0.9556$ | $0.8901$ | $-0.0655$ | $0.9583$ | **+0.0027** | $0.8171$ | $0.8352$ |
| **Babble** | $-2\text{ dB}$ | $-2.00\text{ dB}$ | $0.5837$ | $0.5350$ | $-0.0486$ | $0.5657$ | $-0.0180$ | $0.5004$ | $0.5147$ |
| **Babble** | $0\text{ dB}$ | $0.00\text{ dB}$ | $0.6370$ | $0.5815$ | $-0.0555$ | $0.6220$ | $-0.0150$ | $0.5435$ | $0.5533$ |
| **Babble** | $+2\text{ dB}$ | $+2.00\text{ dB}$ | $0.6897$ | $0.6285$ | $-0.0612$ | $0.6774$ | $-0.0122$ | $0.5854$ | $0.5992$ |
| **Babble** | $+5\text{ dB}$ | $+5.00\text{ dB}$ | $0.7639$ | $0.6965$ | $-0.0673$ | $0.7554$ | $-0.0084$ | $0.6447$ | $0.6642$ |
| **Babble** | $+10\text{ dB}$ | $+10.00\text{ dB}$ | $0.8635$ | $0.7919$ | $-0.0716$ | $0.8599$ | $-0.0036$ | $0.7247$ | $0.7440$ |

---

## Key Theoretical Takeaways for Technical Interviews

1. **Waveform Error $\ne$ Speech Intelligibility:**  
   Point-to-point waveform metrics (MSE, SNR) measure sample alignment, while STOI measures intermediate sub-band temporal envelope correlations driving auditory comprehension. Griffin-Lim achieves $\text{STOI} = 0.9964$ with negative time-domain SNR ($-3.68\text{ dB}$).
2. **Phase Manipulation Does Not Remove Harmonics ($F_0$):**  
   Randomizing phase leaves the magnitude spectrum intact, preserving harmonic peaks at $F_0, 2F_0, 3F_0, \dots$. It destroys the *cross-frequency temporal phase alignment*, smearing glottal pulses into noise-like energy bursts.
3. **Hermitian Symmetry Requirement:**  
   Anti-symmetric (odd) phase with real boundaries at DC and Nyquist ($\phi[0], \phi[N/2] \in \{0, \pi\}$) is necessary and sufficient for real-valued time synthesis.
4. **Metric Design (Evaluation A vs. B):**  
   Band-limiting a signal causes full-band STOI to report an apparent penalty because outer frequency bands are empty. Testing against a bandwidth-matched clean reference decouples bandwidth loss from noise reduction.

---

## Automated Verification & Unit Testing

The test suite in [`test_dsp_pipeline.py`](file:///d:/IISC/test_dsp_pipeline.py) verifies all core DSP functionality:
- `test_audio_duration_and_sr`: Verifies $10.0\text{ s}$ length and $16{,}000\text{ Hz}$ sample rate.
- `test_stft_istft_baseline`: Verifies Perfect Reconstruction SNR $>130\text{ dB}$ and MSE $<10^{-16}$.
- `test_hermitian_symmetry_proof`: Verifies Hermitian error $\max_k |X[k] - X^*[N-k]| = 0.0$ and $\max |\text{Im}| = 0.0$ for odd phase, vs $>14.0$ for arbitrary phase.
- `test_sig1_and_sig2_generation`: Verifies real-valued output and waveform differences.
- `test_snr_mixing`: Verifies accurate noise scaling within $\pm 0.05\text{ dB}$ of target SNR.
- `test_phase_scrambling_and_recovery`: Verifies keyed recovery achieves $\text{STOI} > 0.99$.
- `test_bandpass_filter`: Verifies Butterworth SOS filter stability and zero-phase output.

Run the test suite with:
```bash
python -m unittest test_dsp_pipeline.py
```
