# Speech Signal Processing Analysis Report
**Rigorous Academic Analysis: STFT, Phase Coherence, Hermitian Symmetry, Noise Robustness, Dual-Reference STOI & Filter Design**

---

## 1. Executive Summary
This report presents an exhaustive signal processing investigation into the role of spectral magnitude and phase in speech representation and intelligibility. Using a 10.0-second speech recording (`clean_voice.wav`, sampled at 16 kHz), we evaluate short-time Fourier transform (STFT) properties, analyze reconstruction under zero phase ($\text{sig}_1$) and random odd-symmetric phase ($\text{sig}_2$), mathematically and empirically prove the Hermitian symmetry conditions required for real-valued synthesis, analyze speech unintelligibility and iterative phase retrieval (Griffin-Lim algorithm across 1 to 128 iterations), benchmark Short-Time Objective Intelligibility (STOI) against three NoiseX-92 noises (white, machinegun, babble) across five SNR levels ($-2, 0, 2, 5, 10$ dB), and conduct a dual-reference evaluation of a 4th-order Butterworth Band-Pass Filter (BPF).

---

## 2. STFT Implementation Details & Mathematical Representation

### Precise Implementation Parameters
To guarantee exact reproducibility, the STFT and ISTFT implementation parameters are specified below:
- **Sampling Frequency ($f_s$):** $16{,}000\text{ Hz}$ (audio standardized to 160,000 samples, mono).
- **FFT Length ($N$ / `n_fft`):** $512\text{ samples}$ ($32.0\text{ ms}$ window length).
- **Window Type:** Periodic Hann window, $w[n] = 0.5 - 0.5 \cos(2\pi n / N)$ for $0 \le n < N$.
- **Hop Size ($R$ / `hop_length`):** $128\text{ samples}$ ($8.0\text{ ms}$, 75% frame overlap).
- **Framing & Padding:** Centered analysis (`center=True`), where the signal is padded on both ends by $N/2 = 256$ samples using reflection padding.
- **Spectral Representation:** One-sided discrete spectrum comprising $K = N/2 + 1 = 257$ non-negative frequency bins ($0\text{ Hz}$ to $8{,}000\text{ Hz}$).
- **Magnitude Modification:** Magnitude spectrum $|X(m, k)|$ is preserved exactly without dynamic compression or spectral subtraction prior to ISTFT.
- **Synthesis:** Inverse STFT is performed via normalized overlap-add (OLA) with dual synthesis windowing and inverse real FFT (`irfft`).

### Mathematical Formulation
For the windowed discrete-time speech signal, the complex STFT at frame index $m$ and frequency bin $k$ is:
$$X(m, k) = \sum_{n=0}^{N-1} x[mR + n] \, w[n] \, e^{-j \frac{2\pi}{N} k n}$$
Decomposing into polar form:
- **Magnitude Spectrogram:**
  $$\text{MG}(m, k) = |X(m, k)| = \sqrt{\text{Re}\{X(m, k)\}^2 + \text{Im}\{X(m, k)\}^2}$$
- **Phase Spectrogram:**
  $$\text{PH}(m, k) = \angle X(m, k) = \text{atan2}(\text{Im}\{X(m, k)\}, \text{Re}\{X(m, k)\}) \in (-\pi, \pi]$$

### STFT $\to$ ISTFT Identity Reconstruction Baseline
Before conducting phase manipulations, we verified the numerical precision of the STFT $\to$ ISTFT pipeline on the speech signal:
- **Waveform Mean Squared Error (MSE):** $5.36 \times 10^{-17}$
- **Maximum Absolute Sample Error:** $1.19 \times 10^{-7}$
- **Reconstruction SNR:** $\mathbf{132.55\text{ dB}}$
- **Pearson Waveform Correlation ($\rho$):** $1.000000$
- **STOI Score:** $1.0000$

This confirms that the STFT analysis-synthesis engine satisfies the Perfect Reconstruction (PR) condition up to floating-point machine precision.

---

## 3. Analysis of $\text{sig}_1$ (Zero-Phase Reconstruction)

### Construction
Setting the spectral phase to zero across all time-frequency bins:
$$X_1(m, k) = \text{MG}(m, k) \, e^{j \cdot 0} = \text{MG}(m, k) \in \mathbb{R}^+$$
Inverse STFT is performed via overlap-add:
$$\text{sig}_1[n] = \text{ISTFT}\{\text{MG}(m, k)\}$$

### Defensible Physical and Acoustic Reasoning
- **Temporal Alignment of Cosine Components:**
  Setting all spectral phases to zero aligns the cosine components, producing a waveform with a strong even-symmetry/time-localization characteristic that differs substantially from the original speech frame.
- **Frame-Rate Structure & Timbral Artifacts:**
  Zero-phase reconstruction removes the original phase relationships and can introduce strong temporal concentration and periodic artifacts related to the frame structure. In our reconstruction, these artifacts were perceived as a robotic/buzzy quality.
- **Phonetic Intelligibility vs Naturalness:**
  Although speech naturalness and glottal pulse shaping are degraded, basic speech intelligibility remains largely intact ($\text{STOI} = 0.8400$). The preserved spectral envelope and temporal energy patterns retain information relevant to speech intelligibility, even though the original fine phase relationships are disrupted.

---

## 4. Analysis of $\text{sig}_2$ (Odd Random Phase $\text{PH}_1$) & Real-Valued Reconstruction

### Condition for Real-Valued Time Domain Signals
A discrete-time sequence $x[n]$ is real-valued ($x[n] \in \mathbb{R}$) if and only if its Fourier spectrum satisfies **Hermitian (conjugate) symmetry**.

#### 1. For the Full $N$-Point FFT Representation:
The conjugate symmetry relation across all $N$ bins is:
$$X[k] = X^*[(-k) \bmod N] = X^*[N - k] \quad \text{for } k = 1, \dots, N-1$$
In polar coordinates ($X[k] = |X[k]| e^{j \phi[k]}$):
1. **Magnitude Condition (Even Symmetry):**
   $$|X[k]| = |X[N - k]| \quad \forall k$$
2. **Phase Condition (Odd Symmetry):**
   $$\phi[k] \equiv -\phi[N - k] \pmod{2\pi} \quad \text{for } 1 \le k \le \frac{N}{2}-1$$
3. **Boundary Bins (DC and Nyquist):**
   - At $k = 0$ (DC): $X[0] = X^*[0] \implies \text{Im}\{X[0]\} = 0 \implies \phi[0] \in \{0, \pi\}$.
   - At $k = N/2$ (Nyquist, for even $N$): $X[N/2] = X^*[N/2] \implies \text{Im}\{X[N/2]\} = 0 \implies \phi[N/2] \in \{0, \pi\}$.

#### 2. For the One-Sided STFT Representation:
In practical audio implementations using real-FFT libraries (`rfft` / `irfft`), the STFT stores only the non-negative frequencies $k = 0, \dots, N/2$. The inverse real FFT (`irfft`) algebraically enforces conjugate symmetry for negative frequencies by definition:
$$X[N - k] \triangleq X^*[k]$$
However, for an arbitrary phase matrix, if non-zero imaginary components are supplied at the DC or Nyquist bins, `irfft` discards them. Furthermore, in an explicit full $N$-point inverse FFT (`ifft`), failure to satisfy odd phase symmetry results in a non-zero imaginary time sequence.

### Quantitative Numerical Verification of Hermitian Symmetry
We tested both an arbitrary random phase and an odd-symmetric random phase on active speech frame 50 using a full two-sided 512-point DFT:

| Phase Configuration | Hermitian Error $\max_k |X[k] - X^*[N-k]|$ | Time-Domain Imaginary Error $\max |\text{Im}(x)|$ | Real-Valued? |
| :--- | :---: | :---: | :---: |
| **Arbitrary Random Phase** | $\mathbf{14.2224}$ | $\mathbf{1.4783 \times 10^{-1}}$ | **NO** (Strongly Complex) |
| **Odd-Symmetric Phase** | $\mathbf{0.000000}$ | $\mathbf{0.000000}$ | **YES** (Strictly Real) |

This provides mathematical and numerical proof that odd phase symmetry guarantees a real-valued time-domain signal.

### How Many Valid Phase Patterns Are Possible?
1. **Continuous Phase Space ($\phi \in (-\pi, \pi]$):**
   There are **uncountably infinite ($2^{\aleph_0}$)** valid phase patterns per frame. For $N=512$, there are $N/2 - 1 = 255$ independent interior positive frequency bins. Choosing any arbitrary set of phases $\phi[1], \dots, \phi[255] \in (-\pi, \pi]$, assigning $\phi[N-k] = -\phi[k]$, and setting $\phi[0], \phi[N/2] \in \{0, \pi\}$ guarantees a purely real time signal.
2. **Binary Phase Quantization ($\phi \in \{0, \pi\}$):**
   If phase is restricted to binary values $\{0, \pi\}$ (representing sign flips $e^{j 0} = +1, e^{j \pi} = -1$), then $\sin(0) = \sin(\pi) = 0$, so $\phi[N-k] \equiv -\phi[k] \pmod{2\pi}$ is trivially satisfied for every bin!
   - Independent degrees of freedom per frame: 1 DC bin + 1 Nyquist bin + $(N/2 - 1)$ interior bins = $N/2 + 1 = 257$ bins.
   - Total binary patterns per frame:
     $$2 \times 2 \times 2^{N/2 - 1} = 2^{N/2 + 1} = 2^{257} \approx \mathbf{2.3158 \times 10^{77} \text{ patterns}}$$
3. **Discrete $M$-Level Quantization (e.g. IEEE 754 Floating-Point with $M \approx 2^{64}$):**
   Total patterns per frame: $4 \times M^{N/2 - 1}$. Across $T$ time frames: $(4 \cdot M^{N/2 - 1})^T$.

---

## 5. Distinction: Original vs $\text{sig}_1$ vs $\text{sig}_2$

### Pitch, Harmonics, and the $F_0$ Distinction
An essential theoretical distinction is that **randomizing the phase does NOT mathematically remove the fundamental frequency $F_0$ or harmonic peaks from the magnitude spectrum**.
- The magnitude spectrum $|X(m, k)|$ preserves the spectral harmonic peaks spaced at approximately $F_0, 2F_0, 3F_0, \dots$.
- However, random phase destroys the **temporal phase coherence across harmonics** that is responsible for aligning the waves into distinct pitch periods and glottal pulse impulses.
- Consequently, the reconstruction retains the vocal tract spectral envelope but loses the original temporal phase coherence, producing a noise-like or unnatural speech percept (often subjectively perceived as whispered or noise-vocoded speech), substantially degrading perceived pitch and voicing.

### Unified Metric Suite Comparison
To rigorously decouple waveform similarity, perceptual similarity, and speech intelligibility, we evaluate all signals against the clean original across four complementary metrics:

| Signal Condition | Waveform MSE | Pearson Correlation ($\rho$) | Time-Domain SNR (dB) | STOI Score | Perceptual Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ISTFT (Original STFT)** | $5.36 \times 10^{-17}$ | $1.0000$ | $+145.26\text{ dB}$ | $1.0000$ | Identical to clean speech |
| **$\text{sig}_1$ (Zero Phase)** | $1.87 \times 10^{-2}$ | $-0.0044$ | $-0.16\text{ dB}$ | $0.8400$ | Robotic, buzzy, metallic timbre |
| **$\text{sig}_2$ (Random Odd Phase)** | $2.22 \times 10^{-2}$ | $+0.0150$ | $-0.92\text{ dB}$ | $0.8653$ | Noise-like, whispered timbre |
| **Scrambled Speech** | $2.24 \times 10^{-2}$ | $+0.0056$ | $-0.95\text{ dB}$ | $0.8719$ | Unintelligible masked speech |
| **Griffin-Lim (64 iters)** | $4.20 \times 10^{-2}$ | $-0.1693$ | $-3.68\text{ dB}$ | $\mathbf{0.9964}$ | Fully intelligible, clear voice |
| **Keyed Recovered** | $7.91 \times 10^{-17}$ | $1.0000$ | $+143.57\text{ dB}$ | $1.0000$ | Pristine exact recovery |

> [!IMPORTANT]
> **Waveform Similarity $\ne$ Perceptual Similarity $\ne$ Speech Intelligibility:**
> Notice that Griffin-Lim achieves an almost perfect STOI score of **$0.9964$**, yet its waveform MSE is $0.042$ and time-domain SNR is negative ($-3.68\text{ dB}$).
> STOI evaluates intermediate sub-band temporal envelope correlations that drive human auditory intelligibility; it does not measure sample-by-sample waveform identity. High STOI indicates restored speech intelligibility, NOT sample-by-sample waveform equivalence.

---

## 6. Griffin-Lim Iteration Count Study

Under the chosen STFT parameters ($N=512$, $R=128$, Hann window), we investigated how phase reconstruction quality evolves with iteration count:

| Iterations | STOI Score | Waveform MSE | Spectral Convergence Error | Perceptual Quality |
| :---: | :---: | :---: | :---: | :--- |
| **1** | $0.9369$ | $0.033168$ | $0.3844$ | Coarse intelligibility, audible phase artifacts |
| **4** | $0.9673$ | $0.035730$ | $0.2489$ | Noticeable clarity improvement |
| **8** | $0.9794$ | $0.037278$ | $0.1820$ | Clean vowel formants, low roughness |
| **16** | $0.9891$ | $0.037685$ | $0.1213$ | Highly intelligible speech |
| **32** | $0.9936$ | $0.037419$ | $0.0753$ | Natural consonant transitions |
| **64** | $0.9964$ | $0.036375$ | $0.0473$ | Highly clean, natural speech |
| **128** | $\mathbf{0.9984}$ | $0.034173$ | $\mathbf{0.0307}$ | Asymptotically converged magnitude consistency |

**Scientific Takeaway:** Under the chosen STFT parameters and 64 iterations, Griffin-Lim produced a reconstruction with high STOI ($0.9964$), indicating that substantial speech intelligibility can be recovered from the magnitude representation alone. However, Griffin-Lim depends heavily on STFT redundancy (75% overlap provides strong cross-frame consistency constraints), window shape, and iteration count, and does not guarantee recovery of the true original phase.

---

## 7. NoiseX-92 Characterization & Dual-Reference STOI Benchmarking

### Noise Spectral & Temporal Analysis
To avoid unfounded causal claims, we analyzed the empirical characteristics of the three NoiseX-92 noise signals:
1. **Machinegun Noise:** Characterized by bursty, impulsive, highly non-stationary time-domain spikes separated by quiet inter-burst intervals. A likely explanation for its higher STOI scores is that this intermittent structure leaves speech frames between gunshot bursts comparatively unmasked.
2. **White Noise:** Demonstrates a flat Power Spectral Density (PSD) across the entire Nyquist band ($0 - 8\text{ kHz}$) and continuous stationary time-domain variance, producing continuous, uniform masking across all auditory critical bands.
3. **Babble Noise:** Displays a shaped PSD peaking in the $300 - 3000\text{ Hz}$ region and rapid envelope fluctuations matching human speech formants. This creates severe energetic and informational masking against the target speaker.

### Dual-Reference STOI Benchmark Table
We present both:
- **Evaluation A (Practical Comparison):** $\text{STOI}(\text{clean}_{\text{full}}, \text{noisy})$ vs $\text{STOI}(\text{clean}_{\text{full}}, \text{BPF}(\text{noisy}))$
  *Answers: "What happens to the metric when I apply this filter directly to the noisy recording?"*
- **Evaluation B (Controlled Bandwidth-Matched Comparison):** $\text{STOI}(\text{BPF}(\text{clean}), \text{BPF}(\text{noisy}))$
  *Answers: "After controlling for the bandwidth limitation, does the filter improve speech intelligibility?"*

| Noise Type | SNR (dB) | Raw Noisy STOI | BPF Eval A (vs Full Clean) | $\Delta \text{STOI}_A$ | BPF Eval B (vs BPF Clean) | $\Delta \text{STOI}_B$ | $\text{sig}_1$ STOI | $\text{sig}_2$ STOI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **White** | -2 | 0.6603 | 0.5956 | -0.0647 | 0.6480 | -0.0123 | 0.5599 | 0.5635 |
| **White** | 0 | 0.6901 | 0.6246 | -0.0655 | 0.6800 | -0.0101 | 0.5832 | 0.5837 |
| **White** | 2 | 0.7208 | 0.6549 | -0.0659 | 0.7130 | -0.0078 | 0.6078 | 0.6098 |
| **White** | 5 | 0.7687 | 0.7022 | -0.0665 | 0.7639 | -0.0048 | 0.6469 | 0.6529 |
| **White** | 10 | 0.8469 | 0.7789 | -0.0679 | 0.8450 | -0.0018 | 0.7122 | 0.7198 |
| **Machinegun** | -2 | 0.7866 | 0.7407 | -0.0460 | 0.7799 | -0.0068 | 0.7188 | 0.6934 |
| **Machinegun** | 0 | 0.8259 | 0.7759 | -0.0500 | 0.8227 | -0.0032 | 0.7458 | 0.7270 |
| **Machinegun** | 2 | 0.8615 | 0.8075 | -0.0540 | 0.8612 | -0.0003 | 0.7680 | 0.7541 |
| **Machinegun** | 5 | 0.9062 | 0.8467 | -0.0595 | 0.9085 | **+0.0023** | 0.7927 | 0.7960 |
| **Machinegun** | 10 | 0.9556 | 0.8901 | -0.0655 | 0.9583 | **+0.0027** | 0.8171 | 0.8352 |
| **Babble** | -2 | 0.5837 | 0.5350 | -0.0486 | 0.5657 | -0.0180 | 0.5004 | 0.5147 |
| **Babble** | 0 | 0.6370 | 0.5815 | -0.0555 | 0.6220 | -0.0150 | 0.5435 | 0.5533 |
| **Babble** | 2 | 0.6897 | 0.6285 | -0.0612 | 0.6774 | -0.0122 | 0.5854 | 0.5992 |
| **Babble** | 5 | 0.7639 | 0.6965 | -0.0673 | 0.7554 | -0.0084 | 0.6447 | 0.6642 |
| **Babble** | 10 | 0.8635 | 0.7919 | -0.0716 | 0.8599 | -0.0036 | 0.7247 | 0.7440 |

---

## 8. Band-Pass Filter Design Decisions & Analysis

### Design Decisions
1. **Passband Selection (300 Hz – 3400 Hz):**
   - Matches the standard ITU-T telephony band (G.712).
   - Core speech formants (F1: $300 - 900\text{ Hz}$, F2: $1000 - 2500\text{ Hz}$, F3: $2500 - 3400\text{ Hz}$) fall within this passband.
   - The high-pass cutoff ($300\text{ Hz}$) attenuates low-frequency components such as DC offset, room rumble, and the fundamental components of 50/60 Hz mains interference (though higher harmonics of electrical interference can extend above 300 Hz).
   - The low-pass cutoff ($3400\text{ Hz}$) attenuates high-frequency hiss.
2. **Filter Topology (4th-Order Butterworth with SOS):**
   - Butterworth provides a **maximally flat passband** ($0\text{ dB}$ passband ripple), preserving the natural spectral envelope and formant amplitudes of vowels without artificial coloration.
   - Second-Order Sections (SOS) guarantee numerical stability against finite-precision floating-point roundoff errors.
3. **Zero-Phase Filtering (`sosfiltfilt`):**
   - Forward-backward filtering cancels the filter's phase distortion, making it suitable for offline analysis where preservation of temporal envelope alignment is critical for STOI.
   - *Technical caveat:* Forward-backward filtering effectively squares the magnitude response $|H(f)|^2$, doubling the effective filter order to 8 and steepening passband roll-off.

### Why Metric Design Explains the Apparent BPF Penalty
- In **Evaluation A**, STOI evaluates 15 one-third octave bands spanning up to ~5 kHz. The BPF suppresses energy outside 300–3400 Hz, causing STOI to penalize the missing upper bands as correlation loss ($\Delta \text{STOI}_A \approx -0.05$ to $-0.07$).
- In **Evaluation B**, once the reference is matched to the same passband, BPF yields positive gains for Machinegun noise at $5\text{ dB}$ and $10\text{ dB}$ ($\Delta \text{STOI}_B = +0.0023$ and $+0.0027$).
- For **Babble Noise**, filtering fails to improve STOI under either evaluation because babble noise occupies the exact same $300 - 3400\text{ Hz}$ frequency band as target speech. Linear stationary filtering cannot separate overlapping speech signals.
