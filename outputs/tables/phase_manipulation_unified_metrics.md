# Phase Manipulation Unified Metrics

| Signal                  |   Waveform MSE |   Pearson Corr |   SNR (dB) |   STOI |
|:------------------------|---------------:|---------------:|-----------:|-------:|
| ISTFT (Original STFT)   |    5.35529e-17 |         1      |     145.26 | 1      |
| sig1 (Zero Phase)       |    0.0186839   |        -0.0044 |      -0.16 | 0.84   |
| sig2 (Random Odd Phase) |    0.0222268   |         0.015  |      -0.92 | 0.8653 |
| Scrambled Speech        |    0.0223852   |         0.0056 |      -0.95 | 0.8719 |
| Griffin-Lim (64 iters)  |    0.0420238   |        -0.1693 |      -3.68 | 0.9964 |
| Keyed Recovered         |    7.91497e-17 |         1      |     143.57 | 1      |
