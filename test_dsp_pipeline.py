import unittest
import numpy as np
import os
import soundfile as sf
from src.audio_loader import prepare_clean_speech, mix_noise_at_snr, TARGET_SR, TARGET_DURATION_SEC
from src.stft_transforms import (
    compute_stft,
    invert_stft,
    compute_sig1,
    compute_sig2,
    verify_real_valued_condition_dft,
)
from src.phase_recovery import (
    create_unintelligible_speech,
    recover_with_true_phase,
    recover_with_griffin_lim,
)
from src.filter_module import design_speech_bandpass_filter, apply_bandpass_filter
from src.stoi_benchmark import compute_stoi_safe

class TestDSPPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare clean speech if not already present
        cls.speech, cls.sr = prepare_clean_speech(
            input_audio_path="VoiceRecord.mp3",
            output_wav_path="data/clean_voice.wav",
            sr=TARGET_SR,
            duration=TARGET_DURATION_SEC,
        )

    def test_audio_duration_and_sr(self):
        self.assertEqual(self.sr, 16000)
        self.assertEqual(len(self.speech), 160000)
        # Check non-empty
        self.assertGreater(np.max(np.abs(self.speech)), 0.1)

    def test_stft_istft_baseline(self):
        from src.stft_transforms import evaluate_stft_istft_baseline
        res = evaluate_stft_istft_baseline(self.speech, sr=self.sr)
        self.assertLess(res["mse"], 1e-4)
        self.assertGreater(res["snr_db"], 40.0)
        self.assertGreater(res["stoi"], 0.99)

    def test_hermitian_symmetry_proof(self):
        # Pick active speech frame magnitude
        S, MG, PH = compute_stft(self.speech)
        frame_mg = MG[:, 50]
        res = verify_real_valued_condition_dft(frame_mg)
        
        # Arbitrary phase must have significant non-zero imaginary component and Hermitian error
        self.assertGreater(res["max_imag_arbitrary"], 1e-2)
        self.assertGreater(res["herm_err_arbitrary"], 1.0)
        # Odd-symmetric phase must have zero imaginary component and zero Hermitian error up to float precision
        self.assertLess(res["max_imag_odd"], 1e-6)
        self.assertLess(res["herm_err_odd"], 1e-6)

    def test_sig1_and_sig2_generation(self):
        S, MG, PH = compute_stft(self.speech)
        sig1, S1 = compute_sig1(MG, length=len(self.speech))
        sig2, S2, PH1 = compute_sig2(MG, length=len(self.speech))

        self.assertEqual(len(sig1), len(self.speech))
        self.assertEqual(len(sig2), len(self.speech))
        self.assertTrue(np.isrealobj(sig1))
        self.assertTrue(np.isrealobj(sig2))

        # Check difference: sig1 and sig2 should differ significantly in waveform
        diff_1_2 = np.mean((sig1 - sig2) ** 2)
        self.assertGreater(diff_1_2, 1e-4)

    def test_snr_mixing(self):
        noise = np.random.randn(len(self.speech))
        for target_snr in [-2.0, 0.0, 5.0, 10.0]:
            noisy, scaled_n, actual_snr = mix_noise_at_snr(self.speech, noise, target_snr)
            self.assertAlmostEqual(actual_snr, target_snr, places=2)

    def test_phase_scrambling_and_recovery(self):
        scrambled, MG, PH_orig, PH_scramble = create_unintelligible_speech(
            self.speech[:32000], output_path=None
        )
        # Scrambled STOI should be low
        stoi_scrambled = compute_stoi_safe(self.speech[:32000], scrambled)
        
        # Keyed recovery should restore exact signal
        recovered_keyed = recover_with_true_phase(MG, PH_orig, length=32000, output_path=None)
        stoi_keyed = compute_stoi_safe(self.speech[:32000], recovered_keyed)
        self.assertGreater(stoi_keyed, 0.99)
        self.assertGreater(stoi_keyed, stoi_scrambled)

    def test_bandpass_filter(self):
        sos = design_speech_bandpass_filter(300, 3400, sr=self.sr)
        filtered = apply_bandpass_filter(self.speech, sos)
        self.assertEqual(len(filtered), len(self.speech))
        self.assertTrue(np.isrealobj(filtered))

if __name__ == "__main__":
    unittest.main()
