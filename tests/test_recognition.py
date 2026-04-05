import json
import unittest
import uuid
from pathlib import Path
import shutil

import numpy as np

from recognition import (
    PROFILE_FORMAT_VERSION,
    RecognitionEngine,
    derive_threshold_from_self_scores,
    format_recognition_annotation,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "test-temp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def make_test_dir() -> Path:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class RecognitionProfileTests(unittest.TestCase):
    def test_profile_matches_corpus_requires_current_format(self):
        root = make_test_dir()
        try:
            sample = root / "sample.wav"
            sample.write_bytes(b"abc")
            profile = root / "profile.json"
            verifier = RecognitionEngine(profile_path=profile)
            payload = {
                "version": PROFILE_FORMAT_VERSION - 1,
                "samples_fingerprint": verifier.corpus_fingerprint([sample]),
                "samples": [{"path": str(sample), "self_score": 0.9}],
                "reference_embeddings": [[1.0, 0.0]],
                "corpus_self_score_min": 0.9,
                "corpus_self_score_p10": 0.9,
                "corpus_self_score_avg": 0.9,
                "corpus_self_score_max": 0.9,
                "active_threshold": 0.87,
                "threshold_source": "derived",
                "centroid": [1.0, 0.0],
            }
            profile.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(verifier.profile_matches_corpus([sample]))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_profile_matches_corpus_requires_reference_embeddings_and_self_scores(self):
        root = make_test_dir()
        try:
            sample = root / "sample.wav"
            sample.write_bytes(b"abc")
            profile = root / "profile.json"
            verifier = RecognitionEngine(profile_path=profile)
            payload = {
                "version": PROFILE_FORMAT_VERSION,
                "samples_fingerprint": verifier.corpus_fingerprint([sample]),
                "samples": [{"path": str(sample)}],
                "centroid": [1.0, 0.0],
            }
            profile.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(verifier.profile_matches_corpus([sample]))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_profile_matches_corpus_requires_matching_fingerprint(self):
        root = make_test_dir()
        try:
            sample_a = root / "sample-a.wav"
            sample_b = root / "sample-b.wav"
            sample_a.write_bytes(b"abc")
            sample_b.write_bytes(b"xyz")
            profile = root / "profile.json"
            verifier = RecognitionEngine(profile_path=profile)
            payload = {
                "version": PROFILE_FORMAT_VERSION,
                "samples_fingerprint": verifier.corpus_fingerprint([sample_a]),
                "samples": [{"path": str(sample_a), "self_score": 0.9}],
                "reference_embeddings": [[1.0, 0.0]],
                "corpus_self_score_min": 0.9,
                "corpus_self_score_p10": 0.9,
                "corpus_self_score_avg": 0.9,
                "corpus_self_score_max": 0.9,
                "active_threshold": 0.87,
                "threshold_source": "derived",
                "centroid": [1.0, 0.0],
            }
            profile.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(verifier.profile_matches_corpus([sample_b]))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_save_profile_round_trips_reference_embeddings(self):
        root = make_test_dir()
        try:
            profile = root / "profile.json"
            verifier = RecognitionEngine(profile_path=profile)
            verifier.save_profile(
                {
                    "version": PROFILE_FORMAT_VERSION,
                    "centroid": np.asarray([1.0, 2.0], dtype=np.float32),
                    "reference_embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    "samples": [{"path": "a.wav", "self_score": 0.95}],
                    "samples_fingerprint": "abc",
                    "corpus_self_score_min": 0.95,
                    "corpus_self_score_p10": 0.95,
                    "corpus_self_score_avg": 0.95,
                    "corpus_self_score_max": 0.95,
                    "active_threshold": 0.92,
                    "threshold_source": "derived",
                }
            )
            loaded = verifier.load_profile()
            self.assertEqual(loaded["reference_embeddings"].shape, (2, 2))
            self.assertEqual(loaded["centroid"].shape, (2,))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_threshold_derivation_uses_min_minus_margin_with_clamp(self):
        self.assertAlmostEqual(derive_threshold_from_self_scores([0.90, 0.91, 0.93]), 0.87)
        self.assertAlmostEqual(derive_threshold_from_self_scores([0.60, 0.62]), 0.65)

    def test_format_recognition_annotation_includes_score_threshold_and_delta(self):
        self.assertEqual(
            format_recognition_annotation(0.757, 0.803, -0.046),
            "recognition 0.757/0.803 (-0.046)",
        )


if __name__ == "__main__":
    unittest.main()
