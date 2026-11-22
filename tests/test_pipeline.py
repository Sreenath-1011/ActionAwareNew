"""
test_pipeline.py - Comprehensive unit & integration tests for ActionAware Track 04 system.
"""

import os
import unittest
import cv2
import numpy as np
from code.utils import load_config, compute_temporal_iou, set_seed
from code.video_loader import get_video_metadata
from code.frame_extractor import extract_clips
from code.feature_extractor import FeatureExtractor
from code.action_recognition import ActionRecognizer
from code.action_localization import localize_actions
from code.importance_scoring import score_candidate_segments
from code.highlight_generator import select_highlights
from code.label_classifier import LabelClassifier
from code.json_exporter import export_prediction_json
from validate import validate_file, load_manifest, load_supported_classes


class TestActionAwarePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.cfg = load_config("config/config.yaml")
        cls.manifest = load_manifest(cls.cfg["dataset"]["manifest_path"])
        cls.supported_classes = load_supported_classes(cls.cfg["dataset"]["supported_classes_path"])
        
        # Create a 15-second test video
        cls.test_video_path = "tests/test_sample_15s.mp4"
        width, height, fps = 224, 224, 30.0
        dur_sec = 15.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.test_video_path, fourcc, fps, (width, height))
        for f in range(int(dur_sec * fps)):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video_path):
            try:
                os.remove(cls.test_video_path)
            except Exception:
                pass

    def test_01_temporal_iou(self):
        iou_perfect = compute_temporal_iou([0.0, 5.0], [0.0, 5.0])
        self.assertAlmostEqual(iou_perfect, 1.0)
        iou_none = compute_temporal_iou([0.0, 5.0], [6.0, 10.0])
        self.assertAlmostEqual(iou_none, 0.0)
        iou_partial = compute_temporal_iou([0.0, 5.0], [2.5, 7.5])
        self.assertAlmostEqual(iou_partial, 2.5 / 7.5)

    def test_02_video_loader(self):
        meta = get_video_metadata(self.test_video_path)
        self.assertEqual(meta["width"], 224)
        self.assertEqual(meta["height"], 224)
        self.assertAlmostEqual(meta["duration_sec"], 15.0, delta=0.5)

    def test_03_clip_extraction(self):
        meta = get_video_metadata(self.test_video_path)
        clips = extract_clips(self.test_video_path, meta, self.cfg)
        self.assertGreater(len(clips), 0)
        self.assertEqual(clips[0]["tensor"].shape, (3, 16, 224, 224))

    def test_04_feature_and_action_recognition(self):
        meta = get_video_metadata(self.test_video_path)
        clips = extract_clips(self.test_video_path, meta, self.cfg)[:2]
        feat_extractor = FeatureExtractor(self.cfg)
        clips = feat_extractor.extract_clip_predictions(clips)
        self.assertIn("top_categories", clips[0])
        
        act_recognizer = ActionRecognizer(self.cfg)
        clips = act_recognizer.process_clips(clips, dataset="epic_kitchens_100")
        self.assertIn(clips[0]["verb"], self.supported_classes["verbs"])
        self.assertIn(clips[0]["noun"], self.supported_classes["nouns"])

    def test_05_highlight_budget_constraint(self):
        # 100 seconds total video duration
        total_dur = 100.0
        candidates = [
            {"segment": [10.0, 15.0], "verb": "take", "noun": "plate", "score": 0.9, "importance": 0.95},
            {"segment": [20.0, 26.0], "verb": "cut", "noun": "onion", "score": 0.85, "importance": 0.90},
            {"segment": [30.0, 42.0], "verb": "cook", "noun": "food", "score": 0.80, "importance": 0.85},
            {"segment": [50.0, 70.0], "verb": "wash", "noun": "pan", "score": 0.70, "importance": 0.75},
        ]
        highlights = select_highlights(candidates, total_dur, self.cfg)
        self.assertGreater(len(highlights), 0)
        
        # Check budget: total duration <= 15% (15.0s)
        total_hl_dur = sum(h["segment"][1] - h["segment"][0] for h in highlights)
        self.assertLessEqual(total_hl_dur, 15.1)
        
        # Check min segment rule: at least one segment >= 4.0s
        has_min_4s = any((h["segment"][1] - h["segment"][0]) >= 4.0 for h in highlights)
        self.assertTrue(has_min_4s)
        
        # Check strictly non-overlapping & chronologically sorted
        for i in range(len(highlights) - 1):
            self.assertGreaterEqual(highlights[i + 1]["segment"][0], highlights[i]["segment"][1])

    def test_06_official_schema_validation(self):
        output_path = "tests/test_validation_output.json"
        payload = export_prediction_json(
            team_id="T07",
            video_id="test_video_01",
            duration_sec=60.0,
            label={"name": "cooking", "score": 0.95},
            actions=[
                {"segment": [5.0, 10.0], "verb": "take", "noun": "plate", "score": 0.88},
                {"segment": [20.0, 28.0], "verb": "wash", "noun": "knife", "score": 0.82},
            ],
            highlights=[
                {"segment": [5.0, 10.0]},
                {"segment": [20.0, 24.0]},
            ],
            output_path=output_path,
            classes_path=self.cfg["dataset"]["supported_classes_path"],
        )
        errors, warnings = validate_file(output_path, supported_classes=self.supported_classes)
        self.assertEqual(len(errors), 0)
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
