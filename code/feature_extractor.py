"""
feature_extractor.py - Feature extraction using pre-trained video backbone.
Optimized for CPU inference with batching and graceful weight loading.
"""

import os
import torch
import torch.nn as nn
from torchvision.models.video import s3d, S3D_Weights
from code.utils import setup_logger, resolve_device

logger = setup_logger("FeatureExtractor")


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg["general"].get("device", "auto"))
        self.batch_size = cfg["model"].get("batch_size", 4)
        self.model = None
        self.categories = None
        self._init_model()

    def _init_model(self):
        logger.info(f"Loading S3D model on device: {self.device}...")
        try:
            weights = S3D_Weights.DEFAULT
            self.model = s3d(weights=weights)
            self.categories = weights.meta.get("categories", [])
            logger.info(f"Loaded official pre-trained S3D weights with {len(self.categories)} Kinetics classes.")
        except Exception as e:
            logger.warning(f"Could not download pre-trained weights online ({e}). Initializing architecture.")
            self.model = s3d(weights=None)
            self.categories = [f"class_{i}" for i in range(400)]

        self.model.to(self.device)
        self.model.eval()

        # Video normalization constants (Kinetics standard)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37701], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

    def extract_clip_predictions(self, clips):
        """
        Runs batched forward pass on extracted clips.
        Annotates each clip with:
            - 'top_categories': list of (category_name, probability)
            - 'confidence': float (top-1 prob)
        """
        if not clips:
            return clips

        num_clips = len(clips)
        logger.info(f"Running inference across {num_clips} clips (batch size {self.batch_size})...")

        for i in range(0, num_clips, self.batch_size):
            batch_clips = clips[i : i + self.batch_size]
            tensors = [c["tensor"] for c in batch_clips]
            # Stack to [B, C, T, H, W]
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            # Normalize
            batch_tensor = (batch_tensor - self.mean) / self.std

            with torch.no_grad():
                logits = self.model(batch_tensor) # [B, 400]
                probs = torch.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)

            for b_idx, clip in enumerate(batch_clips):
                top_cats = []
                for k in range(5):
                    idx = int(topk_indices[b_idx, k].item())
                    score = float(topk_probs[b_idx, k].item())
                    cat_name = self.categories[idx] if idx < len(self.categories) else f"action_{idx}"
                    top_cats.append((cat_name, score))

                clip["top_categories"] = top_cats
                clip["confidence"] = top_cats[0][1] if top_cats else 0.0

        return clips
