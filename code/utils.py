"""
utils.py - Common utilities for ActionAware pipeline.
Includes logging, IoU computation, timestamp formatting, and reproducibility setup.
"""

import json
import logging
import os
import random
import sys
import numpy as np
import torch
import yaml


def setup_logger(name="ActionAware", level=logging.INFO):
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed=42):
    """Enforces reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path="config/config.yaml"):
    """Loads YAML configuration file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(json_path):
    """Loads a JSON file safely."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, json_path, indent=2):
    """Saves data dictionary to formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def compute_temporal_iou(seg1, seg2):
    """
    Computes 1D temporal Intersection-over-Union between two segments [s1, e1] and [s2, e2].
    """
    s1, e1 = seg1
    s2, e2 = seg2
    inter_s = max(s1, s2)
    inter_e = min(e1, e2)
    intersection = max(0.0, inter_e - inter_s)
    union = (e1 - s1) + (e2 - s2) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def resolve_device(device_setting="auto"):
    """Determines appropriate PyTorch computation device."""
    if device_setting == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_setting == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
