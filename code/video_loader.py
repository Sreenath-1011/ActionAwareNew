"""
video_loader.py - Extracts video metadata and verifies input against manifest.
"""

import os
import cv2
from code.utils import setup_logger

logger = setup_logger("VideoLoader")


def get_video_metadata(video_path, manifest=None):
    """
    Reads basic video properties using OpenCV:
    fps, total_frames, duration_sec, width, height.
    Cross-checks with manifest if available.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    # Cross-reference with manifest if video_id is found
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    manifest_dur = None
    dataset_type = "unknown"

    if manifest and video_id in manifest:
        manifest_dur = manifest[video_id]["duration_sec"]
        dataset_type = manifest[video_id]["dataset"]
        # If OpenCV reports discrepancy due to container indexing, manifest is authoritative
        if abs(duration_sec - manifest_dur) > 0.5:
            logger.warning(
                f"Video {video_id} OpenCV duration ({duration_sec:.2f}s) differs from manifest ({manifest_dur:.2f}s). Using manifest."
            )
            duration_sec = manifest_dur

    logger.info(
        f"Loaded video '{video_id}': {width}x{height} @ {fps:.2f} fps, {total_frames} frames ({duration_sec:.2f}s, dataset: {dataset_type})"
    )

    return {
        "video_path": video_path,
        "video_id": video_id,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "width": width,
        "height": height,
        "dataset": dataset_type,
    }
