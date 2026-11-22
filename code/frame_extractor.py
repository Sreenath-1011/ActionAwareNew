"""
frame_extractor.py - Timestamp-aware clip extraction and frame sampling.
Extracts uniform clips (e.g. 16 frames per 2.0s window) formatted for video models.
"""

import cv2
import numpy as np
import torch
from code.utils import setup_logger

logger = setup_logger("FrameExtractor")


def extract_clips(video_path, metadata, cfg):
    """
    Extracts temporal clips across the video duration with sliding window stride.
    Returns:
        clips: list of dicts, each with:
            - 'tensor': PyTorch FloatTensor [3, T, H, W] normalized to [0, 1]
            - 'start_sec': float
            - 'end_sec': float
            - 'center_sec': float
            - 'motion_score': float (energy/dynamics heuristic for idle suppression)
    """
    fps = metadata["fps"]
    duration = metadata["duration_sec"]
    total_frames = metadata["total_frames"]

    clip_frames = cfg["video"]["clip_frames"]         # typically 16
    stride_sec = cfg["video"]["clip_stride_sec"]      # typically 2.0s
    frame_size = cfg["video"]["frame_size"]           # 224

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")

    clips = []
    current_time = 0.0

    # Minimum window length is 1.0s or clip_frames / fps
    clip_duration = max(1.0, clip_frames / fps)

    while current_time < duration:
        start_sec = current_time
        end_sec = min(duration, start_sec + clip_duration)

        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))

        if end_frame <= start_frame:
            end_frame = min(total_frames, start_frame + clip_frames)

        # Uniformly sample clip_frames indices
        if end_frame - start_frame >= clip_frames:
            frame_indices = np.linspace(start_frame, end_frame - 1, clip_frames, dtype=int)
        else:
            # Short tail segment: repeat or linspace
            frame_indices = np.linspace(start_frame, min(total_frames - 1, start_frame + clip_frames), clip_frames, dtype=int)

        frames = []
        prev_gray = None
        motion_diffs = []

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                # If reading fails, provide a blank frame
                frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            else:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = np.mean(np.abs(frame_gray.astype(float) - prev_gray.astype(float))) / 255.0
                    motion_diffs.append(diff)
                prev_gray = frame_gray

                # Resize to standard input shape
                frame = cv2.resize(frame, (frame_size, frame_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap_frames = np.stack(frames, axis=0) # [T, H, W, C]
        # Transpose to [C, T, H, W] for torchvision video models
        clip_tensor = torch.from_numpy(cap_frames).permute(3, 0, 1, 2).float() / 255.0

        avg_motion = float(np.mean(motion_diffs)) if motion_diffs else 0.05

        clips.append({
            "tensor": clip_tensor,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "center_sec": round((start_sec + end_sec) / 2.0, 2),
            "motion_score": avg_motion,
            "clip_idx": len(clips),
        })

        current_time += stride_sec

    cap.release()
    logger.info(f"Extracted {len(clips)} temporal clips ({clip_frames} frames/clip, {stride_sec}s stride).")
    return clips
