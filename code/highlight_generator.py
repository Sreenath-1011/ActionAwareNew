"""
highlight_generator.py - Selects optimal highlight segments under the 15% duration budget
and compiles highlight MP4 video from original video footage.
"""

import os
import cv2
from code.utils import setup_logger, compute_temporal_iou

logger = setup_logger("HighlightGenerator")


def select_highlights(candidate_segments, total_duration, cfg):
    """
    Selects highlight segments subject to official constraints:
    - Total duration <= 15% of original video duration.
    - At least one segment >= 4.0s (segments < 4.0s are ignored by evaluator).
    - Chronologically sorted.
    - Strictly non-overlapping (no temporal overlap).
    - Maximizes distinct action coverage and consensus importance.
    """
    budget_ratio = cfg["highlight"].get("budget_ratio", 0.15)
    max_budget_sec = total_duration * budget_ratio
    min_segment_sec = cfg["highlight"].get("min_segment_sec", 4.0)

    # Sort candidates by importance descending
    candidates = sorted(candidate_segments, key=lambda x: x["importance"], reverse=True)

    selected = []
    total_sec = 0.0
    covered_actions = set()

    for cand in candidates:
        s, e = cand["segment"]
        dur = e - s

        # Ensure segment satisfies min duration
        if dur < min_segment_sec:
            # Try to expand segment if within video bounds
            expand_needed = min_segment_sec - dur
            new_s = max(0.0, s - expand_needed / 2.0)
            new_e = min(total_duration, e + expand_needed / 2.0)
            if new_e - new_s < min_segment_sec:
                if new_s == 0.0:
                    new_e = min(total_duration, min_segment_sec)
                elif new_e >= total_duration:
                    new_s = max(0.0, total_duration - min_segment_sec)
            s, e = round(new_s, 2), round(new_e, 2)
            dur = e - s

        if dur < min_segment_sec:
            continue

        # Check if adding fits within budget (with tiny margin)
        if total_sec + dur > max_budget_sec + 0.01:
            # If nothing has been selected yet and duration permits, trim to fit budget
            if not selected and max_budget_sec >= min_segment_sec:
                e = round(s + max_budget_sec, 2)
                dur = e - s
            else:
                continue

        # Check for overlap with already selected segments
        has_overlap = False
        for sel in selected:
            sel_s, sel_e = sel["segment"]
            if not (e <= sel_s or s >= sel_e):
                has_overlap = True
                break

        if not has_overlap:
            selected.append({"segment": [round(s, 2), round(e, 2)]})
            total_sec += dur
            action_key = (cand.get("verb"), cand.get("noun"))
            covered_actions.add(action_key)

    # If no candidate satisfied the budget and min_duration, provide a default centered window
    if not selected:
        dur = min(max_budget_sec, min_segment_sec)
        mid = total_duration / 2.0
        s = max(0.0, round(mid - dur / 2.0, 2))
        e = min(total_duration, round(s + dur, 2))
        selected.append({"segment": [s, e]})
        total_sec = e - s

    # Sort strictly chronologically
    selected = sorted(selected, key=lambda x: x["segment"][0])

    # Final guarantee: verify zero overlap and strictly sorted
    clean_highlights = []
    prev_end = 0.0
    for h in selected:
        s, e = h["segment"]
        if s >= prev_end:
            clean_highlights.append({"segment": [s, e]})
            prev_end = e

    pct = (total_sec / total_duration) * 100 if total_duration > 0 else 0
    logger.info(
        f"Selected {len(clean_highlights)} highlight segments ({total_sec:.2f}s total, {pct:.1f}% of {total_duration:.2f}s, budget={max_budget_sec:.2f}s)."
    )

    return clean_highlights


def compile_highlight_video(video_path, highlights, output_mp4_path):
    """
    Extracts and compiles selected highlight segments into an output MP4 video
    using ONLY original footage. Uses OpenCV VideoCapture/VideoWriter.
    """
    if not highlights:
        logger.warning("No highlights to compile.")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video for compilation: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FourCC codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    total_frames_written = 0

    for hl in highlights:
        s, e = hl["segment"]
        start_frame = int(round(s * fps))
        end_frame = int(round(e * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            out.write(frame)
            total_frames_written += 1
            current_frame += 1

    cap.release()
    out.release()

    written_dur = total_frames_written / fps if fps > 0 else 0.0
    logger.info(
        f"Compiled highlight video saved to '{output_mp4_path}' ({written_dur:.2f}s, {total_frames_written} frames)."
    )
    return output_mp4_path
