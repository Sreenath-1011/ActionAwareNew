"""
importance_scoring.py - Multimodal Importance Scoring for Video Summarization.
Combines action confidence, action novelty (distinctiveness), temporal duration,
and motion dynamics while suppressing idle segments.
"""

from collections import defaultdict
from code.utils import setup_logger

logger = setup_logger("ImportanceScoring")


def score_candidate_segments(action_segments, clips, total_duration, cfg):
    """
    Computes an importance score for each action segment and candidate highlight window.
    Weights:
        - Confidence (0.35)
        - Novelty (0.30) - Rewards distinct (verb, noun) pairs, penalizes frequent repeats
        - Motion / Interaction (0.20) - Suppresses idle footage
        - Duration (0.15) - Prefers segments >= 4.0s (official threshold)
    """
    weights = cfg["importance_scoring"]
    w_conf = weights.get("weight_confidence", 0.35)
    w_nov = weights.get("weight_novelty", 0.30)
    w_dur = weights.get("weight_duration", 0.15)
    w_inter = weights.get("weight_interaction", 0.20)

    # Count occurrences of each (verb, noun) pair to compute novelty
    action_counts = defaultdict(int)
    for act in action_segments:
        key = (act["verb"], act["noun"])
        action_counts[key] += 1

    scored_segments = []

    for act in action_segments:
        s, e = act["segment"]
        dur = e - s
        conf = act["score"]

        # 1. Novelty score: inverse frequency
        key = (act["verb"], act["noun"])
        count = action_counts[key]
        novelty_score = 1.0 / count if count > 0 else 0.5

        # 2. Duration score: reward segments >= 4.0s, penalize excessively long (>30s) or short (<4.0s)
        if dur >= 4.0:
            dur_score = min(1.0, 4.0 / max(4.0, dur * 0.5))
        else:
            dur_score = max(0.1, dur / 4.0)

        # 3. Motion score (interaction dynamics)
        motion = act.get("avg_motion", 0.05)
        motion_score = min(1.0, motion * 10.0)

        # Combined weighted score
        importance = (
            w_conf * conf
            + w_nov * novelty_score
            + w_dur * dur_score
            + w_inter * motion_score
        )

        scored_segments.append({
            "segment": [s, e],
            "verb": act["verb"],
            "noun": act["noun"],
            "score": round(conf, 2),
            "importance": round(float(importance), 4),
            "duration": dur,
        })

    # If few or no actions detected (e.g. idle or FPVSum), generate candidate highlight windows from high-motion clips
    if len(scored_segments) < 3 and clips:
        logger.info("Generating candidate highlight segments from dynamic clips.")
        stride = 6.0
        cur = 0.0
        while cur < total_duration:
            s = cur
            e = min(total_duration, s + 6.0)
            if e - s >= 4.0:
                # Find overlapping clips
                matching_clips = [c for c in clips if not (c["end_sec"] <= s or c["start_sec"] >= e)]
                avg_m = sum(c.get("motion_score", 0.05) for c in matching_clips) / max(1, len(matching_clips))
                scored_segments.append({
                    "segment": [round(s, 2), round(e, 2)],
                    "verb": "other",
                    "noun": "other",
                    "score": 0.5,
                    "importance": round(min(1.0, 0.4 + avg_m * 5.0), 4),
                    "duration": round(e - s, 2),
                })
            cur += stride

    scored_segments = sorted(scored_segments, key=lambda x: x["importance"], reverse=True)
    logger.info(f"Scored {len(scored_segments)} candidate highlight segments.")
    return scored_segments
