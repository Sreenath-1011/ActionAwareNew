"""
action_localization.py - Temporal Action Localization and Merging.
Merges contiguous clips with matching actions and applies 1D temporal Non-Maximum Suppression (NMS).
"""

from code.utils import setup_logger, compute_temporal_iou

logger = setup_logger("ActionLocalization")


def merge_contiguous_actions(clips, max_gap_sec=2.5):
    """
    Groups consecutive clips that share the same (verb, noun) pair into continuous segments.
    """
    if not clips:
        return []

    merged = []
    current_seg = None

    for clip in clips:
        verb = clip["verb"]
        noun = clip["noun"]
        score = clip["action_score"]
        s = clip["start_sec"]
        e = clip["end_sec"]

        if current_seg is None:
            current_seg = {
                "verb": verb,
                "noun": noun,
                "start": s,
                "end": e,
                "scores": [score],
                "motion_scores": [clip.get("motion_score", 0.05)],
            }
        else:
            same_action = (current_seg["verb"] == verb and current_seg["noun"] == noun)
            within_gap = (s - current_seg["end"]) <= max_gap_sec

            if same_action and within_gap:
                current_seg["end"] = e
                current_seg["scores"].append(score)
                current_seg["motion_scores"].append(clip.get("motion_score", 0.05))
            else:
                current_seg["score"] = round(float(sum(current_seg["scores"]) / len(current_seg["scores"])), 2)
                current_seg["avg_motion"] = float(sum(current_seg["motion_scores"]) / len(current_seg["motion_scores"]))
                merged.append(current_seg)
                current_seg = {
                    "verb": verb,
                    "noun": noun,
                    "start": s,
                    "end": e,
                    "scores": [score],
                    "motion_scores": [clip.get("motion_score", 0.05)],
                }

    if current_seg is not None:
        current_seg["score"] = round(float(sum(current_seg["scores"]) / len(current_seg["scores"])), 2)
        current_seg["avg_motion"] = float(sum(current_seg["motion_scores"]) / len(current_seg["motion_scores"]))
        merged.append(current_seg)

    return merged


def temporal_nms(actions, iou_threshold=0.3):
    """
    Applies 1D Temporal Non-Maximum Suppression to remove highly overlapping actions.
    """
    if not actions:
        return []

    # Sort descending by score
    sorted_actions = sorted(actions, key=lambda x: x["score"], reverse=True)
    keep = []

    while sorted_actions:
        best = sorted_actions.pop(0)
        keep.append(best)

        remaining = []
        for cand in sorted_actions:
            # Check overlap if same verb-noun or generally
            iou = compute_temporal_iou([best["start"], best["end"]], [cand["start"], cand["end"]])
            if iou < iou_threshold:
                remaining.append(cand)
            elif best["verb"] != cand["verb"] or best["noun"] != cand["noun"]:
                # If distinct actions overlap partially, keep if IoU is moderate
                if iou < 0.6:
                    remaining.append(cand)

        sorted_actions = remaining

    # Return chronologically sorted
    return sorted(keep, key=lambda x: x["start"])


def localize_actions(clips, cfg, max_duration_sec):
    """
    Full temporal localization pipeline:
    1. Merge contiguous clips
    2. Enforce minimum action duration
    3. Run temporal NMS
    4. Format output into official schema: {"segment": [start, end], "verb": ..., "noun": ..., "score": ...}
    """
    merged = merge_contiguous_actions(clips)
    min_dur = cfg["action_recognition"].get("min_action_duration", 1.0)
    min_conf = cfg["action_recognition"].get("min_confidence", 0.05)
    nms_iou = cfg["action_recognition"].get("nms_iou_threshold", 0.3)
    max_actions = cfg["action_recognition"].get("max_actions_per_video", 2000)

    filtered = []
    for m in merged:
        dur = m["end"] - m["start"]
        if dur >= min_dur and m["score"] >= min_conf:
            m["end"] = min(max_duration_sec, m["end"])
            filtered.append(m)

    nms_kept = temporal_nms(filtered, iou_threshold=nms_iou)

    formatted_actions = []
    for act in nms_kept[:max_actions]:
        formatted_actions.append({
            "segment": [round(act["start"], 2), round(act["end"], 2)],
            "verb": act["verb"],
            "noun": act["noun"],
            "score": round(float(act["score"]), 2),
            "avg_motion": act.get("avg_motion", 0.05),
        })

    logger.info(f"Localized {len(formatted_actions)} clean action segments.")
    return formatted_actions
