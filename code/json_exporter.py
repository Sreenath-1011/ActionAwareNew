"""
json_exporter.py - Formats and exports submission-ready JSON files.
Validates the output against official Track 04 specifications before saving.
"""

import os
from code.utils import setup_logger, save_json
from validate import validate_file

logger = setup_logger("JSONExporter")


def export_prediction_json(
    team_id,
    video_id,
    duration_sec,
    label,
    actions,
    highlights,
    output_path,
    manifest_path=None,
    classes_path=None,
):
    """
    Constructs the exact 6-key JSON payload conforming to the Track 04 schema:
    {
      "team_id": "T07",
      "video_id": "P02_109",
      "duration_sec": 512.3,
      "label": {"name": "cooking", "score": 0.96},
      "actions": [...],
      "highlights": [...]
    }
    """
    # Clean action list (exclude internal debugging fields like avg_motion)
    clean_actions = []
    for act in actions:
        clean_actions.append({
            "segment": [round(float(act["segment"][0]), 2), round(float(act["segment"][1]), 2)],
            "verb": str(act["verb"]),
            "noun": str(act["noun"]),
            "score": round(float(act["score"]), 2),
        })

    # Clean highlights
    clean_highlights = []
    for hl in highlights:
        clean_highlights.append({
            "segment": [round(float(hl["segment"][0]), 2), round(float(hl["segment"][1]), 2)]
        })

    payload = {
        "team_id": str(team_id),
        "video_id": str(video_id),
        "duration_sec": round(float(duration_sec), 2),
        "label": {
            "name": str(label["name"]),
            "score": round(float(label["score"]), 2),
        },
        "actions": clean_actions,
        "highlights": clean_highlights,
    }

    save_json(payload, output_path, indent=2)
    logger.info(f"Saved predictions to '{output_path}'.")

    # In-memory validation pass
    from validate import load_manifest, load_supported_classes
    manifest = load_manifest(manifest_path) if manifest_path else None
    classes = load_supported_classes(classes_path) if classes_path else None

    errors, warnings = validate_file(output_path, manifest=manifest, supported_classes=classes)
    if errors:
        logger.error(f"Validation failed on exported file '{output_path}': {errors}")
        raise ValueError(f"Exported JSON violates schema: {errors}")
    else:
        logger.info(f"Exported JSON verified successfully against official schema.")
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")

    return payload
