#!/usr/bin/env python3
"""
validate.py - Standard-library validation script for Track 04 JSON output.
Matches the official schema and constraints specified in TEAM_README.md.
"""

import argparse
import glob
import json
import math
import os
import sys


def load_manifest(manifest_path):
    manifest = {}
    if not os.path.isfile(manifest_path):
        return manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return manifest
    header = [c.strip() for c in lines[0].split(",")]
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            vid = parts[0]
            try:
                dur = float(parts[1])
                dataset = parts[2]
                manifest[vid] = {"duration_sec": dur, "dataset": dataset}
            except ValueError:
                pass
    return manifest


def load_supported_classes(classes_path):
    if not os.path.isfile(classes_path):
        return None
    with open(classes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_file(file_path, manifest=None, supported_classes=None, verbose=False):
    errors = []
    warnings = []

    if not os.path.isfile(file_path):
        return [f"File not found: {file_path}"], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [f"JSON parse error: {e}"], []

    if not isinstance(data, dict):
        return ["Root object must be a dictionary"], []

    # 1. Required top-level fields
    req_fields = ["team_id", "video_id", "duration_sec", "label", "actions", "highlights"]
    for field in req_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors, warnings

    # 2. Field types and bounds
    team_id = data.get("team_id")
    if not isinstance(team_id, str) or not team_id.strip():
        errors.append(f"'team_id' must be a non-empty string, got {type(team_id).__name__}")

    video_id = data.get("video_id")
    if not isinstance(video_id, str) or not video_id.strip():
        errors.append(f"'video_id' must be a non-empty string, got {type(video_id).__name__}")
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name != video_id:
            warnings.append(f"Filename '{base_name}.json' does not match video_id '{video_id}'")

    duration_sec = data.get("duration_sec")
    if not isinstance(duration_sec, (int, float)) or math.isnan(duration_sec) or duration_sec <= 0:
        errors.append(f"'duration_sec' must be a positive number, got {duration_sec}")
        duration_sec = 0.0

    if manifest and video_id in manifest:
        expected_dur = manifest[video_id]["duration_sec"]
        if abs(duration_sec - expected_dur) > 0.5:
            errors.append(
                f"'duration_sec' ({duration_sec}) differs by > 0.5s from manifest ({expected_dur})"
            )

    # 3. Validate label
    label = data.get("label")
    if not isinstance(label, dict):
        errors.append(f"'label' must be an object, got {type(label).__name__}")
    else:
        label_name = label.get("name")
        label_score = label.get("score")
        if not isinstance(label_name, str) or not label_name.strip():
            errors.append(f"'label.name' must be a non-empty string")
        elif supported_classes and "labels" in supported_classes:
            valid_labels = set(supported_classes["labels"])
            # Allow common aliases like 'ski' vs 'skiing' as warning or normalized
            if label_name not in valid_labels and label_name != "skiing":
                warnings.append(f"'label.name' '{label_name}' not in supported_classes labels")

        if not isinstance(label_score, (int, float)) or math.isnan(label_score) or not (0.0 <= label_score <= 1.0):
            errors.append(f"'label.score' must be a number in [0, 1], got {label_score}")

    # 4. Validate actions
    actions = data.get("actions")
    if not isinstance(actions, list):
        errors.append(f"'actions' must be an array, got {type(actions).__name__}")
    else:
        if len(actions) > 2000:
            errors.append(f"'actions' contains {len(actions)} items (max 2000 allowed)")

        valid_verbs = set(supported_classes.get("verbs", [])) if supported_classes else None
        valid_nouns = set(supported_classes.get("nouns", [])) if supported_classes else None

        for idx, act in enumerate(actions):
            if not isinstance(act, dict):
                errors.append(f"action[{idx}] must be an object")
                continue
            seg = act.get("segment")
            if not isinstance(seg, list) or len(seg) != 2:
                errors.append(f"action[{idx}].segment must be [start, end]")
            else:
                s, e = seg[0], seg[1]
                if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                    errors.append(f"action[{idx}].segment timestamps must be numeric")
                else:
                    if s < 0 or s >= e or (duration_sec > 0 and e > duration_sec + 0.5):
                        errors.append(f"action[{idx}].segment invalid bounds: [{s}, {e}] for duration {duration_sec}")

            verb = act.get("verb")
            if not isinstance(verb, str) or not verb.strip():
                errors.append(f"action[{idx}].verb must be non-empty string")
            elif valid_verbs and verb not in valid_verbs:
                warnings.append(f"action[{idx}].verb '{verb}' not in supported_classes")

            noun = act.get("noun")
            if not isinstance(noun, str) or not noun.strip():
                errors.append(f"action[{idx}].noun must be non-empty string")
            elif valid_nouns and noun not in valid_nouns:
                warnings.append(f"action[{idx}].noun '{noun}' not in supported_classes")

            score = act.get("score")
            if not isinstance(score, (int, float)) or math.isnan(score) or not (0.0 <= score <= 1.0):
                errors.append(f"action[{idx}].score must be in [0, 1], got {score}")

    # 5. Validate highlights
    highlights = data.get("highlights")
    if not isinstance(highlights, list):
        errors.append(f"'highlights' must be an array, got {type(highlights).__name__}")
    elif len(highlights) == 0:
        errors.append("'highlights' must have at least 1 item")
    else:
        has_min_4s_segment = False
        total_highlight_duration = 0.0
        prev_end = -1.0

        for idx, hl in enumerate(highlights):
            if not isinstance(hl, dict):
                errors.append(f"highlight[{idx}] must be an object")
                continue
            seg = hl.get("segment")
            if not isinstance(seg, list) or len(seg) != 2:
                errors.append(f"highlight[{idx}].segment must be [start, end]")
                continue
            s, e = seg[0], seg[1]
            if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                errors.append(f"highlight[{idx}].segment timestamps must be numeric")
                continue

            if s < 0 or s >= e or (duration_sec > 0 and e > duration_sec + 0.5):
                errors.append(f"highlight[{idx}].segment invalid: [{s}, {e}] for duration {duration_sec}")

            if s < prev_end:
                errors.append(
                    f"highlight[{idx}] starts at {s} before previous segment ended at {prev_end} (overlap or not chronologically sorted)"
                )
            prev_end = e

            seg_dur = e - s
            total_highlight_duration += seg_dur
            if seg_dur >= 4.0:
                has_min_4s_segment = True
            else:
                warnings.append(f"highlight[{idx}] segment duration {seg_dur:.2f}s < 4.0s (will be ignored during scoring)")

        if not has_min_4s_segment:
            errors.append("highlights must contain at least one segment >= 4.0s")

        if duration_sec > 0:
            budget_limit = duration_sec * 0.15 + 0.1
            if total_highlight_duration > budget_limit:
                pct = (total_highlight_duration / duration_sec) * 100
                warnings.append(
                    f"Total highlight duration ({total_highlight_duration:.2f}s, {pct:.1f}%) exceeds 15% budget ({duration_sec * 0.15:.2f}s)"
                )

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Track 04 Output JSON Validator")
    parser.add_argument("--file", type=str, help="Path to single JSON file to validate")
    parser.add_argument("--dir", type=str, help="Directory containing JSON files to validate")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--classes", type=str, default="data/supported_classes.json", help="Path to supported_classes.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print warnings")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.print_help()
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    classes = load_supported_classes(args.classes)

    files_to_check = []
    if args.file:
        files_to_check.append(args.file)
    if args.dir:
        files_to_check.extend(sorted(glob.glob(os.path.join(args.dir, "*.json"))))

    if not files_to_check:
        print(f"No JSON files found to validate.")
        sys.exit(1)

    total_files = len(files_to_check)
    passed_files = 0
    failed_files = 0

    for path in files_to_check:
        errors, warnings = validate_file(path, manifest=manifest, supported_classes=classes, verbose=args.verbose)
        if errors:
            failed_files += 1
            print(f"[FAIL] {path}")
            for err in errors:
                print(f"       ERROR: {err}")
            if args.verbose:
                for w in warnings:
                    print(f"       WARNING: {w}")
        else:
            passed_files += 1
            print(f"[PASS] {path}")
            if args.verbose:
                for w in warnings:
                    print(f"       WARNING: {w}")

    print("\n" + "=" * 50)
    print(f"Validation Summary: {passed_files}/{total_files} passed, {failed_files} failed.")
    print("=" * 50)

    if failed_files > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
