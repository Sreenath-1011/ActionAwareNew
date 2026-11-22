#!/bin/bash
# ==============================================================================
# run.sh - Official Single-Video Inference Entry Point
# Usage: ./run.sh <video.mp4> <out.json> [out_highlight.mp4]
# ==============================================================================

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <video.mp4> <out.json> [out_highlight.mp4]"
    exit 1
fi

VIDEO_PATH="$1"
OUTPUT_JSON="$2"
HIGHLIGHT_MP4="${3:-}"

echo "[ActionAware] Processing: $VIDEO_PATH"
echo "[ActionAware] Target JSON: $OUTPUT_JSON"

if [ -n "$HIGHLIGHT_MP4" ]; then
    python3 code/main.py --video "$VIDEO_PATH" --output "$OUTPUT_JSON" --highlight-mp4 "$HIGHLIGHT_MP4" --config config/config.yaml
else
    python3 code/main.py --video "$VIDEO_PATH" --output "$OUTPUT_JSON" --config config/config.yaml
fi

echo "[ActionAware] Inference completed."
