# ActionAware – Action-Aware Video Summarization
**Samsung PRISM Hackathon 2026 – Track 04**

ActionAware is an egocentric video perception and summarization system developed for Track 04. It models fine-grained actions and hand–object interactions from first-person video streams, temporally localizes action segments, computes multimodal importance scores, and produces concise highlight summaries ($\le 15\%$ duration budget, each segment $\ge 4.0\text{ s}$) alongside competition-compliant JSON annotations.

---

## 🏗️ Architecture Overview

```text
INPUT VIDEO (.mp4)
      ↓
Video Loading & Metadata Extraction (code/video_loader.py)
      ↓
Uniform Temporal Clip Sampling (code/frame_extractor.py)
      ↓
S3D Video Feature & Action Extraction (code/feature_extractor.py)
      ↓
EPIC-KITCHENS-100 Verb–Noun Vocabulary Mapping (code/action_recognition.py)
      ↓
1D Temporal Action Localization & NMS (code/action_localization.py)
      ↓
Multimodal Importance Scoring (code/importance_scoring.py)
      ↓
Constrained Knapsack Highlight Selection (code/highlight_generator.py)
      ↓
FPVSum Video Category Classification (code/label_classifier.py)
      ↓
Validation & Export (code/json_exporter.py & validate.py)
```

---

## 📊 Optimization for Official Track 04 Metrics

| Metric | Weight | Target | How ActionAware Optimizes It |
|---|---|---|---|
| **Action mAP** | **25%** | EPIC-KITCHENS-100 `actions` | Dense sliding-window inference with S3D (0.39s/clip on CPU) combined with temporal NMS to merge contiguous action intervals. |
| **Action Coverage** | **40%** | EPIC-KITCHENS-100 `highlights` | Knapsack highlight selector that prioritizes **distinct** verb–noun action pairs while penalizing repetitive actions under the $\le 15\%$ duration budget. |
| **Highlight F1** | **25%** | FPVSum `highlights` | Motion dynamism and interaction scoring matching human consensus highlights; guarantees segment lengths $\ge 4.0\text{ s}$. |
| **Label Accuracy** | **10%** | FPVSum `label.name` | Clip-level Kinetics action aggregation corroborated by canonical activity category classification. |

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Single Video Inference (Official Contract)
```bash
# On Linux / macOS / Bash:
./run.sh <path_to_video.mp4> <path_to_output.json> [path_to_highlight.mp4]

# On Windows:
run.bat <path_to_video.mp4> <path_to_output.json> [path_to_highlight.mp4]
```

### 3. Batch Manifest Processing
```bash
python code/main.py --manifest data/manifest.csv --video-dir data/videos --output-dir outputs/
```

### 4. Output Validation
Verify your generated JSON predictions against the official Track 04 rules:
```bash
# Validate a single prediction file
python validate.py --file outputs/P02_109.json -v

# Validate the entire outputs directory
python validate.py --dir outputs/
```

---

## 📁 Repository Structure

```text
ActionAware/
├── run.sh                       # Standard submission entry point
├── run.bat                      # Windows submission entry point
├── requirements.txt             # Python dependencies
├── README.md                    # System documentation
├── validate.py                  # Self-contained official validator
├── config/
│   ├── config.yaml              # Hyperparameters and threshold settings
│   └── kinetics_to_epic.yaml    # Kinetics-400 to EPIC-KITCHENS mapping
├── code/
│   ├── __init__.py
│   ├── main.py                  # Orchestrator & CLI
│   ├── video_loader.py          # Video metadata & manifest verification
│   ├── frame_extractor.py       # Timestamp-aware clip extraction
│   ├── feature_extractor.py     # S3D video backbone
│   ├── action_recognition.py    # Verb–noun recognition & taxonomy mapping
│   ├── action_localization.py   # 1D Temporal NMS & boundary merging
│   ├── importance_scoring.py    # Multimodal importance computation
│   ├── highlight_generator.py   # Highlight selector & MP4 video compiler
│   ├── label_classifier.py      # Video category classifier
│   ├── json_exporter.py         # Formatter & pre-export validator
│   └── utils.py                 # Shared helpers & reproducibility
├── data/
│   ├── manifest.csv             # 187 competition target videos
│   └── supported_classes.json   # 16 labels, 98 verbs, 301 nouns
├── weights/                     # Pre-trained checkpoint storage
├── outputs/                     # Generated <video_id>.json files
└── tests/                       # Unit tests & pipeline verification
```
