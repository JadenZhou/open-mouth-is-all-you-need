# Open Mouth Is All You Need

**[Read the report (PDF)](report/report.pdf)**

**Project description:**
Which hand-crafted facial features remain informative for speaking detection when training and test data come from different recordings? This project trains an interpretable classifier on a single lecture video and evaluates out-of-sample generalization across 20 held-out recordings under varying lighting, head pose, and camera motion. Eight per-frame features are extracted with MediaPipe face landmarking (mouth openness, eye openness, yaw/pitch, motion statistics, brightness, landmark confidence) and used to train logistic regression and SVM classifiers. An ablation shows a single feature — mouth openness — matches the full 8-feature model, and a generalization study quantifies where and how performance degrades.

---

## Overview

A single-subject, video-based study training on one lecture recording (`20260107-1`) and evaluating generalization across 20 test recordings with varying lighting, pose, and motion conditions. The pipeline extracts per-frame facial features using MediaPipe, trains logistic regression and SVM classifiers, and measures F1 degradation across appearance conditions.

**Key findings:**
- Mouth openness alone matches the full 8-feature model (F1 0.883 vs 0.881, 5-fold CV)
- Removing mouth drops F1 by ~18 points (0.881 → 0.705)
- Cross-video F1 range: 0.750–0.950, mean 0.848 across 17 evaluable test videos
- Fine-tuning with 25–100 labeled frames per test video showed no consistent benefit

---

## Setup

**Requirements:** Python 3.x, CPU only (no GPU needed)

```bash
pip install -r requirements.txt
```

You also need the MediaPipe face landmarker model file:

```bash
wget -q -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The pipeline also downloads this file on first run if it's missing.

---

## Data

Raw videos and intermediate artifacts are **not included** in this repository (single-subject lecture recordings, not publicly redistributable). The pipeline below documents how to reproduce on your own footage.

```
data/
├── raw/          # original .mp4 files (gitignored) — never modified
│   ├── 20260107-1.mp4    # training video (45 min lecture)
│   └── ...               # 29 test recordings
├── interim/      # subject-cropped test videos (gitignored)
├── frames/       # extracted frames per video (gitignored)
├── features/     # per-frame feature CSVs (gitignored)
└── labels/       # manually annotated speaking labels (gitignored)
```

**Train/test split is strictly by video.** `20260107-1` is training only; all other videos are test only.

---

## Pipeline

### 1. Crop test videos (find subject on-screen portion)

```bash
# Interactive scrubber to find start/end timestamps
python src/find_crop.py --video data/raw/20260107-2.mp4

# Crop using found timestamps
ffmpeg -i data/raw/20260107-2.mp4 -ss 00:00:00 -t 00:MM:SS -c copy data/interim/20260107-2.mp4
```

### 2. Extract frames

```bash
# Training video (use raw directly — no crop needed)
python src/extract_frames.py --video data/raw/20260107-1.mp4 \
    --output data/frames/20260107-1 --interval 10

# Test video (use cropped interim)
python src/extract_frames.py --video data/interim/20260107-2.mp4 \
    --output data/frames/20260107-2 --interval 10
```

### 3. Extract features

```bash
python src/extract_features.py \
    --frames data/frames/20260107-1 \
    --output data/features/20260107-1.csv
```

Outputs a CSV with columns: `video_id`, `frame_index`, `timestamp`, `mouth_openness`, `eye_openness`, `yaw`, `pitch`, `motion_mean`, `motion_var`, `brightness`, `landmark_confidence`.

### 4. Label frames

```bash
python src/label_frames.py \
    --frames data/frames/20260107-1 \
    --output data/labels/20260107-1.csv
```

Interactive keyboard-driven tool. Labels: `0` = silent, `1` = speaking.

### 5. Train

```bash
python src/train.py \
    --features data/features/20260107-1.csv \
    --labels data/labels/20260107-1.csv \
    --output models/
```

Trains logistic regression and SVM (both with `StandardScaler`). Saves `models/logreg.joblib` and `models/svm.joblib`.

### 6. Evaluate

```bash
python src/evaluate.py \
    --model models/ \
    --features data/features/20260107-2.csv \
    --labels data/labels/20260107-2.csv
```

Reports accuracy, F1, confusion matrix, and per-condition breakdowns (pose, lighting, motion).

---

## Experiments

### Experiment 1 — Ablation (within 20260107-1)

```bash
python src/experiments/ablation.py \
    --train_features data/features/20260107-1.csv \
    --train_labels data/labels/20260107-1.csv
```

5-fold CV comparing feature subsets: mouth only, motion only, pose only, brightness only, eye only, all features, and leave-one-out variants. Results → `results/ablation.csv`.

### Experiment 2 — Generalization under variation

```bash
python src/experiments/generalization.py \
    --model models/ \
    --test_dir data/features/ \
    --labels_dir data/labels/
```

Evaluates trained model on each labeled test video. Groups results by pose (frontal / angled), lighting (high / low), and motion (low / high) conditions. Results → `results/generalization.csv`.

### Experiment 3 — Fine-tuning recovery

```bash
python src/experiments/finetuning.py \
    --train_features data/features/20260107-1.csv \
    --train_labels data/labels/20260107-1.csv \
    --test_dir data/features/ \
    --labels_dir data/labels/
```

Tests whether adding N={25, 50, 100} labeled frames from each test video recovers performance. Results → `results/finetuning.csv`.

---

## Results

Headline numbers:

| Setting | F1 |
|---|---|
| Logistic regression, all 8 features (5-fold CV on train video) | 0.881 |
| Logistic regression, mouth openness only (5-fold CV) | **0.883** |
| Logistic regression, all features minus mouth (5-fold CV) | 0.705 |
| Cross-video evaluation, 17 evaluable test recordings | 0.848 ± 0.05 (range 0.750–0.950) |

Generalization across the 20 test recordings, grouped by appearance condition:

![Generalization across test videos](report/figures/generalization.pdf)

Ablation across feature subsets:

![Ablation across feature subsets](report/figures/ablation.pdf)

Pre-computed result tables are in [`results/`](results/):

| File | Contents |
|---|---|
| [`results/ablation.csv`](results/ablation.csv) | 5-fold CV F1 per feature subset × model |
| [`results/generalization.csv`](results/generalization.csv) | Per-video F1/accuracy + per-condition breakdown |
| [`results/finetuning.csv`](results/finetuning.csv) | Fine-tuning F1 vs. baseline at N=25/50/100 |

Trained classifiers are in [`models/`](models/) (`logreg.joblib`, `svm.joblib`). The compiled report PDF is at [`report/report.pdf`](report/report.pdf).

---

## Condition Thresholds

Calibrated on `20260107-1` distribution (documented in `docs/thresholds.md`):

| Condition | Threshold | Basis |
|---|---|---|
| Angled pose | \|yaw\| ≥ 15° | Standard frontal boundary |
| Low lighting | brightness ≤ 115 | 25th percentile of training distribution |
| High motion | motion_mean ≥ 5.3 | Median of training distribution (~50/50 split) |

---

## Repository Structure

```
open-mouth-is-all-you-need/
├── src/
│   ├── find_crop.py          # interactive crop endpoint finder
│   ├── extract_frames.py     # frame extraction from video
│   ├── extract_features.py   # MediaPipe feature extraction
│   ├── label_frames.py       # manual labeling tool
│   ├── train.py              # train logreg + SVM
│   ├── evaluate.py           # evaluate with condition breakdowns
│   └── experiments/
│       ├── ablation.py       # Experiment 1
│       ├── generalization.py # Experiment 2
│       └── finetuning.py     # Experiment 3
├── models/
│   ├── logreg.joblib         # trained logistic regression pipeline
│   ├── svm.joblib            # trained SVM pipeline
│   └── face_landmarker.task  # MediaPipe model file (fetched on first run)
├── results/
│   ├── ablation.csv
│   ├── generalization.csv
│   └── finetuning.csv
├── report/
│   ├── report.tex
│   ├── report.pdf
│   ├── gen_figures.py        # regenerates report figures from results/
│   └── figures/
├── docs/
│   └── thresholds.md
├── LICENSE
├── requirements.txt
└── README.md
```

---

Northeastern University, Spring 2026 coursework (CS5330).
