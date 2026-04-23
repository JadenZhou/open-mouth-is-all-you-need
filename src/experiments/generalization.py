"""
generalization.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Evaluate the trained model on each test video and report per-video and
         per-condition (pose, motion) performance. Outputs results/generalization.csv.
         Train/test split is strictly by video — 20260107-1 is training only.
"""
import argparse
import csv
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

TRAIN_VIDEO = "20260107-1"

CONDITION_THRESHOLDS = {
    "yaw_angled":    15.0,
    "brightness_low": 115.0,
    "motion_high":    5.3,
}

FIELDNAMES = [
    "video_id", "model",
    "f1", "accuracy",
    "silent_precision", "silent_recall",
    "speaking_precision", "speaking_recall",
    "n_silent", "n_speaking",
    "angled_f1", "frontal_f1",
    "high_motion_f1", "low_motion_f1",
    "angled_silent_recall", "angled_speaking_recall",
    "frontal_silent_recall", "frontal_speaking_recall",
]


def load_model(model_dir: str, model_name: str):
    path = os.path.join(model_dir, f"{model_name}.joblib")
    obj = joblib.load(path)
    return obj["pipeline"], obj["feature_cols"]


def assign_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pose_cond"]   = np.where(df["yaw"].abs() >= CONDITION_THRESHOLDS["yaw_angled"],
                                 "angled", "frontal")
    df["motion_cond"] = np.where(df["motion_mean"] >= CONDITION_THRESHOLDS["motion_high"],
                                 "high", "low")
    return df


def cond_f1(y, pred, mask):
    if mask.sum() == 0:
        return float("nan")
    return round(f1_score(y[mask], pred[mask], zero_division=0), 3)


def cond_recall(y, pred, mask, true_label):
    sub = mask & (y == true_label)
    if sub.sum() == 0:
        return float("nan")
    return round((pred[sub] == true_label).mean(), 3)


def evaluate_video(pipe, feat_cols, features_path, labels_path, model_name):
    feat = pd.read_csv(features_path)
    lab  = pd.read_csv(labels_path)
    df   = feat.merge(lab, on="frame_index", how="inner").dropna(subset=feat_cols)
    df   = assign_conditions(df)

    X    = df[feat_cols].values.astype(np.float32)
    y    = df["label"].values.astype(int)
    pred = pipe.predict(X)

    video_id = str(df["video_id"].iloc[0]) if "video_id" in df.columns \
               else os.path.splitext(os.path.basename(features_path))[0]

    angled  = df["pose_cond"] == "angled"
    frontal = df["pose_cond"] == "frontal"
    hi_mot  = df["motion_cond"] == "high"
    lo_mot  = df["motion_cond"] == "low"

    row = {
        "video_id":               video_id,
        "model":                  model_name,
        "f1":                     round(f1_score(y, pred, zero_division=0), 3),
        "accuracy":               round(accuracy_score(y, pred), 3),
        "silent_precision":       round(precision_score(y, pred, pos_label=0, zero_division=0), 2),
        "silent_recall":          round(recall_score(y, pred, pos_label=0, zero_division=0), 2),
        "speaking_precision":     round(precision_score(y, pred, pos_label=1, zero_division=0), 2),
        "speaking_recall":        round(recall_score(y, pred, pos_label=1, zero_division=0), 2),
        "n_silent":               int((y == 0).sum()),
        "n_speaking":             int((y == 1).sum()),
        "angled_f1":              cond_f1(y, pred, angled),
        "frontal_f1":             cond_f1(y, pred, frontal),
        "high_motion_f1":         cond_f1(y, pred, hi_mot),
        "low_motion_f1":          cond_f1(y, pred, lo_mot),
        "angled_silent_recall":   cond_recall(y, pred, angled,  0),
        "angled_speaking_recall": cond_recall(y, pred, angled,  1),
        "frontal_silent_recall":  cond_recall(y, pred, frontal, 0),
        "frontal_speaking_recall":cond_recall(y, pred, frontal, 1),
    }

    print(f"  {video_id:15s}  F1={row['f1']:.3f}  acc={row['accuracy']:.3f}"
          f"  angled_F1={row['angled_f1']}  frontal_F1={row['frontal_f1']}"
          f"  hi_mot_F1={row['high_motion_f1']}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, help="Directory with .joblib files")
    parser.add_argument("--test_dir",    required=True, help="Directory of test features CSVs")
    parser.add_argument("--labels_dir",  required=True, help="Directory of labels CSVs")
    parser.add_argument("--model_name",  default="logreg", choices=["logreg", "svm"])
    parser.add_argument("--output",      default="results/generalization.csv")
    args = parser.parse_args()

    pipe, feat_cols = load_model(args.model, args.model_name)
    print(f"Loaded {args.model_name} with features: {feat_cols}")

    feature_files = sorted(
        f for f in os.listdir(args.test_dir)
        if f.endswith(".csv") and not f.startswith(TRAIN_VIDEO)
    )
    if not feature_files:
        print("No test feature files found — exiting.")
        return

    rows = []
    print(f"\nEvaluating {len(feature_files)} test video(s)...")
    for fname in feature_files:
        vid = os.path.splitext(fname)[0]
        features_path = os.path.join(args.test_dir, fname)
        labels_path   = os.path.join(args.labels_dir, fname)
        if not os.path.exists(labels_path):
            print(f"  {vid}: no labels found — skipping")
            continue
        row = evaluate_video(pipe, feat_cols, features_path, labels_path, args.model_name)
        rows.append(row)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {args.output}")

    df = pd.DataFrame(rows)
    print(f"\n=== Summary (mean across {len(df)} test videos) ===")
    for col in ["f1", "angled_f1", "frontal_f1", "high_motion_f1", "low_motion_f1"]:
        print(f"  {col:25s}  {df[col].mean():.3f}")


if __name__ == "__main__":
    main()
