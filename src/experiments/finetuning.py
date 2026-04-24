"""
finetuning.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Experiment 3 — test whether adding a small labeled sample from a test
         video recovers performance. For each test video, a stratified sample of
         N frames is merged with the full training data (20260107-1) and the
         classifier is re-trained. F1 on the remaining held-out frames is compared
         to the zero-shot baseline from Experiment 2.
         Outputs results/finetuning.csv.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

TRAIN_VIDEO = "20260107-1"

FEATURE_COLS = [
    "mouth_openness", "eye_openness", "yaw", "pitch",
    "motion_mean", "motion_var", "brightness", "landmark_confidence",
]

FINETUNE_SIZES = [25, 50, 100]

FIELDNAMES = [
    "video_id", "model", "n_finetune",
    "baseline_f1", "finetuned_f1", "f1_delta",
    "n_train_total", "n_test",
]


def load_merged(features_path: str, labels_path: str) -> pd.DataFrame:
    feat = pd.read_csv(features_path)
    lab  = pd.read_csv(labels_path)
    df   = feat.merge(lab, on="frame_index", how="inner").dropna(subset=FEATURE_COLS)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sample, remainder) with balanced class sampling."""
    per_class = n // 2
    sampled = []
    for label_val in [0, 1]:
        subset = df[df["label"] == label_val]
        k = min(per_class, len(subset))
        sampled.append(subset.sample(n=k, random_state=seed))
    sample_df    = pd.concat(sampled)
    remainder_df = df.drop(index=sample_df.index)
    return sample_df, remainder_df


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def baseline_f1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    pipe = build_pipeline()
    pipe.fit(train_df[FEATURE_COLS].values.astype(np.float32),
             train_df["label"].values.astype(int))
    pred = pipe.predict(test_df[FEATURE_COLS].values.astype(np.float32))
    return round(f1_score(test_df["label"].values.astype(int), pred, zero_division=0), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True,
                        help="Features CSV for 20260107-1 (training video)")
    parser.add_argument("--train_labels",   required=True,
                        help="Labels CSV for 20260107-1")
    parser.add_argument("--test_dir",       required=True,
                        help="Directory of test features CSVs")
    parser.add_argument("--labels_dir",     required=True,
                        help="Directory of test labels CSVs")
    parser.add_argument("--output",         default="results/finetuning.csv")
    args = parser.parse_args()

    train_df = load_merged(args.train_features, args.train_labels)
    print(f"Train data (20260107-1): {len(train_df)} frames")

    feature_files = sorted(
        f for f in os.listdir(args.test_dir)
        if f.endswith(".csv") and not f.startswith(TRAIN_VIDEO)
    )

    rows = []
    print(f"\nRunning fine-tuning experiment on {len(feature_files)} test videos...")

    for fname in feature_files:
        vid           = os.path.splitext(fname)[0]
        features_path = os.path.join(args.test_dir, fname)
        labels_path   = os.path.join(args.labels_dir, fname)
        if not os.path.exists(labels_path):
            print(f"  {vid}: no labels — skipping")
            continue

        test_df = load_merged(features_path, labels_path)
        print(f"\n  {vid}  ({len(test_df)} labeled frames)")

        for n in FINETUNE_SIZES:
            # need enough frames left over to evaluate meaningfully
            if len(test_df) < n + 20:
                print(f"    n={n:3d}: too few frames ({len(test_df)}) — skipping")
                continue

            sample_df, remainder_df = stratified_sample(test_df, n)

            # baseline: train on 20260107-1 only, test on remainder
            base_f1 = baseline_f1(train_df, remainder_df)

            # fine-tuned: train on 20260107-1 + sample, test on remainder
            combined_df = pd.concat([train_df, sample_df], ignore_index=True)
            pipe = build_pipeline()
            pipe.fit(combined_df[FEATURE_COLS].values.astype(np.float32),
                     combined_df["label"].values.astype(int))
            pred = pipe.predict(remainder_df[FEATURE_COLS].values.astype(np.float32))
            ft_f1 = round(f1_score(remainder_df["label"].values.astype(int),
                                   pred, zero_division=0), 3)

            delta = round(ft_f1 - base_f1, 3)
            print(f"    n={n:3d}: baseline={base_f1:.3f}  finetuned={ft_f1:.3f}  Δ={delta:+.3f}"
                  f"  (train={len(combined_df)}, test={len(remainder_df)})")

            rows.append({
                "video_id":      vid,
                "model":         "logreg",
                "n_finetune":    n,
                "baseline_f1":   base_f1,
                "finetuned_f1":  ft_f1,
                "f1_delta":      delta,
                "n_train_total": len(combined_df),
                "n_test":        len(remainder_df),
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {args.output}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n=== Mean Δ F1 by fine-tune size ===")
        for n, grp in df.groupby("n_finetune"):
            print(f"  n={n:3d}  Δ={grp['f1_delta'].mean():+.3f}  "
                  f"(baseline={grp['baseline_f1'].mean():.3f}  "
                  f"finetuned={grp['finetuned_f1'].mean():.3f})")


if __name__ == "__main__":
    main()
