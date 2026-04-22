"""
ablation.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Ablation study — train separate models using feature subsets and compare
         cross-validated F1 to isolate which features carry speaking-detection signal.
         Uses 5-fold CV on 20260107-1 (train split only, never touches test videos).
         Outputs results/ablation.csv.
"""
import argparse
import os
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score

ALL_FEATURES = [
    "mouth_openness", "eye_openness", "yaw", "pitch",
    "motion_mean", "motion_var", "brightness", "landmark_confidence",
]

SUBSETS = {
    "mouth_only":      ["mouth_openness"],
    "motion_only":     ["motion_mean", "motion_var"],
    "pose_only":       ["yaw", "pitch"],
    "brightness_only": ["brightness"],
    "eye_only":        ["eye_openness"],
    "all_features":    ALL_FEATURES,
    "no_mouth":        [f for f in ALL_FEATURES if f not in ("mouth_openness",)],
    "no_motion":       [f for f in ALL_FEATURES if f not in ("motion_mean", "motion_var")],
    "no_pose":         [f for f in ALL_FEATURES if f not in ("yaw", "pitch")],
}


def load_data(features_path: str, labels_path: str):
    feat = pd.read_csv(features_path)
    lab  = pd.read_csv(labels_path)
    df   = feat.merge(lab, on="frame_index", how="inner")
    df   = df.dropna(subset=ALL_FEATURES)
    return df


def build_pipeline(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_ablation(df: pd.DataFrame, n_folds: int = 5):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, zero_division=0)

    classifiers = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm":    SVC(kernel="rbf", class_weight="balanced"),
    }

    rows = []
    for subset_name, cols in SUBSETS.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            print(f"  skipping {subset_name} — no columns available")
            continue
        X = df[available].values.astype(np.float32)
        y = df["label"].values.astype(int)

        for clf_name, clf in classifiers.items():
            pipe = build_pipeline(clf)
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            f1_scores = scores["test_score"]
            row = {
                "subset":      subset_name,
                "features":    "|".join(available),
                "n_features":  len(available),
                "model":       clf_name,
                "f1_mean":     round(f1_scores.mean(), 4),
                "f1_std":      round(f1_scores.std(),  4),
                "f1_min":      round(f1_scores.min(),  4),
                "f1_max":      round(f1_scores.max(),  4),
            }
            rows.append(row)
            print(f"  {subset_name:20s} {clf_name:8s}  F1={row['f1_mean']:.3f} ± {row['f1_std']:.3f}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True,
                        help="Path to features CSV for training video (20260107-1)")
    parser.add_argument("--train_labels",   required=True,
                        help="Path to labels CSV for training video (20260107-1)")
    parser.add_argument("--output",         default="results/ablation.csv")
    parser.add_argument("--folds",          type=int, default=5)
    args = parser.parse_args()

    print(f"Loading data from {args.train_features} + {args.train_labels}")
    df = load_data(args.train_features, args.train_labels)
    print(f"  {len(df)} frames  (speaking={df['label'].sum()}, silent={(df['label']==0).sum()})")

    print(f"\nRunning {args.folds}-fold CV ablation over {len(SUBSETS)} feature subsets × 2 models...")
    rows = run_ablation(df, n_folds=args.folds)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = ["subset", "features", "n_features", "model",
                  "f1_mean", "f1_std", "f1_min", "f1_max"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {args.output}")

    # print ranked summary
    results_df = pd.DataFrame(rows)
    print("\n=== Logreg subsets ranked by F1 ===")
    lr = results_df[results_df["model"] == "logreg"].sort_values("f1_mean", ascending=False)
    print(lr[["subset", "f1_mean", "f1_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
