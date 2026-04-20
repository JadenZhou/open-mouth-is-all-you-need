"""
evaluate.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Evaluate a saved model on a test video's features and labels. Reports
         accuracy, F1, confusion matrix, and per-condition breakdowns (pose,
         lighting, motion).
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

CONDITION_THRESHOLDS = {
    "yaw_angled": 15.0,       # |yaw| >= 15 deg → angled pose
    "brightness_low": 127.0,  # mean brightness <= 127 → low light
    "motion_high": 2.0,       # motion_mean >= threshold → high motion (calibrate later)
}


def load_model(model_dir: str, model_name: str = "logreg"):
    path = os.path.join(model_dir, f"{model_name}.joblib")
    obj = joblib.load(path)
    return obj["pipeline"], obj["feature_cols"]


def assign_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pose_condition"]       = np.where(df["yaw"].abs() >= CONDITION_THRESHOLDS["yaw_angled"],
                                          "angled", "frontal")
    df["lighting_condition"]   = np.where(df["brightness"] <= CONDITION_THRESHOLDS["brightness_low"],
                                          "low", "high")
    df["motion_condition"]     = np.where(df["motion_mean"] >= CONDITION_THRESHOLDS["motion_high"],
                                          "high", "low")
    return df


def evaluate(model_dir, features_path, labels_path, model_name="logreg"):
    pipe, feat_cols = load_model(model_dir, model_name)

    feat = pd.read_csv(features_path)
    lab  = pd.read_csv(labels_path)
    df   = feat.merge(lab, on="frame_index", how="inner").dropna(subset=feat_cols)
    df   = assign_conditions(df)

    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    pred = pipe.predict(X)

    video_id = df["video_id"].iloc[0] if "video_id" in df.columns else os.path.basename(features_path)
    print(f"\n=== {model_name} on {video_id} ===")
    print(classification_report(y, pred, target_names=["silent", "speaking"]))
    print("Confusion matrix:\n", confusion_matrix(y, pred))

    # per-condition breakdown
    for cond_col in ["pose_condition", "lighting_condition", "motion_condition"]:
        print(f"\n--- by {cond_col} ---")
        for val in sorted(df[cond_col].unique()):
            mask = df[cond_col] == val
            if mask.sum() == 0:
                continue
            f1  = f1_score(y[mask], pred[mask], zero_division=0)
            acc = accuracy_score(y[mask], pred[mask])
            print(f"  {val:10s}  n={mask.sum():5d}  F1={f1:.3f}  acc={acc:.3f}")

    return {"video_id": video_id, "f1": f1_score(y, pred, zero_division=0),
            "accuracy": accuracy_score(y, pred)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, help="Directory containing .joblib files")
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels",   required=True)
    parser.add_argument("--model_name", default="logreg", choices=["logreg", "svm"])
    args = parser.parse_args()
    evaluate(args.model, args.features, args.labels, args.model_name)


if __name__ == "__main__":
    main()
