"""
train.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Train Logistic Regression and SVM classifiers on extracted facial
         features and ground-truth speaking labels. Saves fitted pipelines to disk.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "mouth_openness", "eye_openness", "yaw", "pitch",
    "motion_mean", "motion_var", "brightness", "landmark_confidence",
]


def load_data(features_path: str, labels_path: str):
    feat = pd.read_csv(features_path)
    lab  = pd.read_csv(labels_path)
    df   = feat.merge(lab, on="frame_index", how="inner")
    df   = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y, df


def build_pipeline(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_and_save(features_path, labels_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X, y, _ = load_data(features_path, labels_path)
    print(f"Training on {len(X)} samples  (speaking={y.sum()}, silent={(y==0).sum()})")

    models = {
        "logreg": build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "svm":    build_pipeline(SVC(kernel="rbf", class_weight="balanced", probability=True)),
    }

    for name, pipe in models.items():
        pipe.fit(X, y)
        pred = pipe.predict(X)
        print(f"\n=== {name} (train) ===")
        print(classification_report(y, pred, target_names=["silent", "speaking"]))
        print("Confusion matrix:\n", confusion_matrix(y, pred))
        path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump({"pipeline": pipe, "feature_cols": FEATURE_COLS}, path)
        print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels",   required=True)
    parser.add_argument("--output",   default="models/")
    args = parser.parse_args()
    train_and_save(args.features, args.labels, args.output)


if __name__ == "__main__":
    main()
