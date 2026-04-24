"""
gen_figures.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Generate publication-ready figures from experiment results for the report.
         Outputs report/figures/ablation.pdf and report/figures/generalization.pdf.
Run from project root: python report/gen_figures.py
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#6B7280"

# videos excluded from summary stats (degenerate label distributions)
DEGENERATE = {"20260108-2", "20260107-2", "20260219"}


def parse_date(video_id: str) -> datetime:
    digits = str(video_id)[:8]
    return datetime.strptime(digits, "%Y%m%d")


# ── Figure 1: Ablation ────────────────────────────────────────────────────────

def fig_ablation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation.csv"))
    lr  = df[df["model"] == "logreg"].set_index("subset")
    svm = df[df["model"] == "svm"].set_index("subset")

    order = lr["f1_mean"].sort_values().index.tolist()
    labels = {
        "eye_only":        "Eye only",
        "motion_only":     "Motion only",
        "brightness_only": "Brightness only",
        "no_mouth":        "No mouth",
        "pose_only":       "Pose only",
        "no_motion":       "No motion",
        "no_pose":         "No pose",
        "all_features":    "All features",
        "mouth_only":      "Mouth only",
    }

    x = np.arange(len(order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - w/2, [lr.loc[s, "f1_mean"]  for s in order], w,
           yerr=[lr.loc[s, "f1_std"]  for s in order],
           label="LogReg", color=BLUE, capsize=3, error_kw={"elinewidth": 0.8})
    ax.bar(x + w/2, [svm.loc[s, "f1_mean"] for s in order], w,
           yerr=[svm.loc[s, "f1_std"] for s in order],
           label="SVM-RBF", color=ORANGE, capsize=3, error_kw={"elinewidth": 0.8})

    ref = lr.loc["all_features", "f1_mean"]
    ax.axhline(ref, color="gray", linestyle="--", linewidth=0.8,
               label=f"All-features baseline ({ref:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(s, s) for s in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean F1 (5-fold CV)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation Study: F1 by Feature Subset", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ablation.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Generalization time series ─────────────────────────────────────

def fig_generalization():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "generalization.csv"))
    df["video_id"] = df["video_id"].astype(str)
    df["date"] = df["video_id"].apply(parse_date)

    # break ties within a day with a small offset so points don't overlap
    day_counts = {}
    offsets = []
    for vid in df["video_id"]:
        day = str(vid)[:8]
        n = day_counts.get(day, 0)
        offsets.append(n)
        day_counts[day] = n + 1
    df["date_offset"] = [d + timedelta(hours=o * 8) for d, o in zip(df["date"], offsets)]

    is_degen = df["video_id"].isin(DEGENERATE)
    valid    = df[~is_degen]
    degen    = df[is_degen]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=False)

    # ── top panel: overall F1 over time ──────────────────────────────────────
    ax1.plot(valid["date_offset"], valid["f1"], color=BLUE, marker="o",
             markersize=5, linewidth=1.2, label="Overall F1")
    ax1.scatter(degen["date_offset"], degen["f1"], color="red", marker="x",
                s=60, zorder=5, label="Excluded (degenerate)")

    mean_f1 = valid["f1"].mean()
    ax1.axhline(mean_f1, color="gray", linestyle="--", linewidth=0.8,
                label=f"Mean F1 = {mean_f1:.3f}")

    ax1.set_ylabel("F1 Score", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Cross-Video Generalization Over Semester (LogReg)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # format x-axis dates
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate(rotation=30, ha="right")

    # ── bottom panel: frontal vs angled F1 over time ──────────────────────────
    ax2.plot(valid["date_offset"], valid["frontal_f1"], color=GREEN, marker="s",
             markersize=4, linewidth=1.0, label="Frontal pose")
    ax2.plot(valid["date_offset"], valid["angled_f1"],  color=ORANGE, marker="^",
             markersize=4, linewidth=1.0, label="Angled pose")

    ax2.axhline(valid["frontal_f1"].mean(), color=GREEN,  linestyle="--",
                linewidth=0.7, alpha=0.6,
                label=f"Frontal mean ({valid['frontal_f1'].mean():.3f})")
    ax2.axhline(valid["angled_f1"].mean(),  color=ORANGE, linestyle="--",
                linewidth=0.7, alpha=0.6,
                label=f"Angled mean ({valid['angled_f1'].mean():.3f})")

    ax2.set_xlabel("Recording Date", fontsize=9)
    ax2.set_ylabel("F1 Score", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Pose-Stratified F1 Over Semester", fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "generalization.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Feature distributions by speaking label ────────────────────────

def fig_feature_dist():
    feat = pd.read_csv(os.path.join("data", "features", "20260107-1.csv"))
    lab  = pd.read_csv(os.path.join("data", "labels",   "20260107-1.csv"))
    df   = feat.merge(lab, on="frame_index", how="inner").dropna()

    features = [
        ("mouth_openness", "Mouth\nOpenness"),
        ("eye_openness",   "Eye\nOpenness"),
        ("yaw",            "Yaw (°)"),
        ("motion_mean",    "Motion\nMean"),
        ("brightness",     "Brightness"),
    ]

    fig, axes = plt.subplots(1, len(features), figsize=(9.0, 3.2))

    for ax, (col, label) in zip(axes, features):
        silent   = df[df["label"] == 0][col].dropna().values
        speaking = df[df["label"] == 1][col].dropna().values

        parts = ax.violinplot([silent, speaking], positions=[0, 1],
                              showmedians=True, widths=0.7)
        parts["bodies"][0].set_facecolor(GRAY)
        parts["bodies"][0].set_alpha(0.65)
        parts["bodies"][1].set_facecolor(BLUE)
        parts["bodies"][1].set_alpha(0.65)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color("black")
            parts[key].set_linewidth(0.9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Silent", "Speaking"], fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Feature Distributions by Speaking Label (Training Video)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "feature_dist.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Class balance per test video ────────────────────────────────────

def fig_class_balance():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "generalization.csv"))
    df["video_id"] = df["video_id"].astype(str)
    df = df.sort_values("video_id").reset_index(drop=True)

    x    = np.arange(len(df))
    degen = df["video_id"].isin(DEGENERATE)

    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    ax.bar(x, df["n_speaking"], label="Speaking", color=BLUE,  alpha=0.85)
    ax.bar(x, df["n_silent"],   label="Silent",   color=GRAY,  alpha=0.85,
           bottom=df["n_speaking"])

    # highlight degenerate videos
    for idx in x[degen]:
        ax.axvspan(idx - 0.45, idx + 0.45, color="red", alpha=0.12, zorder=0)
        ax.get_xticklabels()  # force tick creation

    ax.set_xticks(x)
    ax.set_xticklabels(df["video_id"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Frame Count", fontsize=9)
    ax.set_title("Speaking / Silent Frame Counts per Test Video", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "class_balance.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_ablation()
    fig_generalization()
    fig_feature_dist()
    fig_class_balance()
    print("Done.")
