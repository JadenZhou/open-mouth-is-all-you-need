"""
extract_features.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Extract facial features (mouth openness, eye openness, head pose, motion,
         brightness) from a directory of frames using MediaPipe, and save to CSV.
"""
import argparse
import os
import re
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

# 3D face model points for PnP head pose
MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0],    # nose tip (1)
    [0.0,   -330.0, -65.0],   # chin (152)
    [-225.0, 170.0, -135.0],  # left eye corner (263)
    [225.0,  170.0, -135.0],  # right eye corner (33)
    [-150.0, -150.0, -125.0], # left mouth corner (287)
    [150.0,  -150.0, -125.0], # right mouth corner (57)
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

MOUTH_TOP, MOUTH_BOTTOM = 13, 14
EYE_TOP_L, EYE_BOT_L   = 386, 374
EYE_TOP_R, EYE_BOT_R   = 159, 145


def ensure_model():
    path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading face landmarker model → {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Download complete.")
    return path


def build_detector(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def get_head_pose(lm, h, w):
    image_points = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in LANDMARK_IDS], dtype=np.float64
    )
    focal = w
    cam  = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan, np.nan

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw   = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])) if sy > 1e-6 else 0.0
    return yaw, pitch


def process_frames(frames_dir: str, video_id: str, detector) -> pd.DataFrame:
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    rows = []
    prev_gray = None

    for fname in frame_files:
        m = re.match(r"(\d+)\.jpg$", fname)
        frame_index = int(m.group(1)) if m else -1

        img_bgr = cv2.imread(os.path.join(frames_dir, fname))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
            motion_mean = float(diff.mean())
            motion_var  = float(diff.var())
        else:
            motion_mean = motion_var = np.nan
        prev_gray = gray

        brightness = float(gray.mean())
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result     = detector.detect(mp_image)

        if not result.face_landmarks:
            rows.append({
                "video_id": video_id, "frame_index": frame_index,
                "mouth_openness": np.nan, "eye_openness": np.nan,
                "yaw": np.nan, "pitch": np.nan,
                "motion_mean": motion_mean, "motion_var": motion_var,
                "brightness": brightness, "landmark_confidence": 0.0,
            })
            continue

        lm = result.face_landmarks[0]

        def ldist(a, b):
            return np.sqrt((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) * h

        mouth_openness = ldist(MOUTH_TOP, MOUTH_BOTTOM)
        eye_openness   = (ldist(EYE_TOP_L, EYE_BOT_L) + ldist(EYE_TOP_R, EYE_BOT_R)) / 2
        yaw, pitch     = get_head_pose(lm, h, w)

        rows.append({
            "video_id": video_id, "frame_index": frame_index,
            "mouth_openness": round(mouth_openness, 4),
            "eye_openness":   round(eye_openness, 4),
            "yaw":   round(yaw,   4) if not np.isnan(yaw)   else np.nan,
            "pitch": round(pitch, 4) if not np.isnan(pitch) else np.nan,
            "motion_mean": round(motion_mean, 4) if not np.isnan(motion_mean) else np.nan,
            "motion_var":  round(motion_var,  4) if not np.isnan(motion_var)  else np.nan,
            "brightness":  round(brightness,  4),
            "landmark_confidence": 1.0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",   required=True, help="Directory of .jpg frames")
    parser.add_argument("--output",   required=True, help="Output CSV path")
    parser.add_argument("--video_id", default=None)
    args = parser.parse_args()

    video_id   = args.video_id or os.path.basename(args.frames.rstrip("/"))
    model_path = ensure_model()
    detector   = build_detector(model_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df = process_frames(args.frames, video_id, detector)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()
