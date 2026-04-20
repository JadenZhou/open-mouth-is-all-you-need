"""
extract_frames.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Extract frames from a video file at a fixed interval and save as JPEGs.
"""
import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str, interval: int) -> int:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"{frame_idx:07d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {frame_idx} total (interval={interval}) → {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval)


if __name__ == "__main__":
    main()
