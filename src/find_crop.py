"""
find_crop.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Interactively scrub through a video to find the crop endpoint (e.g. when
         screen share starts). Outputs the ffmpeg crop command to run.

Keys:
  d / right  — step forward 1 second
  a / left   — step backward 1 second
  f          — step forward 10 seconds
  r          — step backward 10 seconds
  SPACE      — mark current time as crop end and print ffmpeg command
  q          — quit without marking
"""
import argparse
import sys
import cv2


WINDOW = "find_crop  |  a/d = ±1s   r/f = ±10s   SPACE = mark crop end   q = quit"


def fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True, help="Path to raw mp4")
    parser.add_argument("--output", default=None,
                        help="Where to save crop (default: data/interim/<name>.mp4)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total     = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration  = total / fps

    import os
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_path   = args.output or os.path.join("data", "interim", f"{video_name}.mp4")

    pos_sec = 0.0

    def show_frame(t: float):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        label = f"{fmt(t)}  /  {fmt(duration)}"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(WINDOW, frame)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1000, 600)
    show_frame(pos_sec)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("d"), 83):    # right
            pos_sec = min(pos_sec + 1.0, duration)
        elif key in (ord("a"), 81):  # left
            pos_sec = max(pos_sec - 1.0, 0.0)
        elif key == ord("f"):
            pos_sec = min(pos_sec + 10.0, duration)
        elif key == ord("r"):
            pos_sec = max(pos_sec - 10.0, 0.0)
        elif key == ord(" "):
            cv2.destroyAllWindows()
            cap.release()
            duration_str = fmt(pos_sec)
            print(f"\nCrop endpoint: {duration_str}  ({pos_sec:.1f}s)\n")
            print("Run this command to crop:")
            print(f"  ffmpeg -i {args.video} -ss 00:00:00 -t {duration_str} -c copy {out_path}\n")
            return
        elif key == ord("q"):
            break
        show_frame(pos_sec)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
