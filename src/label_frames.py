"""
label_frames.py
Name: Jaden Zhou
Date: Apr 2026
Purpose: Keyboard-driven tool for manually labeling frames as speaking (1) or
         silent (0). Resumes from last labeled frame if output CSV already exists.

Keys:
  1 — speaking
  0 — silent
  s — skip (no label written)
  b — go back one frame
  q — quit and save
"""
import argparse
import os
import csv
import cv2


WINDOW = "Label Frames — [1] speaking  [0] silent  [s] skip  [b] back  [q] quit"


def load_existing(output_path: str) -> dict:
    labels = {}
    if os.path.exists(output_path):
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                labels[int(row["frame_index"])] = int(row["label"])
    return labels


def save_labels(output_path: str, labels: dict):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "label"])
        for fi in sorted(labels):
            writer.writerow([fi, labels[fi]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Directory of .jpg frames")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    frame_files = sorted(
        f for f in os.listdir(args.frames) if f.endswith(".jpg")
    )
    if not frame_files:
        raise RuntimeError(f"No .jpg files in {args.frames}")

    labels = load_existing(args.output)
    already = len(labels)

    # jump to first unlabeled
    idx = 0
    for i, fname in enumerate(frame_files):
        fi = int(os.path.splitext(fname)[0])
        if fi not in labels:
            idx = i
            break
    else:
        idx = len(frame_files)  # all done

    print(f"Loaded {already} existing labels. Resuming at frame #{idx}/{len(frame_files)}.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 600)

    while 0 <= idx < len(frame_files):
        fname = frame_files[idx]
        frame_index = int(os.path.splitext(fname)[0])
        img = cv2.imread(os.path.join(args.frames, fname))
        if img is None:
            idx += 1
            continue

        existing = labels.get(frame_index, -1)
        status = f"{'SPEAKING' if existing == 1 else 'SILENT' if existing == 0 else 'unlabeled'}"
        info = f"[{idx+1}/{len(frame_files)}]  frame={frame_index}  label={status}"
        cv2.setWindowTitle(WINDOW, info)

        overlay = img.copy()
        color = (0, 200, 0) if existing == 1 else (0, 0, 200) if existing == 0 else (180, 180, 180)
        cv2.putText(overlay, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(overlay, f"frame {frame_index}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow(WINDOW, overlay)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("1"):
            labels[frame_index] = 1
            idx += 1
        elif key == ord("0"):
            labels[frame_index] = 0
            idx += 1
        elif key == ord("s"):
            idx += 1
        elif key == ord("b"):
            idx = max(0, idx - 1)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    save_labels(args.output, labels)
    print(f"Saved {len(labels)} labels → {args.output}")


if __name__ == "__main__":
    main()
