#!/usr/bin/env python3
"""
Camera-based crowd monitor. One instance per camera/location node.

Install:  python3 -m pip install opencv-python

Run:
  python3 crowd_monitor.py --node bathroom --camera 0 --capacity 4
  python3 crowd_monitor.py --node room_196 --camera rtsp://... --capacity 20 --show
  python3 crowd_monitor.py --node bathroom --capacity 4 --fake-count 3   # no camera (testing)

It writes/updates crowd.json, which navigator.py reads:
  {"zones": {"bathroom": {"people": 3, "level": 0.75, "updated": 1767000000.0}}}

level = people / capacity, clamped to 0..1 (capacity = number of people before the
space feels crowded; see "capacity" in building.json).

NOTE: OpenCV's HOG people detector is a lightweight prototype detector: it misses
seated or partially hidden people and gives false positives. For a real deployment swap
count_people() for a YOLO/ONNX person detector; nothing else needs to change.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from collections import deque
from pathlib import Path


def update_crowd_file(path: Path, node: str, people: int, capacity: int):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("zones", {})[node] = {
        "people": people,
        "level": round(min(1.0, people / max(1, capacity)), 2),
        "updated": time.time(),
    }
    # Atomic replace so navigator.py never reads a half-written file.
    fd, tmp = tempfile.mkstemp(dir=path.parent or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def make_counter():
    import cv2
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def count_people(frame):
        h, w = frame.shape[:2]
        scale = 640 / w if w > 640 else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        rects, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        kept = [(r, float(wt)) for r, wt in zip(rects, weights) if float(wt) > 0.5]
        if not kept:
            return 0, small
        boxes = [[int(x), int(y), int(bw), int(bh)] for (x, y, bw, bh), _ in kept]
        scores = [wt for _, wt in kept]
        idx = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)  # drop overlapping duplicates
        idx = [int(i) for i in (idx.flatten() if hasattr(idx, "flatten") else idx)]
        for i in idx:
            x, y, bw, bh = boxes[i]
            cv2.rectangle(small, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        return len(idx), small

    return count_people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True, help="Node id from building.json this camera watches")
    ap.add_argument("--camera", default="0", help="Camera index or stream URL")
    ap.add_argument("--capacity", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("crowd.json"))
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between updates")
    ap.add_argument("--show", action="store_true", help="Preview window (press q to quit)")
    ap.add_argument("--fake-count", type=int, help="Skip camera; publish this count")
    args = ap.parse_args()

    if args.fake_count is not None:
        while True:
            update_crowd_file(args.out, args.node, args.fake_count, args.capacity)
            print(f"{args.node}: {args.fake_count} people (fake)")
            time.sleep(args.interval)

    import cv2
    count_people = make_counter()
    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    recent = deque(maxlen=5)  # median filter smooths flicker
    last = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.5)
                continue
            if time.time() - last < args.interval:
                continue
            last = time.time()
            n, annotated = count_people(frame)
            recent.append(n)
            people = int(statistics.median(recent))
            update_crowd_file(args.out, args.node, people, args.capacity)
            print(f"{args.node}: {people} people (raw {n})")
            if args.show:
                cv2.imshow(f"crowd: {args.node}", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
