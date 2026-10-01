"""Prepare synchronized mouth and landmark arrays for every valid GRID speaker."""

import argparse
import csv
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


SPEAKER_PATTERN = re.compile(r"^s\d+$", re.IGNORECASE)
MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def read_alignment(path: Path) -> str:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2].lower() not in {"sil", "sp"}:
            words.append(parts[2].lower())
    return " ".join(words)


def detect_points(frame, face_mesh):
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    height, width = frame.shape[:2]
    return np.asarray([[point.x * width, point.y * height]
                       for point in result.multi_face_landmarks[0].landmark], dtype=np.float32)


def normalize_lips(points: np.ndarray) -> np.ndarray:
    lips = points[MOUTH_INDICES]
    center = lips.mean(axis=0)
    width = max(1e-5, float(lips[:, 0].max() - lips[:, 0].min()))
    return ((lips - center) / width).astype(np.float32)


def process_video(video_path: Path, mouth_path: Path, landmark_path: Path) -> bool:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    mouths, landmarks = [], []
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            points = detect_points(frame, face_mesh)
            if points is None:
                continue
            lips = points[MOUTH_INDICES]
            lower = np.maximum(lips.min(axis=0).astype(int) - 20, 0)
            upper = np.minimum(lips.max(axis=0).astype(int) + 20, [frame.shape[1], frame.shape[0]])
            x1, y1 = lower
            x2, y2 = upper
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            mouths.append(crop[..., None])
            landmarks.append(normalize_lips(points))
    finally:
        cap.release()
        face_mesh.close()

    if not mouths:
        return False
    mouth_array = np.asarray(mouths, dtype=np.float32)
    landmark_array = np.asarray(landmarks, dtype=np.float32)
    if not np.isfinite(mouth_array).all() or not np.isfinite(landmark_array).all():
        return False
    mouth_path.parent.mkdir(parents=True, exist_ok=True)
    landmark_path.parent.mkdir(parents=True, exist_ok=True)
    if not mouth_path.exists():
        np.save(mouth_path, mouth_array)
    if not landmark_path.exists():
        np.save(landmark_path, landmark_array)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-videos", type=Path, default=Path("data/raw/grid/videos"))
    parser.add_argument("--raw-alignments", type=Path, default=Path("data/raw/grid/alignments"))
    parser.add_argument("--output", type=Path, default=Path("processed/grid_all"))
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    alignment_map = {path.stem.lower(): path for path in args.raw_alignments.rglob("*.align")}
    videos = sorted(args.raw_videos.rglob("*.mpg"))
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        raise FileNotFoundError(f"No MPG videos below {args.raw_videos.resolve()}")

    mouth_dir = args.output / "mouth"
    landmark_dir = args.output / "landmarks"
    labels_path = args.output / "labels_all.csv"
    records = []
    for video_path in tqdm(videos, desc="Preparing GRID", unit="video"):
        speaker = next((part.lower() for part in video_path.parts if SPEAKER_PATTERN.match(part)), None)
        alignment = alignment_map.get(video_path.stem.lower())
        if speaker is None or alignment is None:
            continue
        output_name = f"{speaker}_{video_path.stem}.npy"
        mouth_path = mouth_dir / output_name
        landmark_path = landmark_dir / output_name
        if not mouth_path.exists() or not landmark_path.exists():
            if not process_video(video_path, mouth_path, landmark_path):
                print(f"[SKIP] Could not process {video_path}")
                continue
        records.append({"video": output_name, "label": read_alignment(alignment),
                        "speaker": speaker, "source_video": video_path.name})

    args.output.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "label", "speaker", "source_video"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[DONE] Prepared {len(records)} samples across {len({r['speaker'] for r in records})} speakers.")


if __name__ == "__main__":
    main()
