"""Create reproducible speaker-independent GRID train/validation/test splits."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video", "label", "speaker"}
    if not required.issubset(rows[0].keys() if rows else set()):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return rows


def save(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "label"])
        writer.writeheader()
        writer.writerows({"video": row["video"], "label": row["label"]} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("processed/grid_all/labels_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("processed/grid_all/splits"))
    parser.add_argument("--val-speakers", default="", help="optional comma-separated override")
    parser.add_argument("--test-speakers", default="", help="optional comma-separated override")
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=4)
    args = parser.parse_args()

    rows = read_rows(args.labels)
    available = sorted({row["speaker"].lower() for row in rows})
    test_speakers = ({x.strip().lower() for x in args.test_speakers.split(",") if x.strip()}
                     if args.test_speakers else set(available[-args.test_count:]))
    remaining = [speaker for speaker in available if speaker not in test_speakers]
    val_speakers = ({x.strip().lower() for x in args.val_speakers.split(",") if x.strip()}
                    if args.val_speakers else set(remaining[-args.val_count:]))
    if val_speakers & test_speakers:
        raise ValueError("Validation and test speakers overlap.")
    available_set = set(available)
    missing = (val_speakers | test_speakers) - available_set
    if missing:
        raise ValueError(f"Requested split speakers are absent: {sorted(missing)}")

    train = [row for row in rows if row["speaker"].lower() not in val_speakers | test_speakers]
    val = [row for row in rows if row["speaker"].lower() in val_speakers]
    test = [row for row in rows if row["speaker"].lower() in test_speakers]
    save(args.output / "train.csv", train)
    save(args.output / "val.csv", val)
    save(args.output / "test.csv", test)
    manifest = {
        "train_speakers": sorted({r["speaker"] for r in train}),
        "val_speakers": sorted(val_speakers),
        "test_speakers": sorted(test_speakers),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "speaker_counts": {split: dict(Counter(r["speaker"] for r in records))
                            for split, records in (("train", train), ("val", val), ("test", test))},
    }
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
