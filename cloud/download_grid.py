"""Download and optionally extract the official GRID Corpus files.

The script reads file URLs and checksums from the Zenodo record instead of
hard-coding mirrors. It is intended to be run manually on an AWS instance.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable
from urllib.request import Request, urlopen


RECORD_API = "https://zenodo.org/api/records/3625687"


def get_record() -> Dict:
    request = Request(RECORD_API, headers={"User-Agent": "self-made-grid-downloader/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "self-made-grid-downloader/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    print(f"[DOWNLOAD] {destination.name} (resume={existing > 0})")
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response, partial.open("ab" if existing else "wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
    except Exception:
        print(f"[ERROR] Download failed; partial file preserved: {partial}")
        raise

    actual = md5(partial)
    if actual != expected_md5:
        raise RuntimeError(f"Checksum mismatch for {destination.name}: expected {expected_md5}, got {actual}")
    partial.replace(destination)
    print(f"[OK] {destination} md5={actual}")


def extract(archive: Path, destination: Path) -> None:
    print(f"[EXTRACT] {archive.name} -> {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/grid"))
    parser.add_argument("--speakers", default="all", help="all valid video speakers or comma-separated names")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio_25k.zip for visual-only preparation.")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    record = get_record()
    files = {item["key"]: item for item in record["files"]}
    valid_speakers = sorted(
        key[:-4].lower() for key in files
        if key.lower().startswith("s") and key.lower().endswith(".zip")
        and key[1:-4].isdigit()
    )
    requested_speakers = args.speakers.strip().lower()
    speakers = valid_speakers if requested_speakers == "all" else [
        item.strip().lower() for item in requested_speakers.split(",") if item.strip()
    ]
    invalid = sorted(set(speakers) - set(valid_speakers))
    if invalid:
        raise ValueError(f"Invalid or unavailable video speaker(s): {invalid}; valid={valid_speakers}")
    requested = ["alignments.zip"] + [f"{speaker}.zip" for speaker in speakers]
    if not args.no_audio:
        requested.append("audio_25k.zip")

    archives = []
    for name in requested:
        if name not in files:
            raise FileNotFoundError(f"{name} was not found in Zenodo record {RECORD_API}")
        item = files[name]
        checksum = item["checksum"].split(":", 1)[-1]
        destination = args.output / "archives" / name
        download(item["links"]["self"], destination, checksum)
        archives.append((name, destination))

    if not args.no_extract:
        for name, archive in archives:
            if name == "alignments.zip":
                target = args.output / "alignments"
            elif name == "audio_25k.zip":
                target = args.output / "audio"
            else:
                target = args.output / "videos"
            extract(archive, target)

    print("[DONE] GRID download and extraction completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted; partial downloads can be resumed.")
