import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ctc_tokenizer import CTCTokenizer, is_ctc_valid


class GRIDSentenceCTCDataset(Dataset):
    def __init__(self, csv_path, mouth_root, landmark_root, vocabulary_path):
        self.csv_path = Path(csv_path)
        self.mouth_root = Path(mouth_root)
        self.landmark_root = Path(landmark_root)
        self.tokenizer = CTCTokenizer(vocabulary_path)
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            self.samples = [row for row in csv.DictReader(handle) if row.get("video") and row.get("label")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        video = row["video"].strip()
        text = row["label"].strip()
        mouth = np.load(self.mouth_root / video)
        landmarks = np.load(self.landmark_root / video)
        if mouth.shape[0] != landmarks.shape[0] or mouth.shape[0] == 0:
            raise ValueError(f"Frame mismatch or empty sequence: {video}")
        if not np.isfinite(mouth).all() or not np.isfinite(landmarks).all():
            raise ValueError(f"NaN or Inf in {video}")
        if mouth.min() < 0 or mouth.max() > 1:
            raise ValueError(f"Mouth values outside [0, 1]: {video}")
        if not is_ctc_valid(mouth.shape[0], text):
            raise ValueError(f"CTC-invalid sample: {video}")
        target = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return (
            torch.from_numpy(mouth).float().permute(0, 3, 1, 2),
            torch.from_numpy(landmarks).float(),
            target,
            mouth.shape[0],
            len(target),
            text,
            video,
        )


def ctc_collate_fn(batch: List[Tuple]):
    lengths = [item[3] for item in batch]
    target_lengths = [item[4] for item in batch]
    max_t = max(lengths)
    batch_size = len(batch)
    _, channels, height, width = batch[0][0].shape
    _, points, coords = batch[0][1].shape
    mouths = torch.zeros(batch_size, max_t, channels, height, width)
    landmarks = torch.zeros(batch_size, max_t, points, coords)
    for index, item in enumerate(batch):
        mouths[index, :item[3]] = item[0]
        landmarks[index, :item[3]] = item[1]
    return (
        mouths,
        landmarks,
        torch.cat([item[2] for item in batch]),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
        [item[5] for item in batch],
        [item[6] for item in batch],
    )
