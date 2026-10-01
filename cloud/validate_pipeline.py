"""Preflight validation for the standalone AWS bundle.

This performs read-only checks and a single CPU forward pass. It never trains,
downloads, writes dataset arrays, or evaluates the test set.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ctc_tokenizer import CTCTokenizer
from gated_ctc_lipreader import GatedCTCLipReader
from sentence_ctc_dataset import GRIDSentenceCTCDataset, ctc_collate_fn


def count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouth-root", type=Path, default=Path("processed/grid_all/mouth"))
    parser.add_argument("--landmark-root", type=Path, default=Path("processed/grid_all/landmarks"))
    parser.add_argument("--vocab", type=Path, default=Path("processed/grid_all/ctc_vocabulary.json"))
    parser.add_argument("--train-csv", type=Path, default=Path("processed/grid_all/splits/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("processed/grid_all/splits/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("processed/grid_all/splits/test.csv"))
    args = parser.parse_args()

    required = [args.mouth_root, args.landmark_root, args.vocab, args.train_csv, args.val_csv, args.test_csv]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(missing))

    tokenizer = CTCTokenizer(args.vocab)
    print(f"[PASS] Vocabulary size={len(tokenizer)}, blank_id={tokenizer.BLANK_ID}")
    datasets = [GRIDSentenceCTCDataset(path, args.mouth_root, args.landmark_root, args.vocab)
                for path in (args.train_csv, args.val_csv, args.test_csv)]
    print(f"[PASS] Split counts: {[len(dataset) for dataset in datasets]}")
    for name, dataset in zip(("train", "val", "test"), datasets):
        sample = dataset[0]
        assert sample[0].shape[1:] == (1, 96, 96)
        assert sample[1].shape[1:] == (40, 2)
        assert sample[0].shape[0] == sample[1].shape[0]
        print(f"[PASS] {name} first sample: frames={sample[0].shape[0]}, target={sample[4]}, video={sample[6]}")

    loader = DataLoader(datasets[0], batch_size=1, shuffle=False, collate_fn=ctc_collate_fn)
    mouths, landmarks, _, lengths, _, _, _ = next(iter(loader))
    model = GatedCTCLipReader(len(tokenizer), landmark_dropout=0.2).eval()
    with torch.no_grad():
        logits = model(mouths, landmarks, lengths)
    assert logits.shape[0] == 1 and logits.shape[1] == int(lengths[0]) and logits.shape[2] == len(tokenizer)
    print(f"[PASS] Forward pass logits shape={tuple(logits.shape)}")
    print(f"[INFO] CUDA available={torch.cuda.is_available()}")
    print("VALIDATION PASSED: the dataset, tokenizer, collate function, and model forward path are compatible.")


if __name__ == "__main__":
    main()
