"""Standalone multi-speaker gated CTC trainer for AWS GPU execution."""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ctc_tokenizer import CTCTokenizer
from gated_ctc_lipreader import GatedCTCLipReader
from sentence_ctc_dataset import GRIDSentenceCTCDataset, ctc_collate_fn
from sequence_metrics import batch_cer_wer, cer, wer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, tokenizer, device, optimizer=None, records=False):
    training = optimizer is not None
    model.train(training)
    space_id = tokenizer.token_to_id[" "]
    loss_sum = samples = pred_chars = target_chars = exact = 0
    references, hypotheses, output_rows = [], [], []
    counts = Counter()
    timesteps = blank_prob = space_prob = 0.0
    gradients = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for mouths, landmarks, targets, lengths, target_lengths, texts, videos in loader:
            mouths, landmarks, targets = mouths.to(device), landmarks.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(mouths, landmarks, lengths)
            log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=-1)
            loss = criterion(log_probs, targets, lengths, target_lengths)
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                gradients.append(float(norm.detach().cpu()))
                optimizer.step()
            probabilities = F.softmax(logits, dim=-1)
            batch_size = mouths.size(0)
            loss_sum += float(loss.item()) * batch_size
            samples += batch_size
            for index in range(batch_size):
                valid = int(lengths[index].item())
                ids = logits[index, :valid].argmax(dim=-1).detach().cpu().tolist()
                prediction = tokenizer.ctc_decode(ids)
                reference = texts[index]
                references.append(reference)
                hypotheses.append(prediction)
                pred_chars += len(prediction)
                target_chars += int(target_lengths[index].item())
                exact += int(reference.strip() == prediction.strip())
                counts.update(ids)
                timesteps += valid
                blank_prob += float(probabilities[index, :valid, 0].sum().item())
                space_prob += float(probabilities[index, :valid, space_id].sum().item())
                if records:
                    output_rows.append({"video": videos[index], "ground_truth": reference,
                                        "prediction": prediction, "cer": cer(reference, prediction),
                                        "wer": wer(reference, prediction),
                                        "exact_match": reference.strip() == prediction.strip()})
    dataset_cer, dataset_wer = batch_cer_wer(references, hypotheses)
    denominator = max(1.0, timesteps)
    blank_count = counts[0]
    space_count = counts[space_id]
    return {
        "loss": loss_sum / max(1, samples), "cer": dataset_cer, "wer": dataset_wer,
        "exact": exact / max(1, samples), "pred_length": pred_chars / max(1, samples),
        "target_length": target_chars / max(1, samples), "blank_ratio": blank_count / denominator,
        "space_ratio": space_count / denominator,
        "other_ratio": max(0, timesteps - blank_count - space_count) / denominator,
        "mean_gradient": float(np.mean(gradients)) if gradients else 0.0,
        "max_gradient": float(np.max(gradients)) if gradients else 0.0,
        "records": output_rows, "top_tokens": counts.most_common(10), "timesteps": timesteps,
    }


def write_history(path, history):
    fields = list(history[0]) if history else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def print_diagnostics(epoch, metrics, tokenizer):
    print(f"Diagnostics epoch {epoch}: top raw tokens")
    for token_id, count in metrics["top_tokens"]:
        token = tokenizer.id_to_token.get(token_id, "<unknown>")
        token = "<blank>" if token_id == 0 else ("space" if token == " " else token)
        print(f"  {token:>8}: {100 * count / max(1, metrics['timesteps']):.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("processed/grid_all"))
    parser.add_argument("--split-root", type=Path, default=Path("processed/grid_all/splits"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-root", type=Path, default=Path("results/multispeaker_gated_ctc"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/multispeaker_gated_ctc"))
    args = parser.parse_args()
    set_seed()

    mouth_root, landmark_root = args.data_root / "mouth", args.data_root / "landmarks"
    vocab_path = args.data_root / "ctc_vocabulary.json"
    train_csv, val_csv, test_csv = [args.split_root / name for name in ("train.csv", "val.csv", "test.csv")]
    required = [mouth_root, landmark_root, vocab_path, train_csv, val_csv, test_csv]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    history_path = args.output_root / "history.csv"
    best_path = args.checkpoint_root / "best.pt"
    tokenizer = CTCTokenizer(vocab_path)
    datasets = [GRIDSentenceCTCDataset(csv_path, mouth_root, landmark_root, vocab_path)
                for csv_path in (train_csv, val_csv, test_csv)]
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=index == 0,
                          num_workers=args.workers, persistent_workers=args.workers > 0,
                          pin_memory=torch.cuda.is_available(), collate_fn=ctc_collate_fn)
               for index, dataset in enumerate(datasets)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCTCLipReader(len(tokenizer), landmark_dropout=0.2).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    history, best_cer, best_loss, no_improvement = [], float("inf"), float("inf"), 0

    print(f"Device: {device} | Train/Val/Test: {[len(dataset) for dataset in datasets]}")
    print(f"Batch size: {args.batch_size} | Workers: {args.workers} | Initial LR: {args.lr}")
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders[0], criterion, tokenizer, device, optimizer)
        val = run_epoch(model, loaders[1], criterion, tokenizer, device)
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": train["loss"], "train_cer": train["cer"],
               "train_wer": train["wer"], "train_exact": train["exact"],
               "val_loss": val["loss"], "val_cer": val["cer"], "val_wer": val["wer"],
               "val_exact": val["exact"], "train_pred_length": train["pred_length"],
               "val_pred_length": val["pred_length"], "target_length": val["target_length"],
               "learning_rate": lr, "mean_gradient_norm": train["mean_gradient"],
               "max_gradient_norm": train["max_gradient"], "train_blank_ratio": train["blank_ratio"],
               "train_space_ratio": train["space_ratio"], "train_other_ratio": train["other_ratio"],
               "val_blank_ratio": val["blank_ratio"], "val_space_ratio": val["space_ratio"],
               "val_other_ratio": val["other_ratio"]}
        history.append(row)
        write_history(history_path, history)
        improved = val["cer"] < best_cer or (val["cer"] == best_cer and val["loss"] < best_loss)
        if improved:
            best_cer, best_loss, no_improvement = val["cer"], val["loss"], 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "train": train, "val": val, "vocabulary": tokenizer.token_to_id}, best_path)
        else:
            no_improvement += 1
        print(f"Epoch {epoch:03d}/{args.epochs} | LR {lr:.6g} | Train CER {train['cer']:.2%} | Val CER {val['cer']:.2%} | Val WER {val['wer']:.2%} | Exact {val['exact']:.2%}")
        if epoch % 5 == 0:
            print_diagnostics(epoch, val, tokenizer)
        if epoch >= 15:
            scheduler.step(val["cer"])
        if no_improvement >= 15 and optimizer.param_groups[0]["lr"] <= 1e-5:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = run_epoch(model, loaders[2], criterion, tokenizer, device, records=True)
    with (args.output_root / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["video", "ground_truth", "prediction", "cer", "wer", "exact_match"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(test["records"])
    print(f"BEST EPOCH: {checkpoint['epoch']}")
    print(f"TEST CER: {test['cer']:.2%} | TEST WER: {test['wer']:.2%} | EXACT: {test['exact']:.2%}")
    print(f"CHECKPOINT: {best_path}")


if __name__ == "__main__":
    main()
