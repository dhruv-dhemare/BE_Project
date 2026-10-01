# AWS Multi-Speaker GRID Training Guide

This guide prepares the scalable visual-speech experiment. It does not contain
automatic AWS provisioning and it does not require putting AWS credentials in
the repository. Run each command manually on an AWS GPU instance.

If you push the contents of `cloud/` as the repository root, run commands as
`python download_grid.py`, `python validate_pipeline.py`, and
`python train_gated_ctc_multispeaker.py`. If you keep the folder nested inside
the original project, prefix those commands with `cloud/` as shown below.

## Dataset source

The official GRID Corpus record is:

`https://zenodo.org/records/3625687`

It contains speaker video archives, audio, and word alignments. The official
record describes 34 talkers and notes that T21 has no video archive. The
download script therefore targets all available video speakers by default and
verifies every archive against the checksum published by Zenodo.

## Expected storage

Use a sufficiently large EBS volume. The raw archives, extracted videos,
processed mouth arrays, landmarks, checkpoints, and logs require substantially
more space than the compressed download alone. Keep raw and processed data on a
persistent volume so an instance can be stopped and resumed.

## Copy the project

Copy this project to the instance using your preferred private method. Do not
put AWS access keys, private dataset credentials, or personal tokens in source
files.

## Install dependencies

Create a virtual environment and install the project requirements. Install the
CUDA-enabled PyTorch build recommended for the selected AWS image using the
current official PyTorch installation command for that image, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r cloud/requirements-aws.txt
```

Verify the environment manually before starting a long run:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Download and extract GRID

The default discovers and downloads every valid video speaker listed by the
official Zenodo record, plus alignments and audio:

```bash
python cloud/download_grid.py --output data/raw/grid
```

For a visual-only preparation smoke test, download one speaker first:

```bash
python cloud/download_grid.py --speakers s1 --no-audio --output data/raw/grid
python cloud/prepare_grid_multispeaker.py --max-videos 20
```

After the smoke test succeeds, remove the smoke-test output or use a new
output directory before preparing the complete dataset.

## Prepare mouth sequences and landmarks

```bash
python cloud/prepare_grid_multispeaker.py \
  --raw-videos data/raw/grid/videos \
  --raw-alignments data/raw/grid/alignments \
  --output processed/grid_all
```

This creates:

- `processed/grid_all/mouth/`
- `processed/grid_all/landmarks/`
- `processed/grid_all/labels_all.csv`

The preparation step is CPU-heavy and can take a long time. It is resumable at
the per-video level because existing `.npy` files are preserved.

## Create speaker-independent splits

The default keeps four speakers for validation and four different speakers for
final testing:

```bash
python cloud/create_speaker_splits.py \
  --labels processed/grid_all/labels_all.csv \
  --output processed/grid_all/splits
```

By default, the last four valid speakers are test speakers and the preceding
four valid speakers are validation speakers. Every other valid speaker is used
for training. Inspect `processed/grid_all/splits/split_manifest.json` before
training; it is the authoritative record of which speakers belong to each split.

## Build the fixed vocabulary

The vocabulary must be constructed from training labels only:

```bash
python -c "import sys; sys.path.insert(0, 'cloud'); from ctc_tokenizer import CTCTokenizer; CTCTokenizer.build_vocab_from_csv('processed/grid_all/splits/train.csv', 'processed/grid_all/ctc_vocabulary.json')"
```

## Run the preflight validation

Run this before starting the long GPU job:

```bash
python cloud/validate_pipeline.py
```

It must print `VALIDATION PASSED`. It checks the split files, vocabulary,
mouth/landmark shapes, CTC compatibility, collate function, and one model
forward pass. It does not train or download anything.

Do not rebuild this vocabulary after training begins.

## Start multi-speaker training

```bash
python cloud/train_gated_ctc_multispeaker.py \
  --data-root processed/grid_all \
  --split-root processed/grid_all/splits \
  --batch-size 8 \
  --workers 4 \
  --epochs 100
```

The first run should use batch size 8 for comparability with the S1 experiment.
If GPU memory allows, a later controlled run may compare a larger batch size.
Do not change the architecture and batch size in the same experiment without
recording the change.

## Outputs

The multi-speaker run writes:

- `checkpoints/multispeaker_gated_ctc/best.pt`
- `results/multispeaker_gated_ctc/history.csv`
- `results/multispeaker_gated_ctc/test_predictions.csv`
- `results/multispeaker_gated_ctc/cer_curve.png`
- `results/multispeaker_gated_ctc/loss_curve.png`

## Interpretation rule

Do not claim speaker-independent generalization from training/test samples that
share speakers. The split manifest and unseen-speaker test results must be
reported with the final metrics.

This visual model is still one component of the later audio-visual authenticity
detector. After multi-speaker visual evaluation, the next separate stage is the
audio ASR branch and timed audio/visual alignment.
