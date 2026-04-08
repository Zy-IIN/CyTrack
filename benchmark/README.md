# VR Cybersickness Dual-Task Benchmark

A modular, open-source benchmark for multimodal VR cybersickness prediction using physiological and eye-tracking signals.

## Overview

This repository provides a dual-task prediction framework:

- **Task 1** — Minute-level continuous tracking: predict cybersickness score (0–8) and binary sick/not-sick label per minute.
- **Task 2** — Session-level prediction: predict SSQ Total Score (0–209) from a complete VR session.

The core model is an **SQI-Aware Multimodal Fusion Network** that uses Signal Quality Index (SQI) validity flags as hard-gating masks, suppressing features from modalities with unreliable signal at each time step.

## Repository Structure

```
vr_cybersickness/
├── models/
│   ├── baselines.py          # XGBoost and Linear Regression wrappers
│   ├── sequence_models.py    # GRUPredictor (Task 1), SequenceToOneLSTM (Task 2)
│   └── fusion_network.py     # ModalityGate, MultimodalFusion, SequenceToOnePredictor
├── features/
│   ├── physio_processing.py  # ECG/EDA/RESP feature extraction from raw CSVs
│   └── sequence_builder.py   # Sliding-window builder (Task 1), session sequences (Task 2)
├── src/
│   ├── utils.py              # LOSO split, preprocessing, evaluation, training loop
│   ├── train_task1.py        # Task 1 LOSO-CV entry point
│   └── train_task2.py        # Task 2 LOSO-CV entry point
├── data/
│   └── raw/                  # Sample data (2 participants) for code verification
└── benchmark_dual_task_cleaned.ipynb
```

## Data

> **Privacy Notice**: Due to participant privacy constraints, this repository includes data from **2 participants only** (F01, M01) for the purpose of verifying that the code runs correctly. The complete dataset (24 participants, 144 sessions) will be made available via **Zenodo** upon paper acceptance, or upon reasonable request to the corresponding author.

The sample data covers:
- `data/raw/dataset_sample.xlsx` — Task 1 minute-level features (192 rows)
- `data/raw/X_sequences.npy` — Task 2 session sequences, shape (12, 16, 12)
- `data/raw/y_labels.npy` — SSQ Total Score labels
- `data/raw/seq_lengths.npy` — Actual sequence lengths
- `data/raw/session_info.pkl` — Session metadata

## Installation

```bash
pip install numpy pandas scikit-learn xgboost torch tqdm openpyxl neurokit2
```

## Usage

### Command-line

```bash
# Task 1 — all models
python src/train_task1.py --model all --data data/raw/dataset_sample.xlsx --epochs 50

# Task 1 — single model
python src/train_task1.py --model gru --data data/raw/dataset_sample.xlsx --epochs 50

# Task 2 — all models
python src/train_task2.py --model all --data-dir data/raw/ --epochs 50
```

### Notebook

Open `benchmark_dual_task_cleaned.ipynb` for an end-to-end walkthrough.

### Physiological feature extraction (requires raw CSVs)

```bash
python features/physio_processing.py \
    --data-root /path/to/participant_csvs/ \
    --physio-ref /path/to/physio_reference.xlsx \
    --output physio_extracted.xlsx
```

## Modalities and Features

| Modality | Features |
|---|---|
| ECG | HR, SDNN, RMSSD |
| EDA | SCL, SCR_AMP, SCR_COUNT |
| Respiration | RESP_RATE, RESP_AMP |
| Eye-tracking | Blink_Frequency, Mean_Blink_Duration, PERCLOS_Proxy, Tracking_Loss_Ratio |

## Experimental Design

- 24 participants (13 male, 11 female)
- 6 conditions per participant: 2 locomotion types (Teleportation / Walk-in-Place) × 3 odor conditions (None / Sweet Orange / Motion Sickness Prevention)
- 16 minutes per condition → 144 sessions, 2304 minute-level data points

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{vr_cybersickness_2026,
  title     = {VR Cybersickness Dual-Task Benchmark},
  booktitle = {ACM Multimedia},
  year      = {2026}
}
```
