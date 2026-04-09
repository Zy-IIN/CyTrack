# CyTrack: A Multimodal Dataset for Continuous Cybersickness Tracking in Virtual Reality

This repository serves as the official project page for the CyTrack dataset, including documentation, preprocessing scripts, and benchmark implementations required for reproducibility.

## Overview

CyTrack is a high-fidelity multimodal dataset for cybersickness analysis in virtual reality (VR).
It contains synchronized physiological signals and eye-tracking data collected under controlled VR locomotion scenarios.

- **Participants:** 24
- **Sessions:** 6 per participant
- **Total Duration:** ~36 hours
- **Sampling:** Minute-level annotations
- **Modalities:** Eye-tracking, ECG, EDA, Respiration

The dataset is designed to support:

- Fine-grained cybersickness tracking
- Multimodal physiological modeling
- Longitudinal analysis across repeated exposures

------

## Dataset Structure

Each record corresponds to a **1-minute time window**, aligned with subjective labels.

### Data Fields

| Category               | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| Metadata               | participant ID, session ID, condition, time index            |
| Physiological Features | HR, SDNN, RMSSD, SCL, SCR_AMP, SCR_COUNT                     |
| Respiration            | RESP_RATE, RESP_AMP                                          |
| Eye Features           | Blink frequency, blink duration, PERCLOS proxy, tracking loss |
| Labels                 | MISC (minute-level), SSQ (session-level)                     |
| Validity Flags         | ecg_valid, eda_valid, resp_valid, eye_valid                  |

------

## Modalities

- **ECG** → Heart rate and HRV (SDNN, RMSSD)
- **EDA** → Skin conductance (SCL, SCR features)
- **Respiration** → Breathing rate and amplitude
- **Eye Tracking** → Blink behavior and attention-related metrics

------

## Preprocessing

We provide processed features with the following pipeline:

- **Baseline normalization** (relative to minute 0)
- **Signal quality filtering**
- **Mean imputation (train-set only, LOSO-CV)**
- **Z-score normalization**
- **Sequence padding with masking**

------

## Benchmark Tasks

### Task 1: Minute-level Cybersickness Tracking

- Input: Multimodal time-series
- Output: MISC score (0–8)
- Type: Regression

### Task 2: Session-level SSQ Prediction

- Input: Full session sequence
- Output: SSQ score
- Type: Regression

------

## Baseline Models

- Linear Regression (LR)
- XGBoost
- GRU
- Sequence-to-One LSTM
- SQI-Aware Multimodal Fusion (proposed)

------

## Code

### Preprocessing

```
/preprocessing/
```

### Benchmark Models

```
/benchmark/
```

### Example Usage

```
python train_task1.py
```

------

## Data Access

Due to the large size of the dataset (~25GB), we provide:

- **Sample data** (included in this repository):\sample_data
- **Full dataset** (raw data and preprocessed data, hosted on a secure server): <u>https://gofile.me/6YCsl/TciBR8E8S</u>, entry password upon request.

### Post-Acceptance Access

After publication, the dataset will be publicly available. Detailed access instructions will be updated here.

------

## Citation

If you use this dataset, please cite:

```
@inproceedings{cytrack2026,
  title={CyTrack: A Multimodal Dataset for Continuous Cybersickness Tracking in Virtual Reality},
  author={...},
  booktitle={ACM Multimedia},
  year={2026}
}
```

------

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

------

