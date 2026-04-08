#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Preprocessing Pipeline
===================================
Transforms raw Entity_Recording CSVs into training-ready datasets.

Five-stage pipeline:
    1. physio_extraction  : Raw CSV -> ECG/EDA/RESP features (per minute)
    2. physio_cleaning    : Hard threshold -> single-gap interpolation -> quality flags
    3. eye_tracking       : Raw CSV -> eye-tracking features (per minute)
    4. Data fusion        : Merge physio + eye + SSQ + MISC scores
    5. Sequence building  : Flat table -> padded session-level sequences

Outputs:
    - dataset_final.xlsx       : Task 1 minute-level dataset (2304 rows)
    - dataset_merged_full.csv  : Full merged dataset with SSQ scores
    - X_sequences.npy          : Task 2 feature sequences [144, 16, 12]
    - y_labels.npy             : Task 2 SSQ Total Score labels [144]
    - seq_lengths.npy          : Actual sequence lengths [144]
    - session_info.pkl         : Session metadata
    - scaler.pkl               : StandardScaler fitted on valid data

Usage:
    python process_data.py \\
        --raw-dir   /path/to/participant_folders \\
        --physio-ref /path/to/physio.xlsx \\
        --ssq       /path/to/ssq.xlsx \\
        --misc      /path/to/misc.xlsx \\
        --output-dir ./output
"""

import argparse
import glob
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from physio_extraction import load_data, compute_metrics, get_windows, ALL_PARTICIPANTS, SR
from physio_cleaning import clean_physio_full, quality_report
from eye_tracking import EyeTrackingProcessor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

PHYSIO_FEATURE_COLS = [
    "HR", "SDNN", "RMSSD",
    "SCL", "SCR_AMP", "SCR_COUNT",
    "RESP_RATE", "RESP_AMP",
]

EYE_FEATURE_COLS = [
    "Blink_Frequency", "Mean_Blink_Duration",
    "PERCLOS_Proxy", "Tracking_Loss_Ratio",
]

ALL_FEATURE_COLS = PHYSIO_FEATURE_COLS + EYE_FEATURE_COLS
MAX_SEQ_LEN = 16


# ======================================================================
# Stage 1: Physiological feature extraction
# ======================================================================

def extract_physio_features(raw_dir, physio_ref_path):
    """Extract per-minute physio features from all participants.

    Returns a raw (uncleaned) DataFrame with 8 physio features.
    """
    ref = pd.read_excel(physio_ref_path)
    design = (
        ref[["participant", "order", "movement", "odor"]]
        .drop_duplicates()
        .set_index(["participant", "order"])
    )

    all_rows = []
    for participant in ALL_PARTICIPANTS:
        csv_files = sorted(glob.glob(str(raw_dir / participant / "Entity_Recording_*.csv")))
        if not csv_files:
            log.warning("  %s: no CSVs found, skipping", participant)
            continue

        for order_idx, csv_path in enumerate(csv_files[:6], start=1):
            log.info("  %s order=%d  %s", participant, order_idx, Path(csv_path).name)
            try:
                df = load_data(csv_path)
                windows = get_windows(df)

                try:
                    mv = design.loc[(participant, order_idx), "movement"]
                    od = design.loc[(participant, order_idx), "odor"]
                except KeyError:
                    mv, od = np.nan, np.nan

                for t_idx, seg in windows:
                    m = compute_metrics(seg)
                    all_rows.append({
                        "participant": participant,
                        "order": order_idx,
                        "movement": mv,
                        "odor": od,
                        "time": t_idx,
                        **m,
                    })
            except Exception as e:
                log.error("  FAILED %s order=%d: %s", participant, order_idx, e)

    col_order = ["participant", "order", "movement", "odor", "time"] + PHYSIO_FEATURE_COLS
    result = pd.DataFrame(all_rows)
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan
    return result[col_order].sort_values(["participant", "order", "time"]).reset_index(drop=True)


# ======================================================================
# Stage 3: Eye-tracking feature extraction
# ======================================================================

def extract_eye_features(raw_dir):
    """Extract per-minute eye features from all participants.

    Returns a DataFrame keyed by (participant, order, time) with 4 eye features.
    """
    processor = EyeTrackingProcessor(fps=90, invalid_threshold=0.0,
                                      min_blink_ms=50.0, max_blink_ms=1000.0)
    all_rows = []
    for participant in ALL_PARTICIPANTS:
        csv_files = sorted(glob.glob(str(raw_dir / participant / "Entity_Recording_*.csv")))
        if not csv_files:
            continue
        for order_idx, csv_path in enumerate(csv_files[:6], start=1):
            try:
                df = pd.read_csv(csv_path)
                if processor.left_col not in df.columns:
                    continue
                eye_result = processor.process(df)
                for minute_idx, row in eye_result.iterrows():
                    all_rows.append({
                        "participant": participant,
                        "order": order_idx,
                        "time": minute_idx - 1,  # Minute is 1-indexed, time is 0-indexed
                        "Blink_Frequency": row["Blink_Frequency"],
                        "Mean_Blink_Duration": row["Mean_Blink_Duration"],
                        "PERCLOS_Proxy": row["PERCLOS_Proxy"],
                        "Tracking_Loss_Ratio": row["Tracking_Loss_Ratio"],
                    })
            except Exception as e:
                log.error("  Eye extraction failed %s order=%d: %s", participant, order_idx, e)

    return pd.DataFrame(all_rows)


# ======================================================================
# Stage 4: Data fusion
# ======================================================================

def fuse_data(df_physio_clean, df_eye, ssq_path):
    """Merge cleaned physio features with eye features and SSQ scores.

    Also generates row-level eye_valid flags (0 if any eye feature is NaN).
    """
    # Merge eye features
    merge_keys = ["participant", "order", "time"]
    df = df_physio_clean.merge(df_eye, on=merge_keys, how="left")

    # Row-level eye validity: any NaN in eye features -> invalid
    df["eye_valid"] = (~df[EYE_FEATURE_COLS].isna().any(axis=1)).astype(int)

    # Merge SSQ scores
    df_ssq = pd.read_excel(ssq_path)
    ssq_merge_keys = ["participant", "order", "movement", "odor"]
    ssq_cols = ["nausea", "ocular", "disorientation", "tscore"]
    df = df.merge(
        df_ssq[ssq_merge_keys + ssq_cols],
        on=ssq_merge_keys, how="left"
    )

    missing = df["tscore"].isna().sum()
    if missing > 0:
        log.warning("  %d rows have missing SSQ after merge", missing)

    return df


# ======================================================================
# Stage 5: Task 2 sequence building
# ======================================================================

def build_task2_sequences(df, output_dir):
    """Build zero-padded session-level sequences for Task 2.

    Each session (participant x order) yields one sequence of up to 16
    time steps. Features are standardized using only valid (non-padded) data.
    """
    session_keys = ["participant", "order", "movement", "odor"]
    sequences, labels, seq_lengths, session_info = [], [], [], []

    for session_id, group in df.groupby(session_keys):
        X_seq = group[ALL_FEATURE_COLS].values
        actual_len = len(X_seq)

        if actual_len < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - actual_len
            X_seq = np.vstack([X_seq, np.zeros((pad_len, X_seq.shape[1]))])

        y = group["tscore"].iloc[0]

        sequences.append(X_seq)
        labels.append(y)
        seq_lengths.append(actual_len)
        session_info.append({
            "participant": session_id[0],
            "order": session_id[1],
            "movement": session_id[2],
            "odor": session_id[3],
            "seq_length": actual_len,
        })

    # Standardize on valid data only (exclude padding)
    valid_data = []
    for seq, length in zip(sequences, seq_lengths):
        valid_data.append(seq[:length])
    all_valid_data = np.vstack(valid_data)

    scaler = StandardScaler()
    scaler.fit(all_valid_data)

    sequences_scaled = []
    for seq, length in zip(sequences, seq_lengths):
        seq_scaled = np.zeros_like(seq)
        seq_scaled[:length] = scaler.transform(seq[:length])
        sequences_scaled.append(seq_scaled)

    X = np.array(sequences_scaled)
    y = np.array(labels)

    output_dir = Path(output_dir)
    np.save(output_dir / "X_sequences.npy", X)
    np.save(output_dir / "y_labels.npy", y)
    np.save(output_dir / "seq_lengths.npy", np.array(seq_lengths))
    with open(output_dir / "session_info.pkl", "wb") as f:
        pickle.dump(session_info, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    log.info("  Task 2 sequences saved: X=%s  y=%s", X.shape, y.shape)


# ======================================================================
# Main pipeline
# ======================================================================

def run_pipeline(raw_dir, physio_ref_path, ssq_path, output_dir):
    raw_dir    = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract raw physio features
    log.info("[1/5] Extracting physiological features ...")
    df_physio_raw = extract_physio_features(raw_dir, physio_ref_path)
    log.info("  Raw physio: %d rows", len(df_physio_raw))

    # Stage 2: Clean physio data (threshold + interpolation + quality flags)
    log.info("[2/5] Cleaning physiological data ...")
    df_physio_clean = clean_physio_full(df_physio_raw)
    report = quality_report(df_physio_clean)
    n_invalid = (~report["is_valid_session"]).sum()
    log.info("  Cleaned. %d/%d sessions invalid", n_invalid, len(report))
    for vc in ["ecg_valid", "eda_valid", "resp_valid"]:
        rate = df_physio_clean[vc].mean() * 100
        log.info("    %s: %.1f%% valid", vc, rate)

    # Stage 3: Extract eye features
    log.info("[3/5] Extracting eye-tracking features ...")
    df_eye = extract_eye_features(raw_dir)
    log.info("  Eye features: %d rows", len(df_eye))

    # Stage 4: Merge everything
    log.info("[4/5] Fusing data ...")
    df_merged = fuse_data(df_physio_clean, df_eye, ssq_path)
    log.info("  eye_valid: %.1f%% valid", df_merged["eye_valid"].mean() * 100)

    # Save Task 1 dataset
    task1_path = output_dir / "dataset_final.xlsx"
    df_merged.to_excel(task1_path, index=False)
    log.info("  Task 1 dataset: %s (%d rows)", task1_path, len(df_merged))

    merged_csv = output_dir / "dataset_merged_full.csv"
    df_merged.to_csv(merged_csv, index=False)
    log.info("  Merged CSV: %s", merged_csv)

    # Stage 5: Build Task 2 sequences
    log.info("[5/5] Building Task 2 sequences ...")
    build_task2_sequences(df_merged, output_dir)

    log.info("Pipeline complete.")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end preprocessing: raw CSVs -> training-ready data"
    )
    parser.add_argument("--raw-dir",    type=Path, required=True,
                        help="Root directory with per-participant CSV folders")
    parser.add_argument("--physio-ref", type=Path, required=True,
                        help="Reference Excel for condition mapping")
    parser.add_argument("--ssq",        type=Path, required=True,
                        help="SSQ questionnaire scores Excel file")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"),
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    run_pipeline(args.raw_dir, args.physio_ref, args.ssq, args.output_dir)
