#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physiological Data Cleaning Module
====================================
Three-step strategy: hard thresholding -> single-gap interpolation ->
modality quality flagging.

Usage:
    from physio_cleaning import clean_physio_full
    df_clean = clean_physio_full(df_raw)
"""

import numpy as np
import pandas as pd
from typing import Optional

# ======================================================================
# Physiologically plausible bounds (hard thresholding)
# ======================================================================
BOUNDS: dict[str, tuple[float, float]] = {
    # ECG
    "HR"        : (40,  180),   # bpm
    "SDNN"      : (5,   200),   # ms, lower bound excludes near-zero artifacts
    "RMSSD"     : (5,   200),   # ms
    # EDA
    "SCL"       : (0.1, 24.5),  # uS, 0.1 excludes electrode detachment; 24.5 excludes saturation
    "SCR_AMP"   : (0,   10),    # uS
    "SCR_COUNT" : (0,   1.0),   # peaks/second
    # RESP
    "RESP_RATE" : (4,   60),    # breaths/min, normal 12-20 with margin
    "RESP_AMP"  : (0,   100),   # arbitrary units
}

# Feature groups for quality flagging
ECG_COLS  = ["HR", "SDNN", "RMSSD"]
EDA_COLS  = ["SCL", "SCR_AMP", "SCR_COUNT"]
RESP_COLS = ["RESP_RATE", "RESP_AMP"]

# Per-modality quality threshold: remaining NaN count > this -> modality invalid
# Based on 16-minute sessions, ~20% tolerance
NAN_THRESHOLD = 3


# ======================================================================
# Step 1: Hard thresholding
# ======================================================================

def _hard_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Replace values outside physiologically plausible bounds with NaN."""
    df = df.copy()
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        mask_bad = (df[col] < lo) | (df[col] > hi)
        n_bad = mask_bad.sum()
        if n_bad > 0:
            df.loc[mask_bad, col] = np.nan
    return df


# ======================================================================
# Step 2: Single-gap linear interpolation (per session)
# ======================================================================

def _single_gap_interpolate(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate within a single session (15-16 rows):
      - Isolated NaN (gap length == 1): linearly interpolated
      - Consecutive NaN (gap length >= 2): left as NaN

    Uses pandas interpolate(method='linear', limit=1) which fills at most
    one consecutive NaN.  No extrapolation beyond the first/last valid value.
    """
    df = session_df.copy()
    feat_cols = [c for c in BOUNDS.keys() if c in df.columns]

    for col in feat_cols:
        df[col] = df[col].interpolate(
            method="linear",
            limit=1,
            limit_direction="both",
            limit_area="inside",
        )
    return df


# ======================================================================
# Step 3: Per-modality quality flagging (per session)
# ======================================================================

def _quality_flag(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count remaining NaN per modality and generate validity flags:

      ecg_valid  : 1 if SDNN NaN count <= NAN_THRESHOLD, else 0
      eda_valid  : 1 if SCL NaN count <= NAN_THRESHOLD, else 0
      resp_valid : 1 if RESP_RATE NaN count <= NAN_THRESHOLD, else 0

    Uses the representative feature of each modality (SDNN, SCL, RESP_RATE)
    as the quality indicator, since these are the primary metrics most
    sensitive to signal quality.
    """
    df = session_df.copy()

    def _nan_n(col):
        return int(df[col].isna().sum()) if col in df.columns else np.nan

    nan_sdnn      = _nan_n("SDNN")
    nan_scl       = _nan_n("SCL")
    nan_resp_rate = _nan_n("RESP_RATE")

    def _avg_nan(cols):
        vals = [df[c].isna().sum() for c in cols if c in df.columns]
        return round(np.mean(vals), 2) if vals else np.nan

    ecg_valid  = bool(nan_sdnn      <= NAN_THRESHOLD)
    eda_valid  = bool(nan_scl       <= NAN_THRESHOLD)
    resp_valid = bool(nan_resp_rate <= NAN_THRESHOLD)

    df["nan_n_ecg"]        = _avg_nan(ECG_COLS)
    df["nan_n_eda"]        = _avg_nan(EDA_COLS)
    df["nan_n_resp"]       = _avg_nan(RESP_COLS)
    df["ecg_valid"]        = ecg_valid
    df["eda_valid"]        = eda_valid
    df["resp_valid"]       = resp_valid
    df["is_valid_session"] = ecg_valid and eda_valid

    return df


# ======================================================================
# Single-session cleaning entry point
# ======================================================================

def clean_session(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full three-step cleaning pipeline on one session
    (~16 rows for one participant x one condition).
    """
    df = _hard_threshold(session_df)       # Step 1
    df = _single_gap_interpolate(df)       # Step 2
    df = _quality_flag(df)                 # Step 3
    return df


# ======================================================================
# Full dataset cleaning (grouped by participant x order)
# ======================================================================

def clean_physio_full(
    df: pd.DataFrame,
    participant_col: str = "participant",
    order_col: str = "order",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Clean the entire physio DataFrame by applying three-step cleaning
    independently to each session (participant x order).

    Returns the cleaned DataFrame with quality flag columns appended.
    """
    df = df.sort_values([participant_col, order_col, time_col]).copy()

    result = (
        df.groupby([participant_col, order_col], sort=False, group_keys=False)
        .apply(clean_session)
    )
    return result.reset_index(drop=True)


# ======================================================================
# Quality summary report
# ======================================================================

def quality_report(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Generate a per-session quality summary table."""
    cols = ["nan_n_ecg", "nan_n_eda", "nan_n_resp",
            "ecg_valid", "eda_valid", "resp_valid", "is_valid_session"]
    report = (
        df_clean.groupby(["participant", "order"])[cols]
        .first()
        .reset_index()
    )
    return report


# ======================================================================
# Standalone usage
# ======================================================================

if __name__ == "__main__":
    from pathlib import Path

    physio_path = Path(__file__).parent / "ACM" / "physio_ecgfix.xlsx"
    out_path    = Path(__file__).parent / "ACM" / "physio_final.xlsx"

    print("Reading physio data ...")
    df_raw = pd.read_excel(physio_path)

    print(f"Raw data: {df_raw.shape}")
    print("Running three-step cleaning ...")
    df_clean = clean_physio_full(df_raw)
    print(f"Cleaned data: {df_clean.shape}")

    # Quality report
    report = quality_report(df_clean)
    total_sessions = len(report)
    n_valid   = report["is_valid_session"].sum()
    n_invalid = total_sessions - n_valid

    print()
    print("=" * 56)
    print("  Cleaning Quality Report")
    print("=" * 56)
    print(f"  Total sessions:    {total_sessions}")
    print(f"  is_valid_session:  True={n_valid}  False={n_invalid}  "
          f"({n_invalid/total_sessions*100:.1f}% invalid)")
    print(f"  ecg_valid=False:   {(~report['ecg_valid']).sum()} sessions")
    print(f"  eda_valid=False:   {(~report['eda_valid']).sum()} sessions")
    print(f"  resp_valid=False:  {(~report['resp_valid']).sum()} sessions")
    print()
    print("  Invalid sessions (is_valid_session=False):")
    invalid = report[~report["is_valid_session"]][
        ["participant","order","nan_n_ecg","nan_n_eda","ecg_valid","eda_valid"]
    ]
    print(invalid.to_string(index=False))

    # NaN change statistics
    feat_cols = list(BOUNDS.keys())
    print()
    print("  NaN changes per feature (raw -> thresholded -> interpolated):")
    df_step1 = df_raw.copy()
    for col, (lo, hi) in BOUNDS.items():
        if col in df_step1.columns:
            df_step1.loc[(df_step1[col] < lo) | (df_step1[col] > hi), col] = np.nan

    print(f"  {'Feature':<12} {'Raw NaN':>8} {'Thresh':>8} {'Interp':>8} {'Recovered':>10}")
    for col in feat_cols:
        if col not in df_raw.columns:
            continue
        n0 = df_raw[col].isna().sum()
        n1 = df_step1[col].isna().sum()
        n2 = df_clean[col].isna().sum()
        print(f"  {col:<12} {n0:>8} {n1:>8} {n2:>8} {n1-n2:>+10}")

    # Save
    df_clean.to_excel(out_path, index=False)
    print()
    print(f"  Saved: {out_path}")
    print("=" * 56)
