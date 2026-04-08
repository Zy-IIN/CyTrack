#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physiological Feature Extraction
=================================
Extract ECG / EDA / RESP features from raw Entity_Recording_*.csv files.
Each session is segmented into 16 one-minute windows, and standard HRV,
electrodermal, and respiration metrics are computed per window.

ECG artifact handling:
    R-peaks producing RR intervals < 300 ms (HR > 200 bpm) are iteratively
    removed before HRV computation.  Gaps > 2000 ms are treated as signal
    interruptions and left intact.

Output: physio_ecgfix.xlsx
"""

import os, sys, glob, warnings, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk

warnings.filterwarnings("ignore")

# ======================================================================
# Path configuration
# ======================================================================
SCRIPT_DIR = Path(__file__).parent
ACM_DIR    = SCRIPT_DIR / "ACM"
PHYSIO_REF = ACM_DIR / "physio.xlsx"       # only for movement/odor mapping
PHYSIO_OUT = ACM_DIR / "physio_ecgfix.xlsx"
LOG_PATH   = SCRIPT_DIR / "physio_ecgfix_progress.log"

ECG_COL  = "PhysioLAB Pro1(00:07:80:8C:AD:AA)|CH1-ECG"
EDA_COL  = "PhysioLAB Pro1(00:07:80:8C:AD:AA)|CH2-EDA"
RESP_COL = "PhysioLAB Pro1(00:07:80:8C:AD:AA)|CH3-RESP"

SR = 1000   # Sampling rate (Hz)

ALL_PARTICIPANTS = [
    "F01","F02","F03","F05","F06","F07","F08","F10","F12","F13","F14","F15",
    "M01","M02","M03","M04","M05","M06","M07","M09","M10","M11","M12","M13",
]

# ======================================================================
# Logging
# ======================================================================
_log_fh = None

def log(msg=""):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()

# ======================================================================
# Data loading
# ======================================================================

def load_data(csv_path):
    """Read raw CSV and compute cumulative experiment time (seconds)."""
    df = pd.read_csv(csv_path)
    df["device_time"] = pd.to_datetime(
        df["StorageTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce"
    )
    df = df.dropna(subset=["device_time"]).copy()

    dt             = df["device_time"].diff().dt.total_seconds()
    df["exp_time"] = dt.fillna(0).clip(lower=0).cumsum()

    for col in (ECG_COL, EDA_COL, RESP_COL):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ======================================================================
# ECG R-peak artifact filtering (manual RR interval constraint)
# ======================================================================

def _filter_ecg_peaks(peaks: np.ndarray, sr: int) -> np.ndarray:
    """
    Iteratively remove spurious R-peaks that cause physiologically
    impossible RR intervals.

    Strategy:
      - RR < 300 ms (HR > 200 bpm): likely artifact, remove the later peak
      - RR > 2000 ms (HR < 30 bpm): signal interruption, keep the gap
      - Iterate until all remaining RR are within [300, 2000] ms
    """
    rr_min = int(0.300 * sr)   # 300 ms
    rr_max = int(2.000 * sr)   # 2000 ms
    peaks  = np.array(peaks, dtype=np.int64)

    for _ in range(len(peaks)):
        if len(peaks) < 2:
            break
        rr   = np.diff(peaks)
        bad  = np.where(rr < rr_min)[0]
        if len(bad) == 0:
            break
        worst = bad[np.argmin(rr[bad])]
        peaks = np.delete(peaks, worst + 1)

    return peaks


# ======================================================================
# Per-window metric computation (ECG with RR filtering)
# ======================================================================

def compute_metrics(data):
    m = {}

    # ECG
    try:
        ecg = data[ECG_COL].dropna()
        signals, info = nk.ecg_process(ecg, sampling_rate=SR)

        # Filter spurious R-peaks: remove those causing RR < 300 ms
        peaks_clean = _filter_ecg_peaks(info["ECG_R_Peaks"], SR)

        hrv = nk.hrv_time(peaks_clean, sampling_rate=SR)

        # Recompute HR from cleaned RR intervals for consistency with SDNN/RMSSD
        if len(peaks_clean) > 1:
            rr_ms = np.diff(peaks_clean.astype(float)) / SR * 1000
            m["HR"] = float(60000.0 / np.mean(rr_ms))
        else:
            m["HR"] = float(signals["ECG_Rate"].mean())

        m["SDNN"]  = float(hrv["HRV_SDNN"].iloc[0])
        m["RMSSD"] = float(hrv["HRV_RMSSD"].iloc[0])
    except Exception:
        m["HR"] = m["SDNN"] = m["RMSSD"] = np.nan

    # EDA
    try:
        eda = data[EDA_COL].dropna()
        signals, _ = nk.eda_process(eda, sampling_rate=SR)
        m["SCL"]     = float(signals["EDA_Tonic"].mean())
        amp          = signals["SCR_Amplitude"]
        m["SCR_AMP"] = float(amp[amp > 0].mean()) if (amp > 0).any() else np.nan
        dur          = data["exp_time"].max() - data["exp_time"].min()
        m["SCR_COUNT"] = float(signals["SCR_Peaks"].sum()) / dur if dur > 0 else np.nan
    except Exception:
        m["SCL"] = m["SCR_AMP"] = m["SCR_COUNT"] = np.nan

    # RESP
    try:
        rsp = data[RESP_COL].dropna()
        signals, _ = nk.rsp_process(rsp, sampling_rate=SR)
        m["RESP_RATE"] = float(signals["RSP_Rate"].mean())
        m["RESP_AMP"]  = float(signals["RSP_Amplitude"].mean())
    except Exception:
        m["RESP_RATE"] = m["RESP_AMP"] = np.nan

    return m

# ======================================================================
# 16 time windows
# ======================================================================

def get_windows(df):
    windows = []
    windows.append((0, df[(df["exp_time"] >= 0)  & (df["exp_time"] < 60)]))
    for i in range(15):
        s = 60 + i * 60
        windows.append((i + 1, df[(df["exp_time"] >= s) & (df["exp_time"] < s + 60)]))
    return windows

# ======================================================================
# Main
# ======================================================================

def get_csv_files(participant):
    return sorted(glob.glob(str(SCRIPT_DIR / participant / "Entity_Recording_*.csv")))


def main():
    global _log_fh
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    start = datetime.now()

    log("=" * 64)
    log("  Physiological Feature Extraction")
    log(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 64)

    # Load experimental design mapping
    log()
    log("[1/4] Loading experimental design mapping ...")
    ref = pd.read_excel(PHYSIO_REF)
    design = (ref[["participant","order","movement","odor"]]
              .drop_duplicates()
              .set_index(["participant","order"]))
    log(f"      {len(design)} mappings loaded")

    # Batch extraction
    total   = len(ALL_PARTICIPANTS) * 6
    done    = 0
    success = 0
    warn_list = []
    all_rows  = []

    log()
    log(f"[2/4] Processing {len(ALL_PARTICIPANTS)} participants x 6 sessions = {total} CSVs ...")
    log("-" * 64)

    for participant in ALL_PARTICIPANTS:
        csv_files = get_csv_files(participant)

        if not csv_files:
            log(f"  {participant}: no CSVs found, skipping")
            done += 6
            continue

        for order_idx, csv_path in enumerate(csv_files[:6], start=1):
            done += 1
            pct = done / total * 100
            log(f"  [{done:3d}/{total}] {pct:5.1f}%  {participant} order={order_idx}"
                f"  <- {Path(csv_path).name}")

            try:
                df      = load_data(csv_path)
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
                        "order":       order_idx,
                        "movement":    mv,
                        "odor":        od,
                        "time":        t_idx,
                        **m,
                    })

                # Brief summary
                hr_vals   = [r["HR"]    for r in all_rows[-16:] if pd.notna(r.get("HR"))]
                sdnn_vals = [r["SDNN"]  for r in all_rows[-16:] if pd.notna(r.get("SDNN"))]
                rsp_vals  = [r["RESP_RATE"] for r in all_rows[-16:] if pd.notna(r.get("RESP_RATE"))]
                log(f"           HR_mean={np.mean(hr_vals):.1f}  "
                    f"SDNN_mean={np.mean(sdnn_vals):.1f}  "
                    f"RSP_mean={np.mean(rsp_vals):.1f}")
                success += 1

            except Exception as e:
                log(f"           ERROR: {e}")
                warn_list.append(f"ERROR {participant} order={order_idx}: {e}")

    log("-" * 64)

    # Assemble DataFrame
    log()
    log("[3/4] Assembling output ...")
    result = pd.DataFrame(all_rows)
    col_order = ["participant","order","movement","odor","time",
                 "HR","SDNN","RMSSD","SCL","SCR_AMP","SCR_COUNT","RESP_RATE","RESP_AMP"]
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan
    result = result[col_order].sort_values(["participant","order","time"]).reset_index(drop=True)

    log(f"  Total rows: {len(result)}")
    log("  NaN counts per feature:")
    for c in col_order[5:]:
        n = result[c].isna().sum()
        log(f"    {c:<12}: {n:4d}  ({n/len(result)*100:.1f}%)")

    # Save
    log()
    log(f"[4/4] Writing {PHYSIO_OUT} ...")
    result.to_excel(PHYSIO_OUT, index=False)
    log(f"  Saved")

    elapsed = (datetime.now() - start).total_seconds()
    log()
    log("=" * 64)
    log(f"  Done.  success={success}  elapsed={elapsed:.0f}s")
    if warn_list:
        for w in warn_list:
            log(f"  {w}")
    log("=" * 64)
    _log_fh.close()


if __name__ == "__main__":
    main()
