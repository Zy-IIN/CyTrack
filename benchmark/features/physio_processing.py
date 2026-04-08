"""Physiological feature extraction from raw Entity_Recording CSV files.

Extracts ECG, EDA, and RESP features across 16 one-minute windows per session.
ECG R-peak artifacts are filtered using physiologically constrained RR-interval
bounds before HRV computation.
"""

import argparse
import glob
import logging
import warnings
from datetime import datetime
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ECG_COL  = "PhysioLAB Pro1(00:07:80:8C:AD:AA)|CH1-ECG"
EDA_COL  = "PhysioLAB Pro1(00:07:80:8C:AD:AA)|CH2-EDA"
RESP_COL = "PhysioLAB Pro1(00:07:80:8C:AD:AA)|CH3-RESP"
SR = 1000  # Sampling rate in Hz

ALL_PARTICIPANTS = [
    "F01","F02","F03","F05","F06","F07","F08","F10","F12","F13","F14","F15",
    "M01","M02","M03","M04","M05","M06","M07","M09","M10","M11","M12","M13",
]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load a raw recording CSV and compute cumulative experiment time.

    Args:
        csv_path: Path to the Entity_Recording_*.csv file.

    Returns:
        DataFrame with an added 'exp_time' column (seconds from start).
    """
    df = pd.read_csv(csv_path)
    df["device_time"] = pd.to_datetime(
        df["StorageTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce"
    )
    df = df.dropna(subset=["device_time"]).copy()
    dt = df["device_time"].diff().dt.total_seconds()
    df["exp_time"] = dt.fillna(0).clip(lower=0).cumsum()
    for col in (ECG_COL, EDA_COL, RESP_COL):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _filter_ecg_peaks(peaks: np.ndarray, sr: int) -> np.ndarray:
    """Remove spurious R-peaks that produce physiologically impossible RR intervals.

    Iteratively deletes the peak responsible for the shortest RR interval when
    that interval falls below 300 ms (HR > 200 bpm). Gaps above 2000 ms
    (HR < 30 bpm) are treated as signal interruptions and left intact.

    Args:
        peaks: Array of R-peak sample indices.
        sr: Sampling rate in Hz.

    Returns:
        Filtered array of R-peak indices.
    """
    rr_min = int(0.300 * sr)
    peaks  = np.array(peaks, dtype=np.int64)
    for _ in range(len(peaks)):
        if len(peaks) < 2:
            break
        rr  = np.diff(peaks)
        bad = np.where(rr < rr_min)[0]
        if len(bad) == 0:
            break
        worst = bad[np.argmin(rr[bad])]
        peaks = np.delete(peaks, worst + 1)
    return peaks


def compute_metrics(data: pd.DataFrame) -> dict:
    """Compute ECG, EDA, and RESP features for a single time window.

    Args:
        data: DataFrame slice for one time window, containing raw signal columns.

    Returns:
        Dictionary with keys: HR, SDNN, RMSSD, SCL, SCR_AMP, SCR_COUNT,
        RESP_RATE, RESP_AMP. Missing values are NaN.
    """
    m: dict = {}

    try:
        ecg = data[ECG_COL].dropna()
        signals, info = nk.ecg_process(ecg, sampling_rate=SR)
        peaks_clean = _filter_ecg_peaks(info["ECG_R_Peaks"], SR)
        hrv = nk.hrv_time(peaks_clean, sampling_rate=SR)
        if len(peaks_clean) > 1:
            rr_ms = np.diff(peaks_clean.astype(float)) / SR * 1000
            m["HR"] = float(60000.0 / np.mean(rr_ms))
        else:
            m["HR"] = float(signals["ECG_Rate"].mean())
        m["SDNN"]  = float(hrv["HRV_SDNN"].iloc[0])
        m["RMSSD"] = float(hrv["HRV_RMSSD"].iloc[0])
    except Exception:
        m["HR"] = m["SDNN"] = m["RMSSD"] = np.nan

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

    try:
        rsp = data[RESP_COL].dropna()
        signals, _ = nk.rsp_process(rsp, sampling_rate=SR)
        m["RESP_RATE"] = float(signals["RSP_Rate"].mean())
        m["RESP_AMP"]  = float(signals["RSP_Amplitude"].mean())
    except Exception:
        m["RESP_RATE"] = m["RESP_AMP"] = np.nan

    return m


def get_windows(df: pd.DataFrame) -> list:
    """Partition a session recording into 16 one-minute windows.

    Window 0 covers [0, 60) s; windows 1–15 cover [60+i*60, 120+i*60) s.

    Args:
        df: Session DataFrame with 'exp_time' column.

    Returns:
        List of (window_index, window_df) tuples.
    """
    windows = [(0, df[(df["exp_time"] >= 0) & (df["exp_time"] < 60)])]
    for i in range(15):
        s = 60 + i * 60
        windows.append((i + 1, df[(df["exp_time"] >= s) & (df["exp_time"] < s + 60)]))
    return windows


def extract_all(
    data_root: Path,
    physio_ref: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Extract physiological features for all participants and sessions.

    Args:
        data_root: Directory containing per-participant subdirectories with CSVs.
        physio_ref: Reference Excel file providing movement/odor condition mapping.
        output_path: Path to write the output Excel file.

    Returns:
        DataFrame with columns: participant, order, movement, odor, time,
        HR, SDNN, RMSSD, SCL, SCR_AMP, SCR_COUNT, RESP_RATE, RESP_AMP.
    """
    ref    = pd.read_excel(physio_ref)
    design = (ref[["participant", "order", "movement", "odor"]]
              .drop_duplicates()
              .set_index(["participant", "order"]))

    col_order = ["participant", "order", "movement", "odor", "time",
                 "HR", "SDNN", "RMSSD", "SCL", "SCR_AMP", "SCR_COUNT", "RESP_RATE", "RESP_AMP"]
    all_rows: list = []

    for participant in ALL_PARTICIPANTS:
        csv_files = sorted(glob.glob(str(data_root / participant / "Entity_Recording_*.csv")))
        if not csv_files:
            logger.warning("No CSVs found for %s, skipping.", participant)
            continue

        for order_idx, csv_path in enumerate(csv_files[:6], start=1):
            logger.info("Processing %s order=%d", participant, order_idx)
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
                    all_rows.append({"participant": participant, "order": order_idx,
                                     "movement": mv, "odor": od, "time": t_idx, **m})
            except Exception as exc:
                logger.error("Failed %s order=%d: %s", participant, order_idx, exc)

    result = pd.DataFrame(all_rows)
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan
    result = result[col_order].sort_values(["participant", "order", "time"]).reset_index(drop=True)
    result.to_excel(output_path, index=False)
    logger.info("Saved %d rows to %s", len(result), output_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract physiological features from raw CSVs.")
    parser.add_argument("--data-root",   type=Path, required=True, help="Root dir with participant subdirs.")
    parser.add_argument("--physio-ref",  type=Path, required=True, help="Reference Excel for condition mapping.")
    parser.add_argument("--output",      type=Path, default=Path("physio_extracted.xlsx"))
    args = parser.parse_args()
    extract_all(args.data_root, args.physio_ref, args.output)
