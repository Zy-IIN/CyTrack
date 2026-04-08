#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR Eye-Tracking Feature Extraction
====================================
Aggregate high-frequency (90/120 Hz) raw eye-tracking data into per-minute
standardized features, temporally aligned with ECG / EDA / RESP signals.

Output features (per minute):
    - Blink_Frequency     : valid blink count (blinks/min)
    - Mean_Blink_Duration : mean blink duration (ms), NaN if no blinks
    - PERCLOS_Proxy       : total blink duration / 60000 ms (percentage)
    - Tracking_Loss_Ratio : tracking loss duration / 60000 ms (percentage)
    - Data_Coverage       : actual data span / 60000 ms (percentage)
"""

import numpy as np
import pandas as pd
from typing import Optional


class EyeTrackingProcessor:
    """
    High-frequency VR eye-tracking data processor.

    Processing pipeline:
        1. Binocular fusion  -> per-frame "both_closed" boolean mask
        2. Run-length encoding (RLE) -> O(N) detection of all closure segments
        3. Duration classification -> valid blink / tracking loss / artifact
        4. Per-minute aggregation -> output feature DataFrame

    Parameters
    ----------
    fps : float
        Sampling rate (frames/sec). Used to generate time axis when no
        Timestamp column is available. Default 90 Hz.
    left_col : str
        Left-eye pupil diameter column name.
    right_col : str
        Right-eye pupil diameter column name.
    timestamp_col : str, optional
        Timestamp column name.  If None, equidistant timestamps are generated
        from fps.  Unit auto-detected: span < 1e6 treated as seconds.
    invalid_threshold : float
        Pupil diameter <= this value is treated as "invalid/closed". Default 0.0.
    min_blink_ms : float
        Minimum valid blink duration (ms). Below this -> artifact. Default 50 ms.
    max_blink_ms : float
        Maximum valid blink duration (ms). Above this -> tracking loss. Default 1000 ms.
    window_size_s : float
        Aggregation window size (seconds). Default 60 s.
    """

    DEFAULT_LEFT_COL  = 'VREyeTracker|EyeData_VR_LeftEye_pupilDiameter_D'
    DEFAULT_RIGHT_COL = 'VREyeTracker|EyeData_VR_RightEye_pupilDiameter_D'

    def __init__(
        self,
        fps: float = 90.0,
        left_col: str = DEFAULT_LEFT_COL,
        right_col: str = DEFAULT_RIGHT_COL,
        timestamp_col: Optional[str] = None,
        invalid_threshold: float = 0.0,
        min_blink_ms: float = 50.0,
        max_blink_ms: float = 1000.0,
        window_size_s: float = 60.0,
    ):
        self.fps               = fps
        self.left_col          = left_col
        self.right_col         = right_col
        self.timestamp_col     = timestamp_col
        self.invalid_threshold = invalid_threshold
        self.min_blink_ms      = min_blink_ms
        self.max_blink_ms      = max_blink_ms
        self.window_size_s     = window_size_s
        self._ms_per_frame: float = 1000.0 / fps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_timestamps_ms(self, df: pd.DataFrame) -> np.ndarray:
        """Build a millisecond-precision timeline normalized to start at 0."""
        if self.timestamp_col and self.timestamp_col in df.columns:
            ts = df[self.timestamp_col].values.astype(np.float64)
            sample = ts[:min(1000, len(ts))]
            median_interval = float(np.median(np.diff(sample))) if len(sample) > 1 else 1.0
            if median_interval < 1.0:
                ts = ts * 1000.0
            ts = ts - ts[0]
        else:
            ts = np.arange(len(df), dtype=np.float64) * self._ms_per_frame
        return ts

    def _run_length_encode_closed(
        self,
        both_closed: np.ndarray,
        timestamps_ms: np.ndarray,
    ) -> pd.DataFrame:
        """
        Run-length encoding: efficiently extract all consecutive "both eyes
        closed" segments.

        Pads the boolean sequence with False sentinels, then uses np.diff()
        to locate rising (+1) and falling (-1) edges in O(N) time.

        Returns DataFrame with columns: start_idx, end_idx, frame_count,
        duration_ms, start_ms, event_type ('blink'|'tracking_loss'|'artifact').
        """
        n = len(both_closed)
        if n == 0:
            return pd.DataFrame(
                columns=['start_idx', 'end_idx', 'frame_count',
                         'duration_ms', 'start_ms', 'event_type']
            )

        padded = np.empty(n + 2, dtype=np.int8)
        padded[0]    = 0
        padded[1:-1] = both_closed.astype(np.int8)
        padded[-1]   = 0

        diff = np.diff(padded)

        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        if len(starts) == 0:
            return pd.DataFrame(
                columns=['start_idx', 'end_idx', 'frame_count',
                         'duration_ms', 'start_ms', 'event_type']
            )

        frame_counts = (ends - starts).astype(np.int32)
        duration_ms  = frame_counts.astype(np.float64) * self._ms_per_frame
        start_ms     = timestamps_ms[starts]

        # Three-way classification
        event_type = np.where(
            duration_ms < self.min_blink_ms,   'artifact',
            np.where(
                duration_ms <= self.max_blink_ms, 'blink',
                'tracking_loss'
            )
        )

        return pd.DataFrame({
            'start_idx'  : starts,
            'end_idx'    : ends,
            'frame_count': frame_counts,
            'duration_ms': duration_ms,
            'start_ms'   : start_ms,
            'event_type' : event_type,
        })

    # ------------------------------------------------------------------
    # Main processing interface
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw eye-tracking DataFrame into per-minute features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw eye-tracking data with left_col and right_col.
            Rows must be in acquisition-time order.

        Returns
        -------
        pd.DataFrame indexed by Minute (1-indexed) with columns:
            Blink_Frequency, Mean_Blink_Duration, PERCLOS_Proxy,
            Tracking_Loss_Ratio, Data_Coverage.
        """
        if df.empty:
            return pd.DataFrame(
                columns=['Blink_Frequency', 'Mean_Blink_Duration',
                         'PERCLOS_Proxy', 'Tracking_Loss_Ratio', 'Data_Coverage']
            )

        # Step 1: Build millisecond timeline
        timestamps_ms = self._build_timestamps_ms(df)
        total_duration_ms = timestamps_ms[-1] + self._ms_per_frame

        # Step 2: Binocular fusion -> per-frame closed mask
        left  = pd.to_numeric(df[self.left_col],  errors='coerce').fillna(0.0).values
        right = pd.to_numeric(df[self.right_col], errors='coerce').fillna(0.0).values

        both_closed: np.ndarray = (
            (left  <= self.invalid_threshold) &
            (right <= self.invalid_threshold)
        )

        # Step 3: RLE extraction and classification
        events = self._run_length_encode_closed(both_closed, timestamps_ms)

        # Step 4: Per-minute window aggregation
        window_ms     = self.window_size_s * 1000.0
        total_minutes = int(np.ceil(total_duration_ms / window_ms))

        rows = []
        for minute_idx in range(total_minutes):
            win_start_ms = minute_idx * window_ms
            win_end_ms   = win_start_ms + window_ms

            data_end_ms    = min(total_duration_ms, win_end_ms)
            data_cover_pct = (data_end_ms - win_start_ms) / window_ms * 100.0

            if len(events) > 0:
                in_win  = (
                    (events['start_ms'] >= win_start_ms) &
                    (events['start_ms'] <  win_end_ms)
                )
                win_ev  = events[in_win]
                blinks  = win_ev[win_ev['event_type'] == 'blink']
                losses  = win_ev[win_ev['event_type'] == 'tracking_loss']

                blink_count    = len(blinks)
                mean_blink_dur = blinks['duration_ms'].mean() if blink_count > 0 else np.nan
                perclos        = blinks['duration_ms'].sum() / window_ms * 100.0
                loss_ratio     = losses['duration_ms'].sum() / window_ms * 100.0
            else:
                blink_count    = 0
                mean_blink_dur = np.nan
                perclos        = 0.0
                loss_ratio     = 0.0

            rows.append({
                'Minute'              : minute_idx + 1,
                'Blink_Frequency'     : blink_count,
                'Mean_Blink_Duration' : round(mean_blink_dur, 2)
                                        if not np.isnan(mean_blink_dur) else np.nan,
                'PERCLOS_Proxy'       : round(perclos,    4),
                'Tracking_Loss_Ratio' : round(loss_ratio, 4),
                'Data_Coverage'       : round(data_cover_pct, 2),
            })

        return pd.DataFrame(rows).set_index('Minute')


# ======================================================================
# Convenience wrapper
# ======================================================================

def extract_eye_features(
    df: pd.DataFrame,
    fps: float = 90.0,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function: returns per-minute eye feature DataFrame."""
    return EyeTrackingProcessor(fps=fps, **kwargs).process(df)
