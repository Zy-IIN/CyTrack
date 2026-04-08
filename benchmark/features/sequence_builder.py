"""Sequence construction utilities for Task 1 (sliding window) and Task 2 (session-level)."""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEAT_COLS = [
    "HR", "SDNN", "RMSSD",
    "SCL", "SCR_AMP", "SCR_COUNT",
    "RESP_RATE", "RESP_AMP",
    "Blink_Frequency", "Mean_Blink_Duration", "PERCLOS_Proxy", "Tracking_Loss_Ratio",
]
MAX_SEQ_LEN = 16


def make_windows(
    df: pd.DataFrame, window: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window samples for Task 1 (minute-level prediction).

    For each session, a window of `window` consecutive time steps is extracted
    and the label is the score at the final step.

    Args:
        df: DataFrame with columns 'participant', 'order', 'time', 'score', and FEAT_COLS.
        window: Number of consecutive time steps per sample.

    Returns:
        (X, y) where X has shape (n_samples, window, n_features) and y has shape (n_samples,).
    """
    Xs, ys = [], []
    for (_, _), grp in df.groupby(["participant", "order"]):
        grp = grp.sort_values("time")
        X, y = grp[FEAT_COLS].values, grp["score"].values
        for t in range(window - 1, len(X)):
            Xs.append(X[t - window + 1 : t + 1])
            ys.append(y[t])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_sequences(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Build padded session-level sequences for Task 2 and save artifacts.

    Each session (participant × order) becomes one sequence of up to MAX_SEQ_LEN
    time steps. Shorter sessions are zero-padded. Features are standardized using
    statistics computed only on valid (non-padded) time steps.

    Args:
        df: Merged DataFrame with FEAT_COLS, 'tscore', 'participant', 'order',
            'movement', 'odor' columns.
        output_dir: Directory to write X_sequences.npy, y_labels.npy,
            seq_lengths.npy, session_info.pkl, scaler.pkl.

    Returns:
        (X, y, seq_lengths, session_info) tuple.
    """
    session_keys = ["participant", "order", "movement", "odor"]
    sequences, labels, seq_lengths, session_info = [], [], [], []

    for session_id, group in df.groupby(session_keys):
        X_seq      = group[FEAT_COLS].values
        actual_len = len(X_seq)
        if actual_len < MAX_SEQ_LEN:
            X_seq = np.vstack([X_seq, np.zeros((MAX_SEQ_LEN - actual_len, X_seq.shape[1]))])
        sequences.append(X_seq)
        labels.append(group["tscore"].iloc[0])
        seq_lengths.append(actual_len)
        session_info.append({
            "participant": session_id[0], "order": session_id[1],
            "movement": session_id[2],   "odor":  session_id[3],
            "seq_length": actual_len,
        })

    valid_data = np.vstack([s[:l] for s, l in zip(sequences, seq_lengths)])
    scaler     = StandardScaler().fit(valid_data)

    sequences_scaled = []
    for seq, length in zip(sequences, seq_lengths):
        s = np.zeros_like(seq)
        s[:length] = scaler.transform(seq[:length])
        sequences_scaled.append(s)

    X = np.array(sequences_scaled, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    lengths = np.array(seq_lengths, dtype=np.int64)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_sequences.npy", X)
    np.save(output_dir / "y_labels.npy", y)
    np.save(output_dir / "seq_lengths.npy", lengths)
    with open(output_dir / "session_info.pkl", "wb") as f:
        pickle.dump(session_info, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X, y, lengths, session_info


def generate_dummy_data(
    n_sessions: int = 12,
    n_participants: int = 4,
    seq_len: int = 16,
    n_features: int = 12,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Generate synthetic data for pipeline testing without real recordings.

    Args:
        n_sessions: Total number of sessions to generate.
        n_participants: Number of simulated participants.
        seq_len: Fixed sequence length (no padding applied).
        n_features: Number of physiological/eye-tracking features.
        seed: Random seed.

    Returns:
        (X_seq, y_ssq, seq_lengths, session_info) with shapes
        (n_sessions, seq_len, n_features), (n_sessions,), (n_sessions,), list.
    """
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_sessions, seq_len, n_features)).astype(np.float32)
    y   = rng.uniform(0, 100, n_sessions).astype(np.float32)
    lengths = np.full(n_sessions, seq_len, dtype=np.int64)
    participants = [f"P{i+1:02d}" for i in range(n_participants)]
    info = [
        {"participant": participants[i % n_participants], "order": (i // n_participants) + 1,
         "movement": "W", "odor": "NO", "seq_length": seq_len}
        for i in range(n_sessions)
    ]
    return X, y, lengths, info
