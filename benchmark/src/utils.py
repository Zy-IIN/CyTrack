"""Shared utilities: data splitting, preprocessing, evaluation, and training."""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SICK_THRESHOLD = 4.0


def loso_split(df: pd.DataFrame, test_p: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-subject-out split.

    Args:
        df: Full dataset with a 'participant' column.
        test_p: Participant ID to hold out as the test set.

    Returns:
        (train_df, test_df) tuple.
    """
    return df[df["participant"] != test_p].copy(), df[df["participant"] == test_p].copy()


def fit_transform_X(
    X_tr: np.ndarray, X_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute missing values and standardize features using training statistics.

    Args:
        X_tr: Training features.
        X_te: Test features.

    Returns:
        (X_tr_scaled, X_te_scaled) as float32 arrays.
    """
    imp = SimpleImputer(strategy="mean")
    sc  = StandardScaler()
    X_tr_out = sc.fit_transform(imp.fit_transform(X_tr))
    X_te_out = sc.transform(imp.transform(X_te))
    return X_tr_out.astype(np.float32), X_te_out.astype(np.float32)


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and Pearson r for regression outputs.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys 'RMSE', 'MAE', 'r'.
    """
    r = pearsonr(y_true, y_pred)[0] if len(np.unique(y_pred)) > 1 else 0.0
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "r":    float(r),
    }


def eval_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Accuracy, macro-F1, and AUC for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary with keys 'Acc', 'F1', 'AUC'.
    """
    auc = (
        roc_auc_score(y_true, y_prob)
        if y_prob is not None and len(np.unique(y_true)) > 1
        else 0.0
    )
    return {
        "Acc": float(accuracy_score(y_true, y_pred)),
        "F1":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "AUC": float(auc),
    }


def train_pytorch_once(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    task: str = "reg",
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict]:
    """Train a PyTorch model with early stopping on a held-out validation split.

    Validation set is the last 15% of training data (sequential split to avoid
    temporal leakage).

    Args:
        model: PyTorch model to train (must accept a single tensor input).
        X_tr: Training features as a numpy array.
        y_tr: Training targets as a numpy array.
        task: 'reg' for MSE loss, 'cls' for BCEWithLogitsLoss.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        device: Torch device.

    Returns:
        (trained_model, history) where history contains 'train_loss' and 'val_loss' lists.
    """
    n_val = max(8, int(len(X_tr) * 0.15))
    X_sub, X_val = X_tr[:-n_val], X_tr[-n_val:]
    y_sub, y_val = y_tr[:-n_val], y_tr[-n_val:]

    ds = TensorDataset(
        torch.tensor(X_sub).to(device), torch.tensor(y_sub).to(device)
    )
    loader   = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit     = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
    X_val_t  = torch.tensor(X_val).to(device)
    y_val_t  = torch.tensor(y_val).to(device)

    best_val, best_wts, pat_cnt = float("inf"), copy.deepcopy(model.state_dict()), 0
    history: Dict = {"train_loss": [], "val_loss": []}

    for _ in range(1, max_epochs + 1):
        model.train()
        epoch_loss = sum(
            (lambda loss: (loss.backward(), nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                           opt.step(), opt.zero_grad(), loss.item())[-1])(
                crit(model(xb), yb)
            )
            for xb, yb in loader
        )
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val_t), y_val_t).item()

        history["train_loss"].append(epoch_loss / len(loader))
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val, best_wts, pat_cnt = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    model.load_state_dict(best_wts)
    return model, history
