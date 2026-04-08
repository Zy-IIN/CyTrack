"""Task 2 LOSO-CV training: session-level SSQ Total Score prediction."""

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baselines import train_xgb
from models.sequence_models import SequenceToOneLSTM
from src.utils import eval_regression, fit_transform_X

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_data(data_dir: Path):
    X          = np.load(data_dir / "X_sequences.npy")
    y          = np.load(data_dir / "y_labels.npy")
    lengths    = np.load(data_dir / "seq_lengths.npy")
    with open(data_dir / "session_info.pkl", "rb") as f:
        info = pickle.load(f)
    return X, y, lengths, info


def _run_xgb(X, y, info, participants, seed):
    y_true, y_pred = [], []
    for p in tqdm(participants, desc="Agg-XGB"):
        tr_idx = [i for i, s in enumerate(info) if s["participant"] != p]
        te_idx = [i for i, s in enumerate(info) if s["participant"] == p]
        X_tr = np.hstack([X[tr_idx].mean(1), X[tr_idx].std(1)])
        X_te = np.hstack([X[te_idx].mean(1), X[te_idx].std(1)])
        X_tr, X_te = fit_transform_X(X_tr, X_te)
        m = train_xgb(X_tr, y[tr_idx], seed=seed)
        y_pred.extend(m.predict(X_te)); y_true.extend(y[te_idx])
    return eval_regression(np.array(y_true), np.array(y_pred))


def _run_lstm(X, y, lengths, info, participants, epochs):
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)
    y_true_all, y_pred_all = [], []

    for p in tqdm(participants, desc="LSTM"):
        tr_idx = [i for i, s in enumerate(info) if s["participant"] != p]
        te_idx = [i for i, s in enumerate(info) if s["participant"] == p]

        X_tr_f, X_te_f = fit_transform_X(
            X[tr_idx].reshape(-1, X.shape[-1]),
            X[te_idx].reshape(-1, X.shape[-1]))
        X_tr_f = X_tr_f.reshape(len(tr_idx), X.shape[1], X.shape[2])
        X_te_f = X_te_f.reshape(len(te_idx), X.shape[1], X.shape[2])

        y_tr = y_norm[tr_idx].astype(np.float32)
        len_tr, len_te = lengths[tr_idx], lengths[te_idx]

        n_val = max(4, int(len(tr_idx) * 0.15))
        X_sub, X_val = X_tr_f[:-n_val], X_tr_f[-n_val:]
        y_sub, y_val = y_tr[:-n_val], y_tr[-n_val:]
        len_sub, len_val = len_tr[:-n_val], len_tr[-n_val:]

        def _t(a, dtype=torch.float32): return torch.tensor(a, dtype=dtype).to(DEVICE)

        loader = DataLoader(
            TensorDataset(_t(X_sub), torch.tensor(len_sub, dtype=torch.int64), _t(y_sub)),
            batch_size=32, shuffle=True)

        model = SequenceToOneLSTM(X.shape[-1]).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit  = nn.MSELoss()

        best_val, best_wts, pat = float("inf"), copy.deepcopy(model.state_dict()), 0
        for _ in range(epochs):
            model.train()
            for xb, lb, yb in loader:
                opt.zero_grad(); loss = crit(model(xb, lb), yb); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                v = crit(model(_t(X_val), torch.tensor(len_val, dtype=torch.int64)),
                         _t(y_val)).item()
            if v < best_val - 1e-4:
                best_val, best_wts, pat = v, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= 10: break

        model.load_state_dict(best_wts)
        model.eval()
        with torch.no_grad():
            pred_norm = model(_t(X_te_f), torch.tensor(len_te, dtype=torch.int64)).cpu().numpy()
        y_pred_all.extend(pred_norm * (y_max - y_min) + y_min)
        y_true_all.extend(y[te_idx])

    return eval_regression(np.array(y_true_all), np.array(y_pred_all))


def main():
    parser = argparse.ArgumentParser(description="Task 2: Session-level SSQ prediction.")
    parser.add_argument("--model",    choices=["xgb", "lstm", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs",   type=int,  default=50)
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    X, y, lengths, info = _load_data(args.data_dir)
    participants = sorted({s["participant"] for s in info})

    results = {}
    run_all = args.model == "all"
    if run_all or args.model == "xgb":
        results["Agg-XGBoost"] = _run_xgb(X, y, info, participants, args.seed)
    if run_all or args.model == "lstm":
        results["LSTM"] = _run_lstm(X, y, lengths, info, participants, args.epochs)

    print("\n[Task 2 Results]")
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'r':>8}")
    for name, m in results.items():
        print(f"{name:<20} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} {m['r']:>8.3f}")


if __name__ == "__main__":
    main()
