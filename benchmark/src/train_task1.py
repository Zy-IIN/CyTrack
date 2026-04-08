"""Task 1 LOSO-CV training: minute-level cybersickness score prediction."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.sequence_builder import FEAT_COLS, make_windows
from models.baselines import train_lr, train_xgb
from models.fusion_network import MultimodalFusion
from models.sequence_models import GRUPredictor
from src.utils import eval_regression, fit_transform_X, loso_split

ECG_COLS  = ["HR", "SDNN", "RMSSD"]
EDA_COLS  = ["SCL", "SCR_AMP", "SCR_COUNT"]
RESP_COLS = ["RESP_RATE", "RESP_AMP"]
EYE_COLS  = ["Blink_Frequency", "Mean_Blink_Duration", "PERCLOS_Proxy", "Tracking_Loss_Ratio"]
VALID_COLS = ["ecg_valid", "eda_valid", "resp_valid", "eye_valid"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_xgb(df, participants, seed):
    y_true, y_pred = [], []
    for p in tqdm(participants, desc="XGBoost"):
        tr, te = loso_split(df, p)
        X_tr, X_te = fit_transform_X(tr[FEAT_COLS].values, te[FEAT_COLS].values)
        m = train_xgb(X_tr, tr["score"].values, seed=seed)
        y_pred.extend(m.predict(X_te)); y_true.extend(te["score"].values)
    return eval_regression(np.array(y_true), np.array(y_pred))


def _run_lr(df, participants):
    y_true, y_pred = [], []
    for p in tqdm(participants, desc="LR"):
        tr, te = loso_split(df, p)
        X_tr, X_te = fit_transform_X(tr[FEAT_COLS].values, te[FEAT_COLS].values)
        m = train_lr(X_tr, tr["score"].values)
        y_pred.extend(m.predict(X_te)); y_true.extend(te["score"].values)
    return eval_regression(np.array(y_true), np.array(y_pred))


def _run_gru(df, participants, epochs, window=5):
    y_true, y_pred = [], []
    for p in tqdm(participants, desc="GRU"):
        tr, te = loso_split(df, p)
        X_tr, y_tr = make_windows(tr, window)
        X_te, y_te = make_windows(te, window)
        if len(X_te) == 0:
            continue
        n_feat = X_tr.shape[-1]
        X_tr_f, X_te_f = fit_transform_X(X_tr.reshape(-1, n_feat), X_te.reshape(-1, n_feat))
        X_tr_f = X_tr_f.reshape(-1, window, n_feat)
        X_te_f = X_te_f.reshape(-1, window, n_feat)

        n_val = max(8, int(len(X_tr_f) * 0.15))
        X_sub, X_val = X_tr_f[:-n_val], X_tr_f[-n_val:]
        y_sub, y_val = y_tr[:-n_val], y_tr[-n_val:]

        model = GRUPredictor(n_feat).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        crit  = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_sub).to(DEVICE), torch.tensor(y_sub).to(DEVICE)),
            batch_size=128, shuffle=True)

        best_val, best_wts, pat = float("inf"), copy.deepcopy(model.state_dict()), 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                v = crit(model(torch.tensor(X_val).to(DEVICE)),
                         torch.tensor(y_val).to(DEVICE)).item()
            if v < best_val - 1e-4:
                best_val, best_wts, pat = v, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= 10: break
        model.load_state_dict(best_wts)
        model.eval()
        with torch.no_grad():
            y_pred.extend(model(torch.tensor(X_te_f).to(DEVICE)).cpu().numpy())
        y_true.extend(y_te)
    return eval_regression(np.array(y_true), np.array(y_pred))


def _run_fusion(df, participants, epochs):
    y_true, y_pred = [], []
    for p in tqdm(participants, desc="Fusion"):
        tr, te = loso_split(df, p)

        def _arrays(d):
            return (d[ECG_COLS].values, d[EDA_COLS].values,
                    d[RESP_COLS].values, d[EYE_COLS].values,
                    d["ecg_valid"].values, d["eda_valid"].values,
                    d["resp_valid"].values, d["eye_valid"].values)

        arrs_tr, arrs_te = _arrays(tr), _arrays(te)
        scaled = [fit_transform_X(arrs_tr[i], arrs_te[i]) for i in range(4)]
        xtr = [s[0] for s in scaled]; xte = [s[1] for s in scaled]
        y_tr, y_te = tr["score"].values.astype(np.float32), te["score"].values.astype(np.float32)

        n_val = max(8, int(len(y_tr) * 0.15))
        def _t(a): return torch.tensor(a, dtype=torch.float32).to(DEVICE)

        ds = TensorDataset(*[_t(x[:-n_val]) for x in xtr],
                           *[_t(arrs_tr[i+4][:-n_val]) for i in range(4)],
                           _t(y_tr[:-n_val]))
        loader = DataLoader(ds, batch_size=128, shuffle=True)

        model = MultimodalFusion().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
        crit  = nn.MSELoss()

        best_val, best_wts, pat = float("inf"), copy.deepcopy(model.state_dict()), 0
        val_inputs = [_t(x[-n_val:]) for x in xtr] + [_t(arrs_tr[i+4][-n_val:]) for i in range(4)]

        for _ in range(epochs):
            model.train()
            for batch in loader:
                inputs, yb = batch[:-1], batch[-1]
                opt.zero_grad(); loss = crit(model(*inputs), yb); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                v = crit(model(*val_inputs), _t(y_tr[-n_val:])).item()
            if v < best_val - 1e-4:
                best_val, best_wts, pat = v, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= 10: break
        model.load_state_dict(best_wts)
        model.eval()
        with torch.no_grad():
            te_inputs = [_t(x) for x in xte] + [_t(arrs_te[i+4]) for i in range(4)]
            y_pred.extend(model(*te_inputs).cpu().numpy())
        y_true.extend(y_te)
    return eval_regression(np.array(y_true), np.array(y_pred))


def main():
    parser = argparse.ArgumentParser(description="Task 1: Minute-level cybersickness prediction.")
    parser.add_argument("--model",  choices=["xgb", "lr", "gru", "fusion", "all"], default="all")
    parser.add_argument("--data",   type=Path, default=Path("data/raw/dataset_sample.xlsx"))
    parser.add_argument("--epochs", type=int,  default=50)
    parser.add_argument("--seed",   type=int,  default=42)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_excel(args.data)
    df["sick"] = (df["score"] >= 4.0).astype(int)
    participants = sorted(df["participant"].unique())

    results = {}
    run_all = args.model == "all"
    if run_all or args.model == "xgb":
        results["XGBoost"] = _run_xgb(df, participants, args.seed)
    if run_all or args.model == "lr":
        results["LR"] = _run_lr(df, participants)
    if run_all or args.model == "gru":
        results["GRU"] = _run_gru(df, participants, args.epochs)
    if run_all or args.model == "fusion":
        results["Fusion"] = _run_fusion(df, participants, args.epochs)

    print("\n[Task 1 Results]")
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'r':>8}")
    for name, m in results.items():
        print(f"{name:<20} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} {m['r']:>8.3f}")


if __name__ == "__main__":
    main()
