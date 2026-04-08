"""Baseline regression models for VR cybersickness prediction."""

import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def train_xgb(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 42, **kwargs) -> xgb.XGBRegressor:
    """Fit an XGBoost regressor.

    Args:
        X_tr: Training features, shape (n_samples, n_features).
        y_tr: Training targets.
        seed: Random seed for reproducibility.
        **kwargs: Additional keyword arguments passed to XGBRegressor.

    Returns:
        Fitted XGBRegressor instance.
    """
    defaults = dict(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=seed)
    defaults.update(kwargs)
    model = xgb.XGBRegressor(**defaults)
    model.fit(X_tr, y_tr)
    return model


def train_lr(X_tr: np.ndarray, y_tr: np.ndarray) -> LinearRegression:
    """Fit an ordinary least-squares linear regression model.

    Args:
        X_tr: Training features, shape (n_samples, n_features).
        y_tr: Training targets.

    Returns:
        Fitted LinearRegression instance.
    """
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model
