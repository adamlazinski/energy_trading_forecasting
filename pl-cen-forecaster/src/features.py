"""
Leakage-safe feature engineering for the CEN forecaster.

Golden rule: a feature for delivery period t may only use info available at
t - H (H = horizon_minutes). Autoregressive price features are therefore
shifted by ceil(H / 15) quarter-hours plus a publication-lag buffer.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
    _PL_HOL = _holidays.Poland()
except Exception:            # optional dependency; degrade gracefully
    _PL_HOL = None


def _qh_shift(h_minutes: int, pub_lag_qh: int = 1) -> int:
    """How many 15-min steps to lag AR features so nothing leaks."""
    return math.ceil(h_minutes / 15) + pub_lag_qh


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["qh_of_day"] = loc.dt.hour * 4 + loc.dt.minute // 15    # 0..95
    df["hour"] = loc.dt.hour
    df["dow"] = loc.dt.dayofweek
    df["month"] = loc.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    # cyclical encodings
    df["qh_sin"] = np.sin(2 * np.pi * df["qh_of_day"] / 96)
    df["qh_cos"] = np.cos(2 * np.pi * df["qh_of_day"] / 96)
    if _PL_HOL is not None:
        df["is_holiday"] = loc.dt.date.map(lambda d: d in _PL_HOL).astype("int8")
    else:
        df["is_holiday"] = 0
    # crude sunrise/sunset ramp flags — the hard hours for the model.
    # (replace with a real solar-position calc when you wire Open-Meteo)
    df["ramp_am"] = df["hour"].between(4, 8).astype("int8")
    df["ramp_pm"] = df["hour"].between(16, 21).astype("int8")
    return df


def ar_features(df: pd.DataFrame, y_col: str, h_minutes: int) -> pd.DataFrame:
    """Lagged / rolling price features, shifted to respect the horizon."""
    df = df.sort_values("ts").copy()
    lag0 = _qh_shift(h_minutes)          # last value visible at decision time
    base = df[y_col].shift(lag0)
    df["ar_last"] = base
    df["ar_lag_1d"] = df[y_col].shift(lag0 + 96)
    df["ar_lag_1w"] = df[y_col].shift(lag0 + 96 * 7)
    df["ar_roll_mean_4"] = base.rolling(4).mean()
    df["ar_roll_std_16"] = base.rolling(16).std()
    df["ar_roll_mean_1d"] = base.rolling(96).mean()
    return df


def forecast_error_features(df: pd.DataFrame,
                            realised: dict[str, str],
                            forecast: dict[str, str],
                            h_minutes: int) -> pd.DataFrame:
    """RES/demand forecast-error features — the imbalance driver.

    `realised` and `forecast` map a name -> column. Realised is lagged (only
    the already-published past is visible); forecast for period t is allowed
    as-is because it's known ahead.
    """
    df = df.sort_values("ts").copy()
    lag = _qh_shift(h_minutes)
    for name, rcol in realised.items():
        fcol = forecast.get(name)
        if rcol in df and fcol in df:
            err = (df[rcol] - df[fcol]).shift(lag)
            df[f"ferr_{name}"] = err
            df[f"ferr_{name}_roll4"] = err.rolling(4).mean()
    # keep forward-looking forecasts themselves as features (known ahead)
    for name, fcol in forecast.items():
        if fcol in df:
            df[f"fc_{name}"] = df[fcol]
    return df


def build(df: pd.DataFrame, cfg: dict,
          realised: dict[str, str] | None = None,
          forecast: dict[str, str] | None = None) -> pd.DataFrame:
    """Assemble the full feature panel. Expects 'ts' and target 'y' present."""
    h = cfg["horizon_minutes"]
    out = calendar_features(df)
    out = ar_features(out, "y", h)
    if realised and forecast:
        out = forecast_error_features(out, realised, forecast, h)
    # drop warmup rows with NaNs introduced by the longest lag/rolling window
    feat_cols = [c for c in out.columns if c.startswith(("ar_", "ferr_", "fc_"))]
    out = out.dropna(subset=feat_cols, how="any").reset_index(drop=True)
    return out
