"""
Target construction, regime handling, and leakage-safe time-series CV.

The single-price reform (2024-06-14) is a hard regime boundary: we train only
on data at/after it. The 15-min-DA change (2025-09-30) is encoded as a feature,
not a split, because there isn't yet enough post-date data to fit separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


def add_regime_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add regime feature(s). Assumes a tz-aware 'ts' (UTC) column."""
    df = df.copy()
    da15 = pd.Timestamp(cfg["regime"]["da_15min_start"], tz="Europe/Warsaw").tz_convert("UTC")
    df["regime_15min_da"] = (df["ts"] >= da15).astype("int8")
    return df


def clip_to_regime(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop everything before the single-price train_start."""
    start = pd.Timestamp(cfg["regime"]["train_start"], tz="Europe/Warsaw").tz_convert("UTC")
    return df.loc[df["ts"] >= start].reset_index(drop=True)


def make_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rename the configured price column to 'y' (CEN for the delivery period)."""
    col = cfg["target"]["value_col"]
    if col not in df.columns:
        raise KeyError(f"target column {col!r} not in frame; cols={list(df.columns)}")
    out = df.copy()
    out["y"] = pd.to_numeric(out[col], errors="coerce")
    return out


@dataclass
class Split:
    train_idx: pd.Index
    val_idx: pd.Index


def forward_chaining_splits(ts: pd.Series, n_splits: int = 5,
                            embargo_minutes: int = 60,
                            val_weeks: int = 4):
    """Expanding-window splits with an embargo gap so no feature built from the
    tail of train can peek into val. `ts` must be sorted, tz-aware.

    Yields Split(train_idx, val_idx). The final untouched test block should be
    carved off BEFORE calling this (see holdout_test).
    """
    ts = ts.sort_values()
    embargo = timedelta(minutes=embargo_minutes)
    val_span = timedelta(weeks=val_weeks)
    t_end = ts.max()
    # place n_splits validation windows back to back at the end of the sample
    for k in range(n_splits, 0, -1):
        val_hi = t_end - val_span * (k - 1)
        val_lo = val_hi - val_span
        train_hi = val_lo - embargo
        train_idx = ts.index[ts < train_hi]
        val_idx = ts.index[(ts >= val_lo) & (ts < val_hi)]
        if len(train_idx) and len(val_idx):
            yield Split(train_idx, val_idx)


def holdout_test(df: pd.DataFrame, cfg: dict):
    """Return (dev_df, test_df); test = most recent `test.weeks` contiguous."""
    span = pd.Timedelta(weeks=cfg["test"]["weeks"])
    cut = df["ts"].max() - span
    dev = df.loc[df["ts"] <= cut].reset_index(drop=True)
    test = df.loc[df["ts"] > cut].reset_index(drop=True)
    return dev, test
