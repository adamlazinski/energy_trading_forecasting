"""
Split-conformal recalibration of the quantile GBM.

The raw quantile GBM is underdispersed on CEN (P10-P90 empirical coverage
~0.56 vs nominal 0.80 on the first run): quantile loss fits the bulk and
under-weights the fat tails. Fix: hold back a calibration slice (never seen
by the model), measure the residual y - q_hat(x) per nominal quantile, and
shift each predicted quantile by the empirical q-quantile of those residuals.
Optionally grouped (e.g. by hour block) so night intervals don't inherit
midday's width.

This is marginal per-quantile split conformal - simple, distribution-free,
and it preserves the quantile interpretation. Group-conditional offsets get
conditional coverage closer to nominal where the error structure is strongly
time-of-day dependent (it is, midday solar hours dominate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import QUANTILES


def hour_block(hours: pd.Series) -> pd.Series:
    """The evaluation's hour blocking, reused for group-conditional offsets."""
    return pd.cut(hours, [0, 6, 10, 14, 18, 22, 24], right=False,
                  labels=["night", "am_ramp", "midday", "pm", "ev_ramp", "late"])


def fit_offsets(y_cal: pd.Series, pred_cal: pd.DataFrame,
                groups: pd.Series | None = None,
                quantiles=QUANTILES) -> pd.DataFrame:
    """Per-quantile additive offsets from a calibration slice.

    offset[q] = empirical q-quantile of (y - pred_q). Zero for a perfectly
    calibrated q. Returns a DataFrame indexed by group (single row '_all'
    when ungrouped), columns = quantiles.
    """
    resid = pd.DataFrame({q: y_cal - pred_cal[q] for q in quantiles})
    if groups is None:
        return pd.DataFrame(
            {q: [resid[q].quantile(q)] for q in quantiles}, index=["_all"])
    out = {}
    for g, idx in resid.groupby(groups, observed=True).groups.items():
        sub = resid.loc[idx]
        out[g] = {q: sub[q].quantile(q) for q in quantiles}
    return pd.DataFrame(out).T


def apply_offsets(pred: pd.DataFrame, offsets: pd.DataFrame,
                  groups: pd.Series | None = None,
                  quantiles=QUANTILES) -> pd.DataFrame:
    """Shift predicted quantiles, then re-sort rows to keep them monotone."""
    out = pred.copy()
    if groups is None or list(offsets.index) == ["_all"]:
        for q in quantiles:
            out[q] = pred[q] + offsets.loc["_all", q]
    else:
        for g, idx in groups.groupby(groups, observed=True).groups.items():
            if g in offsets.index:
                for q in quantiles:
                    out.loc[idx, q] = pred.loc[idx, q] + offsets.loc[g, q]
    out[:] = np.sort(out[list(quantiles)].values, axis=1)
    return out
