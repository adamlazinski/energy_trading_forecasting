"""
Baselines and the quantile LightGBM for the CEN forecaster.

Baselines (in SPEC order, all leakage-honest at the gate):
  B0 persistence  — same-qh CEN from the last fully published day
                    (cen_pub_same_qh; at H=60 that day is 1-2 days back).
  B1 DA anchor    — CEN(t) = csdac_pln(t), known D-1.
  B2 climatology quantiles — per-qh-of-day empirical quantiles of CEN over the
                    trailing train window (captures the midday-negative /
                    evening-spike shape).
  B_pse           — PSE's own cen_fcst. FINAL VINTAGE (published around/after
                    delivery): an upper-bound reference, not a fair H=60
                    competitor. Reported separately, never trained on.

Model: one LightGBM per quantile (P10/25/50/75/90), monotone-rearranged at
predict time to fix quantile crossing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

LGB_PARAMS = dict(
    objective="quantile",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


def predict_baselines(train: pd.DataFrame, test: pd.DataFrame,
                      quantiles=QUANTILES) -> dict[str, pd.DataFrame]:
    """Return {name: DataFrame[q...] aligned to test.index}."""
    out: dict[str, pd.DataFrame] = {}

    def flat(point: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({q: point for q in quantiles}, index=test.index)

    if "cen_pub_same_qh" in test:
        out["B0_persistence"] = flat(test["cen_pub_same_qh"])
    if "fx_da" in test:
        out["B1_da_anchor"] = flat(test["fx_da"])

    # B2: per-qh climatology of the train window
    clim = train.groupby("qh_of_day")["y"].quantile(list(quantiles)).unstack()
    b2 = test["qh_of_day"].map(clim.to_dict("index"))
    out["B2_climatology"] = pd.DataFrame(
        [v if isinstance(v, dict) else {q: np.nan for q in quantiles} for v in b2],
        index=test.index)[list(quantiles)]

    if "price_fcst__cen_fcst" in test:
        out["B_pse_final_vintage"] = flat(
            pd.to_numeric(test["price_fcst__cen_fcst"], errors="coerce"))
    return out


class QuantileGBM:
    """Per-quantile LightGBM, optionally on a variance-stabilized target.

    vst=True applies the asinh VST of Uniejewski & Weron (median/MAD
    normalization, z = asinh((y - m)/s)) before fitting and inverts the
    predicted quantiles exactly (quantiles are equivariant under monotone
    transforms). Motivated by CEN's +/-45k PLN/MWh spikes; see
    arXiv:2511.13603 for the volatile-regime evidence.
    """

    def __init__(self, quantiles=QUANTILES, vst: bool = False, **overrides):
        self.quantiles = quantiles
        self.vst = vst
        self.params = {**LGB_PARAMS, **overrides}
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []
        self._m = 0.0
        self._s = 1.0

    def _fwd(self, y: pd.Series) -> pd.Series:
        return np.arcsinh((y - self._m) / self._s)

    def _inv(self, z: np.ndarray) -> np.ndarray:
        return self._s * np.sinh(z) + self._m

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        if self.vst:
            self._m = float(y.median())
            mad = float((y - self._m).abs().median())
            self._s = max(1.4826 * mad, 1e-6)
            y = self._fwd(y)
        for q in self.quantiles:
            m = lgb.LGBMRegressor(alpha=q, **self.params)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pred = pd.DataFrame(
            {q: self.models[q].predict(X[self.feature_names]) for q in self.quantiles},
            index=X.index)
        if self.vst:
            pred[:] = self._inv(pred.values)
        # rearrange to enforce monotone quantiles (Chernozhukov et al.)
        pred[:] = np.sort(pred.values, axis=1)
        return pred

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            q: pd.Series(m.feature_importances_, index=self.feature_names)
            for q, m in self.models.items()}).sort_values(0.5, ascending=False)
