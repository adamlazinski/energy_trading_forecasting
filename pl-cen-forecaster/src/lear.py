"""
LEAR benchmark (Lasso-Estimated AutoRegressive), Weron-group style, adapted
to a probabilistic 15-min CEN forecast.

    python -m src.lear [--refit-every N] [--windows 56,84,112]

Method:
- Target: price directly (default). The asinh VST (--vst) is available but
  PATHOLOGICAL on CEN: its sinh inverse extrapolates on CEN's +/-45k PLN
  spikes and blows up a handful of predictions, doubling pinball (F9). It
  helps day-ahead prices but not this balancing price.
- Regressors: the strict feature panel (features.feature_cols), z-scored on
  each calibration window (LASSO needs standardized inputs).
- Estimator: per quantile, L1-penalized linear quantile regression
  (sklearn QuantileRegressor, pinball objective + L1) — this is the
  quantile analogue of LEAR's LASSO point model.
- Calibration-window averaging (Marcjasz/Uniejewski/Weron): refit on
  several trailing window lengths and average the predicted VST quantiles
  before inverting — the cheap, robust ensembling that makes LEAR strong.
- Rolling: refit every `refit_every` days on the trailing windows, predict
  forward until the next refit. Evaluated on the same 8-week holdout as
  src.evaluate for a like-for-like pinball comparison against the GBM.

Writes reports/lear_eval.{json,txt}.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from .build_dataset import load_cfg
from .features import build as build_features, feature_cols
from .models import QUANTILES

HOLDOUT_WEEKS = 8
DEFAULT_WINDOWS = (56, 84, 112)     # calibration lengths, days
L1_ALPHA = 1e-3                     # LASSO strength on standardized features
PERIODS_PER_DAY = 96


def load_panel():
    """Feature panel with NaNs filled — shared by holdout and walk-forward."""
    cfg = load_cfg()
    panel = pd.read_parquet("data/proc/panel_15min.parquet")
    df = build_features(panel, cfg).dropna(subset=["y"]).reset_index(drop=True)
    feats = feature_cols(df)
    df[feats] = df[feats].astype(float)
    df[feats] = df[feats].fillna(df[feats].median())
    return df, feats


def rolling_predict(df, feats, pred_days, windows, refit_every,
                    use_vst=False) -> np.ndarray:
    """Rolling quantile-LEAR over `pred_days` (tz-aware normalized days).

    Refits every `refit_every` days on trailing `windows`; predicts each day
    from data strictly before its gate. Returns (len(df), 5) with NaN outside
    predicted rows.
    """
    ts = df["ts"]
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    preds = np.full((len(df), len(QUANTILES)), np.nan)
    refit_anchor, cache = None, None
    for d in pred_days:
        d_utc = pd.Timestamp(d).tz_convert("UTC")
        blk = df.index[(ts >= d_utc) & (ts < d_utc + pd.Timedelta(days=1))]
        if len(blk) == 0:
            continue
        gate = ts.iloc[blk[0]]
        if refit_anchor is None or (gate - refit_anchor).days >= refit_every:
            refit_anchor = gate
            tm0 = ((ts < gate) & (ts >= gate - pd.Timedelta(days=max(windows)))).to_numpy()
            ya = y[tm0]
            m0 = np.median(ya)
            s0 = max(1.4826 * np.median(np.abs(ya - m0)), 1e-6)
            cache = (m0, s0)
        m0, s0 = cache
        Xp = X[blk]
        stack = []
        for w in windows:
            tm = ((ts < gate) & (ts >= gate - pd.Timedelta(days=w))).to_numpy()
            if tm.sum() < 20 * PERIODS_PER_DAY:
                continue
            ytr = np.arcsinh((y[tm] - m0) / s0) if use_vst else y[tm]
            raw = _fit_window(X[tm], ytr, Xp)
            stack.append(s0 * np.sinh(raw) + m0 if use_vst else raw)
        if stack:
            preds[blk] = np.sort(np.mean(stack, axis=0), axis=1)
    return preds


def _fit_window(Xtr, ytr, Xpred):
    """Standardize, fit one QR per quantile, return VST-space predictions."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Ztr, Zpred = (Xtr - mu) / sd, (Xpred - mu) / sd
    out = np.empty((len(Xpred), len(QUANTILES)))
    for j, q in enumerate(QUANTILES):
        m = QuantileRegressor(quantile=q, alpha=L1_ALPHA, solver="highs")
        m.fit(Ztr, ytr)
        out[:, j] = m.predict(Zpred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-every", type=int, default=2)
    ap.add_argument("--windows", type=str,
                    default=",".join(map(str, DEFAULT_WINDOWS)))
    ap.add_argument("--vst", action="store_true",
                    help="fit on asinh-VST target (PATHOLOGICAL on CEN: the "
                         "sinh inverse explodes on the fat tails -> ~2x worse "
                         "pinball; kept only for diagnostics; see F9)")
    args = ap.parse_args()
    use_vst = args.vst
    windows = [int(w) for w in args.windows.split(",")]

    df, feats = load_panel()
    ts = df["ts"]
    y = df["y"].to_numpy()
    cut_hold = ts.max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    days = sorted(ts[ts >= cut_hold].dt.tz_convert("Europe/Warsaw")
                  .dt.normalize().unique())
    preds = rolling_predict(df, feats, days, windows, args.refit_every, use_vst)

    # evaluate on holdout rows that got predictions
    mask = ~np.isnan(preds[:, 0])
    hmask = mask & (ts >= cut_hold).to_numpy()
    yy = y[hmask]
    pp = preds[hmask]
    per_q, tot = {}, 0.0
    for j, q in enumerate(QUANTILES):
        dq = yy - pp[:, j]
        pb = float(np.mean(np.maximum(q * dq, (q - 1) * dq)))
        per_q[str(q)] = round(pb, 2)
        tot += pb
    cov = {"P10_90": round(float(np.mean((yy >= pp[:, 0]) & (yy <= pp[:, 4]))), 3),
           "P25_75": round(float(np.mean((yy >= pp[:, 1]) & (yy <= pp[:, 3]))), 3)}
    out = {"n_holdout": int(hmask.sum()), "windows": windows, "vst": use_vst,
           "refit_every_days": args.refit_every, "l1_alpha": L1_ALPHA,
           "mean_pinball": round(tot / len(QUANTILES), 2),
           "per_quantile": per_q, "coverage": cov}
    pathlib.Path("reports/lear_eval.json").write_text(json.dumps(out, indent=2))
    txt = (f"LEAR (vst={use_vst}, windows={windows}d, refit/{args.refit_every}d, "
           f"L1={L1_ALPHA}) holdout n={hmask.sum()}\n"
           f"  mean pinball = {out['mean_pinball']}  per-q={per_q}\n"
           f"  coverage P10-90={cov['P10_90']} P25-75={cov['P25_75']}\n"
           f"  (GBM+hour-conformal reference: 68.5 / 0.76)")
    pathlib.Path("reports/lear_eval.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
