"""
Temporal hierarchy for CEN (Weron/Kourentzes THieF idea), adapted to our
quantile GBM + conformal recipe.

    python -m src.temporal_hierarchy

CEN at 15-min is very noisy; its hourly and 4-hour-block means are smoother
and forecast more accurately. We forecast the aggregate levels with their own
GBMs (features aggregated to the level) and reconcile the bottom-level median
onto them, keeping the conformal spread as the within-level shape:

  base_med_qh                       bottom (15-min) GBM median
  hourly_fc / block_fc              aggregate-level GBM medians
  reconciled_med = base_med
        + lambda_h (hourly_fc  - implied_hourly_mean(base_med))
        + lambda_b (block_fc   - implied_block_mean(base_med))
  final_q = reconciled_med + (base_conformal_q - base_conformal_med)

lambda_h, lambda_b in [0,1] are chosen on the calibration tail (grid) to
minimize pinball — lambda=1 fully trusts the aggregate level, 0 ignores it.
Evaluated on the same 8-week holdout as src.evaluate.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from . import conformal, features, models
from .build_dataset import load_cfg
from .evaluate import CAL_WEEKS
from .models import QUANTILES


def _pinball_med(y, med) -> float:
    d = y - med
    return float(np.mean(np.maximum(0.5 * d, -0.5 * d)))


def _pinball_all(y, q: pd.DataFrame) -> float:
    tot = 0.0
    for qq in QUANTILES:
        d = y - q[qq].to_numpy()
        tot += np.mean(np.maximum(qq * d, (qq - 1) * d))
    return tot / len(QUANTILES)


def _agg_frame(df, feats, key):
    """Mean-aggregate y and features to a temporal key (per key: one row)."""
    g = df.groupby(key)
    out = g[feats].mean()
    out["y"] = g["y"].mean()
    return out.reset_index()


def _level_key(df, level):
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    day = loc.dt.strftime("%Y-%m-%d")
    if level == "hour":
        return day + "H" + loc.dt.hour.astype(str).str.zfill(2)
    if level == "block":                       # 4-hour blocks -> 6/day
        return day + "B" + (loc.dt.hour // 4).astype(str)
    raise ValueError(level)


def main():
    cfg = load_cfg()
    panel = pd.read_parquet("data/proc/panel_15min.parquet")
    df = features.build(panel, cfg).dropna(subset=["y"]).reset_index(drop=True)
    feats = features.feature_cols(df)
    ts = df["ts"]
    cut_hold = ts.max() - pd.Timedelta(weeks=cfg["test"]["weeks"])
    cut_cal = cut_hold - pd.Timedelta(weeks=CAL_WEEKS)
    embargo = pd.Timedelta(minutes=cfg["horizon_minutes"])
    train = df[ts < cut_cal - embargo].reset_index(drop=True)
    cal = df[(ts >= cut_cal) & (ts < cut_hold - embargo)].reset_index(drop=True)
    hold = df[ts >= cut_hold].reset_index(drop=True)

    # bottom-level GBM + hour-conformal (the baseline we try to beat)
    gbm = models.QuantileGBM().fit(train[feats], train["y"])
    off = conformal.fit_offsets(cal["y"], gbm.predict(cal[feats]),
                                groups=conformal.hour_block(cal["hour"]))

    def base_q(part):
        return conformal.apply_offsets(gbm.predict(part[feats]), off,
                                       groups=conformal.hour_block(part["hour"]))

    # aggregate-level GBMs (median only) for hourly and 4h-block means
    level_models = {}
    for lvl in ("hour", "block"):
        key = _level_key(train, lvl)
        agg = _agg_frame(train.assign(_k=key), feats, "_k")
        m = models.QuantileGBM(quantiles=(0.5,)).fit(agg[feats], agg["y"])
        level_models[lvl] = m

    def level_fc(part, lvl):
        """Broadcast the level-mean median forecast back to 15-min rows."""
        key = _level_key(part, lvl)
        uniq = pd.Index(key.unique())
        agg = part.assign(_k=key).groupby("_k")[feats].mean().reindex(uniq)
        pred = level_models[lvl].predict(agg)[0.5]
        return key.map(pred.to_dict()).to_numpy()

    def implied(part, med, lvl):
        key = _level_key(part, lvl)
        s = pd.Series(med, index=part.index)
        return key.map(s.groupby(key).mean().to_dict()).to_numpy()

    def reconcile(part, bq, lam_h, lam_b):
        med = bq[0.5].to_numpy()
        rec = med.copy()
        if lam_h:
            rec = rec + lam_h * (level_fc(part, "hour") - implied(part, med, "hour"))
        if lam_b:
            rec = rec + lam_b * (level_fc(part, "block") - implied(part, med, "block"))
        shift = rec - med
        out = bq.copy()
        for qq in QUANTILES:
            out[qq] = bq[qq].to_numpy() + shift
        out[:] = np.sort(out.values, axis=1)
        return out

    # tune lambdas on the calibration tail
    cal_bq = base_q(cal)
    yc = cal["y"].to_numpy()
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best, best_pb = (0.0, 0.0), _pinball_all(yc, cal_bq)
    for lh in grid:
        for lb in grid:
            pb = _pinball_all(yc, reconcile(cal, cal_bq, lh, lb))
            if pb < best_pb:
                best_pb, best = pb, (lh, lb)
    lam_h, lam_b = best

    # holdout evaluation
    hold_bq = base_q(hold)
    yh = hold["y"].to_numpy()
    base_pb = _pinball_all(yh, hold_bq)
    rec_bq = reconcile(hold, hold_bq, lam_h, lam_b)
    rec_pb = _pinball_all(yh, rec_bq)
    cov = lambda q: float(np.mean((yh >= q[0.10]) & (yh <= q[0.90])))

    out = {
        "lambda_hour": lam_h, "lambda_block": lam_b,
        "base_pinball": round(base_pb, 2), "base_cov": round(cov(hold_bq), 3),
        "reconciled_pinball": round(rec_pb, 2), "reconciled_cov": round(cov(rec_bq), 3),
        "delta": round(rec_pb - base_pb, 2),
        "median_pinball_base": round(_pinball_med(yh, hold_bq[0.5].to_numpy()), 2),
        "median_pinball_recon": round(_pinball_med(yh, rec_bq[0.5].to_numpy()), 2),
    }
    pathlib.Path("reports/temporal_hierarchy.json").write_text(json.dumps(out, indent=2))
    print(f"tuned lambda: hour={lam_h}, block={lam_b}")
    print(f"base        pinball {out['base_pinball']}  cov {out['base_cov']}")
    print(f"reconciled  pinball {out['reconciled_pinball']}  cov {out['reconciled_cov']}"
          f"   delta {out['delta']:+.2f}")
    print(f"(median-only pinball: base {out['median_pinball_base']} -> "
          f"recon {out['median_pinball_recon']})")


if __name__ == "__main__":
    main()
