"""
Rolling split-conformal recalibration of the CMBP forecaster (F28 fix).

    python -m src.cmbp_conformal

F28's raw quantile GBM under-covers (P10-90 ~0.60 vs 0.80 nominal) — the
standard quantile-loss-fits-the-bulk failure the CEN model had, with the
same fix (src.conformal): shift each quantile by the empirical q-quantile
of trailing OOS residuals, grouped by hour block.

Honest timing: the offsets for month m are fitted ONLY on out-of-sample
predictions from days before m (trailing CAL_DAYS window). Since the wf
predictions are themselves gate-honest and CMBP for a day publishes D-1
09:10, residuals for day d are computable from d-1 onward — a trailing
window ending "yesterday" is bid-gate-legal.

Evaluates coverage and pinball before/after, per product and by quarter.
Writes reports/cmbp_conformal.json and the recalibrated predictions
reports/cmbp_wf_{target}_conf.parquet (for downstream users, e.g. shadow).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .conformal import apply_offsets, fit_offsets, hour_block

W = "Europe/Warsaw"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
TARGETS = ("afrr_g", "afrr_d")
# 90d picked from {30,60,90}: coverage rises monotonically with window on
# afrr_d (0.65/0.70/0.73) — the 2026Q2 regime break punishes fast adaptation,
# longer memory averages over regimes. afrr_g insensitive (0.77 at 60 and 90).
CAL_DAYS = 90


def pinball(y, pred):
    return float(np.mean([np.mean(np.where(y - pred[t] >= 0,
                                           t * (y - pred[t]),
                                           (t - 1) * (y - pred[t]))) for t in QS]))


def coverage(y, pred, lo, hi):
    return float(np.mean((y >= pred[lo]) & (y <= pred[hi])))


def recalibrate(target: str) -> dict:
    wf = pd.read_parquet(f"reports/cmbp_wf_{target}.parquet").sort_values("ts")
    wf["ts"] = pd.to_datetime(wf["ts"], utc=True)
    pred = pd.DataFrame({t: wf[f"q{t}"].to_numpy() for t in QS}, index=wf.index)
    y = wf[target]
    hb = hour_block(wf["ts"].dt.tz_convert(W).dt.hour)
    day = wf["ts"].dt.tz_convert(W).dt.normalize()

    conf = pred.copy()
    months = sorted(wf["ts"].dt.tz_convert(W).dt.strftime("%Y-%m").unique())
    mon = wf["ts"].dt.tz_convert(W).dt.strftime("%Y-%m")
    for m in months:
        te = wf.index[mon == m]
        m0 = day[te].min()
        cal = wf.index[(day < m0) & (day >= m0 - pd.Timedelta(days=CAL_DAYS))]
        if len(cal) < 20 * 24:
            continue
        off = fit_offsets(y.loc[cal], pred.loc[cal], groups=hb.loc[cal])
        conf.loc[te] = apply_offsets(pred.loc[te], off, groups=hb.loc[te]).values

    res = {"n": len(wf),
           "raw": {"pinball": round(pinball(y, pred), 2),
                   "cov_10_90": round(coverage(y, pred, 0.1, 0.9), 3),
                   "cov_25_75": round(coverage(y, pred, 0.25, 0.75), 3)},
           "conf": {"pinball": round(pinball(y, conf), 2),
                    "cov_10_90": round(coverage(y, conf, 0.1, 0.9), 3),
                    "cov_25_75": round(coverage(y, conf, 0.25, 0.75), 3)},
           "by_quarter": {}}
    for q, g in wf.groupby("quarter"):
        res["by_quarter"][q] = {
            "cov_raw": round(coverage(y.loc[g.index], pred.loc[g.index], 0.1, 0.9), 3),
            "cov_conf": round(coverage(y.loc[g.index], conf.loc[g.index], 0.1, 0.9), 3),
            "pb_raw": round(pinball(y.loc[g.index], pred.loc[g.index]), 2),
            "pb_conf": round(pinball(y.loc[g.index], conf.loc[g.index]), 2)}

    out = wf[["ts", target, "quarter"]].copy()
    for t in QS:
        out[f"q{t}"] = conf[t].to_numpy()
    out.to_parquet(f"reports/cmbp_wf_{target}_conf.parquet")
    return res


def main():
    out = {"cal_days": CAL_DAYS, "targets": {}}
    for target in TARGETS:
        r = recalibrate(target)
        out["targets"][target] = r
        print(f"== {target}: pinball {r['raw']['pinball']} -> {r['conf']['pinball']}   "
              f"cov10-90 {r['raw']['cov_10_90']} -> {r['conf']['cov_10_90']}   "
              f"cov25-75 {r['raw']['cov_25_75']} -> {r['conf']['cov_25_75']}")
        for q, v in r["by_quarter"].items():
            print(f"   {q}: cov {v['cov_raw']} -> {v['cov_conf']}   "
                  f"pb {v['pb_raw']} -> {v['pb_conf']}")
    pathlib.Path("reports/cmbp_conformal.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/cmbp_conformal.json")


if __name__ == "__main__":
    main()
