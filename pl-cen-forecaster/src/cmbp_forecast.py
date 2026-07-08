"""
CMBP forecaster: D-1 bid-gate-honest quantiles of the aFRR capacity prices.

    python -m src.cmbp_forecast

The balancing-capacity auction for day D clears the morning of D-1 (CMBP
publication_ts ~09:10 D-1, verified). Offers are due before that; we assume a
07:30 D-1 information gate and admit only features published by then:

  legal at the gate                       NOT legal (later than the gate)
  ------------------------------------   -------------------------------
  CMBP/zmb for D-1 and older (pub D-2)    CMBP/zmb for D itself
  CEN days <= D-3 (D-2 lands D-1 14:00)   csdac for D (D-1 ~13:50)
  NWP 48h-ahead run (w3_d2, ghi_d2)       latest NWP run, pk5l for D

This is a *harder* information set than the CEN forecaster's H=60 gate —
notably the DA price anchor is missing. Targets: afrr_g, afrr_d (hourly,
PLN/MW/h) — the two legs behind the stack's dominant 2.7M PLN/MW/yr.

Model: pooled per-quantile LightGBM, expanding walk-forward with monthly
refits (train from 2024-06-15, test 2025-01 →). Benchmarks: same-hour D-1 /
D-7 persistence (both gate-legal) for MAE; trailing-90-day same-hour
climatology quantiles for pinball. Everything reported by quarter.

Economics reported: clearing-capture curves — bid at forecast q10/q25/q50,
paid CMBP when CMBP >= bid (uniform pricing) — and stated honestly: under
uniform pricing a price-taker's optimal bid is its reservation cost, forecast
or no forecast; the forecast's value is revenue projection and the D-1
commit-capacity-vs-energy portfolio decision (the DP on top of F27), for
which these quantiles are the missing input.
Writes reports/cmbp_forecast.json.
"""
from __future__ import annotations

import json
import pathlib
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

W = "Europe/Warsaw"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
TARGETS = ("afrr_g", "afrr_d")
TEST_START = "2025-01-01"
CLIM_DAYS = 90
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
              verbose=-1)


def build() -> pd.DataFrame:
    cp = pd.read_parquet("data/raw/pse_reserve_prices_basic.parquet")[
        ["ts", "afrr_g", "afrr_d"]].dropna().sort_values("ts")
    for c in TARGETS:
        cp[c] = pd.to_numeric(cp[c], errors="coerce")
    df = cp.set_index("ts").resample("1h").mean()

    zmb = pd.read_parquet("data/raw/pse_reserve_req.parquet")[
        ["ts", "zmb_afrrg", "zmb_afrrd"]].sort_values("ts")
    for c in ("zmb_afrrg", "zmb_afrrd"):
        zmb[c] = pd.to_numeric(zmb[c], errors="coerce")
    df = df.join(zmb.set_index("ts").resample("1h").mean())

    # same-hour lags (24h steps); D-1 values were published D-2 -> legal
    for c in ("afrr_g", "afrr_d", "zmb_afrrg", "zmb_afrrd"):
        for lag in (1, 2, 7):
            df[f"{c}_l{lag}"] = df[c].shift(24 * lag)
    for c in TARGETS:  # D-1 day-level shape
        day = df[c].shift(24).rolling(24)
        df[f"{c}_l1_mean"], df[f"{c}_l1_max"] = day.mean(), day.max()

    # CEN published history: freshest fully-published day at 07:30 D-1 is D-3
    cen = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "cen_cost"]]
    cen["cen_cost"] = pd.to_numeric(cen["cen_cost"], errors="coerce")
    ch = cen.set_index("ts")["cen_cost"].resample("1h").mean()
    df["cen_l3"] = ch.shift(24 * 3)
    d3 = ch.shift(24 * 3).rolling(24)
    df["cen_l3_mean"], df["cen_l3_max"] = d3.mean(), d3.max()
    df["cen_l3_spikes"] = (ch > 1500).astype(float).shift(24 * 3).rolling(24 * 7).sum()

    # NWP 48h-ahead run for delivery hour (issued D-2 -> legal at the gate)
    wx = pd.read_parquet("data/raw/weather_prev_runs.parquet")
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    df = df.join(wx.set_index("ts")[["w3_d2", "ghi_d2"]])

    loc = df.index.tz_convert(W)
    df["hour"], df["dow"], df["month"] = loc.hour, loc.dayofweek, loc.month
    df["quarter"] = pd.Series(loc.year.astype(str), index=df.index) + "Q" + \
        pd.Series(loc.quarter.astype(str), index=df.index)
    return df.reset_index()


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.where(d >= 0, tau * d, (tau - 1) * d)))


def walkforward(df: pd.DataFrame, target: str) -> pd.DataFrame:
    feats = [c for c in df.columns if c.endswith(tuple("1237")) or
             c.endswith(("_mean", "_max", "_spikes")) or
             c in ("w3_d2", "ghi_d2", "hour", "dow", "month")]
    d = df.dropna(subset=feats + [target]).reset_index(drop=True)
    mon = d["ts"].dt.tz_convert(W).dt.strftime("%Y-%m")
    months = sorted(m for m in mon.unique() if m >= TEST_START[:7])
    preds = []
    for m in months:
        tr = d[mon < m]
        te = d[mon == m]
        if len(tr) < 3000 or te.empty:
            continue
        block = te[["ts", target, "quarter"]].copy()
        for tau in QS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = lgb.LGBMRegressor(objective="quantile", alpha=tau, **PARAMS)
                mdl.fit(tr[feats], tr[target])
            block[f"q{tau}"] = mdl.predict(te[feats])
        preds.append(block)
    out = pd.concat(preds, ignore_index=True)
    qcols = [f"q{t}" for t in QS]                 # monotone rearrangement
    out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


def climatology(df: pd.DataFrame, target: str, wf: pd.DataFrame) -> pd.DataFrame:
    """Trailing CLIM_DAYS same-hour empirical quantiles, shifted 1 day (legal)."""
    s = df.set_index("ts")[target]
    out = wf[["ts"]].copy()
    grp = s.groupby(s.index.tz_convert(W).hour)
    clim = {}
    for h, g in grp:
        r = g.rolling(CLIM_DAYS, min_periods=30)
        clim[h] = pd.concat(
            {t: r.quantile(t).shift(1) for t in QS}, axis=1)
    hrs = out["ts"].dt.tz_convert(W).dt.hour
    for t in QS:
        out[f"q{t}"] = [clim[h][t].asof(ts) for h, ts in zip(hrs, out["ts"])]
    return out


def evaluate(wf, clim, d, target):
    y = wf[target].to_numpy()
    res = {"n": len(wf)}
    res["pinball_gbm"] = round(np.mean([pinball(y, wf[f"q{t}"].to_numpy(), t) for t in QS]), 2)
    cl = clim.set_index("ts").reindex(wf["ts"]).reset_index()
    res["pinball_clim"] = round(np.mean([pinball(y, cl[f"q{t}"].to_numpy(), t) for t in QS]), 2)
    res["mae_gbm"] = round(float(np.mean(np.abs(y - wf["q0.5"]))), 1)
    naive = d.set_index("ts")[f"{target}_l1"].reindex(wf["ts"]).to_numpy()
    naive7 = d.set_index("ts")[f"{target}_l7"].reindex(wf["ts"]).to_numpy()
    res["mae_naive1d"] = round(float(np.nanmean(np.abs(y - naive))), 1)
    res["mae_naive7d"] = round(float(np.nanmean(np.abs(y - naive7))), 1)
    res["cover_10_90"] = round(float(np.mean((y >= wf["q0.1"]) & (y <= wf["q0.9"]))), 3)

    byq = {}
    for q, g in wf.groupby("quarter"):
        yq = g[target].to_numpy()
        nq = d.set_index("ts")[f"{target}_l1"].reindex(g["ts"]).to_numpy()
        byq[q] = {"mae_gbm": round(float(np.mean(np.abs(yq - g["q0.5"]))), 1),
                  "mae_naive1d": round(float(np.nanmean(np.abs(yq - nq))), 1),
                  "pinball_gbm": round(np.mean(
                      [pinball(yq, g[f"q{t}"].to_numpy(), t) for t in QS]), 2)}
    res["by_quarter"] = byq

    # clearing capture: bid at forecast quantile, paid CMBP iff CMBP >= bid
    full = float(np.mean(y))
    res["capture"] = {}
    for t in (0.1, 0.25, 0.5):
        bid = wf[f"q{t}"].to_numpy()
        clr = y >= bid
        res["capture"][f"bid_q{t}"] = {
            "clear_rate": round(float(clr.mean()), 3),
            "revenue_share_vs_bid0": round(float(np.mean(y * clr) / full), 3)}
    return res


def main():
    df = build()
    out = {"gate": "D-1 07:30 Warsaw (CMBP publishes ~09:10 D-1; zmb ~08:00 -> lagged only; "
                   "CEN freshest legal day D-3; no DA anchor)", "targets": {}}
    for target in TARGETS:
        wf = walkforward(df, target)
        cl = climatology(df, target, wf)
        res = evaluate(wf, cl, df, target)
        out["targets"][target] = res
        wf.to_parquet(f"reports/cmbp_wf_{target}.parquet")
        print(f"== {target}: n={res['n']}  pinball GBM {res['pinball_gbm']} vs clim "
              f"{res['pinball_clim']}  |  MAE GBM {res['mae_gbm']} vs naive1d "
              f"{res['mae_naive1d']} / naive7d {res['mae_naive7d']}  |  "
              f"cover10-90 {res['cover_10_90']}")
        for q, r in res["by_quarter"].items():
            print(f"   {q}: MAE {r['mae_gbm']:>6} vs naive {r['mae_naive1d']:>6}  "
                  f"pinball {r['pinball_gbm']}")
        print(f"   capture: {res['capture']}\n")
    pathlib.Path("reports/cmbp_forecast.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/cmbp_forecast.json")


if __name__ == "__main__":
    main()
