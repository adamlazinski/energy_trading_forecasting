"""
CEN tail classifier: P(spike) and P(negative) at the H=60 gate (F31).

    python -m src.spike_classifier

The quantile forecaster is weakest exactly where money and grid stress live —
the tails. Dedicated binary classifiers for the two tail events:

  spike  CEN > 1500 PLN/MWh   (base rate ~0.8% — BESS discharge positioning)
  neg    CEN < 0               (base rate ~7.8% — BESS charge positioning)

Features: the strict leakage-honest panel (features.build, same H=60 gate as
the forecaster) plus the F26 NWP run-to-run revisions — at a 60-min gate even
the freshest run (2-5h lead) is legal, unlike at the IDA3 gate that killed
F26 as a trade. Revisions z-scored on the first 30% (no lookahead).

Model: LightGBM binary, scale_pos_weight, expanding walk-forward with monthly
refits (2025-01 →). Benchmark: trailing-90d hour-of-day climatology, lagged
2 days (CEN publishes D+1 ~14:00 — the same publication gate as everything).
Metrics: AUC, average precision (PR-AUC), Brier vs climatology, top-decile
capture (share of true events inside the top 10% of predicted risk), all by
quarter. Writes reports/spike_classifier.json + wf probs parquet.
"""
from __future__ import annotations

import json
import pathlib
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .lear import load_panel

W = "Europe/Warsaw"
TEST_START = "2025-01"
SPIKE = 1500.0
CLIM_D = 90
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
              verbose=-1)


def add_nwp_revisions(df: pd.DataFrame) -> pd.DataFrame:
    wx = pd.read_parquet("data/raw/weather_prev_runs.parquet")
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    wx = wx.set_index("ts")
    cut = int(len(wx) * 0.3)
    for name, a, b in (("rev_w3_21", "w3_d1", "w3_d2"),
                       ("rev_w3_10", "w3_d0", "w3_d1"),
                       ("rev_ghi_21", "ghi_d1", "ghi_d2"),
                       ("rev_ghi_10", "ghi_d0", "ghi_d1")):
        r = wx[a] - wx[b]
        mu, sd = r.iloc[:cut].mean(), r.iloc[:cut].std()
        wx[name] = (r - mu) / sd
    rev = wx[["rev_w3_21", "rev_w3_10", "rev_ghi_21", "rev_ghi_10"]]
    df["hour_ts"] = df["ts"].dt.floor("1h")
    df = df.merge(rev, left_on="hour_ts", right_index=True, how="left")
    return df.drop(columns=["hour_ts"])


def climatology_prob(df: pd.DataFrame, ycol: str) -> np.ndarray:
    """Trailing CLIM_D-day event rate per hour-of-day, lagged 2 days."""
    loc = df["ts"].dt.tz_convert(W)
    daily = (df.assign(h=loc.dt.hour, d=loc.dt.normalize())
             .groupby(["d", "h"])[ycol].mean().unstack())
    clim = daily.rolling(CLIM_D, min_periods=20).mean().shift(2)
    return clim.stack().rename("clim").reset_index().merge(
        pd.DataFrame({"d": loc.dt.normalize(), "h": loc.dt.hour}),
        on=["d", "h"], how="right")["clim"].to_numpy()


def walkforward(df, feats, ycol):
    mon = df["ts"].dt.tz_convert(W).dt.strftime("%Y-%m")
    months = sorted(m for m in mon.unique() if m >= TEST_START)
    prob = np.full(len(df), np.nan)
    for m in months:
        tr, te = df.index[mon < m], df.index[mon == m]
        ytr = df.loc[tr, ycol]
        if ytr.sum() < 30 or len(te) == 0:
            continue
        spw = float((1 - ytr.mean()) / ytr.mean())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = lgb.LGBMClassifier(scale_pos_weight=spw, **PARAMS)
            mdl.fit(df.loc[tr, feats], ytr)
        prob[te] = mdl.predict_proba(df.loc[te, feats])[:, 1]
    return prob


def evaluate(df, ycol, prob, clim):
    m = ~np.isnan(prob) & ~np.isnan(clim)
    y, p, c = df.loc[m, ycol].to_numpy(), prob[m], clim[m]
    top = p >= np.quantile(p, 0.9)
    res = {"n": int(m.sum()), "base_rate": round(float(y.mean()), 4),
           "auc": round(float(roc_auc_score(y, p)), 3),
           "auc_clim": round(float(roc_auc_score(y, c)), 3),
           "ap": round(float(average_precision_score(y, p)), 3),
           "ap_clim": round(float(average_precision_score(y, c)), 3),
           "brier": round(float(np.mean((p - y) ** 2)), 5),
           "brier_clim": round(float(np.mean((c - y) ** 2)), 5),
           "top_decile_capture": round(float(y[top].sum() / max(y.sum(), 1)), 3),
           "by_quarter": {}}
    qs = df.loc[m, "ts"].dt.tz_convert(W)
    qlab = qs.dt.year.astype(str) + "Q" + qs.dt.quarter.astype(str)
    for q in sorted(qlab.unique()):
        i = (qlab == q).to_numpy()
        if y[i].sum() < 5:
            res["by_quarter"][q] = {"n_events": int(y[i].sum())}
            continue
        res["by_quarter"][q] = {
            "n_events": int(y[i].sum()),
            "auc": round(float(roc_auc_score(y[i], p[i])), 3),
            "ap": round(float(average_precision_score(y[i], p[i])), 3),
            "ap_clim": round(float(average_precision_score(y[i], c[i])), 3)}
    return res


def main():
    df, feats = load_panel()
    df = add_nwp_revisions(df)
    feats = feats + ["rev_w3_21", "rev_w3_10", "rev_ghi_21", "rev_ghi_10"]
    df[feats] = df[feats].astype(float).fillna(0.0)
    df["y_spike"] = (df["y"] > SPIKE).astype(int)
    df["y_neg"] = (df["y"] < 0).astype(int)

    out = {"spike_threshold": SPIKE, "targets": {}}
    wf_cols = {"ts": df["ts"]}
    for ycol in ("y_spike", "y_neg"):
        prob = walkforward(df, feats, ycol)
        clim = climatology_prob(df, ycol)
        res = evaluate(df, ycol, prob, clim)
        out["targets"][ycol] = res
        wf_cols[f"p_{ycol}"] = prob
        print(f"== {ycol}: n={res['n']} base {res['base_rate']:.2%}  "
              f"AUC {res['auc']} (clim {res['auc_clim']})  "
              f"AP {res['ap']} (clim {res['ap_clim']})  "
              f"Brier {res['brier']} (clim {res['brier_clim']})  "
              f"top-decile capture {res['top_decile_capture']:.0%}")
        for q, v in res["by_quarter"].items():
            extra = (f"AUC {v['auc']}  AP {v['ap']} (clim {v['ap_clim']})"
                     if "auc" in v else "too few events")
            print(f"   {q}: events {v['n_events']:>4}  {extra}")
        print()

    pd.DataFrame(wf_cols).to_parquet("reports/spike_wf_probs.parquet")
    pathlib.Path("reports/spike_classifier.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/spike_classifier.json")


if __name__ == "__main__":
    main()
