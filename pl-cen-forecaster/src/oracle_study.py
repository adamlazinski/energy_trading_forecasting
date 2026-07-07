"""
ORACLE / CEILING STUDY (idea: "if we knew the exact conditions, how much
would CEN improve?"). LEAKAGE-VIOLATING BY DESIGN — every ora_ feature is a
REALIZED actual, known only after delivery. This is an upper-bound
attribution, never a deployable model.

    python -m src.oracle_study

Ladder of feature sets on the fixed 8-week holdout (GBM + hour-conformal,
same recipe as src.evaluate), reported against the strict baseline (~68.5)
and PSE's final-vintage forecast (27.5, the practical info ceiling):

  strict           the deployable leakage-honest panel (fx_/cen_/imb_/cal)
  + perfect RES    realized PV + wind (kse_actuals pv, wi)
  + perfect load   realized demand (kse_load load_actual)
  + perfect netload realized (load - PV - wind), the residual-demand truth
  + perfect tight  realized reserve margins + non-activated gen
                   (kse_snapshot rez_under, rez_over_demand, gen_not_activ_part)
  + all oracle     everything above

The gaps between rungs attribute the forecastable-with-fundamentals part of
CEN to RES / load / system-tightness, and bound what a perfect near-delivery
nowcast could buy over the day-ahead anchor (cf. F10: day-ahead RES is
already in the DA price).
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


def _pinball(y, pred) -> float:
    tot = 0.0
    for q in QUANTILES:
        d = y - pred[q].to_numpy()
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(QUANTILES)


def oracle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach realized-actual (leakage-violating) columns, ora_ prefixed."""
    act = pd.read_parquet("data/raw/pse_kse_actuals.parquet")[
        ["ts", "pv", "wi", "demand", "swm_p", "swm_np"]]
    snap = pd.read_parquet("data/raw/pse_kse_snapshot.parquet")[
        ["ts", "rez_under", "rez_over_demand", "gen_not_activ_part"]]
    load = pd.read_parquet("data/raw/pse_kse_load.parquet")[["ts", "load_actual"]]
    o = act.merge(snap, on="ts", how="outer").merge(load, on="ts", how="outer")
    for c in o.columns:
        if c != "ts":
            o[c] = pd.to_numeric(o[c], errors="coerce")
    df = df.merge(o, on="ts", how="left")
    df["ora_pv"] = df["pv"]
    df["ora_wi"] = df["wi"]
    df["ora_res"] = df["pv"].fillna(0) + df["wi"].fillna(0)
    df["ora_load"] = df["load_actual"].fillna(df["demand"])
    df["ora_net_load"] = df["ora_load"] - df["ora_res"]
    df["ora_rez_under"] = df["rez_under"]
    df["ora_rez_over"] = df["rez_over_demand"]
    df["ora_gen_slack"] = df["gen_not_activ_part"]
    df["ora_xborder"] = df["swm_p"].fillna(0) + df["swm_np"].fillna(0)
    return df


def fit_eval(train, cal, hold, feats) -> dict:
    gbm = models.QuantileGBM().fit(train[feats], train["y"])
    off = conformal.fit_offsets(cal["y"], gbm.predict(cal[feats]),
                                groups=conformal.hour_block(cal["hour"]))
    p = conformal.apply_offsets(gbm.predict(hold[feats]), off,
                                groups=conformal.hour_block(hold["hour"]))
    y = hold["y"].to_numpy()
    return {
        "mean_pinball": round(_pinball(y, p), 2),
        "cov_P10_90": round(float(np.mean((y >= p[0.10]) & (y <= p[0.90]))), 3),
        "n_feats": len(feats),
    }


def main():
    cfg = load_cfg()
    panel = pd.read_parquet("data/proc/panel_15min.parquet")
    df = features.build(panel, cfg).dropna(subset=["y"]).reset_index(drop=True)
    df = oracle_columns(df)
    base = features.feature_cols(df)

    ladder = {
        "strict": base,
        "+perfect_RES": base + ["ora_pv", "ora_wi"],
        "+perfect_load": base + ["ora_load"],
        "+perfect_netload": base + ["ora_net_load"],
        "+perfect_tight": base + ["ora_rez_under", "ora_rez_over", "ora_gen_slack"],
        "+perfect_xborder": base + ["ora_xborder"],
        "+all_oracle": base + ["ora_pv", "ora_wi", "ora_load", "ora_net_load",
                               "ora_rez_under", "ora_rez_over", "ora_gen_slack",
                               "ora_xborder"],
    }

    ts = df["ts"]
    cut_hold = ts.max() - pd.Timedelta(weeks=cfg["test"]["weeks"])
    cut_cal = cut_hold - pd.Timedelta(weeks=CAL_WEEKS)
    embargo = pd.Timedelta(minutes=cfg["horizon_minutes"])
    train = df[ts < cut_cal - embargo]
    cal = df[(ts >= cut_cal) & (ts < cut_hold - embargo)]
    hold = df[ts >= cut_hold]

    res = {}
    print(f"train {len(train)}, cal {len(cal)}, holdout {len(hold)}\n")
    print(f"{'feature set':20s} {'pinball':>8s} {'cov10-90':>9s}  vs strict")
    strict_pb = None
    for name, feats in ladder.items():
        r = fit_eval(train, cal, hold, feats)
        res[name] = r
        if strict_pb is None:
            strict_pb = r["mean_pinball"]
        delta = r["mean_pinball"] - strict_pb
        print(f"{name:20s} {r['mean_pinball']:8.2f} {r['cov_P10_90']:9.3f}  "
              f"{delta:+6.2f}")
    print(f"\n(PSE final-vintage forecast = 27.47 — the practical info ceiling)")

    out = {"baseline_strict": strict_pb, "pse_final_vintage": 27.47,
           "ladder": res,
           "note": "LEAKAGE-VIOLATING oracle study; ora_ = realized actuals"}
    pathlib.Path("reports/oracle_study.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/oracle_study.json")


if __name__ == "__main__":
    main()
