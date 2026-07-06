"""
Project A with the REAL intraday leg: threshold rule on S = CEN - P_id where
P_id is a TGE intraday price (IDA auctions / RDB continuous VWAP), evaluated
on the walk-forward OOS quantile forecasts of CEN.

    python -m src.backtest_spread_ida

Rule per 15-min period, decided at the H=60 gate:
  long  S (buy id, settle at CEN)  iff P25(CEN) - p_id > cost
  short S (sell id, settle at CEN) iff P75(CEN) - p_id < -cost
Gate-honesty of each leg:
  ida1/ida2 clear D-1 (after csdac; both known at any same-day gate)  - HONEST
  ida3 clears D ~10:30 for afternoon delivery                         - HONEST
    for periods starting >= 12:00 local (filtered accordingly)
  rdb_vwap includes trades after the gate                             - PROXY

Writes reports/spread_backtest_ida.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

COSTS = (10.0, 20.0, 40.0)      # PLN/MWh round-trip cost stress
Q_LO, Q_HI = 0.25, 0.75

WF = pathlib.Path("reports/walkforward_predictions.parquet")
TGE = pathlib.Path("data/raw/tge_rdb.parquet")


def load_legs() -> pd.DataFrame:
    tge = pd.read_parquet(TGE)
    q = tge[tge["dur_min"] == 15].set_index("ts")
    h = tge[tge["dur_min"] == 60].set_index("ts")
    idx = q.index.union(h.index.repeat(4) + pd.to_timedelta(
        np.tile([0, 15, 30, 45], len(h)), unit="m")).unique().sort_values()
    out = pd.DataFrame(index=idx)
    # quarter-hour value where present, else the covering hourly instrument
    for col in ("ida1_pln", "ida2_pln", "ida3_pln", "rdb_vwap"):
        v = q[col].reindex(idx)
        hh = h[col].reindex(idx.floor("h"))
        out[col] = v.to_numpy()
        out[col] = out[col].fillna(pd.Series(hh.to_numpy(), index=idx))
    return out.reset_index().rename(columns={"index": "ts"})


def run_rule(df: pd.DataFrame, leg: str, cost: float) -> pd.DataFrame:
    p = df[leg]
    lo, hi = df[f"q{Q_LO}"], df[f"q{Q_HI}"]
    pos = np.where(lo - p > cost, 1, np.where(hi - p < -cost, -1, 0))
    pnl = pos * (df["y"] - p) - np.abs(pos) * cost
    return df.assign(pos=pos, pnl=pnl)


def summarize(bt: pd.DataFrame) -> dict:
    tr = bt[bt["pos"] != 0]
    if tr.empty:
        return {"n_trades": 0}
    daily = tr.groupby(tr["ts"].dt.tz_convert("Europe/Warsaw").dt.date)["pnl"].sum()
    top = daily.nlargest(5).sum() / daily.sum() if daily.sum() > 0 else np.nan
    eq = daily.cumsum()
    dd = float((eq - eq.cummax()).min())
    return {
        "n_trades": int(len(tr)),
        "share_periods": round(float(len(tr) / len(bt)), 3),
        "hit_rate": round(float((tr["pnl"] > 0).mean()), 3),
        "pnl_per_trade": round(float(tr["pnl"].mean()), 2),
        "pnl_per_quarter_of_year": round(float(tr["pnl"].sum()
                                         / max(len(daily) / 91.25, 1e-9)), 0),
        "top5day_share": round(float(top), 3) if top == top else None,
        "max_drawdown": round(dd, 0),
    }


def main():
    wf = pd.read_parquet(WF).rename(columns={
        f"gbm_conf_q{q}": f"q{q}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
    legs = load_legs()
    df = wf.merge(legs, on="ts", how="inner").dropna(subset=["y"])
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["quarter"] = loc.dt.year.astype(str) + "Q" + loc.dt.quarter.astype(str)
    afternoon = loc.dt.hour >= 12

    res: dict = {"n_joint": int(len(df)),
                 "span": f"{df['ts'].min()} .. {df['ts'].max()}", "legs": {}}
    for leg, honest, mask in (
            ("ida1_pln", "honest_D-1", None),
            ("ida2_pln", "honest_D-1", None),
            ("ida3_pln", "honest_same-day-afternoon", afternoon),
            ("rdb_vwap", "PROXY_includes_post-gate_trades", None)):
        sub = df[mask] if mask is not None else df
        sub = sub.dropna(subset=[leg])
        if len(sub) < 500:
            res["legs"][leg] = {"note": f"only {len(sub)} joint periods"}
            continue
        entry = {"gate": honest, "n": int(len(sub)),
                 "coverage_of_wf": round(float(len(sub) / len(df)), 3),
                 "mean_S": round(float((sub["y"] - sub[leg]).mean()), 2),
                 "by_cost": {}}
        for cost in COSTS:
            bt = run_rule(sub, leg, cost)
            s = summarize(bt)
            s["by_quarter_pnl"] = {
                g: round(float(v), 0) for g, v in
                bt[bt["pos"] != 0].groupby("quarter")["pnl"].sum().items()}
            entry["by_cost"][str(cost)] = s
        res["legs"][leg] = entry

    pathlib.Path("reports/spread_backtest_ida.json").write_text(
        json.dumps(res, indent=2))
    for leg, e in res["legs"].items():
        print(f"\n== {leg} ({e.get('gate','-')}), n={e.get('n')} "
              f"mean_S={e.get('mean_S')}")
        for cost, s in e.get("by_cost", {}).items():
            print(f"  cost {cost:>5s}: trades={s['n_trades']:6d} "
                  f"hit={s.get('hit_rate')} pnl/trade={s.get('pnl_per_trade')} "
                  f"pnl/qtr={s.get('pnl_per_quarter_of_year')} "
                  f"top5day={s.get('top5day_share')}")
            print(f"    by quarter: {s.get('by_quarter_pnl')}")
    print("\nwrote reports/spread_backtest_ida.json")


if __name__ == "__main__":
    main()
