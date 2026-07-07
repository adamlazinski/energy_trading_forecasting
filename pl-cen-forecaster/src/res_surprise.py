"""
Intraday-weather gate: does the RES SURPRISE move prices?

    python -m src.res_surprise

F10/F15 killed the day-ahead RES *forecast* as a feature — it's in the DA
price. But the forecast ERROR (realized − day-ahead forecast) is by
construction NOT in the DA price, and it is revealed through the intraday
session as live forecasts update. This tests the necessary precondition for
any intraday-weather trade: does that surprise actually drive the intraday /
balancing price move away from day-ahead?

  surprise = (realized_wind − consensus_wind) + (realized_solar − cons_solar)
             [MW; consensus = ENTSO-E day-ahead RES forecast]
  id_move  = intraday_price − day_ahead_price     (TGE blend − csdac)
  cen_move = CEN − day_ahead_price                (balancing − csdac)

We expect a NEGATIVE relation: more RES than expected ⇒ system longer ⇒
prices fall below day-ahead. Slope = PLN/MWh per GW of surprise = the prize a
faster live-forecast read would capture. (This uses realized RES, so it is a
mechanism/ceiling test — the tradeable version must forecast the surprise
early; see notes.)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd


def load() -> pd.DataFrame:
    act = pd.read_parquet("data/raw/pse_kse_actuals.parquet")[["ts", "pv", "wi"]]
    cons = pd.read_parquet("data/raw/entsoe_res.parquet")   # DA forecast
    da = pd.read_parquet("data/raw/pse_da_price.parquet")[["ts", "csdac_pln"]]
    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "cen_cost"]]
    tge = pd.read_parquet("data/raw/tge_rdb.parquet")
    tge = tge[tge.dur_min == 15].drop_duplicates("ts")
    for c in ("rdb_vwap", "ida3_pln", "ida2_pln", "ida1_pln"):
        tge[c] = pd.to_numeric(tge[c], errors="coerce")
    tge["id"] = (tge["rdb_vwap"].fillna(tge["ida3_pln"])
                 .fillna(tge["ida2_pln"]).fillna(tge["ida1_pln"]))

    df = (act.merge(cons, on="ts", how="inner")
          .merge(da, on="ts", how="inner")
          .merge(crb, on="ts", how="left")
          .merge(tge[["ts", "id"]], on="ts", how="left"))
    for c in ("pv", "wi", "res_solar", "res_wind", "csdac_pln", "cen_cost", "id"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["surprise"] = ((df["wi"] - df["res_wind"]) + (df["pv"] - df["res_solar"]))
    df["id_move"] = df["id"] - df["csdac_pln"]
    df["cen_move"] = df["cen_cost"] - df["csdac_pln"]
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["quarter"] = loc.dt.year.astype(str) + "Q" + loc.dt.quarter.astype(str)
    return df


def _fit(x: pd.Series, y: pd.Series) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 200:
        return {}
    xv, yv = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    b1, b0 = np.polyfit(xv, yv, 1)
    r = float(np.corrcoef(xv, yv)[0, 1])
    # hit rate: does sign(surprise) predict sign(move)? (expect opposite)
    hit = float(np.mean(np.sign(xv) != np.sign(yv)))
    return {"n": len(d), "corr": round(r, 3),
            "slope_pln_per_gw": round(b1 * 1000, 1),   # per GW
            "sign_hit_rate": round(hit, 3)}


def backtest(df: pd.DataFrame, lag_periods: int, cost: float,
             deadband_mw: float = 300.0) -> dict:
    """Gate-honest signal: at the gate for period t, use the RES surprise from
    `lag_periods`×15min earlier (bounded by RES-actuals latency; ENTSO-E
    actual generation is ~1.2h, so lag≈9-10 periods is realistic). RES
    over-delivering ⇒ expect CEN < DA ⇒ short the CEN−DA spread.
    """
    d = df.dropna(subset=["surprise", "cen_move"]).sort_values("ts").copy()
    sig = d.set_index("ts")["surprise"].shift(lag_periods).to_numpy()
    d = d.assign(sig=sig).dropna(subset=["sig"])
    pos = np.where(d["sig"] > deadband_mw, -1,
                   np.where(d["sig"] < -deadband_mw, 1, 0))
    pnl = pos * d["cen_move"].to_numpy() - np.abs(pos) * cost
    tr = pos != 0
    byq = (pd.Series(pnl[tr], index=d["quarter"].to_numpy()[tr])
           .groupby(level=0).mean().round(1).to_dict())
    return {"lag_periods": lag_periods, "cost": cost, "n_trades": int(tr.sum()),
            "pnl_per_trade": round(float(pnl[tr].mean()), 1),
            "hit_rate": round(float((pnl[tr] > 0).mean()), 3),
            "total_pnl": round(float(pnl.sum()), 0),
            "quarters_positive": f"{sum(v > 0 for v in byq.values())}/{len(byq)}",
            "by_quarter": byq}


def main():
    df = load()
    print(f"{len(df)} periods, {df['ts'].min()} .. {df['ts'].max()}")
    print(f"surprise (MW): mean {df['surprise'].mean():.0f}  std {df['surprise'].std():.0f}  "
          f"P10 {df['surprise'].quantile(.1):.0f}  P90 {df['surprise'].quantile(.9):.0f}\n")

    out = {}
    for tgt in ("id_move", "cen_move"):
        overall = _fit(df["surprise"], df[tgt])
        by_q = {}
        for q, g in df.groupby("quarter"):
            r = _fit(g["surprise"], g[tgt])
            if r:
                by_q[q] = {"corr": r["corr"], "slope": r["slope_pln_per_gw"]}
        out[tgt] = {"overall": overall, "by_quarter": by_q}
        print(f"== surprise -> {tgt} ==")
        print(f"  overall: {overall}")
        print("  by quarter (corr / slope PLN·MWh⁻¹ per GW):")
        for q, r in by_q.items():
            print(f"    {q}: corr {r['corr']:+.3f}  slope {r['slope']:+.1f}")
        print()

    # gate-honest backtest at realistic ENTSO-E latency (~2h ≈ 8 periods)
    print("== gate-honest spread backtest (short CEN−DA when RES over-delivers) ==")
    bt = {}
    for lag, lbl in ((8, "2h"), (10, "2.5h"), (16, "4h")):
        for cost in (20.0, 40.0):
            r = backtest(df, lag, cost)
            bt[f"{lbl}_cost{int(cost)}"] = r
            print(f"  lag {lbl:>4} cost {cost:>4.0f}: pnl/trade {r['pnl_per_trade']:+6.1f}  "
                  f"hit {r['hit_rate']}  quarters+ {r['quarters_positive']}  "
                  f"total {r['total_pnl']:+,.0f}")
    out["backtest"] = bt
    pathlib.Path("reports/res_surprise.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/res_surprise.json")


if __name__ == "__main__":
    main()
