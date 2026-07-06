"""
Project A (fallback leg): threshold-rule backtest of the CEN - price_ref
spread on the untouched holdout, from the saved quantile predictions.

    python -m src.backtest_spread

PRICE REFERENCE CAVEAT (flag on every output): price_ref = csdac_pln — the
day-ahead SDAC price — i.e. this tests the DA<->CEN spread, the spec's
fallback (c). It is a PROXY for the tradeable intraday leg: the DA leg is not
actually executable at t-60. Numbers here measure whether the forecast's
tail asymmetry clears realistic costs — the signal's economics, not a
fully implementable strategy. Swap `PRICE_REF` to ida3/rdb index columns
when the TGE/ENTSO-E loader lands; nothing else changes.

Rule (per spec): one decision per quarter-hour, fixed 1 MWh clip.
  long spread  (carry long to CEN)  iff  P25(S) > +cost
  short spread (carry short to CEN) iff  P75(S) < -cost
where S = CEN - price_ref, and the predicted quantiles of S are the CEN
quantiles minus the (gate-known) price_ref. P&L = dir * S_realized - cost.
Costs are half-spread + fees, with the spec's 2x / 4x stress.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

PRICE_REF = "b1_da"                 # csdac day-ahead; IDA columns slot in here
COSTS = (10.0, 20.0, 40.0)          # PLN/MWh: base, 2x, 4x
Q_LO, Q_HI = 0.25, 0.75             # conservative entry quantiles


def max_drawdown(pnl: pd.Series) -> float:
    eq = pnl.cumsum()
    return float((eq - eq.cummax()).min())


def run_rule(df: pd.DataFrame, cost: float) -> pd.DataFrame:
    s_real = df["y"] - df[PRICE_REF]
    s_lo = df[f"gbm_conf_q{Q_LO}"] - df[PRICE_REF]
    s_hi = df[f"gbm_conf_q{Q_HI}"] - df[PRICE_REF]
    direction = pd.Series(0, index=df.index)
    direction[s_lo > cost] = 1
    direction[s_hi < -cost] = -1
    out = df[["ts", "hour", "month"]].copy()
    out["dir"] = direction
    out["s_real"] = s_real
    out["pnl"] = direction * s_real - direction.abs() * cost
    return out


def summarize(trades: pd.DataFrame, cost: float) -> dict:
    t = trades[trades["dir"] != 0]
    if not len(t):
        return {"cost": cost, "n_trades": 0}
    return {
        "cost": cost,
        "n_trades": int(len(t)),
        "share_periods": round(len(t) / len(trades), 4),
        "hit_rate": round(float((t["pnl"] > 0).mean()), 4),
        "pnl_total": round(float(t["pnl"].sum()), 0),
        "pnl_per_mwh": round(float(t["pnl"].mean()), 2),
        "pnl_p5": round(float(t["pnl"].quantile(0.05)), 1),
        "max_drawdown": round(max_drawdown(t["pnl"]), 0),
        "long_share": round(float((t["dir"] == 1).mean()), 3),
    }


def main():
    df = pd.read_parquet("reports/holdout_predictions.parquet")
    df = df.dropna(subset=["y", PRICE_REF]).reset_index(drop=True)
    s_real = df["y"] - df[PRICE_REF]

    print("=" * 72)
    print("SPREAD BACKTEST — price_ref = DAY-AHEAD (csdac): PROXY LEG, see docstring")
    print(f"holdout {df['ts'].min()} .. {df['ts'].max()}  ({len(df)} periods)")
    print("=" * 72)
    print(f"\nrealized S=CEN-DA:  mean={s_real.mean():.1f}  std={s_real.std():.1f}  "
          f"P(S>0)={float((s_real > 0).mean()):.3f}")

    # naive benchmarks: always carry, both directions, no cost (upper bounds)
    print(f"always-long-spread  pnl/MWh = {s_real.mean():7.2f}   (no cost)")
    print(f"always-short-spread pnl/MWh = {-s_real.mean():7.2f}   (no cost)\n")

    report = {"price_ref": "csdac_da_PROXY", "entry_quantiles": [Q_LO, Q_HI],
              "rules": []}
    rows = []
    for cost in COSTS:
        trades = run_rule(df, cost)
        s = summarize(trades, cost)
        report["rules"].append(s)
        rows.append(s)
        if s["n_trades"]:
            t = trades[trades["dir"] != 0]
            by_hour = t.groupby(pd.cut(t["hour"], [0, 6, 10, 14, 18, 22, 24],
                                       right=False), observed=True)["pnl"].agg(["count", "mean"])
            print(f"-- cost {cost:.0f} PLN/MWh: {s['n_trades']} trades "
                  f"({s['share_periods']:.1%} of periods), hit {s['hit_rate']:.1%}, "
                  f"pnl/MWh {s['pnl_per_mwh']:+.1f}, total {s['pnl_total']:+,.0f} PLN, "
                  f"maxDD {s['max_drawdown']:,.0f}")
            print(by_hour.round(1).to_string(), "\n")

    print(pd.DataFrame(rows).to_string(index=False))
    out = pathlib.Path("reports/spread_backtest_daref.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
