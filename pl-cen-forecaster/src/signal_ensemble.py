"""
Signal ensemble on the CEN−DA spread (passive balancing): compose several
orthogonal, decorrelated signals and conviction-vote — the Weron
distribution-averaging insight applied to trading (F22).

    python -m src.signal_ensemble

Pool (each a gate-honest sign in {−1,0,+1}, ~2.5h RES/load-actuals latency):
  S1 structural carry  — trailing per-hour mean of CEN−DA (the imbalance risk
                         premium; system is structurally long, F21)
  S2 RES surprise      — realized RES vs day-ahead forecast, persistent (F19)
  S3 load surprise     — realized demand vs forecast (the demand-side twin)
Conviction = sum of signs; trade when |Σ| ≥ threshold. Signals validated
individually first — a 4th (spread-regime momentum) was dropped: it
over-trades for a thin edge and loses to cost either sign, the honest lesson
that not every signal earns its place.

Instrument: CEN−csdac spread. Numbers are GROSS of real frictions
(execution, slippage, the ENTSO-E-actuals substitution) — halve for reality —
and the F19 2026 softening applies to the book.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .res_surprise import load

COST = 20.0
MWH = 0.25          # 1 MW for a 15-min period
LAG = 10            # 2.5h effective gate latency on realized actuals


def build_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["hour"] = loc.dt.hour
    kl = pd.read_parquet("data/raw/pse_kse_load.parquet")[
        ["ts", "load_fcst", "load_actual"]]
    for c in ("load_fcst", "load_actual"):
        kl[c] = pd.to_numeric(kl[c], errors="coerce")
    kl["ls"] = kl["load_actual"] - kl["load_fcst"]
    df = df.merge(kl[["ts", "ls"]], on="ts", how="left")

    def lagsign(col, dead):
        v = df.set_index("ts")[col].shift(LAG).to_numpy()
        return np.where(v > dead, 1, np.where(v < -dead, -1, 0))

    tmean = df.groupby("hour")["cen_move"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=10).mean())
    return {
        "S1_carry": np.nan_to_num(np.sign(tmean.to_numpy())),
        "S2_res": np.nan_to_num(-lagsign("surprise", 300)),   # over-deliver → short
        "S3_load": np.nan_to_num(lagsign("ls", 200)),          # over-demand → long
    }


def perf(df, pos) -> dict:
    pnl = (pos * df["cen_move"] - np.abs(pos) * COST) * MWH
    daily = pnl.groupby(df["ts"].dt.tz_convert("Europe/Warsaw").dt.date).sum().dropna()
    ann = daily.mean() * 365
    sharpe = ann / (daily.std() * np.sqrt(365)) if daily.std() else 0.0
    dd = float((daily.cumsum() - daily.cumsum().cummax()).min())
    mo = pnl.groupby(df["ts"].dt.tz_convert("Europe/Warsaw").dt.to_period("M")).sum()
    return {"ann_pln_per_mw": round(float(ann), 0), "sharpe": round(float(sharpe), 2),
            "max_dd": round(dd, 0), "months_pos": f"{int((mo > 0).sum())}/{len(mo)}",
            "n_trades": int((pos != 0).sum())}


def main():
    df = load().dropna(subset=["cen_move"]).sort_values("ts").reset_index(drop=True)
    sigs = build_signals(df)
    for k, v in sigs.items():
        df[k] = v

    out = {"cost": COST, "lag_periods": LAG, "singles": {}, "ensemble": {},
           "orthogonality": pd.DataFrame(sigs).corr().round(2).to_dict()}
    print("single signals (cost 20, gate-honest):")
    for k in sigs:
        out["singles"][k] = perf(df, df[k].to_numpy())
        r = out["singles"][k]
        print(f"  {k:10} ann {r['ann_pln_per_mw']:>8,.0f}  Sharpe {r['sharpe']:>5}  "
              f"maxDD {r['max_dd']:>9,.0f}  months+ {r['months_pos']}")

    score = sum(df[k].to_numpy() for k in sigs)
    print("\nconviction-stacked ensemble:")
    for k, lbl in ((1, ">=1 any"), (2, ">=2 majority"), (3, "=3 all-agree")):
        pos = np.where(score >= k, 1, np.where(score <= -k, -1, 0))
        out["ensemble"][lbl] = perf(df, pos)
        r = out["ensemble"][lbl]
        print(f"  {lbl:12} ann {r['ann_pln_per_mw']:>8,.0f}  Sharpe {r['sharpe']:>5}  "
              f"maxDD {r['max_dd']:>9,.0f}  months+ {r['months_pos']}  n {r['n_trades']}")

    pathlib.Path("reports/signal_ensemble.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/signal_ensemble.json")


if __name__ == "__main__":
    main()
