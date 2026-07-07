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

    # NB: the F21 "calendar carry" (sign of trailing per-hour cen_move) was
    # DROPPED after the F23 leakage audit — with honest timing (shift(2); CEN
    # publishes D+1 ~14:00 so D-2 is the freshest gate-available day) it is a
    # losing signal (Sharpe -0.56). The pool is the two clean, gate-honest
    # signals below (realized RES/load lagged 2.5h, never cen_move as input).
    return {
        "S2_res": np.nan_to_num(-lagsign("surprise", 300)),   # over-deliver → short
        "S3_load": np.nan_to_num(lagsign("ls", 200)),          # over-demand → long
    }


def slepaczuk_metrics(df, pos, capital: float = 2_000_000.0, N: int = 365) -> dict:
    """Ślepaczuk-style performance suite (WNE UW algo-trading methodology).

    ARC  annualized return compounded; ASD annualized std of returns;
    MD   max drawdown (%); MLD max loss duration (years underwater);
    IR*  = ARC/ASD  (== the P&L Sharpe with rf=0); the primary ratio.
    IR** = sign(ARC)·ARC²/(ASD·MD)  — the drawdown-adjusted quality ratio.
    ARC/ASD/MD scale with `capital` (a passive-balancing strategy has no NAV,
    so a notional is assumed); IR* is scale-robust and IR** converges to the
    C-free value ARC-equiv ann_PnL²/(ASD_PnL·MD_PnL) in the large-C regime.
    """
    pnl = (pos * df["cen_move"] - np.abs(pos) * COST) * MWH
    day = df["ts"].dt.tz_convert("Europe/Warsaw").dt.normalize()
    daily = pnl.groupby(day).sum()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D",
                        tz="Europe/Warsaw")
    daily = daily.reindex(idx, fill_value=0.0)
    n = len(daily)
    eq = capital + daily.cumsum()
    r = eq.pct_change().dropna()
    arc = (eq.iloc[-1] / capital) ** (N / n) - 1
    asd = float(r.std() * np.sqrt(N))
    peak = eq.cummax()
    md = float(-((eq - peak) / peak).min())
    under = (eq < peak).astype(int).to_numpy()
    mld = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        mld = max(mld, cur)
    ir1 = arc / asd if asd else 0.0
    ir2 = np.sign(arc) * arc ** 2 / (asd * md) if (asd and md) else 0.0
    return {"ARC_pct": round(arc * 100, 2), "ASD_pct": round(asd * 100, 2),
            "MD_pct": round(md * 100, 2), "MLD_yr": round(mld / N, 2),
            "IR1": round(float(ir1), 2), "IR2": round(float(ir2), 1)}


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
    print("\nconviction-stacked ensemble (2 clean signals):")
    for k, lbl in ((1, ">=1 any"), (2, "=2 both-agree")):
        pos = np.where(score >= k, 1, np.where(score <= -k, -1, 0))
        out["ensemble"][lbl] = perf(df, pos)
        r = out["ensemble"][lbl]
        print(f"  {lbl:12} ann {r['ann_pln_per_mw']:>8,.0f}  Sharpe {r['sharpe']:>5}  "
              f"maxDD {r['max_dd']:>9,.0f}  months+ {r['months_pos']}  n {r['n_trades']}")

    print("\nŚlepaczuk metric suite (C=2M PLN/MW notional; IR* == P&L Sharpe):")
    print(f"  {'strategy':>14}{'IR*':>7}{'IR**':>8}{'ARC%':>7}{'ASD%':>7}{'MD%':>6}{'MLD':>6}")
    out["slepaczuk"] = {}
    for name, pos in (*((k, df[k].to_numpy()) for k in sigs),
                      ("majority", np.where(score >= 2, 1, np.where(score <= -2, -1, 0))),
                      ("any", np.where(score >= 1, 1, np.where(score <= -1, -1, 0)))):
        m = slepaczuk_metrics(df, pos)
        out["slepaczuk"][name] = m
        print(f"  {name:>14}{m['IR1']:>7}{m['IR2']:>8}{m['ARC_pct']:>7}{m['ASD_pct']:>7}"
              f"{m['MD_pct']:>6}{m['MLD_yr']:>6}")

    pathlib.Path("reports/signal_ensemble.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/signal_ensemble.json")


if __name__ == "__main__":
    main()
