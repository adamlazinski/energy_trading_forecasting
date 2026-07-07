"""
Intraday opportunity sizing for a battery: how much price spread can a
1 MW / 2 MWh unit harvest by trading the TGE intraday market, and does that
beat trading day-ahead only?  (First cut before paying for order-book data.)

    python -m src.intraday_mm

We have per-15-min CLEARING prices, not the order book, so this measures the
*arbitrage* a battery can extract from the intraday price curve (a lower
bound on the market-making opportunity — true bid/ask spread capture needs
tick data and is upside on top). Three executable price curves per period:
  da   csdac_pln              (day-ahead, known D-1)
  id   TGE intraday           (rdb_vwap, else ida3/ida2/ida1)
  cen  CEN                    (balancing settlement, the fallback leg)

For each curve we run the same battery arbitrage:
  - PF   perfect-foresight SoC DP (the daily ceiling)
  - THR  causal threshold policy: charge when price < that day's rolling P25,
         discharge when > P75 (a deployable, no-lookahead estimate)
net of round-trip efficiency and degradation. The intraday-minus-day-ahead
gap is the value of trading intraday at all; the MM spread (RDB intra-period
range) is reported as the additional, tick-data-gated upside.

Writes reports/intraday_mm.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .bess_optimizer import C_DEG, E_MWH, ETA, P_MW

DT = 0.25
DE = P_MW * DT                      # MWh grid exchange per period
DDIS = DE / ETA                     # SoC drawn per discharge
DCHG = DE * ETA                     # SoC gained per charge
N_SOC = 21
SOC = np.linspace(0.0, E_MWH, N_SOC)


def arb_pf(price: np.ndarray) -> float:
    """Perfect-foresight daily arbitrage value, start empty (leftover free)."""
    T = len(price)
    NEG = -1e9
    V = np.zeros(N_SOC)                              # terminal: leftover worth 0
    for t in range(T - 1, -1, -1):
        p = price[t]
        # discharge: SoC s -> s-DDIS, gain (p-cdeg)*DE  (needs s >= DDIS)
        nxt_d = SOC - DDIS
        val_dis = np.where(nxt_d >= -1e-9,
                           (p - C_DEG) * DE + np.interp(nxt_d, SOC, V), NEG)
        # charge: SoC s -> s+DCHG, cost p*DE  (needs s+DCHG <= E)
        nxt_c = SOC + DCHG
        val_chg = np.where(nxt_c <= E_MWH + 1e-9,
                           -p * DE + np.interp(nxt_c, SOC, V), NEG)
        V = np.maximum.reduce([V, val_dis, val_chg])
    return float(V[0])                               # start empty


def arb_threshold(price: np.ndarray, lo: float, hi: float) -> float:
    """Causal policy: charge below lo, discharge above hi, SoC-constrained."""
    s, pnl = 0.0, 0.0
    for p in price:
        if p > hi and s >= DDIS - 1e-9:
            pnl += (p - C_DEG) * DE
            s -= DDIS
        elif p < lo and s <= E_MWH - DCHG + 1e-9:
            pnl -= p * DE
            s += DCHG
    return pnl


def load() -> pd.DataFrame:
    da = pd.read_parquet("data/raw/pse_da_price.parquet")[["ts", "csdac_pln"]]
    da["da"] = pd.to_numeric(da["csdac_pln"], errors="coerce")
    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "cen_cost"]]
    crb["cen"] = pd.to_numeric(crb["cen_cost"], errors="coerce")
    t = pd.read_parquet("data/raw/tge_rdb.parquet")
    t = t[t.dur_min == 15].drop_duplicates("ts")
    for c in ("rdb_vwap", "ida3_pln", "ida2_pln", "ida1_pln", "rdb_min", "rdb_max"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["id"] = (t["rdb_vwap"].fillna(t["ida3_pln"]).fillna(t["ida2_pln"])
               .fillna(t["ida1_pln"]))
    t["id_range"] = t["rdb_max"] - t["rdb_min"]
    df = (da[["ts", "da"]].merge(crb[["ts", "cen"]], on="ts", how="outer")
          .merge(t[["ts", "id", "id_range"]], on="ts", how="outer"))
    df["day"] = df["ts"].dt.tz_convert("Europe/Warsaw").dt.date
    return df.sort_values("ts")


def main():
    df = load()
    rows = {"da": [], "id": [], "cen": []}
    mm_capture = []
    for _, g in df.groupby("day", sort=True):
        if len(g) < 90:
            continue
        for col in ("da", "id", "cen"):
            p = g[col].to_numpy()
            if np.isnan(p).mean() > 0.1:
                continue
            p = pd.Series(p).interpolate(limit_direction="both").to_numpy()
            lo, hi = np.nanpercentile(p, 25), np.nanpercentile(p, 75)
            rows[col].append({"pf": arb_pf(p), "thr": arb_threshold(p, lo, hi)})
        # MM proxy: half the intra-period RDB range, on the ~cycles/day we run
        r = g["id_range"].dropna()
        if len(r):
            mm_capture.append(float(r.median()) * 0.5 * 2.0)   # ~2 cycles/day

    res = {}
    for col, v in rows.items():
        if not v:
            continue
        pf = np.array([x["pf"] for x in v])
        thr = np.array([x["thr"] for x in v])
        res[col] = {
            "n_days": len(v),
            "pf_pln_per_day": round(float(pf.mean()), 1),
            "thr_pln_per_day": round(float(thr.mean()), 1),
            "thr_ann_per_mw": round(float(thr.mean() * 365), 0),
        }
    intraday_premium = (res["id"]["thr_pln_per_day"] - res["da"]["thr_pln_per_day"]
                        if "id" in res and "da" in res else None)
    out = {
        "battery": f"{P_MW} MW / {E_MWH} MWh, eta_rt {ETA**2:.2f}, c_deg {C_DEG}",
        "arbitrage_by_curve": res,
        "intraday_minus_dayahead_thr_pln_per_day": round(intraday_premium, 1)
        if intraday_premium is not None else None,
        "mm_spread_upside_pln_per_day_proxy": round(float(np.mean(mm_capture)), 1)
        if mm_capture else None,
        "note": "arb = SoC DP on each cleared-price curve (lower bound on MM); "
                "mm proxy = 0.5 x median RDB intra-period range x ~2 cycles "
                "(needs order-book data to realize).",
    }
    pathlib.Path("reports/intraday_mm.json").write_text(json.dumps(out, indent=2))
    print(f"battery {out['battery']}\n")
    print(f"{'price curve':14s} {'PF /day':>9s} {'THR /day':>9s} {'THR ann/MW':>12s}")
    for col, r in res.items():
        print(f"{col:14s} {r['pf_pln_per_day']:9.0f} {r['thr_pln_per_day']:9.0f} "
              f"{r['thr_ann_per_mw']:12,.0f}")
    print(f"\nintraday over day-ahead (THR): "
          f"{out['intraday_minus_dayahead_thr_pln_per_day']:+.0f} PLN/day")
    print(f"MM spread upside (proxy, needs tick data): "
          f"{out['mm_spread_upside_pln_per_day_proxy']:.0f} PLN/day")
    print("\nwrote reports/intraday_mm.json")


if __name__ == "__main__":
    main()
