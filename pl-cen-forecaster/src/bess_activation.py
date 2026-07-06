"""
Project B, Layer 1 (v1): an empirical model of PSE's dispatcher.

    python -m src.bess_activation

For a battery submitting balancing-energy offers into ZPG/RBN, revenue is
gated by ACTIVATION. Source: data/raw/pse_bpkdbo_marginals.parquet
(src.pull_bpkdbo) — per 15-min period the dispatcher's net direction
(dir: G=up/delivery, D=down/withdrawal) and the marginal activated price
`marg` (oeb-bpkdbo merit ladder crossed at the eb-rozl activated volume;
corr 0.88 with CEN). Offers are committed D-1, so any later conditional
model may only use D-1-known features (the fx_ set).

Deliverable (marginal-pricing approximation):
  pi_up(p | block)  = P(dir=G and marg >= p | block)   — discharge offer at p
  pi_dn(p | block)  = P(dir=D and marg <= p | block)   — charge offer at p
plus marginal-price distributions by block and by quarter (regime check
across the 2025-09-30 SDAC 15-min reform).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

BLOCKS = [0, 6, 10, 14, 18, 22, 24]
BLOCK_LABELS = ["night", "am_ramp", "midday", "pm", "ev_ramp", "late"]
PRICE_GRID_UP = np.arange(0, 2001, 50)        # PLN/MWh discharge offers
PRICE_GRID_DN = np.arange(-500, 1001, 50)     # PLN/MWh charge offers

MARG = pathlib.Path("data/raw/pse_bpkdbo_marginals.parquet")


def load() -> pd.DataFrame:
    df = pd.read_parquet(MARG)
    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "cen_cost"]]
    da = pd.read_parquet("data/raw/pse_da_price.parquet")[["ts", "csdac_pln"]]
    df = df.merge(crb, on="ts", how="left").merge(da, on="ts", how="left")
    for c in ("cen_cost", "csdac_pln"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["hour"] = loc.dt.hour
    df["block"] = pd.cut(df["hour"], BLOCKS, right=False, labels=BLOCK_LABELS)
    df["quarter"] = loc.dt.to_period("Q").astype(str)
    return df.sort_values("ts").reset_index(drop=True)


def activation_curves(df: pd.DataFrame) -> dict:
    """pi(p | block): unconditional on direction — the D-1 bidder does not
    know the direction, so P(dir) x P(marg clears p | dir) is the object."""
    out = {"up": {}, "down": {}}
    for blk, g in df.groupby("block", observed=True):
        up = g[g["dir"] == "G"]["marg"].dropna()
        dn = g[g["dir"] == "D"]["marg"].dropna()
        p_up, p_dn = len(up) / len(g), len(dn) / len(g)
        out["up"][str(blk)] = {
            "p_dir": round(p_up, 4),
            "curve": {int(p): round(p_up * float((up >= p).mean()), 4)
                      for p in PRICE_GRID_UP} if len(up) else {},
        }
        out["down"][str(blk)] = {
            "p_dir": round(p_dn, 4),
            "curve": {int(p): round(p_dn * float((dn <= p).mean()), 4)
                      for p in PRICE_GRID_DN} if len(dn) else {},
        }
    return out


def main():
    df = load()
    print(f"panel {len(df)} periods, {df['ts'].min()} .. {df['ts'].max()}, "
          f"marg nulls {df['marg'].isna().mean():.1%}, "
          f"saturated {df['saturated'].mean():.1%}\n")

    print("== net dispatch direction by block ==")
    print(pd.crosstab(df["block"], df["dir"], normalize="index").round(3), "\n")

    print("== marginal activated price by block x direction ==")
    t = (df.groupby(["dir", "block"], observed=True)["marg"]
           .describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
           .round(0))
    print(t.to_string(), "\n")

    print("== quarterly evolution (up-direction marginal, median / P90) ==")
    qt = (df[df["dir"] == "G"].groupby("quarter")["marg"]
            .agg(n="size", p50="median",
                 p90=lambda s: s.quantile(0.9)).round(0))
    print(qt.to_string(), "\n")

    print("== implied BESS spread: up-marg minus down-marg, daily medians ==")
    day = df["ts"].dt.tz_convert("Europe/Warsaw").dt.date
    du = df[df["dir"] == "G"].groupby(day)["marg"].median()
    dd = df[df["dir"] == "D"].groupby(day)["marg"].median()
    sp = (du - dd).dropna()
    print(f"n_days={len(sp)}  median={sp.median():.0f}  "
          f"P10={sp.quantile(.1):.0f}  P90={sp.quantile(.9):.0f} PLN/MWh\n")

    curves = activation_curves(df)
    pathlib.Path("reports/activation_curves_v1.json").write_text(
        json.dumps(curves, indent=2))

    print("== P(activation) for a discharge offer at price p, by block ==")
    hdr = [400, 500, 600, 800, 1000, 1500]
    print("block      " + "".join(f"{p:>7d}" for p in hdr))
    for blk in BLOCK_LABELS:
        c = curves["up"].get(blk, {}).get("curve", {})
        if c:
            print(f"{blk:10s} " + "".join(f"{c.get(p, float('nan')):7.3f}" for p in hdr))
    print("\n== P(activation) for a charge offer at price p, by block ==")
    hdr = [-100, 0, 50, 100, 200, 300]
    print("block      " + "".join(f"{p:>7d}" for p in hdr))
    for blk in BLOCK_LABELS:
        c = curves["down"].get(blk, {}).get("curve", {})
        if c:
            print(f"{blk:10s} " + "".join(f"{c.get(p, float('nan')):7.3f}" for p in hdr))
    print("\nwrote reports/activation_curves_v1.json")


if __name__ == "__main__":
    main()
