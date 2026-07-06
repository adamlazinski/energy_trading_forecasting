"""
Project B, Layer 1 (v0): an empirical model of PSE's dispatcher.

    python -m src.bess_activation

For a battery submitting balancing-energy offers into ZPG/RBN, revenue is
gated by ACTIVATION: offer at price p, direction up (discharge) is activated
iff the dispatcher's marginal accepted price m_t reaches p. Public data gives
us, per 15-min period:
  poeb-rbn: ofcg / ofcd  — marginal accepted offer price, up / down
  eb-rozl:  eb_afrrg/d, eb_w_pp/eb_d_pp — activated volumes by product/dir
  crb-rozl: ceb_sr_afrrg/d — settlement price of activated aFRR energy
  zeb-rozl: pzeb_afrrg/d  — PICASSO cross-border aFRR platform prices

v0 deliverable (discriminatory-price approximation, caveat as per spec):
  pi_up(p | block)  = P(activated up)   x P(m_up >= p | activated)
  pi_down(p | block) = P(activated down) x P(m_down <= p | activated)
estimated empirically per hour block + a summary of how PICASSO coupling
moves the aFRR marginal price dispersion. Offers are committed D-1, so any
later conditional model may only use D-1-known features (the fx_ set).
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


def load() -> pd.DataFrame:
    raw = pathlib.Path("data/raw")
    poeb = pd.read_parquet(raw / "pse_poeb_rbn.parquet")[["ts", "ofcg", "ofcd", "ofp"]]
    eb = pd.read_parquet(raw / "pse_bal_energy.parquet")[
        ["ts", "eb_afrrg", "eb_afrrd", "eb_w_pp", "eb_d_pp"]]
    crb = pd.read_parquet(raw / "pse_imbalance.parquet")[
        ["ts", "cen_cost", "ceb_sr_afrrg_cost", "ceb_sr_afrrd_cost"]]
    pic = pd.read_parquet(raw / "pse_picasso.parquet")[
        ["ts", "pzeb_afrrg_cost", "pzeb_afrrd_cost", "zebpp"]]
    da = pd.read_parquet(raw / "pse_da_price.parquet")[["ts", "csdac_pln"]]
    df = poeb
    for other in (eb, crb, pic, da):
        df = df.merge(other, on="ts", how="outer")
    df = df.sort_values("ts").reset_index(drop=True)
    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["hour"] = loc.dt.hour
    df["block"] = pd.cut(df["hour"], BLOCKS, right=False, labels=BLOCK_LABELS)
    return df


def activation_curves(df: pd.DataFrame) -> dict:
    """Empirical pi(p | block) per direction, discriminatory approximation."""
    out = {"up": {}, "down": {}}
    for blk, g in df.groupby("block", observed=True):
        up_act = g["eb_w_pp"].fillna(0) > 0          # energy delivered upward
        dn_act = g["eb_d_pp"].fillna(0) > 0
        m_up = g.loc[up_act, "ofcg"].dropna()
        m_dn = g.loc[dn_act, "ofcd"].dropna()
        out["up"][str(blk)] = {
            "p_act": round(float(up_act.mean()), 4),
            "curve": {int(p): round(float(up_act.mean() * (m_up >= p).mean()), 4)
                      for p in PRICE_GRID_UP} if len(m_up) else {},
        }
        out["down"][str(blk)] = {
            "p_act": round(float(dn_act.mean()), 4),
            "curve": {int(p): round(float(dn_act.mean() * (m_dn <= p).mean()), 4)
                      for p in PRICE_GRID_DN} if len(m_dn) else {},
        }
    return out


def main():
    df = load()
    have = df.dropna(subset=["ofcg"])
    print(f"panel {len(df)} periods, poeb coverage {len(have)/len(df):.1%}, "
          f"{df['ts'].min()} .. {df['ts'].max()}\n")

    up_act = df["eb_w_pp"].fillna(0) > 0
    dn_act = df["eb_d_pp"].fillna(0) > 0
    print("== activation frequency ==")
    print(f"up (discharge pays): {up_act.mean():.1%} of periods; "
          f"down (charge gets paid): {dn_act.mean():.1%}")
    print(df.groupby('block', observed=True)[['eb_w_pp', 'eb_d_pp']]
            .apply(lambda g: pd.Series({
                'up_freq': (g['eb_w_pp'].fillna(0) > 0).mean(),
                'dn_freq': (g['eb_d_pp'].fillna(0) > 0).mean()})).round(3), "\n")

    print("== marginal accepted price (up, ofcg), activated periods ==")
    m = df.loc[up_act, ["block", "ofcg"]].dropna()
    print(m.groupby("block", observed=True)["ofcg"]
           .describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
           .round(0).to_string(), "\n")

    print("== PICASSO aFRR platform price vs domestic marginal (up) ==")
    both = df.dropna(subset=["pzeb_afrrg_cost", "ceb_sr_afrrg_cost"])
    if len(both):
        d = both["pzeb_afrrg_cost"] - both["ceb_sr_afrrg_cost"]
        print(f"n={len(both)}  spread pzeb-ceb: mean={d.mean():.1f}  "
              f"std={d.std():.1f}  P(|spread|>100)={float((d.abs() > 100).mean()):.2%}\n")

    curves = activation_curves(df)
    pathlib.Path("reports/activation_curves_v0.json").write_text(
        json.dumps(curves, indent=2))
    print("== example: P(activation) for a discharge offer, by block ==")
    hdr = [200, 400, 600, 800, 1000, 1500]
    print("block      " + "".join(f"{p:>7d}" for p in hdr))
    for blk in BLOCK_LABELS:
        c = curves["up"].get(blk, {}).get("curve", {})
        if c:
            print(f"{blk:10s} " + "".join(f"{c.get(p, float('nan')):7.3f}" for p in hdr))
    print("\nwrote reports/activation_curves_v0.json")


if __name__ == "__main__":
    main()
