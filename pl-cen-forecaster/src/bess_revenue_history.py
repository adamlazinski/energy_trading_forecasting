"""
BESS revenue trajectory over the full history — the robustness/decay test of
the F8 headline (which rests on a single 8-week window).

    python -m src.bess_revenue_history

Per quarter, realized revenue for a 1 MW / 2 MWh unit (no forecast needed —
capacity is realized clearing price, energy is a causal threshold DP):
  cap_afrr    aFRR up+down reservation (afrr_g + afrr_d), PLN/MW/h
  cap_best    best up-product + best down-product across aFRR/FCR/mFRR/RR
              (upper ref; true stacking is product-rule-limited)
  energy_arb  CEN threshold-policy arbitrage (the balancing-energy floor)
annualized to PLN/MW/yr. The capacity lines are gross of SoC-feasibility and
assume dual-direction provision is permitted (F8 caveats); the point is the
TRAJECTORY (is the headline stable, and how fast is it decaying?).

Writes reports/bess_revenue_history.json + reports/bess_revenue_history.csv.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .intraday_mm import arb_threshold

UP = ["afrr_g", "fcr_g", "mfrrd_g", "rr_g"]
DN = ["afrr_d", "fcr_d", "mfrrd_d"]
HOURS_YR = 8760


def main():
    cp = pd.read_parquet("data/raw/pse_reserve_prices_basic.parquet")
    for c in UP + DN:
        cp[c] = pd.to_numeric(cp[c], errors="coerce")
    cp["cap_afrr"] = cp["afrr_g"].fillna(0) + cp["afrr_d"].fillna(0)
    cp["cap_best"] = (cp[UP].max(axis=1).fillna(0) + cp[DN].max(axis=1).fillna(0))
    loc = cp["ts"].dt.tz_convert("Europe/Warsaw")
    cp["q"] = loc.dt.year.astype(str) + "Q" + loc.dt.quarter.astype(str)

    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "cen_cost"]]
    crb["cen"] = pd.to_numeric(crb["cen_cost"], errors="coerce")
    lc = crb["ts"].dt.tz_convert("Europe/Warsaw")
    crb["q"] = lc.dt.year.astype(str) + "Q" + lc.dt.quarter.astype(str)
    crb["day"] = lc.dt.date

    # energy arbitrage per day (causal P25/P75 threshold on realized CEN)
    e_day = []
    for (q, day), g in crb.dropna(subset=["cen"]).groupby(["q", "day"]):
        p = g["cen"].to_numpy()
        if len(p) < 90:
            continue
        lo, hi = np.percentile(p, 25), np.percentile(p, 75)
        e_day.append((q, arb_threshold(p, lo, hi)))
    e_df = pd.DataFrame(e_day, columns=["q", "pnl"])
    e_q = e_df.groupby("q")["pnl"].mean() * 365          # PLN/MW/yr

    rows = []
    for q, g in cp.groupby("q"):
        rows.append({
            "quarter": q,
            "cap_afrr_ann": round(float(g["cap_afrr"].mean() * HOURS_YR), 0),
            "cap_best_ann": round(float(g["cap_best"].mean() * HOURS_YR), 0),
            "afrr_g_hr": round(float(g["afrr_g"].mean()), 0),
            "afrr_d_hr": round(float(g["afrr_d"].mean()), 0),
            "energy_arb_ann": round(float(e_q.get(q, np.nan)), 0),
        })
    df = pd.DataFrame(rows).sort_values("quarter")
    df.to_csv("reports/bess_revenue_history.csv", index=False)

    peak = df["cap_afrr_ann"].iloc[:2].mean()
    recent = df["cap_afrr_ann"].iloc[-3:].mean()
    out = {
        "unit": "1 MW / 2 MWh, PLN/MW/yr, gross of SoC-feasibility & fixed costs",
        "by_quarter": df.to_dict("records"),
        "cap_afrr_peak_2024h1": round(peak, 0),
        "cap_afrr_recent_3q": round(recent, 0),
        "decay_peak_to_recent_pct": round((recent / peak - 1) * 100, 1),
    }
    pathlib.Path("reports/bess_revenue_history.json").write_text(json.dumps(out, indent=2))

    print(f"{'quarter':8s} {'aFRR cap':>10s} {'best cap':>10s} {'energy arb':>11s}"
          f"  (PLN/MW/yr)")
    for r in df.to_dict("records"):
        print(f"{r['quarter']:8s} {r['cap_afrr_ann']:10,.0f} {r['cap_best_ann']:10,.0f} "
              f"{r['energy_arb_ann']:11,.0f}")
    print(f"\naFRR capacity: peak (2024H1) {peak:,.0f} -> recent 3q {recent:,.0f} "
          f"PLN/MW/yr  ({out['decay_peak_to_recent_pct']:+.0f}%)")
    print("wrote reports/bess_revenue_history.{json,csv}")


if __name__ == "__main__":
    main()
