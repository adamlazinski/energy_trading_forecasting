"""
BESS co-optimization: NET capacity revenue after the SoC-feasibility haircut.

    python -m src.bess_cooptimize

F8/F16 report GROSS capacity revenue — they assume the battery reserves 1 MW
aFRR up + 1 MW down every hour. But reserved capacity gets ACTIVATED, which
moves SoC; if SoC leaves the reservable band you cannot offer that direction
and lose the payment. Activation also has a slight net charging drift
(+0.009/period measured), so the battery must periodically discharge to stay
feasible — the point where capacity and energy arbitrage actually couple.

Single-path simulation on realized data (2025-07 → 2026-07, the span where
aFRR activation volumes exist):
  - each 15-min period, offer up-capacity if SoC ≥ buffer, down if SoC ≤ E−buffer
  - SoC moves by realized activation: +u_dn (charge) − u_up (discharge),
    u = activated / procured (eb-rozl / zmb), efficiency-adjusted
  - SoC management: when SoC drifts past a target band, forgo one direction's
    reservation that period and trade 1 MW at CEN to recover (the co-opt term)
  - capacity paid at CMBP (afrr_g/d, hourly); activation energy settled at
    ceb_sr_afrr* where available; recovery trades at CEN
Reports NET vs GROSS capacity revenue, the feasibility haircut, activation
cycling, and the SoC distribution.

Writes reports/bess_cooptimize.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .bess_optimizer import C_DEG, E_MWH, ETA, P_MW

DT = 0.25
BUFFER = 0.40                 # MWh headroom to credibly offer a direction
SOC_LO, SOC_HI = 0.70, 1.30  # target band; outside -> recover via CEN trade
HOURS = 8760


def load() -> pd.DataFrame:
    # eb-rozl (activation) is 15-min; zmb (procured) and cmbp (prices) are
    # hourly — ffill the hourly series onto the 15-min activation grid so the
    # simulation runs at a true 15-min step.
    eb = pd.read_parquet("data/raw/pse_bal_energy.parquet")[
        ["ts", "eb_afrrg", "eb_afrrd"]].dropna().sort_values("ts")
    for c in ("eb_afrrg", "eb_afrrd"):
        eb[c] = pd.to_numeric(eb[c], errors="coerce")
    grid = eb.set_index("ts")

    def hourly(path, cols):
        d = pd.read_parquet(path)[["ts"] + cols].sort_values("ts")
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d.set_index("ts").reindex(grid.index, method="ffill")

    zmb = hourly("data/raw/pse_reserve_req.parquet", ["zmb_afrrg", "zmb_afrrd"])
    cp = hourly("data/raw/pse_reserve_prices_basic.parquet", ["afrr_g", "afrr_d"])
    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[
        ["ts", "cen_cost", "ceb_sr_afrrg_cost", "ceb_sr_afrrd_cost"]].sort_values("ts")
    for c in ("cen_cost", "ceb_sr_afrrg_cost", "ceb_sr_afrrd_cost"):
        crb[c] = pd.to_numeric(crb[c], errors="coerce")
    crb = crb.set_index("ts").reindex(grid.index, method="nearest")

    df = pd.concat([grid, zmb, cp, crb], axis=1).reset_index().rename(
        columns={"index": "ts"})
    df = df[(df["zmb_afrrg"] > 0) & (df["zmb_afrrd"] > 0)].dropna(
        subset=["eb_afrrg", "eb_afrrd", "afrr_g", "afrr_d", "cen_cost"]
        ).sort_values("ts")
    # per-period activation as fraction of full power (0..1)
    df["u_up"] = (df["eb_afrrg"] / df["zmb_afrrg"] / DT).clip(0, 1)
    df["u_dn"] = (df["eb_afrrd"].abs() / df["zmb_afrrd"] / DT).clip(0, 1)
    df["ceb_up"] = df["ceb_sr_afrrg_cost"].fillna(df["cen_cost"])
    df["ceb_dn"] = df["ceb_sr_afrrd_cost"].fillna(df["cen_cost"])
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df["quarter"] = loc.dt.year.astype(str) + "Q" + loc.dt.quarter.astype(str)
    return df.reset_index(drop=True)


def simulate(df: pd.DataFrame) -> dict:
    s = E_MWH / 2
    cap_net = cap_gross = act_pnl = rec_pnl = 0.0
    dis_mwh = 0.0
    lost_up = lost_dn = 0
    soc_path = []
    for r in df.itertuples():
        # gross: both directions every period (F8/F16 assumption), per-15min pay
        cap_gross += (r.afrr_g + r.afrr_d) * P_MW * DT

        # SoC recovery trade when out of target band (co-optimization term)
        recover = 0
        if s > SOC_HI:                      # too full -> discharge 1 MW, forgo up
            e = min(P_MW * DT, s * ETA)
            rec_pnl += (r.cen_cost - C_DEG) * e
            s -= e / ETA
            dis_mwh += e
            recover = +1
        elif s < SOC_LO:                    # too empty -> charge 1 MW, forgo down
            e = min(P_MW * DT, (E_MWH - s) / ETA)
            rec_pnl -= r.cen_cost * e
            s += e * ETA
            recover = -1

        # capacity offered this period, subject to SoC feasibility
        off_up = 1 if (s >= BUFFER and recover != +1) else 0
        off_dn = 1 if (s <= E_MWH - BUFFER and recover != -1) else 0
        lost_up += (1 - off_up)
        lost_dn += (1 - off_dn)
        cap_net += (off_up * r.afrr_g + off_dn * r.afrr_d) * P_MW * DT

        # realized activation moves SoC (only for offered directions)
        e_up = off_up * r.u_up * P_MW * DT          # discharge (deliver up)
        e_dn = off_dn * r.u_dn * P_MW * DT          # charge (absorb down)
        act_pnl += r.ceb_up * e_up - r.ceb_dn * e_dn
        s = s - e_up / ETA + e_dn * ETA
        s = min(max(s, 0.0), E_MWH)
        dis_mwh += e_up
        soc_path.append(s)

    n = len(df)
    days = n / 96
    soc = np.array(soc_path)
    net_total = cap_net + act_pnl + rec_pnl
    return {
        "days": round(days, 0),
        "cap_gross_ann": round(cap_gross / days * 365, 0),
        "cap_net_ann": round(cap_net / days * 365, 0),
        "feasibility_haircut_pct": round((cap_net / cap_gross - 1) * 100, 1),
        "activation_pnl_ann": round(act_pnl / days * 365, 0),
        "soc_recovery_pnl_ann": round(rec_pnl / days * 365, 0),
        "net_total_ann": round(net_total / days * 365, 0),
        "lost_up_hours_pct": round(lost_up / n * 100, 1),
        "lost_dn_hours_pct": round(lost_dn / n * 100, 1),
        "cycles_per_day": round(dis_mwh / days / E_MWH, 2),
        "soc_p5_p50_p95": [round(float(np.percentile(soc, p)), 2) for p in (5, 50, 95)],
    }


def main():
    df = load()
    print(f"loaded {len(df)} periods, {df['ts'].min()} .. {df['ts'].max()}\n")
    res = simulate(df)
    out = {"battery": f"{P_MW} MW / {E_MWH} MWh, eta_rt {ETA**2:.2f}, "
                       f"buffer {BUFFER} MWh, target SoC [{SOC_LO},{SOC_HI}]",
           **res}
    pathlib.Path("reports/bess_cooptimize.json").write_text(json.dumps(out, indent=2))
    print(f"battery {out['battery']}  ({res['days']:.0f} days)\n")
    print(f"  capacity GROSS (F16 basis)   {res['cap_gross_ann']:12,.0f} PLN/MW/yr")
    print(f"  capacity NET (feasible only) {res['cap_net_ann']:12,.0f}   "
          f"(haircut {res['feasibility_haircut_pct']:+.1f}%)")
    print(f"  + activation energy settle   {res['activation_pnl_ann']:12,.0f}")
    print(f"  + SoC-recovery CEN trades    {res['soc_recovery_pnl_ann']:12,.0f}")
    print(f"  = NET TOTAL                  {res['net_total_ann']:12,.0f} PLN/MW/yr")
    print(f"\n  lost capacity hours: up {res['lost_up_hours_pct']}%  "
          f"dn {res['lost_dn_hours_pct']}%  | cycles/day {res['cycles_per_day']}  "
          f"| SoC p5/50/95 {res['soc_p5_p50_p95']}")
    print("\nwrote reports/bess_cooptimize.json")


if __name__ == "__main__":
    main()
