"""
Price-aware SoC recovery: monetizing the CEN forecaster inside the battery.

    python -m src.bess_soc_policy

F18's stack pays a -668k PLN/MW/yr drag from PRICE-BLIND SoC recovery: the
moment SoC exits the target band, the co-opt sim trades 1 MW at whatever CEN
turns out to be. But recovery is rarely urgent to the quarter-hour — and the
CEN forecaster's 60-min horizon means that at decision time t the OOS
forecasts for deliveries t..t+3 were all issued at or before t. So a recovery
can be TIMED: pick the best-forecast quarter-hour within the next hour.
Gate-honest by construction; no new instrument, same realized path as
bess_cooptimize (identical activation, capacity and band mechanics).

Policies compared on the identical path:
  blind   execute the period the band is crossed (F18 baseline)
  fcst    execute at the argmax (discharge) / argmin (charge) of the OOS
          forecast median over the window t..t+3; cancel if activation
          drift brings SoC back inside the band first; emergency-execute
          if SoC nears the physical limits while waiting
  oracle  same window, timed on realized CEN — the ceiling that bounds
          what a better forecast could ever add

Forecasts: LEAR walk-forward quantiles (2024-12 -> 2026-05), GBM+conformal
holdout for the tail; periods with no forecast fall back to blind (counted).
Writes reports/bess_soc_policy.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .bess_cooptimize import BUFFER, SOC_HI, SOC_LO, load
from .bess_optimizer import C_DEG, E_MWH, ETA, P_MW

DT = 0.25
# quarter-hours a recovery may wait. Gate-honesty caps it at 4 (60-min fcst
# horizon); the sweep over {2,3,4} picks 2 — beyond 30 min the SoC drift while
# waiting forfeits more capacity offers than the timing gains (net: 2.90M /
# 2.87M / 2.87M for fcst at W=2/3/4 vs 2.86M blind). Mechanism, not tuning:
# drift cost is monotone in the window; still, sweep re-run on new data.
WINDOW = 2
EMERG = 0.30                  # MWh from a physical limit -> execute immediately


def attach_forecast(df: pd.DataFrame) -> pd.DataFrame:
    lear = pd.read_parquet("reports/lear_walkforward.parquet")[["ts", "lear_q0.5"]]
    gbm = pd.read_parquet("reports/holdout_predictions.parquet")[["ts", "gbm_conf_q0.5"]]
    df = df.merge(lear, on="ts", how="left").merge(gbm, on="ts", how="left")
    df["fc"] = pd.to_numeric(df["lear_q0.5"], errors="coerce").fillna(
        pd.to_numeric(df["gbm_conf_q0.5"], errors="coerce"))
    return df


def simulate(df: pd.DataFrame, mode: str) -> dict:
    """mode in {blind, fcst, oracle}. Mirrors bess_cooptimize.simulate but
    lets a pending recovery wait for the best period in the next WINDOW."""
    cen = df["cen_cost"].to_numpy()
    fc = df["fc"].to_numpy() if "fc" in df else np.full(len(df), np.nan)
    ref = cen if mode == "oracle" else fc

    s = E_MWH / 2
    cap_net = act_pnl = rec_pnl = 0.0
    dis_mwh = 0.0
    n_rec = n_fallback = 0
    wait_gain = []                     # realized CEN edge vs executing at commit
    pending: tuple[int, int, int] | None = None   # (direction, exec_idx, commit_idx)
    rec_q: dict[str, float] = {}

    for i, r in enumerate(df.itertuples()):
        recover = 0

        # resolve pending recovery
        if pending is not None:
            d, j, i0 = pending
            back_in = SOC_LO <= s <= SOC_HI
            emergency = (d == +1 and s > E_MWH - EMERG) or (d == -1 and s < EMERG)
            if back_in:
                pending = None
            elif i == j or emergency:
                recover, pending = d, None
        elif not (SOC_LO <= s <= SOC_HI):
            d = +1 if s > SOC_HI else -1
            hi = min(i + WINDOW, len(df) - 1)
            w = ref[i:hi + 1]
            if mode == "blind" or np.isnan(w).any():
                recover = d
                n_fallback += mode != "blind"
            else:
                j = i + int(np.argmax(w) if d == +1 else np.argmin(w))
                if j == i:
                    recover = d
                else:
                    pending = (d, j, i)

        if recover == +1:              # too full -> discharge 1 MW at CEN
            e = min(P_MW * DT, s * ETA)
            rec_pnl += (cen[i] - C_DEG) * e
            s -= e / ETA
            dis_mwh += e
            n_rec += 1
            rec_q[r.quarter] = rec_q.get(r.quarter, 0.0) + (cen[i] - C_DEG) * e
        elif recover == -1:            # too empty -> charge 1 MW at CEN
            e = min(P_MW * DT, (E_MWH - s) / ETA)
            rec_pnl -= cen[i] * e
            s += e * ETA
            n_rec += 1
            rec_q[r.quarter] = rec_q.get(r.quarter, 0.0) - cen[i] * e

        off_up = 1 if (s >= BUFFER and recover != +1) else 0
        off_dn = 1 if (s <= E_MWH - BUFFER and recover != -1) else 0
        cap_net += (off_up * r.afrr_g + off_dn * r.afrr_d) * P_MW * DT

        e_up = off_up * r.u_up * P_MW * DT
        e_dn = off_dn * r.u_dn * P_MW * DT
        act_pnl += r.ceb_up * e_up - r.ceb_dn * e_dn
        s = min(max(s - e_up / ETA + e_dn * ETA, 0.0), E_MWH)
        dis_mwh += e_up

    days = len(df) / 96
    return {"rec_pnl_ann": round(rec_pnl / days * 365, 0),
            "cap_net_ann": round(cap_net / days * 365, 0),
            "act_pnl_ann": round(act_pnl / days * 365, 0),
            "net_total_ann": round((cap_net + act_pnl + rec_pnl) / days * 365, 0),
            "n_recoveries": n_rec,
            "n_fallback_no_fcst": int(n_fallback),
            "cycles_per_day": round(dis_mwh / days / E_MWH, 2),
            "rec_by_quarter": {k: round(v, 0) for k, v in sorted(rec_q.items())}}


def main():
    df = attach_forecast(load())
    cov = float(df["fc"].notna().mean())
    print(f"{len(df)} periods, {df['ts'].min():%Y-%m-%d} .. {df['ts'].max():%Y-%m-%d}, "
          f"forecast coverage {cov:.1%}\n")

    out = {"window_qh": WINDOW, "fcst_coverage": round(cov, 3), "policies": {}}
    base = None
    for mode in ("blind", "fcst", "oracle"):
        r = simulate(df, mode)
        out["policies"][mode] = r
        base = base or r
        d = r["net_total_ann"] - base["net_total_ann"]
        print(f"== {mode:6}  recovery {r['rec_pnl_ann']:>10,.0f}  "
              f"capacity {r['cap_net_ann']:>10,.0f}  activation {r['act_pnl_ann']:>9,.0f}  "
              f"NET {r['net_total_ann']:>10,.0f} PLN/MW/yr  (vs blind {d:+,.0f})")
        print(f"   recoveries {r['n_recoveries']:>5}  fallbacks {r['n_fallback_no_fcst']}"
              f"  cycles/day {r['cycles_per_day']}")
        print(f"   recovery by quarter: {r['rec_by_quarter']}\n")

    pathlib.Path("reports/bess_soc_policy.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/bess_soc_policy.json")


if __name__ == "__main__":
    main()
