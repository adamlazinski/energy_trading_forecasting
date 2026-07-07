"""
Project B, Layer 3 (v1): revenue stack — balancing CAPACITY (RMB) on top of
the Layer 2 balancing-energy policy.

    python -m src.bess_layer3

Mechanics (WDB deck, verified fields):
- Capacity procured hourly in the basic mode; clearing price CMBP per
  product per hour (cmbp-tp: fcr_g/d, afrr_g/d, mfrrd_g/d, rr_g/d),
  uniform to all cleared offers. Gate D-1 08:30-09:00 — BEFORE csdac
  (13:50), so the mode choice cannot condition on the DA price; the
  honest policy uses trailing-4-week CMBP climatology.
- A price-taking 1 MW battery offering capacity at ~0 always clears and
  earns the realized CMBP.
- LER assumption (documented, adjustable): SUSTAIN_H = 0.5 h of full-power
  delivery per reserved direction. Feasible SoC bands for 1 MW reserved:
    G (up):  s >= SUSTAIN_H / eta_d          (0.53 MWh)
    D (down): s <= E - SUSTAIN_H * eta_c      (1.53 MWh)
  Both directions co-provided when s is in the intersection; else the
  feasible single direction; else no revenue that hour (idle, honest).
- v1 simplifications (flagged): SoC unchanged during cap hours (activation
  energy margin and its cycling excluded — roughly offsetting); the energy
  DP is not re-optimized around the cap schedule.

Policies on the Layer 2 holdout:
  energy_only      — Layer 2 as-is
  capacity_only    — reserve every hour (park SoC mid)
  stack_forecast   — cap hour iff CMBP climatology > DP marginal value of
                     the hour (computed by re-running the DP with that
                     hour's energy option blocked) — honest
  stack_oracle_ub  — per-hour max(realized cap revenue, realized energy
                     PnL of the hour), SoC interactions ignored — a LOOSE
                     upper bound, labeled as such

Writes reports/bess_layer3_stack.{json,txt}.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .bess_optimizer import (C_DEG, DE_CHG, DE_DIS, DE_GRID, E_MWH, ETA, QS,
                             SOC, TERMINAL_PLN_MWH, day_policy)

G_PRODUCTS = ["afrr_g", "mfrrd_g", "fcr_g", "rr_g"]
D_PRODUCTS = ["afrr_d", "mfrrd_d", "fcr_d"]
SUSTAIN_H = 0.5
S_MIN_G = SUSTAIN_H * 1.0 / ETA          # SoC needed to back 1 MW up
S_MAX_D = E_MWH - SUSTAIN_H * ETA        # SoC ceiling to back 1 MW down
CLIM_WEEKS = 4


def load_cmbp() -> pd.DataFrame:
    cp = pd.read_parquet("data/raw/pse_reserve_prices_basic.parquet")
    for c in G_PRODUCTS + D_PRODUCTS:
        cp[c] = pd.to_numeric(cp[c], errors="coerce")
    cp["cap_g"] = cp[G_PRODUCTS].max(axis=1).fillna(0.0)
    cp["cap_d"] = cp[D_PRODUCTS].max(axis=1).fillna(0.0)
    return cp[["ts", "cap_g", "cap_d"]].dropna().sort_values("ts")


def cap_hour_revenue(s: float, g: float, d: float) -> float:
    """Realized revenue of a reserved hour given SoC feasibility."""
    ok_g, ok_d = s >= S_MIN_G - 1e-9, s <= S_MAX_D + 1e-9
    if ok_g and ok_d:
        return g + d
    if ok_g:
        return g
    if ok_d:
        return d
    return 0.0


def run_energy(day: pd.DataFrame, r_up, r_dn, soc0: float,
               enabled: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Layer 2 policy on the enabled periods; returns pnl, end SoC and
    per-hour realized energy PnL."""
    s, pnl = soc0, 0.0
    hour_pnl = np.zeros(24)
    for t, (_, row) in enumerate(day.iterrows()):
        if not enabled[t]:
            continue
        ru = np.interp(s, SOC, r_up[t])
        rd = np.interp(s, SOC, r_dn[t])
        x = row["ceb_pp"]
        if row["dir"] == "G" and s >= DE_DIS - 1e-9 and x >= ru:
            gain = (x - C_DEG) * DE_GRID
            pnl += gain
            hour_pnl[t // 4] += gain
            s -= DE_DIS
        elif row["dir"] == "D" and s <= E_MWH - DE_CHG + 1e-9 and x <= rd:
            pnl -= x * DE_GRID
            hour_pnl[t // 4] -= x * DE_GRID
            s += DE_CHG
    return pnl, s, hour_pnl


def dp_value(p_up, xu, xd, soc0: float, mask: np.ndarray | None = None) -> float:
    """Expected DP value of the day with optionally blocked periods."""
    T = len(p_up)
    v = SOC * TERMINAL_PLN_MWH
    W = 1.0 / len(QS)
    for t in range(T - 1, -1, -1):
        if mask is not None and not mask[t]:
            continue
        v_dis = np.interp(np.maximum(SOC - DE_DIS, 0.0), SOC, v)
        v_chg = np.interp(np.minimum(SOC + DE_CHG, E_MWH), SOC, v)
        can_dis = SOC >= DE_DIS - 1e-9
        can_chg = SOC <= E_MWH - DE_CHG + 1e-9
        gain_up = np.zeros(len(SOC))
        gain_dn = np.zeros(len(SOC))
        for j in range(len(QS)):
            gu = xu[t, j] * DE_GRID - C_DEG * DE_GRID + v_dis - v
            gain_up += W * np.where(can_dis & (gu > 0), gu, 0.0)
            gd = -xd[t, j] * DE_GRID + v_chg - v
            gain_dn += W * np.where(can_chg & (gd > 0), gd, 0.0)
        v = v + p_up[t] * gain_up + (1 - p_up[t]) * gain_dn
    return float(np.interp(soc0, SOC, v) - soc0 * TERMINAL_PLN_MWH)


def main():
    pred = pd.read_parquet("data/proc/bess_cond_holdout.parquet")
    cmbp = load_cmbp()
    loc = pd.to_datetime(pred["ts"]).dt.tz_convert("Europe/Warsaw")
    pred["day"] = loc.dt.date
    cm = cmbp.copy()
    cm_loc = cm["ts"].dt.tz_convert("Europe/Warsaw")
    cm["day"], cm["hour"] = cm_loc.dt.date, cm_loc.dt.hour

    policies = ("energy_only", "capacity_only", "stack_forecast",
                "stack_oracle_ub")
    pnl_days = {p: [] for p in policies}
    soc = {p: E_MWH / 2 for p in policies}
    cap_share = []
    for dkey, day in pred.groupby("day", sort=True):
        if len(day) != 96:
            continue
        caph = cm[cm["day"] == dkey].set_index("hour")
        if len(caph) < 24:
            continue
        cap_g = caph["cap_g"].reindex(range(24)).fillna(0.0).to_numpy()
        cap_d = caph["cap_d"].reindex(range(24)).fillna(0.0).to_numpy()
        p_up = day["p_up"].to_numpy()
        xu = day[[f"up_q{q}" for q in QS]].to_numpy()
        xd = day[[f"dn_q{q}" for q in QS]].to_numpy()
        _, r_up, r_dn = day_policy(p_up, xu, xd)

        # honest forecast comparison: DP marginal value per hour vs
        # trailing climatology of cap prices
        day0 = pd.Timestamp(dkey, tz="Europe/Warsaw").tz_convert("UTC")
        w = cmbp[(cmbp["ts"] < day0) &
                 (cmbp["ts"] >= day0 - pd.Timedelta(weeks=CLIM_WEEKS))]
        wloc = w["ts"].dt.tz_convert("Europe/Warsaw")
        clim = ((w["cap_g"] + w["cap_d"]).groupby(wloc.dt.hour).mean()
                .reindex(range(24)).fillna(0.0).to_numpy())
        v_full = dp_value(p_up, xu, xd, soc["stack_forecast"])
        marg = np.empty(24)
        for h in range(24):
            mask = np.ones(96, bool)
            mask[4 * h:4 * h + 4] = False
            marg[h] = v_full - dp_value(p_up, xu, xd, soc["stack_forecast"],
                                        mask)
        fc_cap_hours = clim > marg
        cap_share.append(fc_cap_hours.mean())

        for p in policies:
            if p == "energy_only":
                pnl, s_end, _ = run_energy(day, r_up, r_dn, soc[p],
                                           np.ones(96, bool))
            elif p == "capacity_only":
                pnl = sum(cap_hour_revenue(soc[p], cap_g[h], cap_d[h])
                          for h in range(24))
                s_end = soc[p]
            elif p == "stack_forecast":
                # single walk: energy trades in energy hours, cap revenue
                # (with SoC feasibility) at the start of each cap hour
                pnl, s_run = 0.0, soc[p]
                for t in range(96):
                    h = t // 4
                    if fc_cap_hours[h]:
                        if t % 4 == 0:
                            pnl += cap_hour_revenue(s_run, cap_g[h], cap_d[h])
                        continue
                    row = day.iloc[t]
                    ru = np.interp(s_run, SOC, r_up[t])
                    rd = np.interp(s_run, SOC, r_dn[t])
                    x = row["ceb_pp"]
                    if (row["dir"] == "G" and s_run >= DE_DIS - 1e-9
                            and x >= ru):
                        pnl += (x - C_DEG) * DE_GRID
                        s_run -= DE_DIS
                    elif (row["dir"] == "D"
                          and s_run <= E_MWH - DE_CHG + 1e-9 and x <= rd):
                        pnl -= x * DE_GRID
                        s_run += DE_CHG
                s_end = s_run
            else:  # stack_oracle_ub
                _, s_end, hour_e = run_energy(day, r_up, r_dn, soc[p],
                                              np.ones(96, bool))
                cap_rev = cap_g + cap_d      # feasibility waived: loose UB
                pnl = float(np.maximum(cap_rev, hour_e).sum())
            pnl += (s_end - soc[p]) * TERMINAL_PLN_MWH
            soc[p] = s_end
            pnl_days[p].append(pnl)

    res = {}
    for p, v in pnl_days.items():
        a = np.array(v)
        res[p] = {"n_days": len(a),
                  "pln_per_day": round(float(a.mean()), 1),
                  "median": round(float(np.median(a)), 1),
                  "p10": round(float(np.quantile(a, .1)), 1),
                  "p90": round(float(np.quantile(a, .9)), 1),
                  "share_pos": round(float((a > 0).mean()), 3),
                  "annualized_per_mw": round(float(a.mean() * 365), 0)}
    out = {"sustain_h": SUSTAIN_H, "clim_weeks": CLIM_WEEKS,
           "mean_cap_hour_share_forecast": round(float(np.mean(cap_share)), 3),
           "note": "cap hours: best G + best D product, LER 30-min sustain "
                   "bands; activation energy during cap hours excluded (v1); "
                   "stack_oracle_ub ignores SoC coupling — loose upper bound",
           "results": res}
    pathlib.Path("reports/bess_layer3_stack.json").write_text(
        json.dumps(out, indent=2))
    lines = [f"BESS Layer 3 stack, holdout ({res['energy_only']['n_days']} days), "
             f"1 MW / 2 MWh, sustain {SUSTAIN_H}h "
             f"(cap-hour share, forecast policy: {np.mean(cap_share):.0%})"]
    for p, r in res.items():
        lines.append(f"{p:16s} PLN/day={r['pln_per_day']:8.1f} "
                     f"median={r['median']:8.1f} [P10 {r['p10']:8.1f}, "
                     f"P90 {r['p90']:8.1f}] pos={r['share_pos']:.0%} "
                     f"ann/MW={r['annualized_per_mw']:,.0f}")
    txt = "\n".join(lines)
    pathlib.Path("reports/bess_layer3_stack.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
