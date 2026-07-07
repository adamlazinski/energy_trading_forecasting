"""
Project B, Layer 2b: offer optimizer + honest holdout simulation.

    python -m src.bess_optimizer

Under uniform pricing (WDB: payment = CEB_PP, not the offer) a price-taking
battery's optimal offer is its RESERVATION price, so the day-ahead policy
falls out of a backward-induction DP over SoC:

  r_up(t, s) = c_deg + [V_{t+1}(s) - V_{t+1}(s - de_d)] / de_grid
  r_dn(t, s) =         [V_{t+1}(s + de_c) - V_{t+1}(s)] / de_grid

with the conditional predictive distribution of (dir, ceb_pp) from
src.bess_cond_model entering V as a 5-point quantile approximation.
V is piecewise-linear in SoC (np.interp), so off-grid SoC states from
efficiency losses are handled exactly.

Simulation on the holdout: each day builds the DP policy from the D-1
conditional forecasts, then plays it against realized (dir, ceb_pp):
activated up iff dir=G and ceb_pp >= r_up(t, s_t); down symmetrically.
Benchmarks: perfect-foresight DP (upper bound) and the same DP driven by
train climatology quantiles (a Layer-1-style unconditional policy).

Battery: 1 MW / 2 MWh, one-way efficiency 0.94 (~0.88 round trip),
degradation cost per MWh discharged, terminal SoC valued at 300 PLN/MWh
(~ median cost of re-charging in down periods).

Writes reports/bess_layer2_backtest.{json,txt}.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

P_MW = 1.0
E_MWH = 2.0
ETA = 0.94                    # one-way; round trip ~0.88
C_DEG = 100.0                 # PLN per MWh discharged (throughput cost)
TERMINAL_PLN_MWH = 300.0
N_SOC = 17                    # SoC grid resolution
DT = 0.25                     # hours per period

QS = (0.1, 0.25, 0.5, 0.75, 0.9)
W = np.full(len(QS), 1.0 / len(QS))
SOC = np.linspace(0.0, E_MWH, N_SOC)
DE_GRID = P_MW * DT           # MWh exchanged with the grid per activation
DE_DIS = DE_GRID / ETA        # SoC drawdown per discharge
DE_CHG = DE_GRID * ETA        # SoC gain per charge


def day_policy(p_up: np.ndarray, xq_up: np.ndarray, xq_dn: np.ndarray):
    """Backward induction over one day.

    p_up: (T,) P(dir=G); xq_up/xq_dn: (T, 5) conditional ceb_pp quantiles.
    Returns V0(s) and reservation prices r_up, r_dn of shape (T, N_SOC).
    """
    T = len(p_up)
    v_next = SOC * TERMINAL_PLN_MWH
    r_up = np.zeros((T, N_SOC))
    r_dn = np.zeros((T, N_SOC))
    for t in range(T - 1, -1, -1):
        v_dis = np.interp(np.maximum(SOC - DE_DIS, 0.0), SOC, v_next)
        v_chg = np.interp(np.minimum(SOC + DE_CHG, E_MWH), SOC, v_next)
        can_dis = SOC >= DE_DIS - 1e-9
        can_chg = SOC <= E_MWH - DE_CHG + 1e-9
        # reservation prices (PLN/MWh of grid energy)
        r_up[t] = np.where(can_dis,
                           C_DEG + (v_next - v_dis) / DE_GRID, np.inf)
        r_dn[t] = np.where(can_chg, (v_chg - v_next) / DE_GRID, -np.inf)
        # expected value: activate exactly when beneficial
        gain_up = np.zeros(N_SOC)
        gain_dn = np.zeros(N_SOC)
        for j, w in enumerate(W):
            x_u = xq_up[t, j]
            gu = x_u * DE_GRID - C_DEG * DE_GRID + v_dis - v_next
            gain_up += w * np.where(can_dis & (gu > 0), gu, 0.0)
            x_d = xq_dn[t, j]
            gd = -x_d * DE_GRID + v_chg - v_next
            gain_dn += w * np.where(can_chg & (gd > 0), gd, 0.0)
        v_next = v_next + p_up[t] * gain_up + (1.0 - p_up[t]) * gain_dn
    return v_next, r_up, r_dn


def simulate_day(day: pd.DataFrame, r_up: np.ndarray, r_dn: np.ndarray,
                 soc0: float) -> tuple[float, float, float]:
    """Play policy vs realized (dir, ceb_pp). Returns (pnl, mwh_dis, soc_end)."""
    s, pnl, dis = soc0, 0.0, 0.0
    for t, (_, row) in enumerate(day.iterrows()):
        ru = np.interp(s, SOC, r_up[t])
        rd = np.interp(s, SOC, r_dn[t])
        x = row["ceb_pp"]
        if row["dir"] == "G" and s >= DE_DIS - 1e-9 and x >= ru:
            pnl += (x - C_DEG) * DE_GRID
            s -= DE_DIS
            dis += DE_GRID
        elif row["dir"] == "D" and s <= E_MWH - DE_CHG + 1e-9 and x <= rd:
            pnl -= x * DE_GRID
            s += DE_CHG
    return pnl, dis, s


def perfect_foresight(day: pd.DataFrame, soc0: float) -> float:
    """DP on realized prices (deterministic upper bound)."""
    n = len(day)
    dirs = day["dir"].to_numpy()
    xs = day["ceb_pp"].to_numpy()
    v = SOC * TERMINAL_PLN_MWH
    for t in range(n - 1, -1, -1):
        v_dis = np.interp(np.maximum(SOC - DE_DIS, 0.0), SOC, v)
        v_chg = np.interp(np.minimum(SOC + DE_CHG, E_MWH), SOC, v)
        stay = v.copy()
        if dirs[t] == "G":
            gu = (xs[t] - C_DEG) * DE_GRID + v_dis
            v = np.where((SOC >= DE_DIS - 1e-9) & (gu > stay), gu, stay)
        else:
            gd = -xs[t] * DE_GRID + v_chg
            v = np.where((SOC <= E_MWH - DE_CHG + 1e-9) & (gd > stay), gd, stay)
    return float(np.interp(soc0, SOC, v) - np.interp(soc0, SOC,
                 SOC * TERMINAL_PLN_MWH))


def main():
    pred = pd.read_parquet("data/proc/bess_cond_holdout.parquet")
    clim = pd.read_parquet("data/proc/bess_cond_climatology.parquet")
    loc = pd.to_datetime(pred["ts"]).dt.tz_convert("Europe/Warsaw")
    pred["day"] = loc.dt.date
    pred["qh_of_day"] = (loc.dt.hour * 4 + loc.dt.minute // 15)
    clim = clim.set_index("qh_of_day")

    results = {}
    for name in ("conditional", "climatology", "perfect"):
        pnl_days, dis_tot = [], 0.0
        soc = E_MWH / 2
        for _, day in pred.groupby("day", sort=True):
            if len(day) < 90:      # skip ragged first/last days
                continue
            if name == "perfect":
                pnl = perfect_foresight(day, soc)
                pnl_days.append(pnl)
                continue
            if name == "conditional":
                p_up = day["p_up"].to_numpy()
                xu = day[[f"up_q{q}" for q in QS]].to_numpy()
                xd = day[[f"dn_q{q}" for q in QS]].to_numpy()
            else:
                k = day["qh_of_day"]
                p_up = clim["p_up"].reindex(k).to_numpy()
                xu = clim[[f"up_q{q}" for q in QS]].reindex(k).to_numpy()
                xd = clim[[f"dn_q{q}" for q in QS]].reindex(k).to_numpy()
            _, r_up, r_dn = day_policy(p_up, xu, xd)
            pnl, dis, soc_end = simulate_day(day, r_up, r_dn, soc)
            pnl += (soc_end - soc) * TERMINAL_PLN_MWH   # value SoC drift
            soc = soc_end                                # carry SoC overnight
            pnl_days.append(pnl)
            dis_tot += dis
        arr = np.array(pnl_days)
        results[name] = {
            "n_days": len(arr),
            "pnl_per_day_mean": round(float(arr.mean()), 1),
            "pnl_per_day_median": round(float(np.median(arr)), 1),
            "pnl_per_day_p10": round(float(np.quantile(arr, .1)), 1),
            "pnl_per_day_p90": round(float(np.quantile(arr, .9)), 1),
            "share_days_positive": round(float((arr > 0).mean()), 3),
            "annualized_per_mw": round(float(arr.mean() * 365), 0),
        }
        if name != "perfect":
            results[name]["cycles_per_day"] = round(
                float(dis_tot / max(len(arr), 1) / E_MWH), 2)

    cfg = {"P_MW": P_MW, "E_MWH": E_MWH, "eta_oneway": ETA, "c_deg": C_DEG,
           "terminal_pln_mwh": TERMINAL_PLN_MWH,
           "settlement": "uniform CEB_PP; activation iff dir matches and "
                         "ceb_pp clears reservation price"}
    out = {"config": cfg, "results": results}
    pathlib.Path("reports/bess_layer2_backtest.json").write_text(
        json.dumps(out, indent=2))
    lines = [f"BESS Layer 2 holdout simulation ({results['conditional']['n_days']} days), "
             f"1 MW / 2 MWh, eta_rt~{ETA**2:.2f}, c_deg={C_DEG:.0f}"]
    for name, r in results.items():
        lines.append(
            f"{name:12s} PLN/day mean={r['pnl_per_day_mean']:8.1f} "
            f"median={r['pnl_per_day_median']:8.1f} "
            f"[P10 {r['pnl_per_day_p10']:8.1f}, P90 {r['pnl_per_day_p90']:8.1f}] "
            f"pos={r['share_days_positive']:.0%} "
            f"cycles/d={r.get('cycles_per_day', '-')} "
            f"annualized/MW={r['annualized_per_mw']:,.0f}")
    txt = "\n".join(lines)
    pathlib.Path("reports/bess_layer2_backtest.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
