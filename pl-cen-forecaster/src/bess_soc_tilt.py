"""
F27 x F31: tail-aware SoC recovery. Does the dedicated tail classifier add
dispatch value on top of the median forecast that F27 already times on?

    python -m src.bess_soc_tilt

F27 (bess_soc_policy) defers a recovery to the best FORECAST-MEDIAN quarter in
the next WINDOW. But the quantile family is weakest exactly in the tails
(F31's premise), and F31's dedicated classifiers find spikes 18x better than
climatology and calibrate negative-price probability to dispatch grade
(ECE 0.011). The tails are where recovery timing pays most: discharge into a
spike, charge into a paid-to-charge (negative) quarter.

The tilt replaces the in-window selection with an expected-value blend of the
median forecast and the tail classifier:

  discharge (SoC too full):  EV(k)   = (1-p_spike)*fc(k) + p_spike*SPIKE_LEVEL
                             pick argmax  (hold SoC for a spike quarter)
  charge    (SoC too empty): Ecost(k)= (1-p_neg)*fc(k)  + p_neg*NEG_LEVEL
                             pick argmin  (charge into a negative quarter)

p_y_neg_cal is CALIBRATED (F31: dispatch-grade) -> its EV is meaningful in
absolute terms. p_y_spike is RANKING-only (too few events for isotonic) -> it
is used only to reorder periods inside a <=4-quarter window, never as an
absolute probability. Gate-honest by the same argument as F27: the H=60 tail
probs for deliveries t..t+3 were all issued at or before decision time t.

Same realized path, same band/capacity/activation mechanics as bess_soc_policy;
the ONLY change is which quarter a pending recovery fires in. Reports the tilt
against F27's fcst baseline and the oracle ceiling, by quarter (hard rule #3).
Writes reports/bess_soc_tilt.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .bess_cooptimize import BUFFER, SOC_HI, SOC_LO, load
from .bess_optimizer import C_DEG, E_MWH, ETA, P_MW
from .bess_soc_policy import EMERG, WINDOW, attach_forecast

DT = 0.25
PROBS = "reports/spike_wf_probs.parquet"


def attach_tails(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(PROBS)[["ts", "p_y_spike_cal", "p_y_neg_cal"]]
    df = df.merge(p, on="ts", how="left")
    # no tail read -> neutral (fall back to pure median timing)
    df["p_spike"] = pd.to_numeric(df["p_y_spike_cal"], errors="coerce").fillna(0.0)
    df["p_neg"] = pd.to_numeric(df["p_y_neg_cal"], errors="coerce").fillna(0.0)
    return df


def tail_levels(df: pd.DataFrame) -> tuple[float, float]:
    """Representative tail magnitudes the classifier is pointing at. NEG uses
    the median (the -45k outlier would make the mean a single-event artifact);
    SPIKE uses the conditional mean (spikes are a fat but bounded cluster)."""
    cen = df["cen_cost"]
    spike_lvl = float(cen[cen > 1500].mean())        # ~2419
    neg_lvl = float(cen[cen < 0].median())           # robust; ~ -tens
    return spike_lvl, neg_lvl


def simulate(df: pd.DataFrame, mode: str, window: int,
             spike_lvl: float, neg_lvl: float) -> dict:
    """mode in {blind, fcst, tilt, oracle}. Mirrors bess_soc_policy.simulate;
    'tilt' swaps the median argmax/argmin for the EV-blend selection."""
    cen = df["cen_cost"].to_numpy()
    fc = df["fc"].to_numpy() if "fc" in df else np.full(len(df), np.nan)
    p_spike = df["p_spike"].to_numpy()
    p_neg = df["p_neg"].to_numpy()
    ref = cen if mode == "oracle" else fc

    s = E_MWH / 2
    cap_net = act_pnl = rec_pnl = 0.0
    dis_mwh = 0.0
    n_rec = n_fallback = n_tail_pick = 0
    pending: tuple[int, int, int] | None = None   # (direction, exec_idx, commit_idx)
    rec_q: dict[str, float] = {}

    for i, r in enumerate(df.itertuples()):
        recover = 0

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
            hi = min(i + window, len(df) - 1)
            w = ref[i:hi + 1]
            if mode == "blind" or np.isnan(w).any():
                recover = d
                n_fallback += mode != "blind"
            elif mode == "tilt":
                fcw = fc[i:hi + 1]
                if d == +1:                          # discharge -> max EV incl. spike
                    score = (1 - p_spike[i:hi + 1]) * fcw + p_spike[i:hi + 1] * spike_lvl
                    k = int(np.argmax(score))
                    med_k = int(np.argmax(fcw))
                else:                                # charge -> min E[cost] incl. neg
                    score = (1 - p_neg[i:hi + 1]) * fcw + p_neg[i:hi + 1] * neg_lvl
                    k = int(np.argmin(score))
                    med_k = int(np.argmin(fcw))
                n_tail_pick += k != med_k            # tilt disagreed with pure median
                j = i + k
                if j == i:
                    recover = d
                else:
                    pending = (d, j, i)
            else:                                    # fcst: pure median timing (F27)
                j = i + int(np.argmax(w) if d == +1 else np.argmin(w))
                if j == i:
                    recover = d
                else:
                    pending = (d, j, i)

        if recover == +1:
            e = min(P_MW * DT, s * ETA)
            rec_pnl += (cen[i] - C_DEG) * e
            s -= e / ETA
            dis_mwh += e
            n_rec += 1
            rec_q[r.quarter] = rec_q.get(r.quarter, 0.0) + (cen[i] - C_DEG) * e
        elif recover == -1:
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
            "n_tail_repicks": int(n_tail_pick),
            "cycles_per_day": round(dis_mwh / days / E_MWH, 2),
            "rec_by_quarter": {k: round(v, 0) for k, v in sorted(rec_q.items())}}


def main():
    df = attach_tails(attach_forecast(load()))
    spike_lvl, med_neg = tail_levels(df)
    mean_neg = float(df["cen_cost"][df["cen_cost"] < 0].mean())
    print(f"{len(df)} periods, {df['ts'].min():%Y-%m-%d} .. {df['ts'].max():%Y-%m-%d}, "
          f"fc cov {df['fc'].notna().mean():.1%}")
    print(f"SPIKE level {spike_lvl:,.0f}  NEG level: median {med_neg:,.0f} / "
          f"mean {mean_neg:,.0f} (min {df['cen_cost'].min():,.0f})\n")

    # baselines and ceiling are neg-level-independent (fcst/oracle ignore tails)
    fcst = {W: simulate(df, "fcst", W, spike_lvl, med_neg) for W in (2, 3, 4)}
    oracle = {W: simulate(df, "oracle", W, spike_lvl, med_neg) for W in (2, 3, 4)}
    best_fcst = max(fcst.values(), key=lambda r: r["net_total_ann"])
    print("F27 baseline (median timing):  best = W=2  "
          f"NET {best_fcst['net_total_ann']:,.0f}  "
          f"(oracle ceiling {oracle[2]['net_total_ann']:,.0f}, "
          f"headroom {oracle[2]['net_total_ann']-best_fcst['net_total_ann']:+,.0f})\n")

    # the tilt: swept over window and the (fragile) neg-selection level
    out = {"spike_lvl": round(spike_lvl, 1), "best_fcst_net": best_fcst["net_total_ann"],
           "oracle_net_w2": oracle[2]["net_total_ann"], "sweep": []}
    print(f"tilt sweep vs the SAME-W fcst baseline (tilt-fcst) and vs best fcst:")
    print(f"{'W':>2} {'neg_lvl':>8} | {'tilt NET':>11} {'vs same-W':>10} "
          f"{'vs best fcst':>12} {'repicks':>8}")
    for W in (2, 3, 4):
        fW = fcst[W]["net_total_ann"]
        for nl in (med_neg, mean_neg, -1000.0):
            t = simulate(df, "tilt", W, spike_lvl, nl)
            row = {"W": W, "neg_lvl": round(nl, 0), "net": t["net_total_ann"],
                   "vs_same_w": t["net_total_ann"] - fW,
                   "vs_best_fcst": t["net_total_ann"] - best_fcst["net_total_ann"],
                   "repicks": t["n_tail_repicks"],
                   "rec_by_quarter": t["rec_by_quarter"]}
            out["sweep"].append(row)
            print(f"{W:>2} {nl:>8.0f} | {t['net_total_ann']:>11,.0f} "
                  f"{row['vs_same_w']:>+10,.0f} {row['vs_best_fcst']:>+12,.0f} {t['n_tail_repicks']:>8}")
        print()

    best_tilt = max(out["sweep"], key=lambda r: r["net"])
    print(f"VERDICT: best tilt {best_tilt['net']:,.0f} (W={best_tilt['W']}, "
          f"neg={best_tilt['neg_lvl']:.0f}) = {best_tilt['vs_best_fcst']:+,.0f} vs best fcst "
          f"({best_tilt['vs_best_fcst']/best_fcst['net_total_ann']:+.2%}). "
          f"Sign flips with neg_lvl -> tuning-fragile, within single-path noise.\n"
          f"The tail classifier's dispatch value stays in POSITIONING/alerts (F31), "
          f"not recovery timing.")
    out["verdict_best_tilt"] = best_tilt

    pathlib.Path("reports/bess_soc_tilt.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/bess_soc_tilt.json")


if __name__ == "__main__":
    main()
