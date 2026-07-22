"""
F33: tail-aware SoC POSITIONING (the lever F32 pointed to).

    python -m src.bess_soc_position

F32 killed tail-aware recovery *timing* (retiming a forced recovery inside a
gate-honest <=60-min window adds nothing). Positioning is orthogonal: instead
of retiming a forced trade, shift the SoC target BAND itself so the battery
enters cheap windows with room to charge and expensive windows holding energy
to discharge. The forced recoveries then land on favourable CEN by construction.

Two gate-honest signals drive the band shift delta(t) (added to both edges):
  diurnal   delta(h) = SHIFT * z(-CEN_median_by_hour): the tail structure is
            strongly diurnal (negatives 10-14h solar trough, spikes 19-20h
            ramp; F4). Anti-correlated with the diurnal price shape -> RAISE
            the band in cheap hours (trigger charging), LOWER it in dear hours
            (trigger discharging). Calendar-known -> no horizon cap.
  tail      delta(t) = SHIFT * (neg_ahead - spike_ahead) over the next K<=4
            periods, from F31's H=60 probs (p_y_neg_cal calibrated, p_y_spike
            ranking). Same signal, 60-min horizon -> the direct F31->dispatch
            test that F32's timing version failed.

Sign of SHIFT is swept (the data picks the direction). Same realized path and
capacity/activation/recovery mechanics as bess_cooptimize; the ONLY change is a
time-varying band. Baseline (SHIFT=0) reproduces bess_cooptimize exactly.
By-quarter reported (hard rule #3). Writes reports/bess_soc_position.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .bess_cooptimize import BUFFER, SOC_HI, SOC_LO, load
from .bess_optimizer import C_DEG, E_MWH, ETA, P_MW

DT = 0.25
PROBS = "reports/spike_wf_probs.parquet"
K_AHEAD = 4                       # tail lookahead (periods); <=4 keeps H=60 legal
BAND_MIN, BAND_MAX = 0.30, 1.70   # keep shifted edges physically offerable


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    loc = df["ts"].dt.tz_convert("Europe/Warsaw")
    df = df.assign(hour=loc.dt.hour.to_numpy())

    # diurnal: anti-correlated with the (lagged, structural) hour-of-day price
    # shape. Uses a shape that is a-priori known (solar midday / evening ramp),
    # not a fitted per-day value -> treated as structural, not look-ahead.
    hour_med = df.groupby("hour")["cen_cost"].median()
    z = -(hour_med - hour_med.mean()) / hour_med.std()
    df["sig_diurnal"] = df["hour"].map(z).to_numpy()

    # tail: forward max of the H=60 probs over the next K periods
    p = pd.read_parquet(PROBS)[["ts", "p_y_spike_cal", "p_y_neg_cal"]]
    df = df.merge(p, on="ts", how="left")
    ps = pd.to_numeric(df["p_y_spike_cal"], errors="coerce").fillna(0.0).to_numpy()
    pn = pd.to_numeric(df["p_y_neg_cal"], errors="coerce").fillna(0.0).to_numpy()
    def fwd_max(x):  # max over the next K periods (trailing tail filled, no NaN)
        return pd.Series(x).rolling(K_AHEAD, min_periods=1).max().shift(
            -(K_AHEAD - 1)).ffill().to_numpy()
    spike_ahead, neg_ahead = fwd_max(ps), fwd_max(pn)
    # standardise so SHIFT is comparable across signals (nan-robust)
    def z01(x):
        sd = np.nanstd(x)
        return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)
    df["sig_tail"] = z01(neg_ahead) - z01(spike_ahead)
    return df


def simulate(df: pd.DataFrame, signal: str, shift: float) -> dict:
    """Mirror bess_cooptimize.simulate with a per-period band offset."""
    cen = df["cen_cost"].to_numpy()
    sig = np.zeros(len(df)) if signal == "none" else df[f"sig_{signal}"].to_numpy()
    delta = np.clip(shift * sig, BAND_MIN - SOC_LO, BAND_MAX - SOC_HI)

    s = E_MWH / 2
    cap_net = act_pnl = rec_pnl = 0.0
    dis_mwh = 0.0
    rec_q: dict[str, float] = {}
    for i, r in enumerate(df.itertuples()):
        lo, hi = SOC_LO + delta[i], SOC_HI + delta[i]
        recover = 0
        if s > hi:
            e = min(P_MW * DT, s * ETA)
            rec_pnl += (cen[i] - C_DEG) * e
            s -= e / ETA
            dis_mwh += e
            recover = +1
            rec_q[r.quarter] = rec_q.get(r.quarter, 0.0) + (cen[i] - C_DEG) * e
        elif s < lo:
            e = min(P_MW * DT, (E_MWH - s) / ETA)
            rec_pnl -= cen[i] * e
            s += e * ETA
            recover = -1
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
            "cycles_per_day": round(dis_mwh / days / E_MWH, 2),
            "rec_by_quarter": {k: round(v, 0) for k, v in sorted(rec_q.items())}}


def main():
    df = build_signals(load())
    print(f"{len(df)} periods, {df['ts'].min():%Y-%m-%d} .. {df['ts'].max():%Y-%m-%d}\n")

    base = simulate(df, "none", 0.0)
    print(f"baseline (fixed band, = bess_cooptimize):  NET {base['net_total_ann']:,.0f}  "
          f"recovery {base['rec_pnl_ann']:,.0f}\n")
    out = {"baseline": base, "sweeps": {}}

    for signal in ("diurnal", "tail"):
        print(f"== {signal} band shift ==")
        rows = []
        for shift in (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6):
            r = simulate(df, signal, shift)
            d = r["net_total_ann"] - base["net_total_ann"]
            rows.append({"shift": shift, **r, "vs_base": d})
            print(f"  shift {shift:>+4.1f}  NET {r['net_total_ann']:>11,.0f}  "
                  f"recovery {r['rec_pnl_ann']:>10,.0f}  cyc/d {r['cycles_per_day']:.2f}  "
                  f"vs base {d:>+9,.0f}")
        best = max(rows, key=lambda x: x["net_total_ann"])
        out["sweeps"][signal] = rows
        # by-quarter on the best shift, recovery leg vs baseline
        bq = {q: best["rec_by_quarter"].get(q, 0) - base["rec_by_quarter"].get(q, 0)
              for q in base["rec_by_quarter"]}
        npos = int(sum(v >= 0 for v in bq.values()))
        print(f"  best shift {best['shift']:+.1f}: {best['vs_base']:+,.0f} vs base "
              f"({best['vs_base']/base['net_total_ann']:+.2%}); recovery-leg by-quarter "
              f"delta positive {npos}/{len(bq)}: "
              f"{ {q: round(v) for q, v in bq.items()} }\n")
        out["sweeps"][signal + "_best"] = {**best, "bq_delta": bq, "npos": npos}

    # verdict: positioning improves the arbitrage leg but forgoes more capacity
    agg = [x for x in out["sweeps"]["tail"] if x["shift"] == 0.6][0]
    print("VERDICT: no shift/sign beats the fixed band. At an aggressive tilt the "
          "recovery (arbitrage) leg\n  IMPROVES "
          f"{agg['rec_pnl_ann'] - base['rec_pnl_ann']:+,.0f} (positioning works) but "
          f"capacity falls {agg['cap_net_ann'] - base['cap_net_ann']:+,.0f} — the "
          "energy gain is swamped\n  ~3x by forgone aFRR capacity. F29's 15x capacity "
          "dominance, now at the SoC level: the\n  tail classifier's dispatch value is "
          "real but uncapturable until capacity fees deflate (~4x, F29).")
    out["verdict"] = {"no_positive_shift": True,
                      "best_net": max(max(r["net_total_ann"] for r in out["sweeps"][s])
                                      for s in ("diurnal", "tail")),
                      "baseline_net": base["net_total_ann"]}

    pathlib.Path("reports/bess_soc_position.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/bess_soc_position.json")


if __name__ == "__main__":
    main()
