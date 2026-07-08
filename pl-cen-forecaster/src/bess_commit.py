"""
Commit-or-stay-free: the D-1 morning decision that joins F27 and F28.

    python -m src.bess_commit

Each hour of day D, decided at the D-1 ~07:30 capacity-bid gate: COMMIT the
battery to aFRR standby (earn the capacity fee, accept activation and the
SoC housekeeping of F27) or KEEP THE HOUR FREE and use it for imbalance-price
energy arbitrage. The two forecasters supply the two sides of the ledger:

  fee side     F28 walk-forward CMBP medians, fc(afrr_g)+fc(afrr_d) — the
               only gate-legal view of tomorrow's standby fee
  energy side  EV(h): trailing-30-day same-hour mean of the profitable part
               of "sell the hour's best quarter, refill at the median",
               eta- and degradation-adjusted, lagged 3 days (CEN publishes
               D+1 14:00, so D-3 is the freshest settled day at the gate)

Commit iff fee forecast >= EV(h). Free hours dispatch in real time with the
60-min-gate CEN forecast medians (LEAR wf + GBM holdout, as in F27):
discharge a quarter-hour when fc >= trailing q90 (SoC permitting), recharge
when fc <= trailing q10; settle at realized CEN. Committed hours run the F27
machinery unchanged (capacity offers, realized activation, W=2 timed
recovery).

Policies on the identical realized path (2025-07 -> 2026-07):
  always  commit every hour (the F27 'fcst' stack — baseline)
  fcst    commit by F28 forecast vs EV      (deployable)
  oracle  commit by realized CMBP vs EV     (ceiling for the decision)
Writes reports/bess_commit.json.
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
W = "Europe/Warsaw"
TRAIL_D = 30
FREE_GUARD = 0.65             # SoC clearance to trade a free quarter-hour


def hourly_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour: CMBP fee forecast (F28), realized fee, and EV(h)."""
    h = df.set_index("ts")["cen_cost"].resample("1h").max().rename("hmax").to_frame()
    med = (df.set_index("ts")["cen_cost"].resample("1D").median()
           .rolling(TRAIL_D, min_periods=10).mean().shift(3))       # D-3 legal
    h["med30"] = med.reindex(h.index, method="ffill")
    # profitable part of: discharge at the hour's best quarter, refill at the
    # trailing median (round-trip efficiency + throughput cost)
    h["prof"] = np.maximum(h["hmax"] * ETA**2 - h["med30"] - C_DEG, 0.0) * P_MW * DT
    hod = h.index.tz_convert(W).hour
    h["ev"] = (h["prof"].groupby(hod).transform(
        lambda s: s.rolling(TRAIL_D, min_periods=10).mean()).shift(3 * 24 // 24))
    # ^ same-hour trailing mean of realized prof, shifted 3 days (gate-legal)
    h["ev"] = h.groupby(hod)["prof"].transform(
        lambda s: s.rolling(TRAIL_D, min_periods=10).mean().shift(3))

    for t in ("g", "d"):
        wf = pd.read_parquet(f"reports/cmbp_wf_afrr_{t}.parquet")[["ts", "q0.5"]]
        h = h.join(wf.set_index("ts")["q0.5"].rename(f"fc_{t}"))
    h["fee_fc"] = h["fc_g"] + h["fc_d"]
    return h[["ev", "fee_fc"]]


def free_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Real-time-legal trailing CEN q10/q90 (2-day publication lag)."""
    s = df.set_index("ts")["cen_cost"]
    daily_q90 = s.resample("1D").quantile(0.9).rolling(TRAIL_D, min_periods=10).mean()
    daily_q10 = s.resample("1D").quantile(0.1).rolling(TRAIL_D, min_periods=10).mean()
    out = pd.DataFrame({"q90": daily_q90.shift(2), "q10": daily_q10.shift(2)})
    return out.reindex(df["ts"].dt.floor("1D").unique())


def simulate(df: pd.DataFrame, commit: np.ndarray) -> dict:
    """F27 'fcst' machinery on committed periods; forecast-band arbitrage on
    free ones. commit: bool per 15-min row."""
    cen = df["cen_cost"].to_numpy()
    fc = df["fc"].to_numpy()
    q90 = df["q90"].to_numpy()
    q10 = df["q10"].to_numpy()

    s = E_MWH / 2
    cap = act = rec = free = 0.0
    dis_mwh = 0.0
    n_free_tr = 0
    pending = None
    netq: dict[str, float] = {}

    for i, r in enumerate(df.itertuples()):
        pnl_i = 0.0
        if not commit[i]:
            # free quarter-hour: band-triggered arbitrage on the 60-min forecast
            if fc[i] >= q90[i] and s >= FREE_GUARD:
                e = min(P_MW * DT, s * ETA)
                pnl_i = (cen[i] - C_DEG) * e
                free += pnl_i
                s -= e / ETA
                dis_mwh += e
                n_free_tr += 1
            elif fc[i] <= q10[i] and s <= E_MWH - FREE_GUARD:
                e = min(P_MW * DT, (E_MWH - s) / ETA)
                pnl_i = -cen[i] * e
                free += pnl_i
                s += e * ETA
                n_free_tr += 1
            netq[r.quarter] = netq.get(r.quarter, 0.0) + pnl_i
            pending = None
            continue

        # committed: F27 fcst policy (timed recovery, capacity, activation)
        recover = 0
        if pending is not None:
            d, j = pending
            if SOC_LO <= s <= SOC_HI:
                pending = None
            elif i == j or (d == +1 and s > E_MWH - EMERG) or (d == -1 and s < EMERG):
                recover, pending = d, None
        elif not (SOC_LO <= s <= SOC_HI):
            d = +1 if s > SOC_HI else -1
            hi = min(i + WINDOW, len(df) - 1)
            wnd = fc[i:hi + 1]
            j = i + int(np.argmax(wnd) if d == +1 else np.argmin(wnd))
            if j == i or np.isnan(wnd).any():
                recover = d
            else:
                pending = (d, j)

        if recover == +1:
            e = min(P_MW * DT, s * ETA)
            pnl_i += (cen[i] - C_DEG) * e
            rec += (cen[i] - C_DEG) * e
            s -= e / ETA
            dis_mwh += e
        elif recover == -1:
            e = min(P_MW * DT, (E_MWH - s) / ETA)
            pnl_i -= cen[i] * e
            rec -= cen[i] * e
            s += e * ETA

        off_up = 1 if (s >= BUFFER and recover != +1) else 0
        off_dn = 1 if (s <= E_MWH - BUFFER and recover != -1) else 0
        c = (off_up * r.afrr_g + off_dn * r.afrr_d) * P_MW * DT
        cap += c
        e_up = off_up * r.u_up * P_MW * DT
        e_dn = off_dn * r.u_dn * P_MW * DT
        a = r.ceb_up * e_up - r.ceb_dn * e_dn
        act += a
        s = min(max(s - e_up / ETA + e_dn * ETA, 0.0), E_MWH)
        dis_mwh += e_up
        netq[r.quarter] = netq.get(r.quarter, 0.0) + pnl_i + c + a

    days = len(df) / 96
    k = 365 / days
    return {"cap_ann": round(cap * k, 0), "act_ann": round(act * k, 0),
            "rec_ann": round(rec * k, 0), "free_ann": round(free * k, 0),
            "net_ann": round((cap + act + rec + free) * k, 0),
            "free_share_pct": round(float((~commit).mean()) * 100, 1),
            "n_free_trades": n_free_tr,
            "cycles_per_day": round(dis_mwh / days / E_MWH, 2),
            "net_by_quarter": {q: round(v * 365 / (df["quarter"] == q).sum() * 96, 0)
                               for q, v in sorted(netq.items())}}


def main():
    df = attach_forecast(load())
    hl = hourly_layer(df)
    df["hour_ts"] = df["ts"].dt.floor("1h")
    df = df.merge(hl, left_on="hour_ts", right_index=True, how="left")
    fb = free_bands(df)
    df = df.merge(fb, left_on=df["ts"].dt.floor("1D"), right_index=True, how="left")
    df = df.dropna(subset=["ev", "fee_fc", "q90", "q10"]).reset_index(drop=True)
    fee_real = (df.set_index("ts")[["afrr_g", "afrr_d"]].sum(axis=1)
                .groupby(df.set_index("ts").index.floor("1h")).transform("mean"))
    df["fee_real"] = fee_real.to_numpy()

    print(f"{len(df)} periods, {df['ts'].min():%Y-%m-%d} .. {df['ts'].max():%Y-%m-%d}")
    print(f"hours with EV > fee forecast: "
          f"{float((df['ev'] > df['fee_fc']).mean()):.1%}\n")

    out = {"policies": {}}
    masks = {"always": np.ones(len(df), bool),
             "fcst": (df["fee_fc"] >= df["ev"]).to_numpy(),
             "oracle": (df["fee_real"] >= df["ev"]).to_numpy()}
    base = None
    for name, m in masks.items():
        r = simulate(df, m)
        out["policies"][name] = r
        base = base or r
        print(f"== {name:6} net {r['net_ann']:>10,.0f}  (vs always "
              f"{r['net_ann'] - base['net_ann']:+10,.0f})  free {r['free_share_pct']}% "
              f"({r['n_free_trades']} trades, {r['free_ann']:+,.0f}/yr)")
        print(f"   capacity {r['cap_ann']:>10,.0f}  activation {r['act_ann']:>9,.0f}  "
              f"recovery {r['rec_ann']:>9,.0f}  cycles/day {r['cycles_per_day']}")
        print(f"   net by quarter: {r['net_by_quarter']}\n")

    pathlib.Path("reports/bess_commit.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/bess_commit.json")


if __name__ == "__main__":
    main()
