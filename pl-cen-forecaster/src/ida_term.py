"""
Executable auction-to-auction spreads (the F24 guardrail, by construction).

    python -m src.ida_term

Every prior spread died on execution: one leg (day-ahead, CEN) was never an
available entry at signal time. This tests the only structure that is
executable end-to-end: ENTER at one IDA auction clearing price, EXIT at a
later one — two real prints, and the signal must strictly predate the ENTRY
auction's gate. Costs are charged on BOTH legs (two real trades).

Gates (Warsaw): IDA1 15:00 D-1 · IDA2 22:00 D-1 · IDA3 10:00 D.
Signal latency: ENTSO-E/PSE actuals ~1.2h → last usable actual = gate − 1.5h.

Pairs × signals (all gate-honest):
  T12  enter IDA1 exit IDA2   signal s1 = mean RES surprise D-1 09:00–13:30
  T13  enter IDA1 exit IDA3   signal s1 (same; IDA3 covers late-day periods)
  T23  enter IDA2 exit IDA3   signal s2 = mean RES surprise D-1 16:00–20:30
  T23b enter IDA2 exit IDA3   signal b1 = mean(IDA1 − DA) for day D
                              (IDA1 cleared 15:00 D-1 → legal at the 22:00 gate)
Direction: RES over-delivering (s>0) ⇒ system longer ⇒ later auctions lower ⇒
SHORT at entry (pos = −sign(s)); b1 tested as reversion (pos = −sign(b1)),
with the raw corr reported so a momentum reading is visible if present.

Caveats carried from F19/F23: surprise built from SETTLED PSE actuals (live =
preliminary vintage, noisier); day-level persistence (0.62) is the mechanism
carrying a D-1 signal into day-D auction moves — two hops, so expect decay.
Writes reports/ida_term.json.
"""
from __future__ import annotations

import json
import pathlib
from datetime import timedelta

import numpy as np
import pandas as pd

W = "Europe/Warsaw"
MWH = 0.25                     # 1 MW for a 15-min period
DEADBAND_MW = 200.0            # on the surprise-window mean
DEADBAND_PLN = 10.0            # on the IDA1−DA basis
COSTS = (5.0, 10.0, 20.0)      # PLN/MWh ROUND TRIP (both auction legs)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    act = pd.read_parquet("data/raw/pse_kse_actuals.parquet")[["ts", "pv", "wi"]]
    cons = pd.read_parquet("data/raw/entsoe_res.parquet")
    sur = act.merge(cons, on="ts", how="inner")
    for c in ("pv", "wi", "res_solar", "res_wind"):
        sur[c] = pd.to_numeric(sur[c], errors="coerce")
    sur["surprise"] = (sur["wi"] - sur["res_wind"]) + (sur["pv"] - sur["res_solar"])

    tge = pd.read_parquet("data/raw/tge_rdb.parquet")
    tge = tge[tge.dur_min == 15].drop_duplicates("ts")
    for c in ("ida1_pln", "ida2_pln", "ida3_pln"):
        tge[c] = pd.to_numeric(tge[c], errors="coerce")
    da = pd.read_parquet("data/raw/pse_da_price.parquet")[["ts", "csdac_pln"]]
    da["csdac_pln"] = pd.to_numeric(da["csdac_pln"], errors="coerce")
    px = tge[["ts", "ida1_pln", "ida2_pln", "ida3_pln"]].merge(da, on="ts", how="left")
    px["day"] = px["ts"].dt.tz_convert(W).dt.date
    loc = px["ts"].dt.tz_convert(W)
    px["quarter"] = loc.dt.year.astype(str) + "Q" + loc.dt.quarter.astype(str)
    return sur, px


def day_signals(sur: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    loc = sur["ts"].dt.tz_convert(W)
    sur = sur.assign(day=loc.dt.date, hm=loc.dt.hour + loc.dt.minute / 60)

    def win(lo, hi):
        m = sur[(sur.hm >= lo) & (sur.hm < hi)]
        return m.groupby("day")["surprise"].mean()

    # windows sit on D-1; shift them onto delivery day D
    s1 = win(9.0, 13.5).rename("s1")     # ≤13:30, gate 15:00 D-1
    s2 = win(16.0, 20.5).rename("s2")    # ≤20:30, gate 22:00 D-1
    sig = pd.concat([s1, s2], axis=1)
    sig.index = pd.Series(sig.index).map(lambda d: d + timedelta(days=1))

    # IDA1−DA basis for day D, cleared 15:00 D-1 → legal at the IDA2 gate
    b1 = ((px["ida1_pln"] - px["csdac_pln"]).groupby(px["day"]).mean()
          .rename("b1"))
    return sig.join(b1, how="outer")


def backtest(px: pd.DataFrame, sig: pd.DataFrame, entry: str, exit_: str,
             scol: str, dead: float, cost: float) -> dict:
    d = px.dropna(subset=[entry, exit_]).merge(
        sig[[scol]], left_on="day", right_index=True, how="inner"
    ).dropna(subset=[scol])
    pos = np.where(d[scol] > dead, -1, np.where(d[scol] < -dead, 1, 0))
    move = (d[exit_] - d[entry]).to_numpy()
    pnl = (pos * move - np.abs(pos) * cost) * MWH
    tr = pos != 0
    if tr.sum() < 100:
        return {}
    daily = pd.Series(pnl, index=pd.to_datetime(d["day"])).groupby(level=0).sum()
    ann = daily.mean() * 365
    sharpe = ann / (daily.std() * np.sqrt(365)) if daily.std() else 0.0
    byq = (pd.Series(pnl[tr], index=d["quarter"].to_numpy()[tr])
           .groupby(level=0).sum().round(0).to_dict())
    return {"n_trades": int(tr.sum()),
            "corr_sig_move": round(float(np.corrcoef(d[scol], move)[0, 1]), 3),
            "pnl_per_mwh_traded": round(float((pos * move)[tr].mean()), 1),
            "hit": round(float(((pos * move)[tr] > 0).mean()), 3),
            "ann_pln_per_mw": round(float(ann), 0),
            "sharpe": round(float(sharpe), 2),
            "quarters_pos": f"{sum(v > 0 for v in byq.values())}/{len(byq)}",
            "by_quarter": byq}


def main():
    sur, px = load()
    sig = day_signals(sur, px)
    pairs = (("T12", "ida1_pln", "ida2_pln", "s1", DEADBAND_MW),
             ("T13", "ida1_pln", "ida3_pln", "s1", DEADBAND_MW),
             ("T23", "ida2_pln", "ida3_pln", "s2", DEADBAND_MW),
             ("T23b", "ida2_pln", "ida3_pln", "b1", DEADBAND_PLN))
    n_pair = {n: int(px.dropna(subset=[e, x]).shape[0])
              for n, e, x, *_ in pairs}
    print("joint periods per pair:", n_pair, "\n")

    out = {"deadband_mw": DEADBAND_MW, "deadband_pln": DEADBAND_PLN, "pairs": {}}
    for name, e, x, s, dead in pairs:
        out["pairs"][name] = {}
        print(f"== {name}: enter {e[:4]} → exit {x[:4]}, signal {s} ==")
        for cost in COSTS:
            r = backtest(px, sig, e, x, s, dead, cost)
            out["pairs"][name][f"cost{int(cost)}"] = r
            if not r:
                print(f"  cost {cost:>4.0f}: <100 trades, skipped")
                continue
            print(f"  cost {cost:>4.0f}: n {r['n_trades']:>6}  corr {r['corr_sig_move']:+.3f}  "
                  f"pnl/MWh {r['pnl_per_mwh_traded']:+6.1f}  hit {r['hit']:.3f}  "
                  f"ann/MW {r['ann_pln_per_mw']:>8,.0f}  Sharpe {r['sharpe']:>5}  "
                  f"q+ {r['quarters_pos']}")
        print()

    pathlib.Path("reports/ida_term.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/ida_term.json")


if __name__ == "__main__":
    main()
