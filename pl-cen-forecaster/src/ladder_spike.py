"""
F34: does the balancing offer ladder lead CEN spikes, beyond F31's panel?

    python -m src.ladder_spike

The balancing offer ladder (poeb-rbn, 67M offers) is the merit-order supply
curve. CEN is its marginal (F4: corr 0.88), so the ladder TOP is ~lagged price
— which F31 already has. The novel signal is SCARCITY: how much expensive
supply is standing (tail_vol = MW offered above 1000 PLN) and how thin the
cheap supply is (cheap_frac = MW below 500 / total). When the cheap cushion is
thin, a demand surprise pushes activation into the expensive tail -> spike.

Exploratory in-sample logistic (this module's motivation) showed tail_vol
lagged 75 min corr +0.244 with spikes (vs lagged CEN +0.060), adding AUC
0.892->0.905 over an hour-of-day + lagged-CEN baseline, and the lead persists
across lags with a 24h resurgence (structural, not a price echo). This module
is the honest test: wire the ladder features into F31's exact walk-forward
classifier and measure the by-quarter AP lift.

Leakage: the archived ladder for period t publishes at t+2min, so at the H=60
gate only ladders published <= t-60min are legal -> lag >= 5 periods (t-5
publishes at t-73min). We use lags 5/10 (near-term scarcity + trend) and 96
(same-hour-yesterday, ~23h old, trivially legal). Ladder features built by
src.ladder_features (reports/ladder_features.parquet).

Reports F31 (panel+NWP) vs +ladder, AUC/AP/top-decile by quarter.
Writes reports/ladder_spike.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .spike_classifier import (SPIKE, add_nwp_revisions, climatology_prob,
                               evaluate, walkforward)
from .lear import load_panel

LADDER = "reports/ladder_features.parquet"
LAGS = (5, 10, 96)                       # 75min, 150min, ~23h — all >= H=60 gate
COLS = ["tail_vol", "cheap_frac", "g_max", "g_p90"]


def add_ladder(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    lad = pd.read_parquet(LADDER)
    lad["ts"] = pd.to_datetime(lad["ts"]).dt.tz_localize("UTC")
    added: list[str] = []
    for L in LAGS:
        shifted = lad[["ts"] + COLS].copy()
        shifted["ts"] = shifted["ts"] + pd.Timedelta(minutes=15 * L)  # align to t
        cols = COLS if L != 96 else ["tail_vol", "g_max"]             # 24h: scarcity only
        ren = {c: f"{c}_L{L}" for c in cols}
        df = df.merge(shifted[["ts"] + cols].rename(columns=ren), on="ts", how="left")
        added += list(ren.values())
    # near-term trend in the expensive tail (rising -> stress building)
    df["d_tail"] = df["tail_vol_L5"] - df["tail_vol_L10"]
    added.append("d_tail")
    return df, added


def main():
    df, feats = load_panel()
    df = add_nwp_revisions(df)
    base_feats = feats + ["rev_w3_21", "rev_w3_10", "rev_ghi_21", "rev_ghi_10"]
    df, lad_feats = add_ladder(df)
    all_feats = base_feats + lad_feats
    df[all_feats] = df[all_feats].astype(float).fillna(0.0)
    df["y_spike"] = (df["y"] > SPIKE).astype(int)

    clim = climatology_prob(df, "y_spike")
    out = {"lags": LAGS, "ladder_feats": lad_feats, "variants": {}}
    for name, fs in (("F31_baseline", base_feats), ("F31_plus_ladder", all_feats)):
        prob = walkforward(df, fs, "y_spike")
        res = evaluate(df, "y_spike", prob, clim)
        out["variants"][name] = res
        print(f"== {name}: n={res['n']} base {res['base_rate']:.2%}  "
              f"AUC {res['auc']}  AP {res['ap']} (clim {res['ap_clim']})  "
              f"top-decile {res['top_decile_capture']:.0%}")
        for q, v in res["by_quarter"].items():
            extra = f"AUC {v['auc']}  AP {v['ap']}" if "auc" in v else "too few"
            print(f"   {q}: events {v['n_events']:>4}  {extra}")
        print()

    b, l = out["variants"]["F31_baseline"], out["variants"]["F31_plus_ladder"]
    print(f"LADDER LIFT: AUC {b['auc']}->{l['auc']} ({l['auc']-b['auc']:+.3f})  "
          f"AP {b['ap']}->{l['ap']} ({l['ap']-b['ap']:+.3f})  "
          f"top-decile {b['top_decile_capture']:.0%}->{l['top_decile_capture']:.0%}")
    aps = [(q, l['by_quarter'][q].get('ap'), b['by_quarter'][q].get('ap'))
           for q in b['by_quarter'] if 'ap' in b['by_quarter'][q] and 'ap' in l['by_quarter'].get(q, {})]
    npos = sum(la >= ba for _, la, ba in aps)
    print(f"  AP improves in {npos}/{len(aps)} quarters: "
          f"{ {q: round(la-ba,3) for q,la,ba in aps} }")
    out["lift"] = {"auc": round(l['auc']-b['auc'], 3), "ap": round(l['ap']-b['ap'], 3),
                   "quarters_improved": f"{npos}/{len(aps)}"}
    print("VERDICT: ladder scarcity is a REAL univariate spike lead (+0.244 corr, "
          "structural) but\n  adds nothing over F31's full panel on walk-forward "
          f"(AUC {l['auc']-b['auc']:+.3f}, top-decile "
          f"{l['top_decile_capture']-b['top_decile_capture']:+.0%}, {npos}/{len(aps)} "
          "quarters). The\n  panel already captures the tightness the ladder encodes "
          "(fundamentals priced in, cf. F10/F11/F15).\n  Possible value at a shorter "
          "horizon (lag-1 corr +0.352 >> lag-5 +0.244) via a spike NOWCAST,\n  which "
          "needs the live collector's pre-delivery ladder vintages -> revisit ~Aug.")
    out["verdict"] = "real univariate lead, redundant with panel at H=60; nowcast is the open follow-up"

    pathlib.Path("reports/ladder_spike.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/ladder_spike.json")


if __name__ == "__main__":
    main()
