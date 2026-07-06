"""
Day-by-day pull of oeb-bpkdbo (offers accepted into the current balancing
plan, one row PER OFFER, single activ_direction per period) reduced to a
per-15-min record of the dispatcher's merit-order ladder:

  dir            — G (up/delivery) or D (down/withdrawal)
  marg           — marginal price: ladder crossed at the activated volume
                   from eb-rozl (eb_d_pp / eb_w_pp, MWh/15min -> x4 MW)
  g{V}           — ladder price at cumulative volume V MW (re-crossable later)
  n_offers, ladder_mw, vol_mw, saturated

Validated 2026-07: crossed marginal vs CEN corr ~0.82 both directions
(CEN differs by CKOEB/aFRR components, as expected). Publication is D+1
~14:00 Warsaw — same leakage gate as crb-rozl.

Appends daily; safe to re-run (skips days already present).

    python -m src.pull_bpkdbo [start] [end]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

from .build_dataset import load_cfg
from .pse_client import PSEClient

OUT = pathlib.Path("data/raw/pse_bpkdbo_marginals.parquet")
VOL_GRID = [25, 50, 100, 150, 200, 300, 400, 500,
            750, 1000, 1250, 1500, 2000, 2500, 3000, 4000]


def _load_eb() -> pd.DataFrame:
    eb = pd.read_parquet("data/raw/pse_bal_energy.parquet")[
        ["ts", "eb_d_pp", "eb_w_pp"]]
    for c in ("eb_d_pp", "eb_w_pp"):
        eb[c] = pd.to_numeric(eb[c], errors="coerce")
    return eb.set_index("ts")


def reduce_day(raw: pd.DataFrame, eb: pd.DataFrame) -> pd.DataFrame:
    for c in ("ofc", "ofp"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    recs = []
    for ts, g in raw.groupby("ts"):
        d = g["activ_direction"].iloc[0]
        up = d == "G"
        g = g.dropna(subset=["ofc", "ofp"]).sort_values("ofc", ascending=up)
        prices = g["ofc"].to_numpy()
        cum = g["ofp"].abs().cumsum().to_numpy()
        if not len(prices):
            continue
        rec = {"ts": ts, "dir": d, "n_offers": len(g),
               "ladder_mw": float(cum[-1])}
        for v in VOL_GRID:
            i = np.searchsorted(cum, v)
            rec[f"g{v}"] = float(prices[min(i, len(prices) - 1)])
        vol = np.nan
        if ts in eb.index:
            e = eb.loc[ts]
            vol = float(e["eb_d_pp"]) * 4 if up else abs(float(e["eb_w_pp"])) * 4
        rec["vol_mw"] = vol
        if vol == vol:
            i = np.searchsorted(cum, vol)
            rec["saturated"] = i >= len(prices)
            rec["marg"] = float(prices[min(i, len(prices) - 1)])
        else:
            rec["saturated"] = False
            rec["marg"] = np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    cfg = load_cfg()
    start = sys.argv[1] if len(sys.argv) > 1 else cfg["regime"]["train_start"]
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m-%d")
    cli = PSEClient(pause=0.05)
    eb = _load_eb()

    done = set()
    frames = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        frames.append(prev)
        done = set(prev["ts"].dt.tz_convert("Europe/Warsaw").dt.date.astype(str))

    days = pd.date_range(start, end, freq="D")
    for i, d in enumerate(days):
        ds = d.strftime("%Y-%m-%d")
        if ds in done:
            continue
        try:
            raw = cli.fetch("oeb-bpkdbo", ds, ds, chunk_days=1)
            if raw.empty or "ts" not in raw:
                print(f"[skip] {ds}: empty", flush=True)
                continue
            frames.append(reduce_day(raw, eb))
        except Exception as e:
            print(f"[FAIL] {ds}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if i % 25 == 0 or i == len(days) - 1:
            allp = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
            allp.sort_values("ts").to_parquet(OUT)
            print(f"[ckpt] {ds}: {len(allp)} periods total", flush=True)

    allp = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
    allp.sort_values("ts").to_parquet(OUT)
    print(f"DONE {len(allp)} periods -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
