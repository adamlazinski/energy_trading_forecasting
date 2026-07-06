"""
Day-by-day pull of poeb-rbn (accepted balancing-energy offers, ~88k rows/day
— one row PER OFFER) aggregated on the fly to per-15-min marginals:

  ofcg_max / ofcg_p90  — marginal (and near-marginal) accepted price, up
  ofcd_min / ofcd_p10  — same, down
  n_offers, ofp_sum    — ladder depth and accepted power

Appends daily; safe to re-run (skips days already present).

    python -m src.pull_poeb_marginals [start] [end]
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

from .build_dataset import load_cfg
from .pse_client import PSEClient

OUT = pathlib.Path("data/raw/pse_poeb_marginals.parquet")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("ofcg", "ofcd", "ofp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby("ts")
    out = pd.DataFrame({
        "ofcg_max": g["ofcg"].max(),
        "ofcg_p90": g["ofcg"].quantile(0.9),
        "ofcd_min": g["ofcd"].min(),
        "ofcd_p10": g["ofcd"].quantile(0.1),
        "n_offers": g["ofp"].size(),
        "ofp_sum": g["ofp"].sum(),
    })
    return out.reset_index()


def main():
    cfg = load_cfg()
    start = sys.argv[1] if len(sys.argv) > 1 else cfg["regime"]["train_start"]
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m-%d")
    cli = PSEClient(pause=0.05)

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
            raw = cli.fetch("poeb-rbn", ds, ds, chunk_days=1)
            if raw.empty or "ts" not in raw:
                print(f"[skip] {ds}: empty", flush=True)
                continue
            frames.append(aggregate(raw))
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
