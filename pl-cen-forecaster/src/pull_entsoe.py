"""
Pull ENTSO-E day-ahead wind & solar forecasts for PL — the near-delivery RES
information the CEN forecaster is missing, and (unlike PSE's pk5l PV/wind,
which the API only serves as a post-delivery latest vintage) genuinely
LEAKAGE-SAFE: the day-ahead RES forecast is published D-1 (~18:00 CET), so
it is known at every H=60 gate of day D, like the csdac anchor.

    python -m src.pull_entsoe [start] [end]

Requires ENTSOE_API_TOKEN (see src.entsoe_client). Chunked by 90 days,
appends, safe to re-run (skips days already present). Output (15-min UTC):
  data/raw/entsoe_res.parquet  [ts, res_solar, res_wind]
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

from . import entsoe_client as ec
from .build_dataset import load_cfg

OUT = pathlib.Path("data/raw/entsoe_res.parquet")
CHUNK_DAYS = 90


def main():
    cfg = load_cfg()
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1
                         else cfg["regime"]["train_start"])
    end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2
                       else pd.Timestamp.today().strftime("%Y-%m-%d"))

    done, frames = set(), []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        frames.append(prev)
        done = set(prev["ts"].dt.tz_convert("Europe/Warsaw").dt.date.astype(str))

    cur = start
    while cur < end:
        hi = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        ds, de = cur.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")
        if ds in done and (cur + pd.Timedelta(days=1)).strftime("%Y-%m-%d") in done:
            cur = hi
            continue
        try:
            r = ec.wind_solar_forecast(ds, de, "PL")
            r = r.rename(columns={"Solar": "res_solar",
                                  "Wind Onshore": "res_wind"})
            r.index = r.index.tz_convert("UTC")
            out = r.reset_index().rename(columns={"index": "ts"})
            out = out.rename(columns={out.columns[0]: "ts"})
            frames.append(out[["ts", "res_solar", "res_wind"]])
            print(f"[ok] {ds}..{de}: {len(out)} rows", flush=True)
        except Exception as e:
            print(f"[FAIL] {ds}..{de}: {type(e).__name__}: {str(e)[:140]}",
                  flush=True)
        allp = (pd.concat(frames, ignore_index=True)
                .drop_duplicates("ts").sort_values("ts"))
        allp.to_parquet(OUT)
        cur = hi

    allp = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("ts").sort_values("ts"))
    for c in ("res_solar", "res_wind"):
        allp[c] = pd.to_numeric(allp[c], errors="coerce")
    allp.to_parquet(OUT)
    print(f"DONE {len(allp)} rows {allp['ts'].min()}..{allp['ts'].max()} -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
