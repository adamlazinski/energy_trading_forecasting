"""
Per-period balancing offer-ladder shape features (feeds F34, ladder_spike).

    python -m src.ladder_features

Streams the 67M-row poeb-rbn offer ladder (data/raw/pse_poeb_rbn.parquet) month
by month and reduces each 15-min delivery period to a shape summary. The offer
SET is ~fixed D-1 (vol ~28 GW and n ~934 barely move); all the signal is in the
PRICE structure. Scarcity metrics:
  g_max/g_p90/g_p50  up-ladder (ofcg) top / upper / median offer price
  vol, n             total offered power, offer count (near-constant)
  cheap_vol          MW of up-offers priced < 500 PLN (cheap cushion)
  tail_vol           MW of up-offers priced > 1000 PLN (expensive tail)
  cheap_frac         cheap_vol / vol

Writes reports/ladder_features.parquet (ts in UTC, per 15-min period).
"""
from __future__ import annotations

import warnings

import pandas as pd
import pyarrow.dataset as ds

SRC = "data/raw/pse_poeb_rbn.parquet"
OUT = "reports/ladder_features.parquet"
CHEAP, TAIL = 500.0, 1000.0


def per_period(df: pd.DataFrame) -> pd.DataFrame:
    df["ofcg"] = pd.to_numeric(df["ofcg"], errors="coerce")
    df["ofp"] = pd.to_numeric(df["ofp"], errors="coerce")
    df = df.dropna(subset=["ofcg", "ofp"])
    g = df.groupby("dtime_utc")
    out = g.agg(vol=("ofp", "sum"), n=("ofp", "size"),
                g_max=("ofcg", "max"), g_p90=("ofcg", lambda x: x.quantile(.9)),
                g_p50=("ofcg", "median"))
    out["cheap_vol"] = df[df.ofcg < CHEAP].groupby("dtime_utc")["ofp"].sum(
        ).reindex(out.index).fillna(0)
    out["tail_vol"] = df[df.ofcg > TAIL].groupby("dtime_utc")["ofp"].sum(
        ).reindex(out.index).fillna(0)
    out["cheap_frac"] = out.cheap_vol / out.vol
    return out.reset_index()


def main():
    warnings.filterwarnings("ignore")
    d = ds.dataset(SRC)
    bd = d.to_table(columns=["business_date"]).to_pandas()["business_date"]
    months = sorted(pd.Series(pd.to_datetime(bd.unique())).dt.strftime("%Y-%m").unique())
    print(f"{len(months)} months {months[0]}..{months[-1]}")

    parts = []
    for m in months:
        df = d.to_table(columns=["dtime_utc", "ofcg", "ofp"],
                        filter=(ds.field("business_date") >= f"{m}-01")
                        & (ds.field("business_date") <= f"{m}-31")).to_pandas()
        parts.append(per_period(df))
    lad = (pd.concat(parts).drop_duplicates("dtime_utc").sort_values("dtime_utc")
           .rename(columns={"dtime_utc": "ts"}).reset_index(drop=True))
    lad["ts"] = pd.to_datetime(lad["ts"])
    lad.to_parquet(OUT)
    print(f"wrote {OUT}: {len(lad):,} periods, cols {list(lad.columns)}")


if __name__ == "__main__":
    main()
