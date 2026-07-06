"""
Day-by-day scrape of TGE intraday results (RDB continuous + IDA1/2/3
auctions) from the public results page — the tradeable intraday leg for
Project A. One row per instrument (24x H, 96x Q per day):

  ts, dur_min      — delivery start (UTC) and 15/60 duration
  rdb_min/max/vwap, rdb_vol/vol_b/vol_s          — continuous trading, PLN/MWh & MW
  ida{1,2,3}_eur/pln, ida{k}_vol/vol_b/vol_s     — uniform-price auctions
  tot_min/max/vwap, tot_vol/vol_b/vol_s          — combined, MWh

The page needs browser-ish headers (WAF rejects bare curl). Historic days
via ?dateShow=DD-MM-YYYY, verified back to 2024-07. Appends daily; safe to
re-run (skips days already present).

    python -m src.pull_tge_rdb [start] [end]
"""
from __future__ import annotations

import io
import pathlib
import sys
import time

import pandas as pd
import requests

OUT = pathlib.Path("data/raw/tge_rdb.parquet")
URL = "https://tge.pl/energia-elektryczna-rdb"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    "Referer": "https://tge.pl/",
}
GROUPS = {           # header level-0 substring -> column prefix
    "CT ": "rdb", "IDA 1": "ida1", "IDA 2": "ida2", "IDA 3": "ida3",
    "Łącznie": "tot",
}
FIELDS = {           # header level-1 substring -> suffix
    "Kurs min.": "min", "Kurs max.": "max", "Kurs średni": "vwap",
    "Kurs jednolity [EUR/MWh]": "eur", "Kurs jednolity [PLN/MWh]": "pln",
    "Wolumen kupna": "vol_b", "Wolumen sprzedaży": "vol_s", "Wolumen [": "vol",
}


def _suffix(lvl1: str) -> str | None:
    for key, suf in FIELDS.items():
        if key in lvl1:
            return suf
    return None


def parse_day(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), decimal=".", thousands=",")
    t = max(tables, key=len)
    rows = {"instrument": t.iloc[:, 0].astype(str), "dur_min": t.iloc[:, 1]}
    for i, col in enumerate(t.columns):
        lvl0, lvl1 = str(col[0]), str(col[1])
        pref = next((p for k, p in GROUPS.items() if lvl0.startswith(k)), None)
        suf = _suffix(lvl1)
        if pref and suf:
            rows[f"{pref}_{suf}"] = pd.to_numeric(
                t.iloc[:, i].replace("-", pd.NA), errors="coerce")
    df = pd.DataFrame(rows)
    df = df[df["instrument"].str.contains(r"_\w", na=False)]

    ins = df["instrument"].str.split("_", n=1, expand=True)
    date = pd.to_datetime(ins[0], errors="coerce")
    tail = ins[1]
    start = pd.Series(pd.NaT, index=df.index)
    h = tail.str.fullmatch(r"H\d{1,2}")
    start[h] = date[h] + pd.to_timedelta(tail[h].str[1:].astype(float) - 1, unit="h")
    q = tail.str.fullmatch(r"Q\d{2}:\d{2}")
    if q.any():
        end_min = (tail[q].str[1:3].astype(int) * 60
                   + tail[q].str[4:6].astype(int))
        start[q] = date[q] + pd.to_timedelta(end_min - 15, unit="m")
    df["ts"] = (pd.DatetimeIndex(start)
                .tz_localize("Europe/Warsaw", nonexistent="shift_forward",
                             ambiguous=True)
                .tz_convert("UTC"))
    df["dur_min"] = pd.to_numeric(df["dur_min"], errors="coerce")
    return df.dropna(subset=["ts"])


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-06-15"
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m-%d")

    done, frames = set(), []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        frames.append(prev)
        done = set(prev["instrument"].str[:10].unique())

    ses = requests.Session()
    ses.headers.update(HEADERS)
    days = pd.date_range(start, end, freq="D")
    for i, d in enumerate(days):
        ds = d.strftime("%Y-%m-%d")
        if ds in done:
            continue
        try:
            r = ses.get(URL, params={"dateShow": d.strftime("%d-%m-%Y")}, timeout=30)
            r.raise_for_status()
            day = parse_day(r.text)
            if day.empty:
                print(f"[skip] {ds}: no rows", flush=True)
            else:
                frames.append(day)
        except Exception as e:
            print(f"[FAIL] {ds}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(0.7)
        if (i % 25 == 0 or i == len(days) - 1) and frames:
            allp = pd.concat(frames, ignore_index=True).drop_duplicates("instrument")
            allp.sort_values("ts").to_parquet(OUT)
            print(f"[ckpt] {ds}: {len(allp)} instruments total", flush=True)

    allp = pd.concat(frames, ignore_index=True).drop_duplicates("instrument")
    allp.sort_values("ts").to_parquet(OUT)
    print(f"DONE {len(allp)} instruments -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
