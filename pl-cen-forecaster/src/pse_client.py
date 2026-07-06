"""
Generic client for the PSE v2 report API (https://api.raporty.pse.pl/api/).

Free, no auth (as of 2026-07). JSON, OData-style filtering on `business_date`,
15-minute resolution. The API is a set of named resources ("reports"); this
client is deliberately endpoint-agnostic so you can point it at whatever slug
`discover` reveals for the imbalance price, PV/wind generation+forecast,
cross-border exchange, PK5L, reserve prices, etc.

Usage
-----
    python -m src.pse_client discover                 # list all resources
    python -m src.pse_client peek rce-pln             # one recent day, columns
    python -m src.pse_client fetch rce-pln 2024-06-15 2026-06-30 -o out.parquet

Programmatic
------------
    from src.pse_client import PSEClient
    df = PSEClient().fetch("rce-pln", "2024-06-15", "2024-06-30")
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote

import pandas as pd
import requests

BASE = "https://api.raporty.pse.pl/api"
DATE_FIELD = "business_date"          # OData filter field used by v2 reports
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "pl-cen-forecaster"})


def _daterange_chunks(start: str, end: str, days: int = 31):
    """Yield (from, to) inclusive string chunks; the API caps range per call."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        hi = min(cur + timedelta(days=days - 1), d1)
        yield cur.isoformat(), hi.isoformat()
        cur = hi + timedelta(days=1)


def discover() -> list[str]:
    """Return the list of available report slugs from the API root."""
    r = SESSION.get(BASE, timeout=30)
    r.raise_for_status()
    payload = r.json()
    # The root typically exposes an OData service document under "value";
    # fall back to printing the raw payload if the shape differs.
    items = payload.get("value", payload)
    slugs = []
    if isinstance(items, list):
        for it in items:
            slug = it.get("name") or it.get("url") or it.get("kind")
            if slug:
                slugs.append(slug)
    return slugs or [str(payload)[:2000]]


class PSEClient:
    def __init__(self, base: str = BASE, pause: float = 0.2):
        self.base = base
        self.pause = pause  # be polite between paged requests

    def _get(self, url: str, params: dict | None = None) -> dict:
        r = SESSION.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def fetch(self, resource: str, date_from: str, date_to: str,
              date_field: str = DATE_FIELD, extra_filter: str | None = None,
              chunk_days: int = 31) -> pd.DataFrame:
        """Pull a resource over [date_from, date_to] inclusive as a DataFrame.

        Handles both OData `nextPage`-link pagination and, as a fallback,
        month-by-month chunking so a multi-year pull never trips the API's
        per-request range cap.
        """
        frames: list[pd.DataFrame] = []
        for lo, hi in _daterange_chunks(date_from, date_to, days=chunk_days):
            flt = f"{date_field} ge '{lo}' and {date_field} le '{hi}'"
            if extra_filter:
                flt = f"({flt}) and ({extra_filter})"
            # the API 400s on '+'-encoded spaces, so percent-encode ourselves.
            # $first lifts the default 100-row page cap (verified to 50k).
            url = f"{self.base}/{resource}?$filter={quote(flt)}&$first=50000"
            params = None
            while True:
                js = self._get(url, params=params)
                rows = js.get("value", js if isinstance(js, list) else [])
                if rows:
                    frames.append(pd.DataFrame(rows))
                nxt = js.get("nextLink") or js.get("nextPage") or js.get("@odata.nextLink")
                if not nxt:
                    break
                # follow the absolute next-page link; drop params (baked in)
                url, params = (nxt.get("link") if isinstance(nxt, dict) else nxt), None
                time.sleep(self.pause)
            time.sleep(self.pause)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return _parse_time(df)


def _parse_time(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tz-aware UTC 'ts' column labelling the START of each interval.

    PSE v2 reports label rows with the interval END (dtime 00:15 <-> period
    00:00-00:15), and most expose a ready-made UTC column. Preference order:
    *_utc columns as plain UTC, else local (Europe/Warsaw) columns. The
    end->start shift is inferred from the modal spacing of consecutive stamps
    so hourly reports (e.g. pdgsz) shift by 1h, quarter-hour ones by 15min.
    Publication timestamps are parsed alongside; daily reports (only
    business_date) get ts = business_date midnight Warsaw.
    """
    df = df.copy()
    utc_cand = [c for c in ("dtime_utc", "plan_dtime_utc") if c in df.columns]
    loc_cand = [c for c in ("dtime", "plan_dtime", "udtczas", "udtczas_oreb") if c in df.columns]
    if utc_cand:
        ts = pd.to_datetime(df[utc_cand[0]], errors="coerce").dt.tz_localize("UTC")
    elif loc_cand:
        ts = pd.to_datetime(df[loc_cand[0]], errors="coerce")
        ts = ts.dt.tz_localize("Europe/Warsaw", ambiguous="NaT",
                               nonexistent="shift_forward").dt.tz_convert("UTC")
    elif "business_date" in df.columns:      # daily reports (e.g. rcco2)
        ts = pd.to_datetime(df["business_date"], errors="coerce")
        ts = ts.dt.tz_localize("Europe/Warsaw").dt.tz_convert("UTC")
        df["ts"] = ts
        return df.sort_values("ts").reset_index(drop=True)
    else:
        return df

    # end-of-interval label -> start-of-interval label; modal positive spacing
    # (multi-version reports repeat each stamp, so zero diffs are dropped)
    diffs = ts.dropna().sort_values().diff()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if len(diffs):
        ts = ts - diffs.mode().iloc[0]
    df["ts"] = ts

    if "publication_ts_utc" in df.columns:
        df["pub_ts"] = pd.to_datetime(
            df["publication_ts_utc"], errors="coerce").dt.tz_localize("UTC")
    return df.sort_values("ts").reset_index(drop=True)


def _cli():
    p = argparse.ArgumentParser(description="PSE v2 report API client")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover")
    pk = sub.add_parser("peek"); pk.add_argument("resource")
    fe = sub.add_parser("fetch")
    fe.add_argument("resource"); fe.add_argument("date_from"); fe.add_argument("date_to")
    fe.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    if a.cmd == "discover":
        for s in discover():
            print(s)
    elif a.cmd == "peek":
        y = (date.today() - timedelta(days=2)).isoformat()
        df = PSEClient().fetch(a.resource, y, y)
        print(f"rows={len(df)}  cols={list(df.columns)}")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.head(8))
    elif a.cmd == "fetch":
        df = PSEClient().fetch(a.resource, a.date_from, a.date_to)
        print(f"pulled {len(df)} rows, {df['ts'].min()} .. {df['ts'].max()}"
              if "ts" in df else f"pulled {len(df)} rows")
        if a.out:
            df.to_parquet(a.out); print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(_cli())
