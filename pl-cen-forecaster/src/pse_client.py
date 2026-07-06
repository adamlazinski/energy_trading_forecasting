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
            url = f"{self.base}/{resource}"
            params = {"$filter": flt}
            while True:
                js = self._get(url, params=params)
                rows = js.get("value", js if isinstance(js, list) else [])
                if rows:
                    frames.append(pd.DataFrame(rows))
                nxt = js.get("nextPage") or js.get("@odata.nextLink")
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
    """Best-effort: build a tz-aware UTC 'ts' index from PSE time columns.

    PSE quarter-hour reports carry some of: dtime, udtczas, udtczas_oreb,
    period, business_date. Column names vary by report, so we probe.
    """
    df = df.copy()
    cand = [c for c in ("dtime", "udtczas", "udtczas_oreb", "doba") if c in df.columns]
    if cand:
        col = cand[0]
        ts = pd.to_datetime(df[col], errors="coerce")
        # PSE publishes in local (Europe/Warsaw) civil time
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("Europe/Warsaw", ambiguous="NaT",
                                   nonexistent="shift_forward")
        df["ts"] = ts.dt.tz_convert("UTC")
        df = df.sort_values("ts").reset_index(drop=True)
    return df


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
