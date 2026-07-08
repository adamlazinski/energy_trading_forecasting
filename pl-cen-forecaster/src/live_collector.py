"""
Live vintage collector: the free-data moat (LITERATURE.md §6.3).

    python -m src.live_collector            # one collection pass
    python -m src.live_collector status     # what has been captured so far

Every pass polls the vintage-sensitive PSE v2 resources for business dates
[D-1, D] and appends any row whose content hash has not been seen before to
an append-only store, stamped with capture_ts (UTC, when WE saw it). Nothing
is ever updated or deleted — the store is a record of what was visible when.

Why: settled reports (crb-rozl, his-wlk-cal) silently replace the preliminary
vintages that were visible intraday, and the multi-version reports' own
publication_ts history cannot be audited after the fact. Capturing live is
the only way to (a) measure preliminary-vs-settled revision error (the F19
residual risk), (b) prove publication timing empirically, (c) accumulate a
gate-honest live dataset nobody can buy. Value accrues per day of uptime;
runs are cheap (~20 small requests) and idempotent, so overlap/backoff is
handled with a simple lock file.

Store layout: data/live/{name}/{YYYY-MM}.parquet, all payload columns as
strings (lossless vs the API JSON, schema-drift-proof), plus row_hash and
capture_ts. Dedupe window = current + previous monthly file.

Scheduling: a LaunchAgent (see ops/com.pl-cen.collector.plist) runs this
every 15 min while the Mac is awake. Gaps in capture_ts ARE the uptime log.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import time
from datetime import date, timedelta, datetime, timezone

import pandas as pd

from .pse_client import PSEClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "live"
LOCK = STORE / ".lock"
LOG = STORE / "collector.log"

# vintage-sensitive resources: preliminary values, rolling forecasts, and
# reports whose true publication timing we want proven, not assumed
LIVE_SLUGS = {
    "kse_snapshot": "pdgobpkd",     # 5-min KSE state: PV/wind/demand/reserves
    "kse_actuals":  "his-wlk-cal",  # settled D+1; capture when rows appear
    "kse_load":     "kse-load",     # load actual+fcst, revised intraday
    "imbalance":    "crb-rozl",     # CEN: preliminary vs corrected vintages
    "price_fcst":   "price-fcst",   # PSE rolling CEN forecast (multi-version)
    "plan_pk5l":    "pk5l-wp",      # demand/RES forecast plan (multi-version)
    "contracting":  "sk",           # KSE contracting state
    "poeb_rbn":     "poeb-rbn",     # offer marginals: verify ~20min-post-delivery timing
    "imb_energy":   "en-rozl",      # system length
}

DERIVED = {"ts", "pub_ts", "row_hash", "capture_ts"}  # never hashed


def _hash_rows(df: pd.DataFrame) -> pd.Series:
    cols = sorted(c for c in df.columns if c not in DERIVED)
    # astype("string")+fillna: pandas 3 astype(str) leaves nulls as floats
    joined = df[cols].astype("string").fillna("␀").agg("|".join, axis=1)
    return joined.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def _month_files(name: str, today: date) -> list[pathlib.Path]:
    cur = today.strftime("%Y-%m")
    prev = (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    return [STORE / name / f"{m}.parquet" for m in (prev, cur)]


def _collect_one(cli: PSEClient, name: str, slug: str, today: date) -> int:
    df = cli.fetch(slug, (today - timedelta(days=1)).isoformat(), today.isoformat(),
                   chunk_days=2)
    if df.empty:
        return 0
    payload = [c for c in df.columns if c not in DERIVED]
    df = df[payload].astype("string")
    df["row_hash"] = _hash_rows(df)
    df = df.drop_duplicates("row_hash")

    prev_f, cur_f = _month_files(name, today)
    seen: set[str] = set()
    for f in (prev_f, cur_f):
        if f.exists():
            seen |= set(pd.read_parquet(f, columns=["row_hash"])["row_hash"])
    new = df[~df["row_hash"].isin(seen)].copy()
    if new.empty:
        return 0
    new["capture_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur_f.parent.mkdir(parents=True, exist_ok=True)
    if cur_f.exists():
        new = pd.concat([pd.read_parquet(cur_f), new], ignore_index=True)
    new.to_parquet(cur_f)
    return len(df[~df["row_hash"].isin(seen)])


def collect() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    # stale-safe lock: a crashed run must not block collection forever
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 600:
        return
    LOCK.write_text(str(datetime.now(timezone.utc)))
    try:
        cli = PSEClient()
        today = date.today()
        parts = []
        for name, slug in LIVE_SLUGS.items():
            try:
                n = _collect_one(cli, name, slug, today)
                parts.append(f"{name}:{n}")
            except Exception as e:                    # one failure never kills a pass
                parts.append(f"{name}:ERR({type(e).__name__})")
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {' '.join(parts)}\n"
        with LOG.open("a") as fh:
            fh.write(line)
        print(line, end="")
    finally:
        LOCK.unlink(missing_ok=True)


def status() -> None:
    for name in LIVE_SLUGS:
        files = sorted((STORE / name).glob("*.parquet"))
        if not files:
            print(f"{name:14} —")
            continue
        n = sum(len(pd.read_parquet(f, columns=["row_hash"])) for f in files)
        last = pd.read_parquet(files[-1], columns=["capture_ts"])["capture_ts"].max()
        print(f"{name:14} rows {n:>8,}  last capture {last}")


if __name__ == "__main__":
    status() if (len(sys.argv) > 1 and sys.argv[1] == "status") else collect()
