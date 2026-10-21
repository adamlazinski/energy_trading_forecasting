"""PSE v2 report API (https://api.raporty.pse.pl/api/, free, no auth) + local cache.

Cache layout under <workdir>/raw:
  <slug>/<YYYY-MM>.parquet      15-min settlement series, ts = period START (UTC)
  cordmax/<YYYY-MM>.parquet     daily CORDMax, keyed by business_date
  ladder/<YYYY-MM-DD>.parquet   BPKD/BO plan ladder (oeb-bpkdbo), one file per day
  offers/<YYYY-MM-DD>.parquet   all balancing offers (poeb-rbn), one file per day

Closed months/days are never re-downloaded; the recent ones are refreshed,
because PSE publishes settlement data D+1 and can correct it.
"""
from __future__ import annotations

import pathlib
import time
from urllib.parse import quote

import pandas as pd
import requests

from . import UserError

BASE = "https://api.raporty.pse.pl/api"
WAW = "Europe/Warsaw"

# slug -> columns kept (see docs/CEN_replikacja.md for their meaning)
SOURCES = {
    "crb-rozl": ["cen_cost", "ckoeb_cost", "ceb_pp_cost", "ceb_sr_cost",
                 "ceb_sr_afrrg_cost", "ceb_sr_afrrd_cost"],
    "price-cost": ["cor_pp_cost", "contracting"],
    "zeb-rozl": ["zebpp", "pzeb_afrrg_cost", "pzeb_afrrd_cost"],
    "sk": ["sk_cost"],
    "csdac-pln": ["csdac_pln"],
    "lolp": [f"b{i}" for i in range(10)] + [f"p{i}" for i in range(10)],
    "ro-rozl": ["ro_cost"],
}
LADDER_COLS = ["cdo", "ofc", "ofp", "activ_direction"]
OFFER_COLS = ["ofp", "ofcg", "ofcd"]
REFRESH_DAYS = 7          # anything this recent is always re-downloaded


class PSE:
    def __init__(self, pause: float = 0.2, tries: int = 6):
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json", "User-Agent": "cenmodel"})
        self.pause, self.tries = pause, tries

    def _get(self, url: str) -> dict:
        err = None
        for k in range(self.tries):
            try:
                r = self.s.get(url, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    err = f"HTTP {r.status_code}"
                elif r.status_code >= 400:
                    raise UserError(f"API PSE odrzuciło zapytanie (HTTP {r.status_code}): {url}\n"
                                    f"{r.text[:300]}")
                else:
                    return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                err = type(e).__name__
            time.sleep(min(60, 3 * 2 ** k))
        raise UserError(f"API PSE nie odpowiada ({err}) po {self.tries} próbach. Sprawdź internet "
                        "i https://api.raporty.pse.pl — potem uruchom komendę ponownie "
                        "(pobrane już miesiące zostaną pominięte).")

    def fetch(self, slug: str, lo: str, hi: str) -> pd.DataFrame:
        """All rows of `slug` with business_date in [lo, hi], following nextLink."""
        flt = quote(f"business_date ge '{lo}' and business_date le '{hi}'")
        url = f"{BASE}/{slug}?$filter={flt}&$first=50000"
        rows: list = []
        while url:
            js = self._get(url)
            rows += js.get("value", [])
            nxt = js.get("nextLink") or js.get("@odata.nextLink")
            url = nxt.get("link") if isinstance(nxt, dict) else nxt
            time.sleep(self.pause)
        return pd.DataFrame(rows)

    def ping(self) -> bool:
        try:
            self.s.get(f"{BASE}/sk?$first=1", timeout=20).raise_for_status()
            return True
        except requests.RequestException:
            return False


def period_start_utc(df: pd.DataFrame) -> pd.Series:
    """PSE `dtime_utc` labels the period END (00:15 <-> 00:00-00:15)."""
    return pd.to_datetime(df["dtime_utc"]).dt.tz_localize("UTC") - pd.Timedelta(minutes=15)


def _months(start: str, end: str):
    """Whole months touching [start, end]: a month file is always complete,
    so a short refresh never truncates the cache."""
    for m in pd.period_range(start, end, freq="M"):
        lo = m.start_time.strftime("%Y-%m-%d")
        hi = min(pd.Timestamp(end), m.end_time).strftime("%Y-%m-%d")
        yield str(m), lo, hi, m.end_time


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(WAW).tz_localize(None).normalize()


def fetch_public(workdir: pathlib.Path, start: str, end: str | None = None, log=print) -> None:
    """Download/refresh the settlement series. end defaults to tomorrow (CSDAC for D+1)."""
    end = end or (_today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise UserError(f"Koniec {end} jest przed początkiem {start}.")
    raw = workdir / "raw"
    cli = PSE()
    fresh = _today() - pd.Timedelta(days=REFRESH_DAYS)
    for slug in [*SOURCES, "cordmax"]:
        (raw / slug).mkdir(parents=True, exist_ok=True)
        got = skipped = 0
        for tag, lo, hi, m_end in _months(start, end):
            f = raw / slug / f"{tag}.parquet"
            if f.exists() and m_end < fresh:
                skipped += 1
                continue
            df = cli.fetch(slug, lo, hi)
            if df.empty:
                continue
            if slug == "cordmax":
                df = df[["business_date", "cordmax_cost"]].drop_duplicates("business_date", keep="last")
            else:
                cols = [c for c in SOURCES[slug] if c in df.columns]
                df = (df.assign(ts=period_start_utc(df))[["ts"] + cols]
                        .drop_duplicates("ts", keep="last").sort_values("ts"))
            df.to_parquet(f, index=False)
            got += len(df)
        log(f"  {slug:11s} pobrano {got:6d} wierszy, pominięto {skipped} zamkniętych miesięcy")


def fetch_ladder(workdir: pathlib.Path, start: str, end: str, log=print, slug: str = "oeb-bpkdbo") -> None:
    """Day-by-day ladders: oeb-bpkdbo -> raw/ladder, poeb-rbn -> raw/offers (~10 s per day)."""
    sub, cols = ("ladder", LADDER_COLS) if slug == "oeb-bpkdbo" else ("offers", OFFER_COLS)
    d = workdir / "raw" / sub
    d.mkdir(parents=True, exist_ok=True)
    cli = PSE()
    fresh = _today() - pd.Timedelta(days=REFRESH_DAYS)
    days = pd.date_range(start, end, freq="D")
    for i, day in enumerate(days, 1):
        f = d / f"{day:%Y-%m-%d}.parquet"
        if f.exists() and day < fresh:
            continue
        df = cli.fetch(slug, f"{day:%Y-%m-%d}", f"{day:%Y-%m-%d}")
        if df.empty:
            log(f"  {sub} {day:%Y-%m-%d}: brak danych w API (jeszcze nieopublikowane?)")
            continue
        df = df.assign(ts=period_start_utc(df))[["ts"] + [c for c in cols if c in df.columns]]
        df.to_parquet(f, index=False)
        log(f"  {sub} {day:%Y-%m-%d}: {len(df):7d} wierszy   ({i}/{len(days)})")


def read_slug(workdir: pathlib.Path, slug: str) -> pd.DataFrame:
    files = sorted((workdir / "raw" / slug).glob("*.parquet"))
    if not files:
        raise UserError(f"Brak danych '{slug}' w {workdir / 'raw'}. Uruchom: python -m cenmodel fetch")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_ladder(workdir: pathlib.Path, start: str, end: str) -> pd.DataFrame:
    files = [workdir / "raw" / "ladder" / f"{d:%Y-%m-%d}.parquet"
             for d in pd.date_range(start, end, freq="D")]
    have = [f for f in files if f.exists()]
    if not have:
        raise UserError(f"Brak drabinki BPKD/BO dla {start}..{end}. "
                        f"Uruchom: python -m cenmodel fetch --ladder {start[:7]}")
    return pd.concat([pd.read_parquet(f) for f in have], ignore_index=True)
