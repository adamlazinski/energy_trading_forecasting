"""
Thin wrappers over entsoe-py for the pieces PSE doesn't give you cleanly:
cross-border context (PL vs DE_LU) and an independent imbalance-price series to
validate the PSE CEN target after the 2024-06-14 single-price reform.

Token: register at https://transparency.entsoe.eu then email
transparency@entsoe.eu with subject "RESTful API access". Put the token in the
ENTSOE_API_TOKEN environment variable.

    export ENTSOE_API_TOKEN=xxxxxxxx-xxxx-...
"""
from __future__ import annotations

import os

import pandas as pd

try:
    from entsoe import EntsoePandasClient
except ImportError as e:  # keep import-safe if entsoe-py isn't installed yet
    EntsoePandasClient = None
    _IMPORT_ERR = e


def _client() -> "EntsoePandasClient":
    if EntsoePandasClient is None:
        raise ImportError(f"entsoe-py not installed: {_IMPORT_ERR}")
    tok = os.environ.get("ENTSOE_API_TOKEN")
    if not tok:
        raise RuntimeError("set ENTSOE_API_TOKEN (see module docstring)")
    return EntsoePandasClient(api_key=tok)


def _ts(day: str, tz: str = "Europe/Brussels") -> pd.Timestamp:
    return pd.Timestamp(day, tz=tz)


def imbalance_prices(date_from: str, date_to: str, zone: str = "PL") -> pd.DataFrame:
    """Independent imbalance-price series to cross-check the PSE CEN target."""
    c = _client()
    s = c.query_imbalance_prices(zone, start=_ts(date_from), end=_ts(date_to))
    return s.tz_convert("UTC") if hasattr(s, "tz_convert") else s


def day_ahead(date_from: str, date_to: str, zone: str = "PL") -> pd.Series:
    c = _client()
    s = c.query_day_ahead_prices(zone, start=_ts(date_from), end=_ts(date_to))
    return s.tz_convert("UTC")


def pl_de_spread(date_from: str, date_to: str) -> pd.DataFrame:
    """Day-ahead PL minus DE_LU — the cross-border pull / price-island gauge."""
    pl = day_ahead(date_from, date_to, "PL").rename("da_pl")
    de = day_ahead(date_from, date_to, "DE_LU").rename("da_de")
    df = pd.concat([pl, de], axis=1)
    df["da_pl_de_spread"] = df["da_pl"] - df["da_de"]
    return df


def wind_solar_forecast(date_from: str, date_to: str, zone: str = "PL",
                        intraday: bool = False) -> pd.DataFrame:
    """RES forecasts known ahead of delivery (leakage-safe features)."""
    c = _client()
    fn = (c.query_intraday_wind_and_solar_forecast if intraday
          else c.query_wind_and_solar_forecast)
    df = fn(zone, start=_ts(date_from), end=_ts(date_to))
    return df.tz_convert("UTC")
