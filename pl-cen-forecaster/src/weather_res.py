"""
Weather-alpha feasibility gate: can a weather-driven RES forecast beat the
CONSENSUS (the published day-ahead RES forecast already embedded in the DA
price, F10)?  If not, the whole weather angle is dead; if yes, the residual
is what we could trade.

    python -m src.weather_res

Test (hourly, on realized RES as truth):
  baseline   realized ~ consensus                    (the market's forecast)
  +weather   realized ~ consensus + forecast weather (does weather add?)
  weather    realized ~ forecast weather only        (our standalone forecast)
Run separately for wind and solar; report out-of-sample RMSE (time-split).
An incremental RMSE drop of +weather over baseline = beat-the-consensus edge.

Weather from Open-Meteo Historical Forecast API (forecast-quality, realistic
lead-time error — not reanalysis truth). National proxies: mean wind_speed
_100m (and its cube, wind power ~ v^3) over wind sites; mean shortwave
_radiation over solar sites. Cached to data/raw/weather_pl.parquet.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import requests

WIND_SITES = [(54.4, 17.0), (53.8, 16.0), (53.1, 18.0), (52.4, 16.9), (53.3, 20.5)]
SOLAR_SITES = [(52.2, 21.0), (51.1, 17.0), (50.1, 20.0), (51.2, 22.6), (52.4, 16.9)]
API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CACHE = pathlib.Path("data/raw/weather_pl.parquet")


def _pull_site(lat, lon, start, end, vars_):
    frames = []
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur < end:
        hi = min(cur + pd.Timedelta(days=120), end)
        r = requests.get(API, params={
            "latitude": lat, "longitude": lon,
            "start_date": cur.strftime("%Y-%m-%d"), "end_date": hi.strftime("%Y-%m-%d"),
            "hourly": ",".join(vars_), "timezone": "UTC"}, timeout=60)
        r.raise_for_status()
        h = r.json()["hourly"]
        frames.append(pd.DataFrame(h))
        cur = hi + pd.Timedelta(days=1)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    df["ts"] = pd.to_datetime(df["time"], utc=True)
    return df.drop(columns=["time"])


def build_weather(start="2024-06-15", end=None) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    wind = []
    for lat, lon in WIND_SITES:
        d = _pull_site(lat, lon, start, end, ["wind_speed_100m"])
        wind.append(d.set_index("ts")["wind_speed_100m"])
        print(f"[wind] {lat},{lon}: {len(d)}h", flush=True)
    solar = []
    for lat, lon in SOLAR_SITES:
        d = _pull_site(lat, lon, start, end, ["shortwave_radiation"])
        solar.append(d.set_index("ts")["shortwave_radiation"])
        print(f"[solar] {lat},{lon}: {len(d)}h", flush=True)
    w = pd.DataFrame(index=wind[0].index.union([]))
    ws = pd.concat(wind, axis=1)
    sr = pd.concat(solar, axis=1)
    out = pd.DataFrame({
        "ts": ws.index,
        "wind_ms": ws.mean(axis=1).values,
        "wind_cube": (ws ** 3).mean(axis=1).values,     # power ~ v^3
        "ghi": sr.mean(axis=1).values,
    })
    out.to_parquet(CACHE)
    print(f"cached {len(out)} hours -> {CACHE}", flush=True)
    return out


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _ols_oos(df, ycol, xcols):
    """Time-split OLS: fit on first 70%, RMSE on last 30%."""
    d = df.dropna(subset=[ycol] + xcols).sort_values("ts")
    n = len(d)
    cut = int(n * 0.7)
    Xtr = np.column_stack([np.ones(cut)] + [d[c].to_numpy()[:cut] for c in xcols])
    ytr = d[ycol].to_numpy()[:cut]
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    Xte = np.column_stack([np.ones(n - cut)] + [d[c].to_numpy()[cut:] for c in xcols])
    yte = d[ycol].to_numpy()[cut:]
    return _rmse(yte, Xte @ beta), len(yte)


def main():
    w = build_weather()
    w["ts"] = pd.to_datetime(w["ts"], utc=True)
    # realized RES (truth) and consensus (ENTSO-E DA forecast), to hourly
    act = pd.read_parquet("data/raw/pse_kse_actuals.parquet")[["ts", "pv", "wi"]]
    for c in ("pv", "wi"):
        act[c] = pd.to_numeric(act[c], errors="coerce")
    cons = pd.read_parquet("data/raw/entsoe_res.parquet")
    a_h = act.set_index("ts").resample("1h").mean()
    c_h = cons.set_index("ts").resample("1h").mean().rename(
        columns={"res_solar": "cons_solar", "res_wind": "cons_wind"})
    df = (a_h.join(c_h, how="inner").reset_index()
          .merge(w, on="ts", how="inner"))
    print(f"joined {len(df)} hours  {df['ts'].min()}..{df['ts'].max()}\n")

    for label, ycol, cons_col, wx in (
            ("WIND", "wi", "cons_wind", ["wind_ms", "wind_cube"]),
            ("SOLAR", "pv", "cons_solar", ["ghi"])):
        base, n = _ols_oos(df, ycol, [cons_col])
        both, _ = _ols_oos(df, ycol, [cons_col] + wx)
        wonly, _ = _ols_oos(df, ycol, wx)
        # correlation of weather with the consensus RESIDUAL (tradeable signal)
        d = df.dropna(subset=[ycol, cons_col] + wx)
        resid = d[ycol] - d[cons_col]
        rc = max(abs(resid.corr(d[c])) for c in wx)
        print(f"== {label} (n_oos={n}) ==")
        print(f"  RMSE consensus-only : {base:8.1f} MW")
        print(f"  RMSE consensus+wx   : {both:8.1f} MW   ({(base-both)/base*100:+.1f}%)")
        print(f"  RMSE weather-only   : {wonly:8.1f} MW")
        print(f"  max |corr(weather, consensus residual)| : {rc:.3f}\n")


if __name__ == "__main__":
    main()
