"""
NWP run-to-run revision: the intraday RES-forecast revision PL doesn't publish
(F10), manufactured from consecutive model runs — Open-Meteo Previous Runs API
(same 5 wind + 5 solar national-proxy sites as weather_res).

    python -m src.nwp_revision

For delivery hour (D,h) the API exposes the latest pre-delivery run (lead
~2-5h) and the runs ~24h / ~48h earlier (previous_day1/2). Two revisions:

  rev21 = day1 − day2   run at (D-1,h): with ~2h model-availability lag it is
                        known before the 22:00 D-1 IDA2 gate for h ≤ 19 —
                        STRICT gate-honest. Trade: enter IDA2, exit IDA3.
  rev10 = day0 − day1   the latest run's revision; for afternoon delivery the
                        run may POST-DATE the 10:00 IDA3 gate — a CEILING
                        (mechanism) test only. Trade: enter IDA3, settle CEN.

Direction: wind/solar revised UP ⇒ system longer ⇒ later prices fall ⇒ short
at entry. Wind in power terms (v³), solar as GHI; each z-scored on the first
30% (no lookahead) and summed. Deadband = 1σ of the train segment.
Writes reports/nwp_revision.json; caches data/raw/weather_prev_runs.parquet.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import requests

from .weather_res import SOLAR_SITES, WIND_SITES

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
CACHE = pathlib.Path("data/raw/weather_prev_runs.parquet")
W = "Europe/Warsaw"
MWH = 0.25
COSTS_RT = (5.0, 10.0, 20.0)     # round trip, trade A (two auction legs)
COSTS_ONE = (10.0, 20.0)         # trade B (one auction leg, CEN settlement)


def _pull_site(lat, lon, start, end, base_var):
    vars_ = [base_var, f"{base_var}_previous_day1", f"{base_var}_previous_day2"]
    frames = []
    cur, end = pd.Timestamp(start), pd.Timestamp(end)
    while cur < end:
        hi = min(cur + pd.Timedelta(days=120), end)
        r = requests.get(API, params={
            "latitude": lat, "longitude": lon,
            "start_date": cur.strftime("%Y-%m-%d"), "end_date": hi.strftime("%Y-%m-%d"),
            "hourly": ",".join(vars_), "timezone": "UTC"}, timeout=120)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
        cur = hi + pd.Timedelta(days=1)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    df["ts"] = pd.to_datetime(df["time"], utc=True)
    return df.drop(columns=["time"]).set_index("ts")


def build(start="2024-06-15", end=None) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    w0, w1, w2, s0, s1, s2 = [], [], [], [], [], []
    for lat, lon in WIND_SITES:
        d = _pull_site(lat, lon, start, end, "wind_speed_100m")
        w0.append(d["wind_speed_100m"] ** 3)
        w1.append(d["wind_speed_100m_previous_day1"] ** 3)
        w2.append(d["wind_speed_100m_previous_day2"] ** 3)
        print(f"[wind] {lat},{lon}: {len(d)}h", flush=True)
    for lat, lon in SOLAR_SITES:
        d = _pull_site(lat, lon, start, end, "shortwave_radiation")
        s0.append(d["shortwave_radiation"])
        s1.append(d["shortwave_radiation_previous_day1"])
        s2.append(d["shortwave_radiation_previous_day2"])
        print(f"[solar] {lat},{lon}: {len(d)}h", flush=True)
    out = pd.DataFrame({
        "w3_d0": pd.concat(w0, axis=1).mean(axis=1),
        "w3_d1": pd.concat(w1, axis=1).mean(axis=1),
        "w3_d2": pd.concat(w2, axis=1).mean(axis=1),
        "ghi_d0": pd.concat(s0, axis=1).mean(axis=1),
        "ghi_d1": pd.concat(s1, axis=1).mean(axis=1),
        "ghi_d2": pd.concat(s2, axis=1).mean(axis=1),
    }).reset_index()
    out.to_parquet(CACHE)
    print(f"cached {len(out)} hours -> {CACHE}", flush=True)
    return out


def _zsum(df, pairs, train_frac=0.3):
    """Sum of components z-scored on the first `train_frac` (no lookahead)."""
    cut = int(len(df) * train_frac)
    z = np.zeros(len(df))
    for a, b in pairs:
        rev = (df[a] - df[b]).to_numpy(dtype=float)
        mu, sd = np.nanmean(rev[:cut]), np.nanstd(rev[:cut])
        z += np.nan_to_num((rev - mu) / sd)
    return z, 1.0  # deadband = 1σ per construction


def _perf(days, pnl, pos):
    tr = pos != 0
    if tr.sum() < 100:
        return {}
    daily = pd.Series(pnl, index=pd.to_datetime(days)).groupby(level=0).sum()
    ann = daily.mean() * 365
    sharpe = ann / (daily.std() * np.sqrt(365)) if daily.std() else 0.0
    return {"n_trades": int(tr.sum()),
            "pnl_per_mwh": round(float((pnl[tr] / MWH + 0).mean()), 1),
            "hit": round(float((pnl[tr] > 0).mean()), 3),
            "ann_pln_per_mw": round(float(ann), 0),
            "sharpe": round(float(sharpe), 2)}


def main():
    wx = build()
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True)

    tge = pd.read_parquet("data/raw/tge_rdb.parquet")
    tge = tge[tge.dur_min == 15].drop_duplicates("ts")
    for c in ("ida2_pln", "ida3_pln"):
        tge[c] = pd.to_numeric(tge[c], errors="coerce")
    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "cen_cost"]]
    crb["cen_cost"] = pd.to_numeric(crb["cen_cost"], errors="coerce")
    px = tge[["ts", "ida2_pln", "ida3_pln"]].merge(crb, on="ts", how="left")
    px["hour_ts"] = px["ts"].dt.floor("1h")
    df = px.merge(wx.rename(columns={"ts": "hour_ts"}), on="hour_ts", how="inner")
    loc = df["ts"].dt.tz_convert(W)
    df["day"], df["h"] = loc.dt.date, loc.dt.hour
    df["quarter"] = loc.dt.year.astype(str) + "Q" + loc.dt.quarter.astype(str)
    df = df.sort_values("ts").reset_index(drop=True)

    df["rev21"], dead21 = _zsum(df, [("w3_d1", "w3_d2"), ("ghi_d1", "ghi_d2")])
    df["rev10"], dead10 = _zsum(df, [("w3_d0", "w3_d1"), ("ghi_d0", "ghi_d1")])
    df["m32"] = df["ida3_pln"] - df["ida2_pln"]
    df["mc3"] = df["cen_cost"] - df["ida3_pln"]

    out = {"trades": {}}
    print("validity: corr(rev10, rev21) =",
          round(float(df[["rev10", "rev21"]].corr().iloc[0, 1]), 3))
    for sig, tgt in (("rev21", "m32"), ("rev10", "m32"), ("rev10", "mc3")):
        d = df.dropna(subset=[sig, tgt])
        c = float(np.corrcoef(d[sig], d[tgt])[0, 1])
        out[f"corr_{sig}_{tgt}"] = round(c, 3)
        print(f"diagnostic: corr({sig}, {tgt}) = {c:+.3f}  (n={len(d)})")

    # trade A (STRICT): enter IDA2, exit IDA3, rev21 known pre-gate for h<=19
    a = df.dropna(subset=["m32"]).query("h <= 19")
    pos_a = np.where(a["rev21"] > dead21, -1, np.where(a["rev21"] < -dead21, 1, 0))
    print("\n== A (strict): rev21 -> enter IDA2, exit IDA3, h<=19 ==")
    for cost in COSTS_RT:
        r = _perf(a["day"], (pos_a * a["m32"].to_numpy() - np.abs(pos_a) * cost) * MWH, pos_a)
        out["trades"][f"A_cost{int(cost)}"] = r
        if r:
            print(f"  cost {cost:>4.0f}: n {r['n_trades']:>6}  pnl/MWh {r['pnl_per_mwh']:+6.1f}  "
                  f"hit {r['hit']:.3f}  ann/MW {r['ann_pln_per_mw']:>8,.0f}  Sharpe {r['sharpe']:>5}")

    # trade B (CEILING): enter IDA3, settle CEN — rev10 vintage may leak past
    # the 10:00 gate for afternoon hours; mechanism upper bound only.
    b = df.dropna(subset=["mc3"])
    pos_b = np.where(b["rev10"] > dead10, -1, np.where(b["rev10"] < -dead10, 1, 0))
    print("\n== B (CEILING, vintage-flagged): rev10 -> enter IDA3, settle CEN ==")
    for cost in COSTS_ONE:
        r = _perf(b["day"], (pos_b * b["mc3"].to_numpy() - np.abs(pos_b) * cost) * MWH, pos_b)
        out["trades"][f"B_cost{int(cost)}"] = r
        if r:
            print(f"  cost {cost:>4.0f}: n {r['n_trades']:>6}  pnl/MWh {r['pnl_per_mwh']:+6.1f}  "
                  f"hit {r['hit']:.3f}  ann/MW {r['ann_pln_per_mw']:>8,.0f}  Sharpe {r['sharpe']:>5}")

    pathlib.Path("reports/nwp_revision.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/nwp_revision.json")


if __name__ == "__main__":
    main()
