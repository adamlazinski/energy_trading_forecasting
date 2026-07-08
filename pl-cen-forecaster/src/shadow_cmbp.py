"""
Shadow forecaster: daily gate-honest CMBP forecasts, issued BEFORE the fact.

    python -m src.shadow_cmbp          # one issuance + scoring pass
    python -m src.shadow_cmbp status   # track-record summary

Every morning at 07:20 Warsaw (LaunchAgent, before the ~07:30 capacity-bid
gate and the 09:10 D-1 CMBP publication) this job:
  1. pulls fresh CMBP/zmb/CEN history from the PSE API (self-gating: only
     published rows exist to be pulled) and appends tomorrow's 48h-ahead NWP
     run to a local weather cache,
  2. trains the F28 model on everything available and issues quantile
     forecasts of afrr_g/afrr_d for every hour of TOMORROW,
  3. appends them to an append-only store with issued_ts and a gate_ok flag
     (True iff issued before 08:00 Warsaw on D-1 — the audit criterion),
  4. scores every matured past forecast against realized CMBP.

The point is the timestamp: after N weeks this is a live track record no
backtest can fake — forecasts provably issued before publication. Runs from
~/.pl-cen-collector (TCC-safe) like the collector; store under data/shadow/.
"""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from .pse_client import PSEClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "shadow"
WXCACHE = ROOT / "data" / "weather_prev_runs.parquet"
LOG = STORE / "shadow.log"

W = "Europe/Warsaw"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
TARGETS = ("afrr_g", "afrr_d")
START = "2024-06-15"
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
              verbose=-1)
WX_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
WIND_SITES = [(54.4, 17.0), (53.8, 16.0), (53.1, 18.0), (52.4, 16.9), (53.3, 20.5)]
SOLAR_SITES = [(52.2, 21.0), (51.1, 17.0), (50.1, 20.0), (51.2, 22.6), (52.4, 16.9)]


def _wx_update() -> pd.DataFrame:
    """Local NWP cache: keep w3_d2/ghi_d2 current through tomorrow 23:00 UTC."""
    wx = pd.read_parquet(WXCACHE)
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    need_to = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(hours=47)
    if wx["ts"].max() >= need_to:
        return wx
    lo = (wx["ts"].max() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hi = need_to.strftime("%Y-%m-%d")

    def pull(sites, var):
        cols = []
        for lat, lon in sites:
            r = requests.get(WX_API, params={
                "latitude": lat, "longitude": lon, "start_date": lo, "end_date": hi,
                "hourly": f"{var}_previous_day2", "timezone": "UTC"}, timeout=120)
            r.raise_for_status()
            j = pd.DataFrame(r.json()["hourly"])
            j["ts"] = pd.to_datetime(j["time"], utc=True)
            cols.append(j.set_index("ts")[f"{var}_previous_day2"])
        return pd.concat(cols, axis=1).mean(axis=1)

    new = pd.DataFrame({"w3_d2": pull(WIND_SITES, "wind_speed_100m") ** 3,
                        "ghi_d2": pull(SOLAR_SITES, "shortwave_radiation")}).reset_index()
    wx = (pd.concat([wx, new], ignore_index=True)
          .drop_duplicates("ts", keep="last").sort_values("ts"))
    wx.to_parquet(WXCACHE)
    return wx


def build() -> pd.DataFrame:
    """F28's feature frame, from a fresh API pull, extended through tomorrow."""
    cli = PSEClient()
    today = datetime.now(timezone.utc).date()
    end = (today + timedelta(days=1)).isoformat()
    cp = cli.fetch("cmbp-tp", START, end)[["ts", "afrr_g", "afrr_d"]]
    zb = cli.fetch("zmb", START, end)[["ts", "zmb_afrrg", "zmb_afrrd"]]
    cen = cli.fetch("crb-rozl", START, end)[["ts", "cen_cost"]]
    for d, cs in ((cp, TARGETS), (zb, ("zmb_afrrg", "zmb_afrrd")), (cen, ("cen_cost",))):
        for c in cs:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    idx = pd.date_range(cp["ts"].min(),
                        pd.Timestamp(end, tz=W).tz_convert("UTC") + pd.Timedelta(hours=23),
                        freq="1h")
    df = pd.DataFrame(index=idx)
    df = df.join(cp.set_index("ts").resample("1h").mean())
    df = df.join(zb.set_index("ts").resample("1h").mean())
    for c in ("afrr_g", "afrr_d", "zmb_afrrg", "zmb_afrrd"):
        for lag in (1, 2, 7):
            df[f"{c}_l{lag}"] = df[c].shift(24 * lag)
    for c in TARGETS:
        day = df[c].shift(24).rolling(24)
        df[f"{c}_l1_mean"], df[f"{c}_l1_max"] = day.mean(), day.max()

    ch = cen.set_index("ts")["cen_cost"].resample("1h").mean().reindex(idx)
    df["cen_l3"] = ch.shift(24 * 3)
    d3 = ch.shift(24 * 3).rolling(24)
    df["cen_l3_mean"], df["cen_l3_max"] = d3.mean(), d3.max()
    df["cen_l3_spikes"] = (ch > 1500).astype(float).shift(24 * 3).rolling(24 * 7).sum()

    df = df.join(_wx_update().set_index("ts")[["w3_d2", "ghi_d2"]])
    loc = df.index.tz_convert(W)
    df["hour"], df["dow"], df["month"] = loc.hour, loc.dayofweek, loc.month
    return df.reset_index().rename(columns={"index": "ts"})


def issue(df: pd.DataFrame) -> pd.DataFrame | None:
    now = pd.Timestamp.now(tz=W)
    target_day = (now + pd.Timedelta(days=1)).date()
    feats = [c for c in df.columns if c.endswith(tuple("1237")) or
             c.endswith(("_mean", "_max", "_spikes")) or
             c in ("w3_d2", "ghi_d2", "hour", "dow", "month")]
    te = df[df["ts"].dt.tz_convert(W).dt.date == target_day].dropna(subset=feats)
    if te.empty:
        return None
    rows = []
    for target in TARGETS:
        tr = df.dropna(subset=feats + [target])
        tr = tr[tr["ts"] < te["ts"].min()]
        block = te[["ts"]].copy()
        block["product"], block["target_day"] = target, str(target_day)
        for tau in QS:
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=tau, **PARAMS)
            mdl.fit(tr[feats], tr[target])
            block[f"q{tau}"] = mdl.predict(te[feats])
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    qcols = [f"q{t}" for t in QS]
    out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    out["issued_ts"] = now.tz_convert("UTC").isoformat(timespec="seconds")
    out["gate_ok"] = bool(now.hour < 8)          # audit flag: pre-gate issuance
    f = STORE / "cmbp_forecasts.parquet"
    if f.exists():
        prev = pd.read_parquet(f)
        dup = (prev["target_day"] == str(target_day))
        if dup.any():                            # never re-issue for the same day
            return None
        out = pd.concat([prev, out], ignore_index=True)
    STORE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f)
    return out


def score(df: pd.DataFrame) -> str:
    f = STORE / "cmbp_forecasts.parquet"
    if not f.exists():
        return "no forecasts yet"
    fc = pd.read_parquet(f)
    real = df.set_index("ts")[list(TARGETS)]
    fc = fc.merge(real.reset_index().melt("ts", var_name="product", value_name="y"),
                  on=["ts", "product"], how="left")
    m = fc.dropna(subset=["y"])
    if m.empty:
        return "no matured forecasts yet"
    pb = np.mean([np.mean(np.where(m["y"] - m[f"q{t}"] >= 0,
                                   t * (m["y"] - m[f"q{t}"]),
                                   (t - 1) * (m["y"] - m[f"q{t}"]))) for t in QS])
    mae = float(np.mean(np.abs(m["y"] - m["q0.5"])))
    naive = df.set_index("ts")[list(TARGETS)].shift(24).reset_index().melt(
        "ts", var_name="product", value_name="nv")
    m2 = m.merge(naive, on=["ts", "product"], how="left").dropna(subset=["nv"])
    mae_nv = float(np.mean(np.abs(m2["y"] - m2["nv"])))
    m[["ts", "product", "target_day", "issued_ts", "gate_ok", "y",
       *(f"q{t}" for t in QS)]].to_parquet(STORE / "cmbp_scores.parquet")
    return (f"scored {len(m)} fc-hours over {m['target_day'].nunique()} days: "
            f"pinball {pb:.2f}  MAE {mae:.1f} (naive1d {mae_nv:.1f})  "
            f"gate_ok {float(m['gate_ok'].mean()):.0%}")


def status() -> None:
    f = STORE / "cmbp_forecasts.parquet"
    if not f.exists():
        print("no forecasts issued yet")
        return
    fc = pd.read_parquet(f)
    print(f"issued: {fc['target_day'].nunique()} days, {len(fc)} fc-hours, "
          f"first {fc['target_day'].min()}, last {fc['target_day'].max()}, "
          f"gate_ok {float(fc['gate_ok'].mean()):.0%}")
    s = STORE / "cmbp_scores.parquet"
    if s.exists():
        sc = pd.read_parquet(s)
        for p, g in sc.groupby("product"):
            print(f"  {p}: n={len(g)}  MAE {np.abs(g['y']-g['q0.5']).mean():.1f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
        return
    STORE.mkdir(parents=True, exist_ok=True)
    df = build()
    new = issue(df)
    line = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
            f"{'issued ' + str(new['target_day'].iloc[-1]) if new is not None else 'no-issue'}"
            f" | {score(df)}\n")
    with LOG.open("a") as fh:
        fh.write(line)
    print(line, end="")


if __name__ == "__main__":
    main()
