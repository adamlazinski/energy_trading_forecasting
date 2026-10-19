"""Public PSE panel on a full 15-min UTC grid + leakage-safe public features.

Every feature for delivery period T uses only what was public at the gate
g = T - horizon_min:
  calendar, CSDAC of day D (published D-1 ~13:50),
  SK / CEN of the last fully published day (settlement publishes D+1 ~14:00;
  modelled as available D+1 15:00).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import UserError
from .pse import SOURCES, WAW, read_slug
from .wdb import AFRR_START

RENAME = {
    "cen_cost": "cen", "ckoeb_cost": "ckoeb", "ceb_pp_cost": "ceb_pp",
    "ceb_sr_cost": "ceb_sr", "ceb_sr_afrrg_cost": "ceb_afrr_g",
    "ceb_sr_afrrd_cost": "ceb_afrr_d", "cor_pp_cost": "cor",
    "zebpp": "zeb_pp", "pzeb_afrrg_cost": "pzeb_g", "pzeb_afrrd_cost": "pzeb_d",
    "sk_cost": "sk", "csdac_pln": "csdac", "ro_cost": "ro",
}
PUBLISH_HOUR = 15             # settlement of day D usable from D+1 15:00 Warsaw


def load_public(workdir: pathlib.Path, start: str) -> pd.DataFrame:
    frames = []
    for slug, cols in SOURCES.items():
        d = read_slug(workdir, slug)
        d = d[["ts"] + [c for c in cols if c in d.columns]].drop_duplicates("ts", keep="last")
        frames.append(d.set_index("ts"))
    t0 = pd.Timestamp(start, tz=WAW).tz_convert("UTC")
    t1 = max(f.index.max() for f in frames)
    grid = pd.date_range(t0, t1, freq="15min", name="ts")
    df = pd.concat([f.reindex(grid) for f in frames], axis=1)
    num = [c for c in df.columns if c != "contracting"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    cm = read_slug(workdir, "cordmax").drop_duplicates("business_date", keep="last")
    cm = pd.to_numeric(cm.set_index("business_date").cordmax_cost, errors="coerce")
    df["cordmax"] = df.index.tz_convert(WAW).strftime("%Y-%m-%d").map(cm).astype(float)
    return df.rename(columns=RENAME)


def ns(t) -> np.ndarray:
    """Timestamps as int64 nanoseconds (pandas 2 and 3 differ in default units)."""
    return pd.DatetimeIndex(t).as_unit("ns").asi8


def _published_history(idx: pd.DatetimeIndex, v: pd.Series, prefix: str, h: int) -> pd.DataFrame:
    loc = idx.tz_convert(WAW)
    day = loc.normalize().tz_localize(None)
    qh = loc.hour * 4 + loc.minute // 15
    long = pd.DataFrame({"day": day, "qh": qh, "v": v.to_numpy()}).dropna()
    prof = long.pivot_table(index="day", columns="qh", values="v", aggfunc="last")
    prof = prof[prof.notna().sum(axis=1) >= 90]                    # complete days only
    st = pd.DataFrame({
        f"{prefix}_mean": prof.mean(axis=1), f"{prefix}_std": prof.std(axis=1),
        f"{prefix}_min": prof.min(axis=1), f"{prefix}_max": prof.max(axis=1),
        f"{prefix}_neg_share": (prof < 0).mean(axis=1)})
    avail = (st.index + pd.Timedelta(days=1, hours=PUBLISH_HOUR)).tz_localize(WAW).tz_convert("UTC")
    st = st.assign(avail=avail).sort_values("avail")
    gate = ns(idx - pd.Timedelta(minutes=h))
    k = np.searchsorted(ns(st.avail), gate, side="right") - 1
    ok = k >= 0
    out = pd.DataFrame(index=idx)
    for c in st.columns[:-1]:
        out[c] = np.where(ok, st[c].to_numpy()[np.clip(k, 0, None)], np.nan)
    pday = np.where(ok, st.index.to_numpy()[np.clip(k, 0, None)], np.datetime64("NaT", "ns"))
    out[f"{prefix}_staleness_d"] = (day.to_numpy() - pday) / np.timedelta64(1, "D")
    flat = prof.stack()
    key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(pday), qh])
    out[f"{prefix}_same_qh"] = flat.reindex(key).to_numpy()
    return out


def public_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
    idx = df.index
    loc = idx.tz_convert(WAW)
    f = pd.DataFrame(index=idx)
    f["qh"] = loc.hour * 4 + loc.minute // 15
    f["hour"], f["dow"], f["month"] = loc.hour, loc.dayofweek, loc.month
    f["is_afrr"] = (idx >= AFRR_START).astype(int)
    f["csdac"] = df.csdac
    g = df.csdac.groupby(loc.normalize())
    f["csdac_day_mean"], f["csdac_day_max"] = g.transform("mean"), g.transform("max")
    f["csdac_ramp"] = df.csdac.shift(-4) - df.csdac.shift(4)
    return pd.concat([f, _published_history(idx, df.sk, "skh", h),
                      _published_history(idx, df.cen, "cenh", h)], axis=1)


def simulated_rt_sk(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Backtest-only what-if: a perfect real-time feed of SK for every quarter
    that had ENDED by the gate. Not available in live use."""
    k0 = h // 15 + 1
    f = pd.DataFrame(index=df.index)
    for k in (k0, k0 + 1, k0 + 3, k0 + 7):
        f[f"simsk_l{k}"] = df.sk.shift(k)
    f["simsk_mean4"] = df.sk.shift(k0).rolling(4).mean()
    f["simsk_afrr_px"] = np.where(df.sk.shift(k0) < 0, df.ceb_afrr_g.shift(k0), df.ceb_afrr_d.shift(k0))
    return f


def last_published(df: pd.DataFrame) -> pd.Timestamp:
    s = df.cen.dropna()
    if s.empty:
        raise UserError("W cache nie ma żadnych opublikowanych CEN. Uruchom: python -m cenmodel fetch")
    return s.index.max()
