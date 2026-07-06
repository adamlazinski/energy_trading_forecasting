"""
Leakage-safe, publication-aware feature engineering for the CEN forecaster.

Golden rule: a feature for delivery period t may only use info available at
the decision gate tau = t - H (H = horizon_minutes).

What "available" means per source (verified empirically on the v2 API,
2026-07-06 — see git history / config.yaml notes):

- csdac_pln, rce_pln     published D-1 ~13:50 local -> known for every gate
                         of day D (earliest gate is D-1 ~23:00 local). SAFE.
- load_fcst              PSE's day-ahead KSE load forecast; single vintage by
                         design (the row's pub_ts reflects later actual-load
                         writes, not the forecast). SAFE by design.
- reserve prices / zmb   D-1 balancing-capacity auction results. SAFE.
- sk_d1_fcst             "designated on day D-1" by definition. SAFE.
- rcco2                  daily, published D-1 evening. SAFE.
- CEN / imbalance energy settlement: published D+1 ~14:00 local, and the API
  only stores the LATEST correction's pub_ts. We therefore gate these by a
  fixed availability rule AVAIL(day) = day D+1 15:00 local, and expose the
  staleness. At H=60 the freshest visible CEN is 1-2 days old.
- pk5l-wp forecasts, pdgobpkd, price-fcst: the API serves only the final
  vintage (pub_ts often AFTER delivery). NOT valid as gate features. pk5l /
  snapshot go behind extended=True (leakage-caveat sensitivity runs);
  price_fcst__cen_fcst is kept only as the PSE benchmark column, never a
  feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
    _PL_HOL = _holidays.Poland()
except Exception:            # optional dependency; degrade gracefully
    _PL_HOL = None

WARSAW = "Europe/Warsaw"

# columns that are never legal features (targets, bookkeeping, benchmarks,
# final-vintage forecasts). Prefix match.
NON_FEATURE_PREFIXES = (
    "y", "ts", "imbalance__", "imb_energy__", "bal_energy__", "kse_actuals__",
    "kse_snapshot__", "price_fcst__", "plan_pk5l__", "xborder_flows__",
    "kse_load__load_actual", "contracting__sk_cost", "contracting__sk_d_fcst",
)


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    loc = df["ts"].dt.tz_convert(WARSAW)
    df["qh_of_day"] = loc.dt.hour * 4 + loc.dt.minute // 15    # 0..95
    df["hour"] = loc.dt.hour
    df["dow"] = loc.dt.dayofweek
    df["month"] = loc.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["qh_sin"] = np.sin(2 * np.pi * df["qh_of_day"] / 96)
    df["qh_cos"] = np.cos(2 * np.pi * df["qh_of_day"] / 96)
    df["doy_sin"] = np.sin(2 * np.pi * loc.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * loc.dt.dayofyear / 365.25)
    if _PL_HOL is not None:
        df["is_holiday"] = loc.dt.date.map(lambda d: d in _PL_HOL).astype("int8")
    else:
        df["is_holiday"] = 0
    # crude solar ramp windows (replace with solar-position calc later)
    df["ramp_am"] = df["hour"].between(4, 8).astype("int8")
    df["ramp_pm"] = df["hour"].between(16, 21).astype("int8")
    return df


def anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead price anchors, fully known at every gate of day D."""
    df = df.sort_values("ts").copy()
    da, rce = "da_price__csdac_pln", "rce__rce_pln"
    for c in (da, rce):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if da in df:
        df["fx_da"] = df[da]
        df["fx_da_lag1d"] = df[da].shift(96)
        df["fx_da_diff1d"] = df["fx_da"] - df["fx_da_lag1d"]
        # shape of the delivery day (whole day known at D-1 13:50)
        day = df["ts"].dt.tz_convert(WARSAW).dt.date
        g = df.groupby(day)[da]
        df["fx_da_day_mean"] = g.transform("mean")
        df["fx_da_day_std"] = g.transform("std")
        df["fx_da_rel_day"] = df[da] - df["fx_da_day_mean"]
        # local slope: the ramp the DA curve prices in around t
        df["fx_da_slope"] = df[da].shift(-2) - df[da].shift(2)
    if rce in df:
        df["fx_rce"] = df[rce]
        df["fx_rce_lag1d"] = df[rce].shift(96)
    if da in df and rce in df:
        df["fx_rce_da_spread"] = df[rce] - df[da]
    return df


def passthrough_known_ahead(df: pd.DataFrame) -> pd.DataFrame:
    """D-1-known system columns kept as-is under fx_ names."""
    df = df.copy()
    known = {
        "kse_load__load_fcst": "fx_load_fcst",
        "contracting__sk_d1_fcst": "fx_sk_d1",
        "co2__rcco2_pln": "fx_co2",
        "reserve_prices_basic__afrr_g": "fx_afrr_g",
        "reserve_prices_basic__afrr_d": "fx_afrr_d",
        "reserve_prices_basic__fcr_g": "fx_fcr_g",
        "reserve_prices_basic__mfrrd_g": "fx_mfrr_g",
        "reserve_req__zmb_afrrg": "fx_zmb_afrrg",
        "reserve_req__zmb_afrrd": "fx_zmb_afrrd",
    }
    for src, dst in known.items():
        if src in df:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
    if "fx_load_fcst" in df:
        df["fx_load_fcst_slope"] = df["fx_load_fcst"].shift(-4) - df["fx_load_fcst"].shift(4)
    return df


def _availability(day: pd.Series, publish_next_day_hour: int) -> pd.Series:
    """AVAIL(day) = day + 1 day at HH:00 local, as UTC."""
    avail = (pd.to_datetime(day) + pd.Timedelta(days=1)
             + pd.Timedelta(hours=publish_next_day_hour))
    return avail.dt.tz_localize(WARSAW, ambiguous="NaT",
                                nonexistent="shift_forward").dt.tz_convert("UTC")


def published_history_features(df: pd.DataFrame, value_col: str, prefix: str,
                               h_minutes: int,
                               publish_next_day_hour: int = 15) -> pd.DataFrame:
    """Features from the last FULLY PUBLISHED day of a settlement series.

    For each row t: find the most recent Warsaw day D* whose data was public
    at the gate (AVAIL(D*) <= t - H), then attach same-quarter-hour value,
    day stats, and staleness. Vectorized via merge_asof on the availability
    time of each day.
    """
    if value_col not in df.columns:
        return df
    df = df.sort_values("ts").copy()
    v = pd.to_numeric(df[value_col], errors="coerce")
    loc = df["ts"].dt.tz_convert(WARSAW)
    day = loc.dt.normalize().dt.tz_localize(None)
    qh = loc.dt.hour * 4 + loc.dt.minute // 15

    # per published day: the 96-qh profile + summary stats
    prof = pd.DataFrame({"day": day, "qh": qh, "v": v}).pivot_table(
        index="day", columns="qh", values="v", aggfunc="last")
    stats = pd.DataFrame({
        "day": prof.index,
        f"{prefix}_pubday_mean": prof.mean(axis=1).values,
        f"{prefix}_pubday_std": prof.std(axis=1).values,
        f"{prefix}_pubday_min": prof.min(axis=1).values,
        f"{prefix}_pubday_max": prof.max(axis=1).values,
        f"{prefix}_pubday_neg_share": (prof < 0).mean(axis=1).values,
    })
    stats["avail"] = _availability(stats["day"], publish_next_day_hour)
    stats = stats.dropna(subset=["avail"]).sort_values("avail")

    gate = (df["ts"] - pd.Timedelta(minutes=h_minutes)).rename("gate")
    picked = pd.merge_asof(
        pd.DataFrame({"gate": gate}).sort_values("gate"),
        stats, left_on="gate", right_on="avail", direction="backward")
    picked.index = gate.sort_values().index
    picked = picked.reindex(df.index)

    for c in stats.columns:
        if c.startswith(prefix):
            df[c] = picked[c]
    df[f"{prefix}_staleness_d"] = (day - picked["day"]).dt.days

    # same-qh value from the picked day, via profile lookup
    key = pd.MultiIndex.from_arrays([picked["day"], qh])
    flat = prof.stack()
    df[f"{prefix}_pub_same_qh"] = flat.reindex(key).values
    return df


def extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Final-vintage columns (LEAKAGE CAVEAT — sensitivity runs only)."""
    df = df.copy()
    ext = {
        "plan_pk5l__grid_demand_fcst": "xt_demand_fcst",
        "plan_pk5l__fcst_pv_tot_gen": "xt_pv_fcst",
        "plan_pk5l__fcst_wi_tot_gen": "xt_wind_fcst",
        "plan_pk5l__planned_exchange": "xt_exchange",
        "plan_pk5l__surplus_cap_avail_tso": "xt_surplus",
        "kse_snapshot__gen_fv": "xt_pv_plan",
        "kse_snapshot__gen_wi": "xt_wind_plan",
        "kse_snapshot__rez_over_demand": "xt_rez_over",
    }
    for src, dst in ext.items():
        if src in df:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
    if {"xt_pv_fcst", "xt_demand_fcst"} <= set(df.columns):
        df["xt_res_load_ratio"] = (
            (df["xt_pv_fcst"].fillna(0) + df.get("xt_wind_fcst", 0).fillna(0))
            / df["xt_demand_fcst"].clip(lower=1))
    return df


def feature_cols(df: pd.DataFrame, extended: bool = False) -> list[str]:
    """The model's input columns: fx_/cal/hist prefixes, xt_ if extended."""
    core_prefix = ("fx_", "cen_", "imb_", "qh_", "doy_", "hour", "dow", "month",
                   "is_", "ramp_", "regime_")
    cols = [c for c in df.columns if c.startswith(core_prefix)
            and not c.startswith(NON_FEATURE_PREFIXES)]
    if extended:
        cols += [c for c in df.columns if c.startswith("xt_")]
    return cols


def build(df: pd.DataFrame, cfg: dict, extended: bool = False) -> pd.DataFrame:
    """Assemble the feature panel. Expects 'ts' and target 'y' present."""
    h = cfg["horizon_minutes"]
    out = calendar_features(df)
    out = anchor_features(out)
    out = passthrough_known_ahead(out)
    out = published_history_features(out, "y", "cen", h)
    out = published_history_features(out, "imb_energy__balance", "imb", h)
    if extended:
        out = extended_features(out)
    # require the anchors + published CEN history; keep partial otherwise
    need = [c for c in ("fx_da", "cen_pub_same_qh") if c in out.columns]
    if need:
        out = out.dropna(subset=need).reset_index(drop=True)
    return out
