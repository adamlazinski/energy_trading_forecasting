"""Your own data -> leakage-safe features at the gate g = T - horizon_min.

realtime  rows (time_col, values...): a value is usable from time_col + known_delay_min.
          Per column: last value at g, change over the hour before g, 60-min mean.
          Values older than max_staleness_min at g count as missing.
forecast  rows (issued_col, target_col, values...): for period T the latest
          forecast for T issued at or before g.

Nothing after the gate can enter a feature: that is enforced here, by
construction, not left to the user.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from . import UserError
from .panel import ns


def _read(spec: dict) -> pd.DataFrame:
    p: pathlib.Path = spec["path"]
    if not p.exists():
        raise UserError(f"own_data '{spec['name']}': nie ma pliku {p}")
    suf = p.suffix.lower()
    try:
        if suf == ".parquet":
            return pd.read_parquet(p)
        if suf in (".csv", ".txt"):
            return pd.read_csv(p, sep=spec["sep"], decimal=spec["decimal"], low_memory=False)
    except Exception as e:                                      # noqa: BLE001 - show the cause
        raise UserError(f"own_data '{spec['name']}': nie da się wczytać {p.name}: {e}") from None
    raise UserError(f"own_data '{spec['name']}': obsługiwane są .csv i .parquet "
                    f"(jest {suf}). Excel: zapisz jako CSV.")


def _to_utc(s: pd.Series, spec: dict, col: str) -> pd.Series:
    name = spec["name"]
    if pd.api.types.is_datetime64_any_dtype(s):
        t = s
    else:
        txt = s.astype(str)
        # 18.10.2026 13:45 (Polish/Excel style) must be read day-first
        dayfirst = bool(txt.str.match(r"^\s*\d{1,2}[./-]\d{1,2}[./-]\d{4}").mean() > 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                t = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
            except (ValueError, TypeError):
                t = None
            if t is None or t.dtype == object:          # mixed UTC offsets (DST)
                t = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst, utc=True)
        bad = t.isna() & s.notna()
        if bad.any():
            raise UserError(f"own_data '{name}': {bad.sum()} wartości w '{col}' to nie daty, "
                            f"np. {s[bad].iloc[0]!r}. Użyj formatu 2026-10-18 13:45:00[+02:00].")
    if getattr(t.dt, "tz", None) is None:
        tz = spec["timezone"]
        if not tz:
            raise UserError(
                f"own_data '{name}': daty w '{col}' nie mają strefy czasowej (np. '2026-10-18 13:45'). "
                "Dopisz w konfiguracji `timezone: Europe/Warsaw` (czas polski) albo `timezone: UTC`.")
        try:                                   # sorted data: resolve the repeated DST hour
            t = pd.Series(pd.DatetimeIndex(t).tz_localize(tz, ambiguous="infer", nonexistent="NaT"),
                          index=t.index)
        except Exception:                                       # noqa: BLE001
            try:
                t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            except Exception as e:                              # noqa: BLE001
                raise UserError(f"own_data '{name}': zła strefa '{tz}': {e}") from None
        lost = t.isna() & s.notna()
        if lost.any():
            print(f"  UWAGA own_data '{name}': {lost.sum()} znaczników w godzinach zmiany czasu "
                  f"(niejednoznacznych) pominięto. Podawaj czas w UTC, żeby tego uniknąć.")
    return t.dt.tz_convert("UTC")


def _numeric(df: pd.DataFrame, spec: dict, skip: list[str]) -> pd.DataFrame:
    name = spec["name"]
    if spec["columns"]:
        miss = [c for c in spec["columns"] if c not in df.columns]
        if miss:
            raise UserError(f"own_data '{name}': brak kolumn {miss}. W pliku są: {list(df.columns)}")
        cols = list(spec["columns"])
    else:
        cols = [c for c in df.columns if c not in skip]
    out = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        bad = v.isna() & df[c].notna()
        if bad.mean() > 0.05:
            if spec["columns"]:
                raise UserError(f"own_data '{name}': kolumna '{c}' nie jest liczbowa, np. "
                                f"{df[c][bad].iloc[0]!r}. Przecinek dziesiętny? Ustaw `decimal: \",\"`.")
            continue                                 # auto mode: silently skip text columns
        out[c] = v
    if not out:
        raise UserError(f"own_data '{name}': nie znaleziono żadnej kolumny liczbowej. "
                        f"Kolumny w pliku: {list(df.columns)}. Sprawdź `sep` i `decimal`.")
    return pd.DataFrame(out, index=df.index)


def load_source(spec: dict) -> pd.DataFrame:
    """Clean frame: realtime -> index 'known' (UTC); forecast -> columns issued, target."""
    raw = _read(spec)
    name = spec["name"]
    if spec["kind"] == "realtime":
        tc = spec["time_col"]
        if tc not in raw.columns:
            raise UserError(f"own_data '{name}': brak kolumny czasu '{tc}'. Kolumny: {list(raw.columns)}")
        t = _to_utc(raw[tc], spec, tc) + pd.Timedelta(minutes=spec["known_delay_min"])
        vals = _numeric(raw, spec, [tc])
        df = vals.set_index(pd.DatetimeIndex(t, name="known"))
        df = df[df.index.notna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
    else:
        ic, tc = spec["issued_col"], spec["target_col"]
        for c in (ic, tc):
            if c not in raw.columns:
                raise UserError(f"own_data '{name}': brak kolumny '{c}'. Kolumny: {list(raw.columns)}")
        vals = _numeric(raw, spec, [ic, tc])
        df = vals.assign(issued=_to_utc(raw[ic], spec, ic), target=_to_utc(raw[tc], spec, tc))
        df = df.dropna(subset=["issued", "target"])
        off = df.target.dt.minute % 15 != 0
        if off.any():
            raise UserError(f"own_data '{name}': {off.sum()} wartości '{tc}' nie leży na początku "
                            f"kwadransa (np. {df.target[off].iloc[0]}). target_col = POCZĄTEK okresu.")
        r = spec["target_resolution_min"]
        if r > 15:                                   # spread each row over its quarters
            df = pd.concat([df.assign(target=df.target + pd.Timedelta(minutes=15 * k))
                            for k in range(r // 15)], ignore_index=True)
        late = df.issued > df.target
        if late.mean() > 0.5:
            print(f"  UWAGA own_data '{name}': {late.mean():.0%} prognoz wydano PO początku okresu. "
                  "To raczej pomiary — użyj kind: realtime.")
        df = df.sort_values(["target", "issued"]).drop_duplicates(["target", "issued"], keep="last")
    if df.empty:
        raise UserError(f"own_data '{name}': po wczytaniu nie został żaden wiersz.")
    return df


def _asof(times_ns: np.ndarray, vals: np.ndarray, at_ns: np.ndarray, tol_ns: int):
    k = np.searchsorted(times_ns, at_ns, side="right") - 1
    ok = k >= 0
    kc = np.clip(k, 0, None)
    ok &= (at_ns - times_ns[kc]) <= tol_ns
    return np.where(ok, vals[kc], np.nan), np.where(k >= 0, (at_ns - times_ns[kc]) / 6e10, np.nan)


def gate_features(spec: dict, df: pd.DataFrame, idx: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """Features for delivery periods `idx` (UTC period starts) at gate idx - h."""
    p = spec["name"]
    gate = ns(idx - pd.Timedelta(minutes=h))
    out = pd.DataFrame(index=idx)
    if spec["kind"] == "realtime":
        tol = int(spec["max_staleness_min"] * 6e10)
        t = ns(df.index)
        for c in df.columns:
            s = df[c].dropna()
            if s.empty:
                continue
            ts, v = ns(s.index), s.to_numpy(float)
            last, age = _asof(ts, v, gate, tol)
            prev, _ = _asof(ts, v, gate - int(36e11), tol)
            m60 = s.rolling("60min").mean().to_numpy(float)
            out[f"{p}_{c}_last"] = last
            out[f"{p}_{c}_d60"] = last - prev
            out[f"{p}_{c}_m60"], _ = _asof(ts, m60, gate, tol)
        _, age = _asof(t, np.zeros(len(t)), gate, np.iinfo(np.int64).max // 2)
        out[f"{p}_age_min"] = age
    else:
        left = pd.DataFrame({"target": idx, "gate": pd.DatetimeIndex(gate, tz="UTC").as_unit("ns")})
        right = df.copy()
        right["issued"] = right.issued.dt.as_unit("ns")
        right["target"] = right.target.dt.as_unit("ns")
        left["target"] = left.target.dt.as_unit("ns")
        m = pd.merge_asof(left.reset_index().sort_values("gate"), right.sort_values("issued"),
                          left_on="gate", right_on="issued", by="target", direction="backward")
        m = m.sort_values("index").set_index(idx)
        for c in df.columns.drop(["issued", "target"]):
            out[f"{p}_{c}"] = m[c].to_numpy(float)
        out[f"{p}_lead_min"] = ((m.target - m.issued).dt.total_seconds() / 60).to_numpy()
    return out


def all_features(specs: list[dict], idx: pd.DatetimeIndex, h: int,
                 cache: dict | None = None) -> pd.DataFrame:
    parts = []
    for s in specs:
        df = cache[s["name"]] if cache and s["name"] in cache else load_source(s)
        parts.append(gate_features(s, df, idx, h))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=idx)


def describe(spec: dict, df: pd.DataFrame) -> list[str]:
    name = spec["name"]
    if spec["kind"] == "realtime":
        t = df.index
        step = pd.Series(t).diff().median()
        lines = [f"[{name}] realtime  {len(df)} wierszy  {t.min()} .. {t.max()} (UTC, z opóźnieniem)",
                 f"   typowy krok {step}; kolumny: {', '.join(df.columns)}"]
        gaps = pd.Series(t).diff()
        big = gaps[gaps > pd.Timedelta(minutes=spec["max_staleness_min"])]
        if len(big):
            lines.append(f"   przerwy > {spec['max_staleness_min']} min: {len(big)} "
                         f"(najdłuższa {big.max()}) — wtedy cechy = brak danych")
    else:
        lead = (df.target - df.issued).dt.total_seconds() / 60
        lines = [f"[{name}] forecast  {len(df)} wierszy  okresy {df.target.min()} .. {df.target.max()}",
                 f"   wyprzedzenie prognoz [min]: mediana {lead.median():.0f}, min {lead.min():.0f}; "
                 f"kolumny: {', '.join(df.columns.drop(['issued', 'target']))}"]
    na = df.drop(columns=[c for c in ("issued", "target") if c in df.columns]).isna().mean()
    for c, v in na[na > 0.05].items():
        lines.append(f"   UWAGA: '{c}' ma {v:.0%} pustych wartości")
    return lines
