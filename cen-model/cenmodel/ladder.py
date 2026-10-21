"""Offer-ladder curves: price at fixed cumulative volume, per 15-min period and direction.

Two sources (both cached by `fetch`):
  bpkdbo  oeb-bpkdbo, the BPKD/BO plan ladder that SET the price: bands in the
          activated direction only, price = CDO (offer + start-up correction).
          Published D+1 -> lagged >= LAG_DAYS when used as a forecast input.
  rbn     poeb-rbn, every balancing offer (one MW, an up price ofcg and a down
          price ofcd). Published ~2 min after the period -> usable at the gate.

A curve is the price of the band covering V MW of cumulative merit-order volume
for V in GRID (G: cheapest first, D: highest first). NaN = ladder exhausted
before V; `ladder_mw` is the total. This is exactly how CKOEB is set
(|ZEB_PP| x 4 MW), so crossing a forecast curve at the realised volume
measures the curve error alone.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd

from . import UserError
from .pse import WAW
from .wdb import undo_api_replication

GRID = (25, 50, 100, 150, 200, 300, 400, 500, 600, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000)
PCOLS = [f"p{v}" for v in GRID]
LAG_DAYS = 2                 # bpkdbo of day D is public D+1 afternoon -> D-2 is safe at any gate
PCA_MAX_MISSING = 0.10       # PCA only on grid points where the ladder rarely runs out


# --- bands -> curves ---------------------------------------------------------
def bands_bpkdbo(lad: pd.DataFrame) -> pd.DataFrame:
    """oeb-bpkdbo rows -> bands (ts, dir, price, mw, startup)."""
    lad = lad.copy()
    for c in ("cdo", "ofp", "ofc"):
        if c in lad:
            lad[c] = pd.to_numeric(lad[c], errors="coerce")
    lad = undo_api_replication(lad.dropna(subset=["cdo", "ofp"]))
    out = pd.DataFrame({"ts": lad.ts, "dir": lad.activ_direction, "price": lad.cdo, "mw": lad.ofp.abs()})
    if "ofc" in lad:
        out["startup"] = (lad.cdo - lad.ofc).abs() > 0.005
    return out


def bands_rbn(off: pd.DataFrame) -> pd.DataFrame:
    """poeb-rbn rows -> bands, each offer twice: up at ofcg, down at ofcd."""
    off = off.copy()
    for c in ("ofp", "ofcg", "ofcd"):
        off[c] = pd.to_numeric(off[c], errors="coerce")
    mw = off.ofp.abs()
    return pd.concat([pd.DataFrame({"ts": off.ts, "dir": "G", "price": off.ofcg, "mw": mw}),
                      pd.DataFrame({"ts": off.ts, "dir": "D", "price": off.ofcd, "mw": mw})],
                     ignore_index=True).dropna(subset=["price", "mw"])


def _merit(b: pd.DataFrame):
    """Sort into merit order; return the sorted frame, group ids and per-group keys."""
    up = b["dir"].eq("G").to_numpy()
    b = b.assign(key=np.where(up, b.price, -b.price)).sort_values(["ts", "dir", "key"], kind="mergesort")
    gid = b.groupby(["ts", "dir"], sort=True).ngroup().to_numpy()
    cum = b.mw.groupby(gid).cumsum().to_numpy()
    keys = b.groupby(["ts", "dir"], sort=True).size().index
    return b, gid, cum, keys


def _price_at(b, gid, cum, need_by_group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Price of the first band with cum >= need, per group (vectorised). -> (price, row pos)."""
    big = float(cum.max()) + 1.0 if len(cum) else 1.0
    flat = cum + gid * big
    g = np.arange(len(need_by_group))
    pos = np.searchsorted(flat, g * big + need_by_group - 1e-6)
    ok = (pos < len(flat)) & np.isfinite(need_by_group)
    ok[ok] &= gid[pos[ok]] == g[ok]
    price = np.full(len(g), np.nan)
    price[ok] = b.price.to_numpy()[pos[ok]]
    return price, np.where(ok, pos, -1)


def curves(bands: pd.DataFrame) -> pd.DataFrame:
    """bands -> one row per (ts, dir): p<V> for V in GRID, ladder_mw, p_top."""
    if bands.empty:
        return pd.DataFrame(columns=["ts", "dir", *PCOLS, "ladder_mw", "p_top"])
    b, gid, cum, keys = _merit(bands)
    n = len(keys)
    out = pd.DataFrame(index=keys)
    for v, col in zip(GRID, PCOLS):
        out[col] = _price_at(b, gid, cum, np.full(n, float(v)))[0]
    last = pd.Series(np.arange(len(b))).groupby(gid).max().to_numpy()
    out["ladder_mw"] = cum[last]
    out["p_top"] = b.price.to_numpy()[last]
    return out.reset_index()


def cross(bands: pd.DataFrame, zeb_pp: pd.Series) -> pd.DataFrame:
    """CKOEB from bands at |ZEB_PP| x 4 MW, plus whether the marginal band has a start-up correction."""
    b, gid, cum, keys = _merit(bands[bands.ts.isin(zeb_pp.index)])
    need = pd.Series(keys.get_level_values(0)).map(zeb_pp.abs() * 4).to_numpy(float)
    price, pos = _price_at(b, gid, cum, need)
    out = pd.DataFrame({"ckoeb_hat": price, "need_mw": need}, index=keys)
    if "startup" in b:
        out["startup_marginal"] = np.where(pos >= 0, b.startup.to_numpy()[np.maximum(pos, 0)], False)
    return out


# --- cache -> curve panel ----------------------------------------------------
def _days(d: pathlib.Path, start, end):
    files = sorted(d.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= start]
    if end:
        files = [f for f in files if f.stem <= end]
    return files


def build(workdir: pathlib.Path, source: str, start=None, end=None, zeb_pp=None, log=print) -> pd.DataFrame:
    """Curves for every cached day of `source`, month by month (memory-safe)."""
    sub = {"bpkdbo": "ladder", "rbn": "offers"}[source]
    files = _days(workdir / "raw" / sub, start, end)
    if not files:
        flag = "--ladder" if source == "bpkdbo" else "--offers"
        raise UserError(f"Brak danych '{sub}' w {workdir / 'raw'}. Uruchom: python -m cenmodel fetch {flag} RRRR-MM")
    to_bands = bands_bpkdbo if source == "bpkdbo" else bands_rbn
    out = []
    for month, grp in pd.Series(files).groupby([f.stem[:7] for f in files]):
        raw = pd.concat([pd.read_parquet(f) for f in grp], ignore_index=True)
        b = to_bands(raw)
        c = curves(b)
        if zeb_pp is not None and source == "bpkdbo":
            x = cross(b, zeb_pp.dropna()).reset_index().rename(columns={"level_0": "ts", "level_1": "dir"})
            c = c.merge(x, on=["ts", "dir"], how="left")
        out.append(c)
        log(f"  {source} {month}: {c.ts.nunique():5d} okresów")
    return pd.concat(out, ignore_index=True)


# --- analysis ---------------------------------------------------------------
def _wide(c: pd.DataFrame, d: str, col: str, idx: pd.DatetimeIndex) -> pd.Series:
    return c[c["dir"] == d].set_index("ts")[col].reindex(idx)


def describe(c: pd.DataFrame, pub: pd.DataFrame) -> list[str]:
    idx = pd.date_range(c.ts.min(), c.ts.max(), freq="15min")
    out = []
    for d, name in (("G", "w górę (G)"), ("D", "w dół (D)")):
        x = c[c["dir"] == d]
        if x.empty:
            continue
        out += ["", f"== kierunek {name}: {len(x)} okresów, mediana ladder_mw {x.ladder_mw.median():.0f} MW",
                f"   {'punkt':>6s} {'P10':>8s} {'mediana':>8s} {'P90':>8s} {'brak%':>6s} "
                f"{'ak15m':>6s} {'ak24h':>6s} {'ak7d':>6s} {'corrCSDAC':>9s}"]
        for col in PCOLS:
            s = _wide(c, d, col, idx)
            cs = pub.csdac.reindex(idx)
            out.append(f"   {col:>6s} {x[col].quantile(.1):8.0f} {x[col].median():8.0f} {x[col].quantile(.9):8.0f} "
                       f"{x[col].isna().mean():6.1%} {s.autocorr(1):6.2f} {s.autocorr(96):6.2f} "
                       f"{s.autocorr(672):6.2f} {s.corr(cs):9.2f}")
        if "startup_marginal" in x:
            out.append(f"   pasmo krańcowe z korektą rozruchową: {x.startup_marginal.mean():.1%} okresów")
        loc = pd.DatetimeIndex(x.ts).tz_convert(WAW)
        prof = x.p500.groupby(loc.hour).median()
        out.append("   mediana p500 wg godziny (czas PL): "
                   + " ".join(f"{h}:{v:.0f}" for h, v in prof.items()))
        mon = x.groupby(loc.strftime("%Y-%m"))[["p100", "p500", "p1000", "p2000"]].median()
        out.append("   mediana wg miesiąca  p100 / p500 / p1000 / p2000:")
        out += [f"     {m}  {r.p100:6.0f} {r.p500:6.0f} {r.p1000:6.0f} {r.p2000:6.0f}" for m, r in mon.iterrows()]
    return out


def pca(c: pd.DataFrame, d: str, k: int = 3):
    """PCA on asinh(price) over grid points that are rarely exhausted, complete rows only."""
    x = c[c["dir"] == d]
    cols = [p for p in PCOLS if x[p].isna().mean() <= PCA_MAX_MISSING]
    x = x.dropna(subset=cols)
    if len(x) < 50:
        return None
    a = np.arcsinh(x[cols].to_numpy(float))
    mu = a.mean(0)
    u, s, vt = np.linalg.svd(a - mu, full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    scores = pd.DataFrame((a - mu) @ vt[:k].T, index=pd.DatetimeIndex(x.ts), columns=[f"f{i + 1}" for i in range(k)])
    return {"cols": cols, "mean": mu, "load": vt[:k], "ev": ev[:k], "scores": scores}


def pca_lines(c: pd.DataFrame) -> list[str]:
    out = ["", f"== PCA krzywej (asinh ceny; punkty z brakiem <= {PCA_MAX_MISSING:.0%}): czynniki i ładunki"]
    for d in ("G", "D"):
        r = pca(c, d)
        if r is None:
            continue
        out.append(f"   {d}: wyjaśniona wariancja " + ", ".join(f"f{i + 1} {e:.1%}" for i, e in enumerate(r["ev"])))
        out.append("      " + " ".join(f"{p:>7s}" for p in r["cols"]))
        for i, row in enumerate(r["load"]):
            out.append(f"   f{i + 1} " + " ".join(f"{v:7.2f}" for v in row))
    out.append("   (f1 z ładunkami jednego znaku = poziom; f2 zmieniający znak = nachylenie; f3 = wygięcie/ogon)")
    return out


def _lagged(c: pd.DataFrame, d: str, idx, lag_q: int) -> pd.DataFrame:
    w = c[c["dir"] == d].set_index("ts")[PCOLS]
    full = w.reindex(pd.date_range(min(w.index.min(), idx.min()), idx.max(), freq="15min"))
    return full.shift(lag_q).reindex(idx)


def _median_days(c: pd.DataFrame, d: str, idx, lag_days: int, n_days: int) -> pd.DataFrame:
    w = c[c["dir"] == d].set_index("ts")[PCOLS]
    full = w.reindex(pd.date_range(min(w.index.min(), idx.min()), idx.max(), freq="15min"))
    stack = np.stack([full.shift(96 * (lag_days + i)).reindex(idx).to_numpy(float) for i in range(n_days)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN slices -> NaN, expected
        med = np.nanmedian(stack, axis=0)
    return pd.DataFrame(med, index=idx, columns=PCOLS)


def _monotone(a: np.ndarray, up: bool) -> np.ndarray:
    a = np.where(np.isnan(a), -np.inf if up else np.inf, a)
    a = np.maximum.accumulate(a, axis=1) if up else np.minimum.accumulate(a, axis=1)
    return np.where(np.isinf(a), np.nan, a)


def at_volume(curve: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Read a gridded curve at `need` MW: linear in asinh(price) between grid points, flat outside."""
    g = np.array(GRID, float)
    a = np.arcsinh(curve)
    out = np.full(len(need), np.nan)
    for i in range(len(need)):
        ok = np.isfinite(a[i])
        if ok.sum() == 0 or not np.isfinite(need[i]):
            continue
        out[i] = np.sinh(np.interp(need[i], g[ok], a[i][ok]))
    return out


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[0, 0] -= lam                                         # no penalty on the intercept
    return np.linalg.solve(A, X.T @ y)


def evaluate(c: pd.DataFrame, pub: pd.DataFrame, test_days: int, horizon_min: int,
             rbn: pd.DataFrame | None = None) -> tuple[list[str], pd.DataFrame]:
    """Forecast the bpkdbo curve of the activated direction; score it at the realised volume vs CKOEB."""
    c = c[c.ts.isin(pub.index[pub.zeb_pp.notna() & pub.ckoeb.notna()])]
    t_end = c.ts.max()
    t_test = (t_end - pd.Timedelta(days=test_days)).tz_convert(WAW).normalize().tz_convert("UTC")
    lag_rbn = int(np.ceil((horizon_min + 15 + 5) / 15))    # period T-k ends T-k+15, published ~2 min later
    preds, lines = [], []
    for d in ("G", "D"):
        x = c[c["dir"] == d].set_index("ts").sort_index()
        if len(x) < 500:
            continue
        idx = x.index
        lag2 = _lagged(c, d, idx, 96 * LAG_DAYS)
        med7 = _median_days(c, d, idx, LAG_DAYS, 7)
        cand = {"D-2 ta sama pora (brak -> mediana)": lag2.fillna(med7).to_numpy(float),
                "mediana 7 dni (od D-2)": med7.to_numpy(float)}
        if rbn is not None and not rbn.empty:
            cand["rbn sprzed bramki"] = _lagged(rbn, d, idx, lag_rbn).to_numpy(float)

        loc = idx.tz_convert(WAW)
        cal = np.column_stack([np.ones(len(idx))]
                              + [(loc.hour == h).astype(float) for h in range(1, 24)]
                              + [(loc.dayofweek == k).astype(float) for k in range(1, 7)])
        cs = np.arcsinh(pub.csdac.reindex(idx).to_numpy(float))
        reg = np.full((len(idx), len(PCOLS)), np.nan)
        tr = np.asarray(idx < t_test - pd.Timedelta(days=LAG_DAYS))
        for j, col in enumerate(PCOLS):
            feats = [cal, cs[:, None], np.arcsinh(lag2[col].to_numpy(float))[:, None],
                     np.arcsinh(med7[col].to_numpy(float))[:, None]]
            if "rbn sprzed bramki" in cand:
                feats.append(np.arcsinh(cand["rbn sprzed bramki"][:, j])[:, None])
            X = np.hstack(feats)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)    # a lag that is never observed
                fill = np.nan_to_num(np.nanmedian(X[tr], axis=0))
            X = np.where(np.isnan(X), fill, X)
            y = np.arcsinh(x[col].to_numpy(float))
            ok = tr & np.isfinite(y)
            if ok.sum() < 200:
                continue
            lo, hi = np.nanquantile(y[ok], [0.001, 0.999])  # never extrapolate past the training range
            reg[:, j] = np.sinh(np.clip(X @ _ridge(X[ok], y[ok]), lo, hi))
        cand["regresja (kalendarz+CSDAC+lagi)"] = reg
        cand["[wyrocznia: prawdziwa krzywa]"] = x[PCOLS].to_numpy(float)

        need = (pub.zeb_pp.abs() * 4).reindex(idx).to_numpy(float)
        te = np.asarray(idx >= t_test)
        act = pub.ckoeb.reindex(idx).to_numpy(float)
        for name, cv in cand.items():
            p = at_volume(_monotone(cv[te], d == "G"), need[te])
            preds.append(pd.DataFrame({"ts": idx[te], "dir": d, "model": name, "pred": p, "ckoeb": act[te],
                                       "need_mw": need[te]}))
    if not preds:
        return ["", "== Prognoza krzywej: za mało danych"], pd.DataFrame()
    P = pd.concat(preds, ignore_index=True)
    ok = P.assign(f=np.isfinite(P.pred) & np.isfinite(P.ckoeb)).groupby(["ts", "dir"]).f.all()
    P = P.merge(ok.rename("common").reset_index(), on=["ts", "dir"])
    lines += ["",f"== Prognoza krzywej BPKD/BO, test {t_test.tz_convert(WAW):%Y-%m-%d} .. {t_end.tz_convert(WAW):%Y-%m-%d}"
              f" ({test_days} dni); ocena: krzywa przecięta na RZECZYWISTYM |ZEB_PP|x4 vs CKOEB",
              f"   {'model':34s} {'kier':>4s} {'n':>6s} {'MAE':>7s} {'medAE':>7s} {'P90 AE':>7s} {'<10 zł':>7s}"]
    for (name, d), g in P[P.common].groupby(["model", "dir"], sort=False):
        e = (g.pred - g.ckoeb).abs()
        if e.empty:
            continue
        lines.append(f"   {name:34s} {d:>4s} {len(e):6d} {e.mean():7.1f} {e.median():7.1f} "
                     f"{e.quantile(.9):7.1f} {np.mean(e < 10):7.1%}")
    lines += ["   Wszystkie modele na tych samych okresach (gdzie każdy ma prognozę).",
              "   Wyrocznia = błąd samej dyskretyzacji siatki (dolna granica dla każdego modelu).",
              f"   Lagi: bpkdbo >= {LAG_DAYS} doby (publikacja D+1); rbn >= {lag_rbn} kwadransów "
              f"(publikacja ~2 min po okresie, bramka {horizon_min} min)."]
    return lines, P
