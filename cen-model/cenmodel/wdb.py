"""The CEN price chain of WDB pkt 12-13 (tekst jednolity incl. Zmiany 3/2025).

  CEN    = min(CEBsr, CSDAC) if SK>0, max(CEBsr, CSDAC) if SK<0, CEBsr if SK=0  [13.1]
           (CSDAC = 0 -> the bound is -0.01 / +0.01)
  CEBsr  = volume-weighted price of energy activated in the direction that
           HELPS the system (up when SK<0, down when SK>0)                 [13.3-13.6]
  CEB_PP = CKOEB + COR                                                     [13.2(2)]
  COR    = 0 if LOLP(RO) = 0 else min(GLC - CKOEB, max(CORMin, CORDMax*LOLP))   [12.5]
  CKOEB  = CDO of the band where the BPKD/BO ladder sorted by CDO (G ascending,
           D descending) covers |ZEB_PP| x 4 MW                            [13.2(3)-(5)]

All functions take the period-indexed panel built by `panel.load_public`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WAW = "Europe/Warsaw"
AFRR_START = pd.Timestamp("2025-07-11", tz=WAW).tz_convert("UTC")
TOL = 0.01


def cen_rule(sk, ceb, csdac) -> np.ndarray:
    sk, ceb, csdac = (np.asarray(a, dtype=float) for a in (sk, ceb, csdac))
    lo = np.where(csdac == 0, -0.01, csdac)            # long: min(CEB, CSDAC)
    hi = np.where(csdac == 0, 0.01, csdac)             # short: max(CEB, CSDAC)
    return np.select([sk > 0, sk < 0], [np.minimum(ceb, lo), np.maximum(ceb, hi)], default=ceb)


def lolp(df: pd.DataFrame) -> np.ndarray:
    """Piecewise-linear LOLP curve (b0..b9 [MW], p0..p9) evaluated at RO [12.3(5)]."""
    b = df[[f"b{i}" for i in range(10)]].to_numpy(float)
    p = df[[f"p{i}" for i in range(10)]].to_numpy(float)
    ro = df.ro.to_numpy(float)
    i = np.clip((ro[:, None] >= b).sum(axis=1) - 1, 0, 8)
    r = np.arange(len(df))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (ro - b[r, i]) / (b[r, i + 1] - b[r, i])
    mid = p[r, i] * (1 - w) + p[r, i + 1] * w
    return np.where(ro < b[:, 0], 1.0, np.where(ro >= b[:, 9], 0.0, mid))


def cor_from_lolp(df: pd.DataFrame, ckoeb, cor_min, glc: float = np.inf) -> np.ndarray:
    """COR per WDB 12.5. GLC never binds in 2024-2026, hence +inf."""
    lp = lolp(df)
    cor = np.minimum(glc - np.asarray(ckoeb, float),
                     np.maximum(np.asarray(cor_min, float), df.cordmax.to_numpy(float) * lp))
    return np.round(np.where(lp == 0, 0.0, cor), 2)


def undo_api_replication(lad: pd.DataFrame, thresh: float = 0.3) -> pd.DataFrame:
    """On ~46 days the API serves each period's ladder k-fold. Identical bands
    are otherwise legitimate, so plain dedup is wrong: divide each band's
    multiplicity by the per-period GCD, only where duplication is heavy."""
    cols = ["ts", "cdo", "ofp", "activ_direction"] + (["ofc"] if "ofc" in lad.columns else [])
    cnt = lad.groupby(cols, dropna=False).size().rename("n").reset_index()
    dup = 1 - cnt.groupby("ts").size() / cnt.groupby("ts").n.sum()
    gcd = cnt.groupby("ts").n.agg(lambda s: np.gcd.reduce(s.to_numpy()))
    k = gcd.where(dup > thresh, 1)
    cnt["n"] //= cnt.ts.map(k)
    return cnt.loc[cnt.index.repeat(cnt.n), cols].reset_index(drop=True)


def ckoeb_from_ladder(lad: pd.DataFrame, zeb_pp: pd.Series) -> pd.DataFrame:
    """Cross each period's ladder at |ZEB_PP| x 4 MW (MWh per 15 min -> MW)."""
    lad = undo_api_replication(lad[lad.ts.isin(zeb_pp.index)])
    up = lad.activ_direction.eq("G")
    lad = lad.assign(key=np.where(up, lad.cdo, -lad.cdo)).sort_values(["ts", "key"])
    lad["cum"] = lad.ofp.abs().groupby(lad.ts).cumsum()
    lad["need"] = lad.ts.map(zeb_pp.abs() * 4)
    covered = lad[lad.cum >= lad.need - 1e-6].groupby("ts").head(1)
    last = lad.groupby("ts").tail(1)
    pick = pd.concat([covered, last[~last.ts.isin(covered.ts)]]).set_index("ts")
    return pd.DataFrame({"ckoeb_hat": pick.cdo, "ladder_dir": pick.activ_direction,
                         "ladder_short": ~pick.index.isin(covered.ts)})


def helping(df: pd.DataFrame):
    """(ZEB_PP leg, aFRR volume, aFRR price) in the direction that helps the system."""
    short = (df.sk < 0).to_numpy()
    z = np.where(short, df.zeb_pp.clip(lower=0), df.zeb_pp.clip(upper=0))
    p = np.where(short, df.pzeb_g.fillna(0).clip(lower=0), df.pzeb_d.fillna(0).clip(upper=0))
    c = np.where(short, df.ceb_afrr_g.fillna(0), df.ceb_afrr_d.fillna(0))
    return z, p, c


def ceb_sr_from_parts(df: pd.DataFrame, ceb_pp) -> tuple[np.ndarray, np.ndarray]:
    """CEBsr from 15-min aggregates; NaN in the CUA+COR case (CUA unpublished).
    Exact without an aFRR leg; approximate with one (WDB weights per 4-s cycle)."""
    z, p, c = helping(df)
    den = z + p
    cua = (np.abs(den) < 1e-9) | (df.sk == 0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        est = (z * np.asarray(ceb_pp, float) + p * c) / den
    return np.round(np.where(cua, np.nan, est), 2), cua


def _hit(a, b) -> str:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    if not m.any():
        return "n=     0"
    e = np.abs(a[m] - b[m])
    return (f"n={m.sum():6d}  co do grosza {np.mean(e <= TOL + 1e-9):6.1%}  "
            f"<1 PLN {np.mean(e < 1):6.1%}  MAE {e.mean():8.2f}")


def verify(df: pd.DataFrame, lad: pd.DataFrame | None, cor_min) -> list[str]:
    """Test each level of the chain against what PSE publishes for it."""
    df = df.dropna(subset=["cen", "sk", "csdac", "ceb_pp"])
    out = [f"Okresy: {df.index.min()} .. {df.index.max()}  ({len(df)})", ""]
    lab = df.contracting.map({"short": -1, "long": 1})
    out.append(f"znak SK == etykieta 'contracting' PSE:      {np.mean(np.sign(df.sk) == lab):.2%}")
    out.append("L1 CEN = reguła(SK, CEBśr opubl., CSDAC)  " + _hit(cen_rule(df.sk, df.ceb_sr, df.csdac), df.cen))
    out.append("L3 CEB_PP = CKOEB + COR                   " + _hit(df.ckoeb + df.cor, df.ceb_pp))
    cor_hat = cor_from_lolp(df, df.ckoeb, cor_min_series(df.index, cor_min))
    out.append("L3 COR z LOLP(RO), CORDMax, CORMin         " + _hit(cor_hat, df.cor))
    est, cua = ceb_sr_from_parts(df, df.ceb_pp)
    _, p, _ = helping(df)
    leg = np.abs(p) > 0
    out.append("L2 CEBśr bez członu aFRR                   " + _hit(est[~leg], df.ceb_sr[~leg]))
    out.append("L2 CEBśr z członem aFRR (przybliżenie)     " + _hit(est[leg], df.ceb_sr[leg]))
    out.append(f"   przypadki CUA (niepublikowane): {cua.sum()}")
    if lad is None:
        out.append("L4 CKOEB z drabinki: pominięte (brak drabinki; fetch --ladder YYYY-MM)")
        return out
    ck = ckoeb_from_ladder(lad, df.zeb_pp)
    j = df.join(ck, how="inner")
    if j.empty:
        out.append("L4 CKOEB z drabinki: drabinka nie pokrywa okresów z danymi rozliczeniowymi")
        return out
    out.append(f"L4 CKOEB z drabinki ({j.index.min():%Y-%m-%d}..{j.index.max():%Y-%m-%d}) "
               + _hit(j.ckoeb_hat, j.ckoeb))
    cmin = cor_min_series(j.index, cor_min)
    ceb_pp = j.ckoeb_hat.to_numpy() + cor_from_lolp(j, j.ckoeb_hat, cmin)
    e, _ = ceb_sr_from_parts(j, ceb_pp)
    cen = cen_rule(j.sk, e, j.csdac)
    _, pj, _ = helping(j)
    legj = np.abs(pj) > 0
    out.append("")
    out.append("OD ZERA (drabinka+ZEB_PP+LOLP/RO+aFRR+SK+CSDAC):")
    out.append("   bez członu aFRR                         " + _hit(cen[~legj], j.cen[~legj]))
    out.append("   z członem aFRR                          " + _hit(cen[legj], j.cen[legj]))
    out.append("   wszystkie (bez CUA)                     " + _hit(cen, j.cen))
    return out


def cor_min_series(index: pd.DatetimeIndex, cor_min) -> np.ndarray:
    """cor_min: callable(years) -> values."""
    return np.asarray(cor_min(index.tz_convert(WAW).year), float)
