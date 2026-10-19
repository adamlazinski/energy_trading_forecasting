"""
Rebuild the Polish imbalance price CEN bottom-up from public PSE data,
following the settlement rules in WDB pkt 13.2-13.3 (tekst jednolity
incl. Zmiany 3/2025, aFRR platform from 2025-07-11):

  L1  CEN_t   = min(CEBsr, CSDAC)            SK > 0 (long)    [13.1]
              = max(CEBsr, CSDAC)            SK < 0 (short)
              = CEBsr                        SK = 0
              (CSDAC = 0  ->  bound becomes -0.01 / +0.01)
  L2  CEBsr   = volume-weighted price of balancing energy activated in the
                direction that HELPS the system: down (ZEB_PP<0, aFRR D)
                when SK>0, up (ZEB_PP>0, aFRR G) when SK<0      [13.3-13.6]
                ... and CUA + COR if nothing was activated in that
                direction, or SK = 0                             [13.4/6/7]
  L3  CEB_PP  = CKOEB + COR                                      [13.2(2)]
      COR     = 0 if LOLP(RO)=0, else
                min(GLC - CKOEB, max(CORMin, CORDMax_d * LOLP(RO)))   [12.5]
                LOLP: piecewise-linear curve (b0..b9, p0..p9) at RO  [12.3(5)]
  L4  CKOEB   = CDO of the band at which the BPKD/BO ladder, sorted by CDO
                (ascending for G, descending for D), covers |ZEB_PP| x 4 MW
                                                                 [13.2(3)-(5)]

Each level is tested against what PSE publishes for it, then the chain is
run end-to-end from primitives (ladder, ZEB_PP, COR, aFRR aggregates, SK,
CSDAC). Sources:
  crb-rozl   cen, ckoeb, ceb_pp, ceb_sr (=CEBsr), ceb_sr_afrr{g,d}
  price-cost cor_pp_cost (=COR), contracting (short/long)
  zeb-rozl   zebpp (=ZEB_PP, MWh), pzeb_afrr{g,d}_cost (=PZEB_aFRR, MWh!)
  sk         sk_cost (=SK, MWh)
  csdac-pln  csdac_pln
  lolp       b0..b9 [MW], p0..p9 per period (b0 = ROMin already included)
  ro-rozl    ro_cost (=RO, MW — despite the name)
  cordmax    cordmax_cost (=CORDMax, daily, keyed by business_date)
  oeb-bpkdbo per-band BPKD/BO plan ladder (data/raw/pse_offer_ladder.parquet)

    python -m src.cen_reconstruct fetch [start] [end]   # cache components
    python -m src.cen_reconstruct                       # run all levels
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

from .pse_client import PSEClient

RAW = pathlib.Path("data/raw/cenrec")
LADDER = pathlib.Path("data/raw/pse_offer_ladder.parquet")
OUT = pathlib.Path("data/proc/cen_reconstruction.parquet")
REPORT = pathlib.Path("reports/cen_reconstruction.txt")
START = "2024-06-14"
AFRR_START = pd.Timestamp("2025-07-11", tz="Europe/Warsaw").tz_convert("UTC")
TOL = 0.01                                  # PSE publishes to the grosz

SOURCES = {                                 # slug -> columns kept
    "crb-rozl": ["cen_cost", "ckoeb_cost", "ceb_pp_cost", "ceb_sr_cost",
                 "ceb_sr_afrrg_cost", "ceb_sr_afrrd_cost"],
    "price-cost": ["cor_pp_cost", "contracting"],
    "zeb-rozl": ["zebpp", "pzeb_afrrg_cost", "pzeb_afrrd_cost"],
    "sk": ["sk_cost"],
    "csdac-pln": ["csdac_pln"],
    "lolp": [f"b{i}" for i in range(10)] + [f"p{i}" for i in range(10)],
    "ro-rozl": ["ro_cost"],
}
# CORMin = 10 zł/MW-h CPI-indexed from 2025 (WDB 12.4(3)); published on
# pse.pl, not in the API — values read off the data (min COR with LOLP > 0)
COR_MIN = {2024: 10.00, 2025: 10.50, 2026: 10.82}


def fetch(start: str = START, end: str | None = None) -> None:
    end = end or (pd.Timestamp.now("Europe/Warsaw") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    RAW.mkdir(parents=True, exist_ok=True)
    cli = PSEClient()
    for slug, cols in SOURCES.items():
        df = cli.fetch(slug, start, end)
        df = df[["ts"] + [c for c in cols if c in df.columns]].drop_duplicates("ts", keep="last")
        df.to_parquet(RAW / f"{slug}.parquet", index=False)
        print(f"{slug:11s} {len(df):6d} rows  {df.ts.min()} .. {df.ts.max()}")
    cm = cli.fetch("cordmax", start, end)[["business_date", "cordmax_cost"]]
    cm.to_parquet(RAW / "cordmax.parquet", index=False)


def load_components() -> pd.DataFrame:
    frames = [pd.read_parquet(RAW / f"{s}.parquet", columns=["ts"] + c).set_index("ts")
              for s, c in SOURCES.items()]
    df = frames[0].join(frames[1:], how="inner")
    num = [c for c in df.columns if c != "contracting"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    cm = pd.read_parquet(RAW / "cordmax.parquet").set_index("business_date").cordmax_cost
    df["cordmax"] = pd.to_numeric(df.index.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d").map(cm))
    return df.rename(columns={
        "cen_cost": "cen", "ckoeb_cost": "ckoeb", "ceb_pp_cost": "ceb_pp",
        "ceb_sr_cost": "ceb_sr", "ceb_sr_afrrg_cost": "ceb_afrr_g",
        "ceb_sr_afrrd_cost": "ceb_afrr_d", "cor_pp_cost": "cor",
        "zebpp": "zeb_pp", "pzeb_afrrg_cost": "pzeb_g",
        "pzeb_afrrd_cost": "pzeb_d", "sk_cost": "sk", "csdac_pln": "csdac",
        "ro_cost": "ro"})


# --- L3b: COR from LOLP, RO and CORDMax --------------------------------------
def cor_from_lolp(df: pd.DataFrame, ckoeb: pd.Series, glc: float = np.inf) -> pd.Series:
    """WDB 12.5 with the 12.3(5) LOLP curve. GLC (harmonised balancing price
    cap, EUR->PLN) never binds in the sample, so it defaults to +inf."""
    b = df[[f"b{i}" for i in range(10)]].to_numpy()
    p = df[[f"p{i}" for i in range(10)]].to_numpy()
    ro = df.ro.to_numpy()[:, None]
    i = np.clip((ro >= b).sum(axis=1) - 1, 0, 8)            # segment index
    r = np.arange(len(df))
    w = (ro[:, 0] - b[r, i]) / (b[r, i + 1] - b[r, i])
    lolp = np.where(ro[:, 0] < b[:, 0], 1.0,
                    np.where(ro[:, 0] >= b[:, 9], 0.0, p[r, i] * (1 - w) + p[r, i + 1] * w))
    cor_min = df.index.tz_convert("Europe/Warsaw").year.map(COR_MIN).to_numpy()
    cor = np.minimum(glc - ckoeb.to_numpy(), np.maximum(cor_min, df.cordmax.to_numpy() * lolp))
    return pd.Series(np.where(lolp == 0, 0.0, cor), index=df.index).round(2)


# --- L4: CKOEB from the BPKD/BO ladder ---------------------------------------
def _undo_api_replication(lad: pd.DataFrame, thresh: float = 0.3) -> pd.DataFrame:
    """On ~46 days the API serves each period's ladder k-fold (k ~ 9-14; a
    fresh pull reproduces it). Identical bands are otherwise legitimate
    (different units, same offer), so plain dedup is wrong: divide each
    band's multiplicity by the per-period GCD, only where duplication is heavy.
    """
    cols = ["ts", "cdo", "ofp", "activ_direction"]
    cnt = lad.groupby(cols).size().rename("n").reset_index()
    dup = 1 - cnt.groupby("ts").size() / cnt.groupby("ts").n.sum()
    gcd = cnt.groupby("ts").n.agg(lambda s: np.gcd.reduce(s.to_numpy()))
    k = gcd.where(dup > thresh, 1)
    cnt["n"] //= cnt.ts.map(k)
    return cnt.loc[cnt.index.repeat(cnt.n), cols]


def ckoeb_from_ladder(zeb_pp: pd.Series) -> pd.DataFrame:
    """Cross each period's plan ladder at |ZEB_PP| x 4 MW (MWh/15min -> MW)."""
    lad = pd.read_parquet(LADDER, columns=["ts", "cdo", "ofp", "activ_direction"])
    lad = _undo_api_replication(lad[lad.ts.isin(zeb_pp.index)])
    up = lad.activ_direction.eq("G")
    # one global sort: by ts, then CDO in merit order (G ascending, D descending)
    lad = lad.assign(key=np.where(up, lad.cdo, -lad.cdo)).sort_values(["ts", "key"])
    lad["cum"] = lad.ofp.abs().groupby(lad.ts).cumsum()
    lad["need"] = lad.ts.map(zeb_pp.abs() * 4)
    covered = lad[lad.cum >= lad.need - 1e-6].groupby("ts").head(1)
    last = lad.groupby("ts").tail(1)                    # ladder exhausted
    pick = pd.concat([covered, last[~last.ts.isin(covered.ts)]]).set_index("ts")
    return pd.DataFrame({"ckoeb_hat": pick.cdo,
                         "ladder_dir": pick.activ_direction,
                         "ladder_short": ~pick.index.isin(covered.ts)})


# --- L2: weighted balancing price -------------------------------------------
def ceb_sr_from_parts(df: pd.DataFrame, ceb_pp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """CEBsr per 13.3-13.6 from 15-min aggregates; NaN where it is CUA+COR.

    The rule weights per-4s aFRR cycle (OAaFRR). PSE only publishes 15-min
    aggregates, so the aFRR term uses pzeb_{g,d} x ceb_afrr_{g,d}; exact
    whenever the aFRR leg is absent, an approximation otherwise.
    """
    pg = df.pzeb_g.fillna(0).clip(lower=0)
    pd_ = df.pzeb_d.fillna(0).clip(upper=0)
    cg = df.ceb_afrr_g.fillna(0)
    cd = df.ceb_afrr_d.fillna(0)
    zu, zd = df.zeb_pp.clip(lower=0), df.zeb_pp.clip(upper=0)
    short = df.sk < 0
    num = np.where(short, zu * ceb_pp + pg * cg, zd * ceb_pp + pd_ * cd)
    den = np.where(short, zu + pg, zd + pd_)
    cua_case = pd.Series((np.abs(den) < 1e-9) | (df.sk == 0), index=df.index)
    est = pd.Series(np.where(cua_case, np.nan, num / np.where(den == 0, 1, den)),
                    index=df.index)
    return est.round(2), cua_case


# --- L1: imbalance price rule ----------------------------------------------
def cen_rule(sk: pd.Series, ceb: pd.Series, csdac: pd.Series) -> pd.Series:
    lo = np.where(csdac == 0, -0.01, csdac)            # long: min(CEB, CSDAC)
    hi = np.where(csdac == 0, 0.01, csdac)             # short: max(CEB, CSDAC)
    return pd.Series(np.select([sk > 0, sk < 0], [np.minimum(ceb, lo), np.maximum(ceb, hi)],
                               default=ceb), index=sk.index)


def _hit(a: pd.Series, b: pd.Series) -> str:
    m = a.notna() & b.notna()
    e = (a[m] - b[m]).abs()
    return (f"n={m.sum():6d}  exact(<=0.01) {np.mean(e <= TOL + 1e-9):6.1%}  "
            f"<1 PLN {np.mean(e < 1):6.1%}  MAE {e.mean():8.2f}  P99 {e.quantile(.99):8.2f}")


def run() -> None:
    df = load_components()
    lines = [f"CEN reconstruction  {df.index.min()} .. {df.index.max()}  ({len(df)} periods)", ""]
    say = lines.append

    # contracting state: SK sign vs PSE's own short/long label
    lab = df.contracting.map({"short": -1, "long": 1})
    say(f"SK sign == PSE 'contracting' label: {np.mean(np.sign(df.sk) == lab):.2%}   "
        f"(SK=0 periods: {(df.sk == 0).sum()})")
    say("")

    say("L1  CEN = rule(SK, CEBsr_published, CSDAC)                     [13.1]")
    l1 = cen_rule(df.sk, df.ceb_sr, df.csdac)
    say("    all                " + _hit(l1, df.cen))
    say("    binding = CSDAC    " + f"{np.mean((df.cen - df.csdac).abs() <= TOL):.1%} of periods")
    say("")

    say("L3  CEB_PP = CKOEB + COR                                        [13.2(2)]")
    say("    all                " + _hit(df.ckoeb + df.cor, df.ceb_pp))
    cor_daily = df.cor.groupby(df.index.tz_convert("Europe/Warsaw").date).nunique()
    say("    COR from LOLP(RO), CORDMax, CORMin [12.5]  " + _hit(cor_from_lolp(df, df.ckoeb), df.cor))
    say(f"    COR constant within the day on {np.mean(cor_daily == 1):.1%} of days; "
        f"median {df.cor.median():.2f}, range {df.cor.min():.2f}..{df.cor.max():.2f}")
    say("")

    say("L2  CEBsr from ZEB_PP, CEB_PP, PZEB_aFRR, CEB_aFRR             [13.3-13.6]")
    est, cua = ceb_sr_from_parts(df, df.ceb_pp)
    pre = df.index < AFRR_START
    has_afrr = np.where(df.sk < 0, df.pzeb_g.fillna(0) > 0, df.pzeb_d.fillna(0) < 0)
    say(f"    CUA+COR case (no helping activation / SK=0): {cua.sum()} periods — "
        "CUA not published, left out")
    say("    pre-aFRR  (<2025-07-11)      " + _hit(est[pre], df.ceb_sr[pre]))
    say("    aFRR era, no aFRR leg        " + _hit(est[~pre & ~has_afrr], df.ceb_sr[~pre & ~has_afrr]))
    say("    aFRR era, with aFRR leg      " + _hit(est[~pre & has_afrr], df.ceb_sr[~pre & has_afrr]))
    say(f"    published CEBsr null before 2025-07-11: {df.ceb_sr[pre].isna().mean():.1%}")
    say("")

    say("L4  CKOEB from BPKD/BO ladder crossed at |ZEB_PP| x 4 MW       [13.2(3)-(5)]")
    lad = ckoeb_from_ladder(df.zeb_pp)
    j = df.join(lad, how="inner")
    say(f"    ladder overlap {j.index.min()} .. {j.index.max()}")
    say(f"    ladder direction == sign(ZEB_PP): "
        f"{np.mean(j.ladder_dir.map({'G': 1, 'D': -1}) == np.sign(j.zeb_pp)):.2%}")
    say("    all                " + _hit(j.ckoeb_hat, j.ckoeb))
    say(f"    ZEB_PP == 0 (CKOEB = CUA, not reconstructable): {(j.zeb_pp == 0).sum()} periods")
    say(f"    ladder too short to cover ZEB_PP: {j.ladder_short.mean():.2%}")
    say("")

    say("END-TO-END  CEN from ladder + ZEB_PP + LOLP/RO/CORDMax + aFRR aggregates + SK + CSDAC")
    ceb_pp_hat = j.ckoeb_hat + cor_from_lolp(j, j.ckoeb_hat)
    est_j, cua_j = ceb_sr_from_parts(j, ceb_pp_hat)
    cen_hat = cen_rule(j.sk, est_j, j.csdac)
    pre_j = j.index < AFRR_START
    hasj = pd.Series(np.where(j.sk < 0, j.pzeb_g.fillna(0) > 0, j.pzeb_d.fillna(0) < 0), index=j.index)
    say("    pre-aFRR                     " + _hit(cen_hat[pre_j], j.cen[pre_j]))
    say("    aFRR era, no aFRR leg        " + _hit(cen_hat[~pre_j & ~hasj], j.cen[~pre_j & ~hasj]))
    say("    aFRR era, with aFRR leg      " + _hit(cen_hat[~pre_j & hasj], j.cen[~pre_j & hasj]))
    say(f"    CUA periods: {cua_j.sum()} (counted as misses below)")
    say("    all non-CUA                  " + _hit(cen_hat, j.cen))

    df = df.join(lad).assign(cen_rule_pub=l1, ceb_sr_hat=est, cua_case=cua)
    df["cen_hat"] = cen_hat.reindex(df.index)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch(*sys.argv[2:4])
    else:
        run()
