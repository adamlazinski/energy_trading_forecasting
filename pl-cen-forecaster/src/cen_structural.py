"""
Structural CEN forecaster: forecast the system state SK, push scenarios
through the WDB price chain (F37), read CEN quantiles off the scenarios.

  SK_s      ~ quantile-GBM forecast of SK at H=60, sampled by inverse CDF
  V_s       = helping activation volume = a * (-SK_s) + b   (corr 0.997, F38)
  share_s,
  dprem_s   ~ analog bootstrap from history in the same (sign, hour block,
              |SK| bucket): aFRR share of V, and CEBsr - CEB_PP (the aFRR
              premium, taken as published — it embeds the 4-s weighting
              we cannot replicate)
  Z_s       = (1 - share_s) * V_s                            -> ZEB_PP
  CKOEB_s   = g(Z_s, x_t) + eps_s   g: LightGBM median, monotone in Z;
              eps_s bootstrapped residuals (sign, hour block). Stands in for
              the BPKD/BO ladder, which publishes only D+1.
  CEN_s     = rule(SK_s, CKOEB_s + COR + dprem_s, CSDAC_t)   [WDB 13.1]
  +RC       rolling 14-day conformal shift per quantile, lagged 2 days

Two information sets at the gate t - H:
  public    calendar, CSDAC (D-1), published-history SK/CEN (D+1 15:00 rule)
  intraday  public + SK of the last complete quarters (lag >= 5 qh) and the
            last aFRR price — ASSUMES a near-real-time imbalance estimate
            (ACE / aFRR activation / own data). Not public PSE data today.

    python -m src.cen_structural          # train <= TRAIN_END, test after
"""
from __future__ import annotations

import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cen_reconstruct import COR_MIN, cen_rule, load_components
from .evaluate import pinball
from .features import published_history_features
from .models import QuantileGBM

H = 60
TRAIN_END = pd.Timestamp("2026-08-01", tz="Europe/Warsaw").tz_convert("UTC")
AFRR_START = pd.Timestamp("2025-07-11", tz="Europe/Warsaw").tz_convert("UTC")
SK_Q = tuple(np.round(np.arange(0.025, 1.0, 0.025), 3))      # 39 SK quantiles
OUT_Q = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
N_SCEN = 200
REPORT = pathlib.Path("reports/cen_structural.txt")
PREDS = pathlib.Path("data/proc/cen_structural_preds.parquet")
rng = np.random.default_rng(0)


# --- features ---------------------------------------------------------------
def build_frame() -> pd.DataFrame:
    df = load_components().reset_index()
    loc = df.ts.dt.tz_convert("Europe/Warsaw")
    df["qh"] = loc.dt.hour * 4 + loc.dt.minute // 15
    df["hour"], df["dow"], df["month"] = loc.dt.hour, loc.dt.dayofweek, loc.dt.month
    df["hblock"] = df.hour // 4
    df["is_afrr"] = (df.ts >= AFRR_START).astype(int)
    day = loc.dt.normalize()
    g = df.groupby(day).csdac                                 # whole day D known at D-1
    df["csdac_day_mean"], df["csdac_day_max"] = g.transform("mean"), g.transform("max")
    df["csdac_ramp"] = df.csdac.shift(-4) - df.csdac.shift(4)
    df["y"] = df.cen
    for col, pre in [("sk", "skh"), ("y", "cenh")]:
        df = published_history_features(df, col, pre, H)
    for k in (5, 6, 8, 12):                                  # lag 5 = last quarter complete at t-60
        df[f"rt_sk_l{k}"] = df.sk.shift(k)
    df["rt_sk_mean5_8"] = df.sk.shift(5).rolling(4).mean()
    df["rt_afrr_px_l5"] = np.where(df.sk.shift(5) < 0, df.ceb_afrr_g.shift(5), df.ceb_afrr_d.shift(5))
    return df.set_index("ts")


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    base = ["qh", "hour", "dow", "month", "is_afrr", "csdac", "csdac_day_mean",
            "csdac_day_max", "csdac_ramp"]
    public = base + [c for c in df.columns if c.startswith(("skh_", "cenh_"))]
    return {"public": public, "intraday": public + [c for c in df.columns if c.startswith("rt_")]}


# --- structural pieces --------------------------------------------------------
def helping_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Volume and aFRR share in the direction that helps the system."""
    short = df.sk < 0
    z = np.where(short, df.zeb_pp.clip(lower=0), df.zeb_pp.clip(upper=0))
    p = np.where(short, df.pzeb_g.fillna(0).clip(lower=0), df.pzeb_d.fillna(0).clip(upper=0))
    v = z + p
    return pd.DataFrame({"v": v, "share": np.where(v != 0, p / np.where(v == 0, 1, v), 0.0),
                         "dprem": df.ceb_sr - df.ceb_pp}, index=df.index)


def sk_bucket(sk: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(sk), [50, 100, 200, 400])        # 5 |SK| buckets [MWh]


class Analogs:
    """Joint bootstrap pools keyed by (sign, hour block, |SK| bucket)."""

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        self.values = values
        order = np.argsort(keys, kind="stable")
        self.sorted_idx = order
        uk, start, cnt = np.unique(keys[order], return_index=True, return_counts=True)
        self.lookup = {k: (s, c) for k, s, c in zip(uk, start, cnt)}

    def sample(self, keys: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        out = np.empty((len(keys), self.values.shape[1]))
        for k in np.unique(keys):
            m = keys == k
            s, c = self.lookup.get(k, (None, 0))
            if c < 30:                                      # thin pool -> coarser key
                s, c = self.lookup.get(fallback[m][0], (0, len(self.sorted_idx)))
            pick = self.sorted_idx[s + rng.integers(0, c, m.sum())]
            out[m] = self.values[pick]
        return out


def key_of(sign, hblock, bucket):
    return (np.asarray(sign) > 0).astype(int) * 100 + np.asarray(hblock) * 10 + np.asarray(bucket)


def sample_from_quantiles(qpred: np.ndarray, n: int) -> np.ndarray:
    """Inverse-CDF draws from a row of quantiles; linear tails past the ends."""
    qs = np.array(SK_Q)
    u = rng.uniform(0.005, 0.995, size=(len(qpred), n))
    lo_slope = (qpred[:, 1] - qpred[:, 0]) / (qs[1] - qs[0])
    hi_slope = (qpred[:, -1] - qpred[:, -2]) / (qs[-1] - qs[-2])
    out = np.empty_like(u)
    for i in range(len(qpred)):
        out[i] = np.interp(u[i], qs, qpred[i])
    lo, hi = u < qs[0], u > qs[-1]
    out[lo] = (qpred[:, [0]] - lo_slope[:, None] * (qs[0] - u))[lo]
    out[hi] = (qpred[:, [-1]] + hi_slope[:, None] * (u - qs[-1]))[hi]
    return out


def fit_curve(tr: pd.DataFrame, cols: list[str]) -> tuple[lgb.LGBMRegressor, list[str]]:
    """CKOEB = g(ZEB_PP, x): the dispatcher's supply curve, monotone in ZEB."""
    feats = ["zeb_pp"] + cols
    m = lgb.LGBMRegressor(objective="huber", alpha=100.0, n_estimators=600, learning_rate=0.03,
                          num_leaves=63, min_child_samples=40, verbose=-1,
                          monotone_constraints=[1] + [0] * len(cols))
    m.fit(tr[feats], tr.ckoeb)
    return m, feats


def simulate(te: pd.DataFrame, sk_q: np.ndarray, tr: pd.DataFrame, curve, curve_feats,
             vol_fit, pools) -> pd.DataFrame:
    n = len(te)
    sk = sample_from_quantiles(sk_q, N_SCEN)                          # (n, S)
    hb = np.repeat(te.hblock.to_numpy()[:, None], N_SCEN, 1)
    key = key_of(np.sign(sk), hb, sk_bucket(sk)).ravel()
    fb = key_of(np.sign(sk), hb, np.zeros_like(hb)).ravel() // 10 * 10  # drop |SK| bucket
    share, dprem, eps = pools.sample(key, fb).T
    v = vol_fit[0] * (-sk.ravel()) + vol_fit[1]
    v = np.where(np.sign(v) == -np.sign(sk.ravel()), v, 0.0)          # keep helping direction
    z = (1 - share) * v
    X = pd.DataFrame(np.repeat(te[curve_feats[1:]].to_numpy(), N_SCEN, 0), columns=curve_feats[1:])
    X.insert(0, "zeb_pp", z)
    ckoeb = curve.predict(X) + eps
    cor = te.index.tz_convert("Europe/Warsaw").year.map(COR_MIN).to_numpy()
    ceb = ckoeb + np.repeat(cor, N_SCEN) + dprem
    cen = cen_rule(pd.Series(sk.ravel()), pd.Series(ceb),
                   pd.Series(np.repeat(te.csdac.to_numpy(), N_SCEN))).to_numpy().reshape(n, N_SCEN)
    return pd.DataFrame(np.quantile(cen, OUT_Q, axis=1).T, index=te.index, columns=list(OUT_Q))


def rolling_conformal(y: pd.Series, pred: pd.DataFrame, days: int = 14,
                      lag_days: int = 2) -> pd.DataFrame:
    """Shift each quantile by the trailing empirical q-quantile of its own
    residual. lag_days respects CEN availability (settles D+1 ~14:00)."""
    adj = pred.copy()
    for q in pred.columns:
        r = (y - pred[q]).shift(96 * lag_days)
        adj[q] = pred[q] + r.rolling(96 * days, min_periods=96 * days // 2).quantile(q)
    return pd.DataFrame(np.sort(adj.to_numpy(), axis=1), index=pred.index, columns=pred.columns)


def coverage(y, pred, lo, hi) -> float:
    return float(np.mean((y >= pred[lo]) & (y <= pred[hi])))


def run() -> None:
    df = build_frame().dropna(subset=["sk", "cen", "csdac", "zeb_pp", "ckoeb"])
    fs = feature_sets(df)
    tr, te = df[df.index < TRAIN_END], df[df.index >= TRAIN_END]
    lines = [f"train {tr.index.min()} .. {tr.index.max()} ({len(tr)})",
             f"test  {te.index.min()} .. {te.index.max()} ({len(te)})", ""]
    say = lines.append

    hp = helping_parts(tr)
    vol_fit = np.polyfit(-tr.sk, hp.v, 1)
    say(f"helping volume V = {vol_fit[0]:.3f} * (-SK) + {vol_fit[1]:.1f}   "
        f"corr {np.corrcoef(-tr.sk, hp.v)[0, 1]:.3f}")

    # analog pools: aFRR era only (share/premium do not exist before)
    trA = tr[tr.index >= AFRR_START]
    hpA = hp.loc[trA.index]
    keysA = key_of(np.sign(trA.sk), trA.hblock, sk_bucket(trA.sk.to_numpy()))

    res = {}
    for name, cols in fs.items():
        curve_cols = [c for c in cols if not c.startswith("rt_")]
        curve, cfeats = fit_curve(tr, curve_cols)
        eps = (tr.ckoeb - curve.predict(tr[cfeats])).loc[trA.index]
        pools = Analogs(keysA, np.column_stack([hpA.share, hpA.dprem.fillna(0), eps]))

        skm = QuantileGBM(quantiles=SK_Q, n_estimators=400).fit(tr[cols], tr.sk)
        sk_q = skm.predict(te[cols]).to_numpy()
        sk_med = sk_q[:, SK_Q.index(0.5)]
        p_short = (sk_q < 0).mean(axis=1)
        brier = np.mean((p_short - (te.sk < 0)) ** 2)
        clim = np.mean(((tr.sk < 0).mean() - (te.sk < 0)) ** 2)
        say(f"[{name}] SK model: MAE(median) {np.abs(sk_med - te.sk).mean():.1f} MWh "
            f"(sd SK {te.sk.std():.1f}); P(short) Brier {brier:.3f} vs climatology {clim:.3f}")

        struct = simulate(te, sk_q, tr, curve, cfeats, vol_fit, pools)
        direct = QuantileGBM(quantiles=OUT_Q).fit(tr[cols], tr.cen).predict(te[cols])
        for model, pred in [("structural", struct), ("direct GBM", direct)]:
            pb = pinball(te.cen, pred, OUT_Q)
            res[(name, model)] = pred
            say(f"    {model:11s} pinball {pb.mean():7.2f}  "
                f"cov P10-90 {coverage(te.cen, pred, 0.10, 0.90):.3f}  "
                f"P5-95 {coverage(te.cen, pred, 0.05, 0.95):.3f}  "
                f"width P10-90 {(pred[0.90] - pred[0.10]).median():.0f}")
        for model in ("structural", "direct GBM"):
            pred = res[(name, model)]
            by_m = {str(m): f"{coverage(te.cen[g], pred.loc[g], 0.10, 0.90):.2f}/{pinball(te.cen[g], pred.loc[g], OUT_Q).mean():.0f}"
                    for m, g in te.groupby(te.index.tz_convert("Europe/Warsaw").to_period("M")).groups.items()}
            say(f"      {model:11s} by month cov/pinball: {by_m}")
        for model in ("structural", "direct GBM"):
            adj = rolling_conformal(te.cen, res[(name, model)])
            ok = adj[0.10].notna()
            say(f"    {model + ' +RC':15s} pinball {pinball(te.cen[ok], adj[ok], OUT_Q).mean():7.2f}  "
                f"cov P10-90 {coverage(te.cen[ok], adj[ok], 0.10, 0.90):.3f}  "
                f"P5-95 {coverage(te.cen[ok], adj[ok], 0.05, 0.95):.3f}  "
                f"(raw on same rows {pinball(te.cen[ok], res[(name, model)][ok], OUT_Q).mean():.2f})")
        avg = (res[(name, "structural")] + res[(name, "direct GBM")]) / 2
        say(f"    {'average':11s} pinball {pinball(te.cen, avg, OUT_Q).mean():7.2f}  "
            f"cov P10-90 {coverage(te.cen, avg, 0.10, 0.90):.3f}")

    # oracle: perfect SK knowledge -> how much is left (aFRR + curve error)
    sk_true = np.repeat(te.sk.to_numpy()[:, None], len(SK_Q), 1)
    orc = simulate(te, sk_true, tr, curve, cfeats, vol_fit, pools)
    say(f"[oracle SK] structural pinball {pinball(te.cen, orc, OUT_Q).mean():7.2f}  "
        f"cov P10-90 {coverage(te.cen, orc, 0.10, 0.90):.3f}   (upper bound of SK effort)")

    pd.concat({f"{k[0]}|{k[1]}": v for k, v in res.items()} | {"oracle|structural": orc},
              axis=1).assign(y=te.cen, sk=te.sk).to_parquet(PREDS)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
