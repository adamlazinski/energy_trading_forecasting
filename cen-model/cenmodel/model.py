"""Structural CEN forecaster (see README, F38 in RESEARCH_LOG.md).

  SK_s     ~ quantile-GBM forecast of SK, sampled by inverse CDF
  V_s      = a * (-SK_s) + b           helping activation volume
  share_s,
  dprem_s,
  eps_s    ~ joint bootstrap from history in the same (sign, hour block, |SK| bucket):
             aFRR share of V, aFRR premium CEBsr - CEB_PP, supply-curve residual
  CKOEB_s  = g(ZEB_PP_s = (1-share_s) V_s, x) + eps_s     g monotone in ZEB_PP
  CEN_s    = rule(SK_s, CKOEB_s + CORMin + dprem_s, CSDAC)            [WDB 13.1]
  then a rolling conformal shift per quantile (out-of-sample residuals only).

A direct quantile GBM on CEN is trained alongside as a cross-check / blend.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import UserError, __version__
from .config import cor_min_for
from .panel import public_features, simulated_rt_sk
from .wdb import AFRR_START, WAW, cen_rule, helping

SK_Q = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)
OUT_Q = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QCOLS = [f"q{int(round(q * 100)):02d}" for q in OUT_Q]
LGB = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
TARGETS = ["sk", "cen", "csdac", "zeb_pp", "ckoeb"]
MIN_TRAIN_DAYS = 60


class QGBM:
    def __init__(self, quantiles, seed=0):
        self.q, self.seed, self.models = quantiles, seed, []

    def fit(self, X, y):
        self.models = [lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed,
                                         **LGB).fit(X, y) for q in self.q]
        return self

    def predict(self, X) -> np.ndarray:
        return np.sort(np.column_stack([m.predict(X) for m in self.models]), axis=1)


class Analogs:
    """Joint bootstrap pools keyed by (sign, hour block, |SK| bucket)."""

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        self.values = values
        self.order = np.argsort(keys, kind="stable")
        uk, start, cnt = np.unique(keys[self.order], return_index=True, return_counts=True)
        self.lookup = {int(k): (int(s), int(c)) for k, s, c in zip(uk, start, cnt)}

    def sample(self, keys, fallback, rng) -> np.ndarray:
        out = np.empty((len(keys), self.values.shape[1]))
        for k in np.unique(keys):
            m = keys == k
            s, c = self.lookup.get(int(k), (0, 0))
            if c < 30:                                    # thin pool -> drop |SK| bucket
                s, c = self.lookup.get(int(fallback[m][0]), (0, 0))
            if c < 30:                                    # still thin -> everything
                s, c = 0, len(self.order)
            out[m] = self.values[self.order[s + rng.integers(0, c, m.sum())]]
        return out


def _bucket(sk):
    return np.digitize(np.abs(sk), [50, 100, 200, 400])


def _key(sign, hblock, bucket):
    return (np.asarray(sign) > 0).astype(int) * 100 + np.asarray(hblock) * 10 + np.asarray(bucket)


def _draw(qpred: np.ndarray, n: int, rng) -> np.ndarray:
    """Inverse-CDF draws from rows of SK_Q quantiles; linear tails past the ends."""
    qs = np.array(SK_Q)
    u = rng.uniform(0.005, 0.995, size=(len(qpred), n))
    out = np.empty_like(u)
    for i in range(len(qpred)):
        out[i] = np.interp(u[i], qs, qpred[i])
    lo_s = (qpred[:, 1] - qpred[:, 0]) / (qs[1] - qs[0])
    hi_s = (qpred[:, -1] - qpred[:, -2]) / (qs[-1] - qs[-2])
    lo, hi = u < qs[0], u > qs[-1]
    out[lo] = (qpred[:, [0]] - lo_s[:, None] * (qs[0] - u))[lo]
    out[hi] = (qpred[:, [-1]] + hi_s[:, None] * (u - qs[-1]))[hi]
    return out


# --- features ---------------------------------------------------------------
def make_features(panel: pd.DataFrame, cfg: dict, own: pd.DataFrame | None = None,
                  sim_rt: bool = False) -> pd.DataFrame:
    X = public_features(panel, cfg["horizon_min"])
    if own is not None and len(own.columns):
        X = X.join(own)
    if sim_rt:
        X = X.join(simulated_rt_sk(panel, cfg["horizon_min"]))
    return X


def trainable(panel: pd.DataFrame) -> pd.Series:
    return panel[TARGETS].notna().all(axis=1)


# --- fit / predict ------------------------------------------------------------
def fit(panel: pd.DataFrame, X: pd.DataFrame, rows: pd.Index, cfg: dict, log=print) -> dict:
    tr = panel.loc[rows]
    days = len(tr) / 96
    if days < MIN_TRAIN_DAYS:
        raise UserError(f"Za mało danych do treningu: {days:.0f} dni (minimum {MIN_TRAIN_DAYS}). "
                        "Pobierz dłuższą historię (fetch) albo sprawdź zakres Twoich danych.")
    cols = list(X.columns)
    Xt = X.loc[rows, cols]
    seed = cfg["model"]["seed"]
    z, p, _ = helping(tr)
    v = z + p
    vol = np.polyfit(-tr.sk.to_numpy(float), v, 1)
    log(f"  wolumen pomagający V = {vol[0]:.3f}*(-SK) + {vol[1]:.1f}  "
        f"(korelacja {np.corrcoef(-tr.sk, v)[0, 1]:.3f})")

    curve_cols = [c for c in cols if not c.startswith("simsk_")]
    cf = ["zeb_pp"] + curve_cols
    curve = lgb.LGBMRegressor(objective="huber", alpha=100.0, n_estimators=600, learning_rate=0.03,
                              num_leaves=63, min_child_samples=40, random_state=seed, verbose=-1,
                              monotone_constraints=[1] + [0] * len(curve_cols))
    Xc = pd.concat([tr[["zeb_pp"]], Xt[curve_cols]], axis=1)
    curve.fit(Xc, tr.ckoeb)
    log("  krzywa podaży CKOEB = g(ZEB_PP, x): gotowa")

    afrr = tr.index >= AFRR_START
    if afrr.sum() < 96 * 30:                           # pre-aFRR training: use everything
        afrr = np.ones(len(tr), bool)
    share = np.where(v != 0, p / np.where(v == 0, 1, v), 0.0)
    dprem = (tr.ceb_sr - tr.ceb_pp).fillna(0).to_numpy()
    eps = (tr.ckoeb - curve.predict(Xc)).to_numpy()
    hb = (Xt.qh.to_numpy() // 16)
    keys = _key(np.sign(tr.sk), hb, _bucket(tr.sk.to_numpy()))
    pools = Analogs(keys[afrr], np.column_stack([share, dprem, eps])[afrr])

    log(f"  model SK: {len(SK_Q)} kwantyli ...")
    sk_m = QGBM(SK_Q, seed).fit(Xt, tr.sk)
    log(f"  model bezpośredni CEN: {len(OUT_Q)} kwantyli ...")
    direct = QGBM(OUT_Q, seed).fit(Xt, tr.cen)
    return {"version": __version__, "horizon_min": cfg["horizon_min"], "cols": cols,
            "curve_cols": cf, "vol": vol, "curve": curve, "pools": pools, "sk": sk_m,
            "direct": direct, "train_start": tr.index.min(), "train_end": tr.index.max(),
            "n_train": len(tr), "cor_min": dict(cfg["cor_min"]),
            "own": [(s["name"], s["kind"]) for s in cfg["own_data"]]}


def predict_raw(b: dict, panel: pd.DataFrame, X: pd.DataFrame, idx: pd.Index, cfg: dict,
                rng, keep_scen: bool = False) -> dict:
    miss = [c for c in b["cols"] if c not in X.columns]
    if miss:
        raise UserError(f"Model był trenowany z innymi cechami (brak: {miss[:5]}...). "
                        "Zmieniłeś own_data? Uruchom ponownie: python -m cenmodel train")
    Xp = X.loc[idx, b["cols"]]
    n, S = len(idx), cfg["model"]["n_scenarios"]
    sk_q = b["sk"].predict(Xp)
    sk = _draw(sk_q, S, rng)
    hb = np.repeat((Xp.qh.to_numpy() // 16)[:, None], S, 1)
    key = _key(np.sign(sk), hb, _bucket(sk)).ravel()
    fb = _key(np.sign(sk), hb, np.zeros_like(hb)).ravel()
    share, dprem, eps = b["pools"].sample(key, fb, rng).T
    v = b["vol"][0] * (-sk.ravel()) + b["vol"][1]
    v = np.where(np.sign(v) == -np.sign(sk.ravel()), v, 0.0)      # helping direction only
    Xc = pd.DataFrame(np.repeat(Xp[b["curve_cols"][1:]].to_numpy(), S, 0), columns=b["curve_cols"][1:])
    Xc.insert(0, "zeb_pp", (1 - share) * v)
    cmin = np.asarray(cor_min_for(cfg, idx.tz_convert(WAW).year), float)
    ceb = b["curve"].predict(Xc) + eps + np.repeat(cmin, S) + dprem
    cen = cen_rule(sk.ravel(), ceb, np.repeat(panel.loc[idx, "csdac"].to_numpy(), S)).reshape(n, S)
    struct = pd.DataFrame(np.quantile(cen, OUT_Q, axis=1).T, index=idx, columns=QCOLS)
    direct = pd.DataFrame(b["direct"].predict(Xp), index=idx, columns=QCOLS)
    blend = pd.DataFrame(np.sort((struct.to_numpy() + direct.to_numpy()) / 2, axis=1),
                         index=idx, columns=QCOLS)
    out = {"structural": struct, "direct": direct, "blend": blend,
           "p_short": pd.Series((sk < 0).mean(axis=1), index=idx),
           "sk_q50": pd.Series(sk_q[:, SK_Q.index(0.50)], index=idx)}
    if keep_scen:
        out["scenarios"] = pd.DataFrame(cen, index=idx, columns=[f"s{i}" for i in range(S)])
    return out


# --- calibration ---------------------------------------------------------------
def rolling_conformal(y: pd.Series, pred: pd.DataFrame, days: int, lag_days: int) -> pd.DataFrame:
    """Backtest: shift each quantile by the trailing empirical quantile of its
    own residual, using only residuals published lag_days earlier."""
    grid = pd.date_range(pred.index.min(), pred.index.max(), freq="15min")
    adj = pred.copy()
    for c, q in zip(QCOLS, OUT_Q):
        r = (y - pred[c]).reindex(grid).shift(96 * lag_days)
        off = r.rolling(96 * days, min_periods=96 * days // 2).quantile(q).reindex(pred.index)
        adj[c] = pred[c] + off
    return pd.DataFrame(np.sort(adj.to_numpy(), axis=1), index=pred.index, columns=QCOLS)


def live_offsets(y: pd.Series, pred: pd.DataFrame) -> pd.Series:
    """Live: offsets from the out-of-sample residuals of the latest published days."""
    return pd.Series({c: float(np.nanquantile(y - pred[c], q)) for c, q in zip(QCOLS, OUT_Q)})


def apply_offsets(pred: pd.DataFrame, off: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(np.sort((pred + off).to_numpy(), axis=1), index=pred.index, columns=QCOLS)


# --- scoring ---------------------------------------------------------------------
def pinball(y: pd.Series, pred: pd.DataFrame) -> float:
    d = y.to_numpy()[:, None] - pred[QCOLS].to_numpy()
    q = np.array(OUT_Q)
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, pred, lo="q10", hi="q90") -> float:
    return float(np.mean((y >= pred[lo]) & (y <= pred[hi])))
