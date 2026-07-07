"""
Project B, Layer 2a: conditional model of the balancing settlement price.

    python -m src.bess_cond_model

Rules established from PSE's WDB training deck (see RESEARCH_LOG F7):
- Activated balancing energy settles UNIFORM-PRICE at CEB_PP per 15-min
  period (crb-rozl ceb_pp_cost, 100% published), not pay-as-bid.
- Initial offers bind at the D-1 RBN gate (10:00-14:30); intraday updates
  may only become LESS aggressive. So the conditional model uses D-1-known
  features: fx_ anchors (csdac publishes 13:50, 40 min before gate close)
  and published-history features (1-2 day stale, same as the CEN model).
- Approximation: an up-offer at price p is activated iff the period's net
  direction is G and ceb_pp >= p (uniform-price in-merit condition);
  symmetrically for down. Counter-direction activations ("wymuszone",
  settled min/max(CEBPP, CSDAC)) are ignored - conservative.

Targets on the same leakage-honest feature panel as the CEN model:
  1. P(dir = G)            - LGBMClassifier
  2. ceb_pp | dir=G        - QuantileGBM (P10..P90)
  3. ceb_pp | dir=D        - QuantileGBM

Outputs:
  data/proc/bess_cond_holdout.parquet  (per-period conditional predictions)
  reports/bess_cond_eval.txt
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .build_dataset import load_cfg
from .features import build as build_features, feature_cols
from .models import QUANTILES, QuantileGBM

HOLDOUT_WEEKS = 8
EMBARGO_DAYS = 7

OUT_PRED = pathlib.Path("data/proc/bess_cond_holdout.parquet")
OUT_TXT = pathlib.Path("reports/bess_cond_eval.txt")


def load() -> pd.DataFrame:
    panel = pd.read_parquet("data/proc/panel_15min.parquet")
    df = build_features(panel, load_cfg())
    m = pd.read_parquet("data/raw/pse_bpkdbo_marginals.parquet")[["ts", "dir"]]
    crb = pd.read_parquet("data/raw/pse_imbalance.parquet")[["ts", "ceb_pp_cost"]]
    crb["ceb_pp_cost"] = pd.to_numeric(crb["ceb_pp_cost"], errors="coerce")
    df = df.merge(m, on="ts", how="inner").merge(crb, on="ts", how="inner")
    return df.dropna(subset=["dir", "ceb_pp_cost"]).sort_values("ts")


def pinball(y, pred, quantiles=QUANTILES) -> float:
    tot = 0.0
    for q in quantiles:
        d = y - pred[q].to_numpy()
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(quantiles)


def climatology(train: pd.DataFrame, test: pd.DataFrame, col: str) -> pd.DataFrame:
    clim = train.groupby("qh_of_day")[col].quantile(list(QUANTILES)).unstack()
    rows = test["qh_of_day"].map(clim.to_dict("index"))
    return pd.DataFrame(
        [v if isinstance(v, dict) else {q: np.nan for q in QUANTILES} for v in rows],
        index=test.index)[list(QUANTILES)]


def main():
    df = load()
    feats = feature_cols(df, extended=False)
    cut_hold = df["ts"].max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    cut_train = cut_hold - pd.Timedelta(days=EMBARGO_DAYS)
    train = df[df["ts"] < cut_train]
    hold = df[df["ts"] >= cut_hold]
    lines = [f"train n={len(train)} to {cut_train}; holdout n={len(hold)} "
             f"from {cut_hold}; {len(feats)} features"]

    # 1) direction
    y_dir = (train["dir"] == "G").astype(int)
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                         min_child_samples=100, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, verbose=-1)
    clf.fit(train[feats], y_dir)
    p_up = clf.predict_proba(hold[feats])[:, 1]
    yh = (hold["dir"] == "G").astype(int).to_numpy()
    brier = float(np.mean((p_up - yh) ** 2))
    base = train.groupby("qh_of_day")["dir"].apply(lambda s: (s == "G").mean())
    p_base = hold["qh_of_day"].map(base).to_numpy()
    brier_base = float(np.mean((p_base - yh) ** 2))
    lines.append(f"direction: base rate G={yh.mean():.3f}  Brier model={brier:.4f} "
                 f"vs block-climatology={brier_base:.4f}")

    # 2) conditional price models
    preds = {"ts": hold["ts"].to_numpy(), "p_up": p_up,
             "dir": hold["dir"].to_numpy(),
             "ceb_pp": hold["ceb_pp_cost"].to_numpy()}
    for d, tag in (("G", "up"), ("D", "dn")):
        tr = train[train["dir"] == d]
        te = hold[hold["dir"] == d]
        gbm = QuantileGBM().fit(tr[feats], tr["ceb_pp_cost"])
        ph = gbm.predict(hold[feats])          # predict on ALL holdout rows
        for q in QUANTILES:
            preds[f"{tag}_q{q}"] = ph[q].to_numpy()
        pb_m = pinball(te["ceb_pp_cost"].to_numpy(), ph.loc[te.index])
        pb_c = pinball(te["ceb_pp_cost"].to_numpy(), climatology(tr, te, "ceb_pp_cost"))
        lines.append(f"ceb_pp|{d}: holdout n={len(te)}  pinball model={pb_m:.2f} "
                     f"vs qh-climatology={pb_c:.2f}")

    pd.DataFrame(preds).to_parquet(OUT_PRED)

    # unconditional (train-climatology) reference quantiles for the optimizer
    clim_rows = {"qh_of_day": sorted(train["qh_of_day"].unique())}
    for d, tag in (("G", "up"), ("D", "dn")):
        tr = train[train["dir"] == d]
        cl = tr.groupby("qh_of_day")["ceb_pp_cost"].quantile(list(QUANTILES)).unstack()
        for q in QUANTILES:
            clim_rows[f"{tag}_q{q}"] = [cl.loc[k, q] for k in clim_rows["qh_of_day"]]
    clim_rows["p_up"] = [base[k] for k in clim_rows["qh_of_day"]]
    pd.DataFrame(clim_rows).to_parquet("data/proc/bess_cond_climatology.parquet")

    txt = "\n".join(lines)
    OUT_TXT.write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {OUT_PRED} and data/proc/bess_cond_climatology.parquet")


if __name__ == "__main__":
    main()
