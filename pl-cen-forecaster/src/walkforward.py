"""
Walk-forward out-of-sample predictions over the whole dev window.

    python -m src.walkforward

Expanding-window refits: first prediction block starts WARMUP_WEEKS after the
sample start; each step fits on everything up to the block start (minus the
conformal calibration tail and the horizon embargo), conformalizes on the
tail, predicts the next STEP_WEEKS, and rolls. Output: one parquet of pure
OOS quantile predictions spanning ~18 months — the input for any strategy
evaluation with enough independent tail events to mean something.

The holdout (last test.weeks) stays untouched, as everywhere else.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from . import conformal, features, models, targets
from .build_dataset import load_cfg
from .evaluate import CAL_WEEKS
from .models import QUANTILES

WARMUP_WEEKS = 26
STEP_WEEKS = 4


def main():
    cfg = load_cfg()
    panel = pd.read_parquet(pathlib.Path(cfg["paths"]["proc"]) / "panel_15min.parquet")
    df = features.build(panel, cfg).dropna(subset=["y"]).reset_index(drop=True)
    fcols = features.feature_cols(df)
    dev, _test = targets.holdout_test(df, cfg)
    embargo = pd.Timedelta(minutes=cfg["horizon_minutes"])

    t0 = dev["ts"].min() + pd.Timedelta(weeks=WARMUP_WEEKS)
    t_end = dev["ts"].max()
    blocks = []
    cur = t0
    while cur < t_end:
        blocks.append((cur, min(cur + pd.Timedelta(weeks=STEP_WEEKS), t_end)))
        cur += pd.Timedelta(weeks=STEP_WEEKS)
    print(f"{len(blocks)} walk-forward blocks of {STEP_WEEKS}w from {t0.date()} to {t_end.date()}")

    outs = []
    for i, (lo, hi) in enumerate(blocks):
        cal_cut = lo - pd.Timedelta(weeks=CAL_WEEKS)
        tr = dev[dev["ts"] < cal_cut - embargo]
        cal = dev[(dev["ts"] >= cal_cut) & (dev["ts"] < lo - embargo)].reset_index(drop=True)
        blk = dev[(dev["ts"] >= lo) & (dev["ts"] < hi)].reset_index(drop=True)
        if len(tr) < 5000 or not len(blk):
            continue
        gbm = models.QuantileGBM().fit(tr[fcols], tr["y"])
        off = conformal.fit_offsets(cal["y"], gbm.predict(cal[fcols]),
                                    groups=conformal.hour_block(cal["hour"]))
        p = conformal.apply_offsets(gbm.predict(blk[fcols]), off,
                                    groups=conformal.hour_block(blk["hour"]))
        out = blk[["ts", "y", "hour", "qh_of_day", "month", "fx_da"]].copy()
        for q in QUANTILES:
            out[f"gbm_conf_q{q}"] = p[q].values
        out["block"] = i
        outs.append(out)
        print(f"block {i:2d} {lo.date()}..{hi.date()}  train={len(tr):6d}  "
              f"pred={len(blk):5d}", flush=True)

    allp = pd.concat(outs, ignore_index=True)
    allp = allp.rename(columns={"fx_da": "b1_da"})
    path = pathlib.Path("reports/walkforward_predictions.parquet")
    allp.to_parquet(path)
    print(f"wrote {path}  shape={allp.shape}  span {allp['ts'].min()}..{allp['ts'].max()}")


if __name__ == "__main__":
    main()
