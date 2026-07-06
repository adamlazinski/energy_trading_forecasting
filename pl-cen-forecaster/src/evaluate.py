"""
Evaluation: pinball loss, calibration, and the blocks that matter
(hour-of-day, season) — the spec's sunset-ramp emphasis.

    python -m src.evaluate            # full run: CV + holdout, prints report
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from . import features, models, targets
from .build_dataset import load_cfg
from .models import QUANTILES


def pinball(y: pd.Series, pred: pd.DataFrame, quantiles=QUANTILES) -> pd.Series:
    """Mean pinball loss per quantile."""
    out = {}
    for q in quantiles:
        d = y - pred[q]
        out[q] = np.nanmean(np.maximum(q * d, (q - 1) * d))
    return pd.Series(out).rename("pinball")


def coverage(y: pd.Series, pred: pd.DataFrame) -> dict:
    """Empirical coverage of the central intervals."""
    return {
        "P10-P90": float(((y >= pred[0.10]) & (y <= pred[0.90])).mean()),
        "P25-P75": float(((y >= pred[0.25]) & (y <= pred[0.75])).mean()),
        "below_P50": float((y < pred[0.50]).mean()),
    }


def pinball_by(y, pred, by: pd.Series, quantiles=QUANTILES) -> pd.DataFrame:
    rows = {}
    for key, idx in by.groupby(by).groups.items():
        rows[key] = pinball(y.loc[idx], pred.loc[idx], quantiles)
    return pd.DataFrame(rows).T


def run(extended: bool = False, config: str = "config.yaml") -> dict:
    cfg = load_cfg(config)
    panel = pd.read_parquet(pathlib.Path(cfg["paths"]["proc"]) / "panel_15min.parquet")
    df = features.build(panel, cfg, extended=extended)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    fcols = features.feature_cols(df, extended=extended)
    print(f"panel {df.shape}, {len(fcols)} features, extended={extended}")
    print(f"span {df['ts'].min()} .. {df['ts'].max()}")

    dev, test = targets.holdout_test(df, cfg)
    print(f"dev {len(dev)} rows, holdout-test {len(test)} rows "
          f"(last {cfg['test']['weeks']} weeks untouched until the end)\n")

    # ---- CV on dev ----------------------------------------------------------
    cv_rows = []
    for i, sp in enumerate(targets.forward_chaining_splits(
            dev["ts"], embargo_minutes=cfg["horizon_minutes"])):
        tr, va = dev.loc[sp.train_idx], dev.loc[sp.val_idx]
        gbm = models.QuantileGBM().fit(tr[fcols], tr["y"])
        preds = {"GBM": gbm.predict(va[fcols]), **models.predict_baselines(tr, va)}
        for name, p in preds.items():
            pb = pinball(va["y"], p)
            cv_rows.append({"fold": i, "model": name, **{f"q{q}": v for q, v in pb.items()},
                            "mean_pinball": pb.mean()})
    cv = pd.DataFrame(cv_rows)
    cv_summary = cv.groupby("model")["mean_pinball"].agg(["mean", "std"]).sort_values("mean")
    print("== CV (dev, forward-chaining) mean pinball ==")
    print(cv_summary.round(2), "\n")

    # ---- final fit on all dev, score holdout --------------------------------
    gbm = models.QuantileGBM().fit(dev[fcols], dev["y"])
    preds = {"GBM": gbm.predict(test[fcols]), **models.predict_baselines(dev, test)}

    report = {"extended": extended, "n_features": len(fcols),
              "cv": cv_summary.to_dict(), "holdout": {}}
    print("== HOLDOUT (last %s weeks) ==" % cfg["test"]["weeks"])
    for name, p in preds.items():
        pb = pinball(test["y"], p)
        cov = coverage(test["y"], p)
        report["holdout"][name] = {"pinball": {str(q): float(v) for q, v in pb.items()},
                                   "mean_pinball": float(pb.mean()), "coverage": cov}
        print(f"{name:24s} mean_pinball={pb.mean():8.2f}  "
              f"P10-90 cov={cov['P10-P90']:.2f}  P25-75 cov={cov['P25-P75']:.2f}")

    # blocks: hour of day + season, GBM vs best baseline
    gp = preds["GBM"]
    print("\n== GBM pinball by hour block (holdout) ==")
    hour_block = pd.cut(test["hour"], [0, 6, 10, 14, 18, 22, 24], right=False,
                        labels=["night", "am_ramp", "midday", "pm", "ev_ramp", "late"])
    print(pinball_by(test["y"], gp, hour_block).round(2))
    print("\n== GBM pinball by month (holdout) ==")
    print(pinball_by(test["y"], gp, test["month"]).round(2))

    print("\n== GBM feature importance (P50, top 20) ==")
    print(gbm.importance()[0.50].head(20))

    out = pathlib.Path(cfg["paths"]["proc"]) / f"report_{'ext' if extended else 'strict'}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true",
                    help="include final-vintage pk5l/snapshot features (leakage caveat)")
    a = ap.parse_args()
    run(extended=a.extended)
