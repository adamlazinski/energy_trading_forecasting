"""
Report figures for the CEN forecaster (light-mode PNGs into reports/figs).

    python -m src.report_figs

Fits the final (train-minus-cal) model itself so it can dump holdout
predictions alongside the figures; slow-ish (~1 min of LightGBM).
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import conformal, features, models, targets
from .build_dataset import load_cfg
from .evaluate import CAL_WEEKS, pinball
from .models import QUANTILES

# palette (reference instance, light mode)
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
BLUE_100, BLUE_200, BLUE_450 = "#cde2fb", "#9ec5f4", "#2a78d6"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 11,
    "font.family": "sans-serif",
})


def _save(fig, path: pathlib.Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    cfg = load_cfg()
    out_dir = pathlib.Path("reports/figs"); out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(pathlib.Path(cfg["paths"]["proc"]) / "panel_15min.parquet")
    df = features.build(panel, cfg).dropna(subset=["y"]).reset_index(drop=True)
    fcols = features.feature_cols(df)
    dev, test = targets.holdout_test(df, cfg)

    cal_cut = dev["ts"].max() - pd.Timedelta(weeks=CAL_WEEKS)
    embargo = pd.Timedelta(minutes=cfg["horizon_minutes"])
    tr = dev[dev["ts"] < cal_cut - embargo]
    cal = dev[dev["ts"] >= cal_cut].reset_index(drop=True)

    gbm = models.QuantileGBM().fit(tr[fcols], tr["y"])
    p_cal = gbm.predict(cal[fcols])
    off_h = conformal.fit_offsets(cal["y"], p_cal,
                                  groups=conformal.hour_block(cal["hour"]))
    p_raw = gbm.predict(test[fcols])
    p_conf = conformal.apply_offsets(p_raw, off_h,
                                     groups=conformal.hour_block(test["hour"]))
    base = models.predict_baselines(dev, test)

    # persist holdout predictions for the notebook / later analysis
    dump = test[["ts", "y", "hour", "qh_of_day", "month"]].copy()
    for q in QUANTILES:
        dump[f"gbm_raw_q{q}"] = p_raw[q].values
        dump[f"gbm_conf_q{q}"] = p_conf[q].values
    dump["b1_da"] = base["B1_da_anchor"][0.50].values
    dump.to_parquet("reports/holdout_predictions.parquet")

    # ---- fig 1: intervals over the final 10 days of holdout -----------------
    w = test["ts"] >= test["ts"].max() - pd.Timedelta(days=10)
    t, sub, pc = test.loc[w, "ts"], test.loc[w, "y"], p_conf.loc[w]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(t, pc[0.10], pc[0.90], color=BLUE_100, label="P10–P90")
    ax.fill_between(t, pc[0.25], pc[0.75], color=BLUE_200, label="P25–P75")
    ax.plot(t, pc[0.50], color=BLUE_450, lw=2, label="P50")
    ax.plot(t, sub, ".", color=INK, ms=2.5, alpha=0.6, label="CEN realized")
    ax.set_ylabel("CEN [PLN/MWh]")
    ax.set_title("Conformalized forecast intervals vs realized CEN — final 10 holdout days",
                 loc="left", color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    _save(fig, out_dir / "fig1_intervals.png")

    # ---- fig 2: quantile calibration, raw vs conformal ----------------------
    fig, ax = plt.subplots(figsize=(5.2, 5))
    qs = list(QUANTILES)
    emp_raw = [float((test["y"].values < p_raw[q].values).mean()) for q in qs]
    emp_conf = [float((test["y"].values < p_conf[q].values).mean()) for q in qs]
    ax.plot([0, 1], [0, 1], color=BASE, lw=1, ls="--", zorder=1)
    ax.plot(qs, emp_raw, "o-", color=YELLOW, lw=2, ms=8, label="GBM raw")
    ax.plot(qs, emp_conf, "o-", color=BLUE, lw=2, ms=8, label="GBM conformal (hour)")
    ax.set_xlabel("nominal quantile"); ax.set_ylabel("empirical P(y < q̂)")
    ax.set_title("Quantile calibration on holdout", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out_dir / "fig2_calibration.png")

    # ---- fig 3: pinball by hour of day, model comparison --------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    hours = sorted(test["hour"].unique())
    for pred, color, name in ((p_conf, BLUE, "GBM conformal"),
                              (base["B1_da_anchor"], AQUA, "DA anchor"),
                              (base["B2_climatology"], YELLOW, "climatology")):
        v = [pinball(test.loc[test["hour"] == h, "y"], pred.loc[test["hour"] == h]).mean()
             for h in hours]
        ax.plot(hours, v, "-", color=color, lw=2, label=name)
        ax.annotate(name, (hours[-1], v[-1]), xytext=(4, 0),
                    textcoords="offset points", color=color, fontsize=9, va="center")
    ax.set_xlabel("hour of day (Europe/Warsaw)"); ax.set_ylabel("mean pinball [PLN/MWh]")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("Forecast difficulty by hour — midday solar imbalance dominates",
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, out_dir / "fig3_pinball_by_hour.png")

    # ---- fig 4: CEN day-shape percentile fan (full sample) ------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    prof = df.groupby("qh_of_day")["y"].quantile([.10, .25, .50, .75, .90]).unstack()
    x = prof.index / 4
    ax.fill_between(x, prof[.10], prof[.90], color=BLUE_100, label="P10–P90")
    ax.fill_between(x, prof[.25], prof[.75], color=BLUE_200, label="P25–P75")
    ax.plot(x, prof[.50], color=BLUE_450, lw=2, label="P50")
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xlabel("hour of day (Europe/Warsaw)"); ax.set_ylabel("CEN [PLN/MWh]")
    ax.set_xticks(range(0, 25, 3))
    ax.set_title("CEN distribution by time of day — full post-reform sample",
                 loc="left", color=INK)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    _save(fig, out_dir / "fig4_cen_day_shape.png")

    print("done")


if __name__ == "__main__":
    main()
