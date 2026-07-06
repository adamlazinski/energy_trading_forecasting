"""
Weron-group postprocessing on the walk-forward CEN predictions
(Lipiecki, Uniejewski & Weron 2024: postprocessing + distribution averaging).

    python -m src.postprocess

All methods are fit on a TRAILING calibration window and applied to the next
week only (weekly refit), so every reported number is out-of-sample:

  GBM      — the conformalized quantile GBM as produced by walkforward (ref).
  QRA      — per-quantile regression of y on [1, gbm_q50, b1_da]
             (sklearn QuantileRegressor, alpha=0).
  IDR-b    — binned isotonic distributional regression: empirical quantiles
             of y within quantile-bins of gbm_q50, isotonized across bins.
  Ave-Q    — Vincentization: mean of the three quantile vectors.

Outputs reports/postprocess_walkforward.{json,txt}.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
CAL_WEEKS = 8
N_BINS = 20

WF = pathlib.Path("reports/walkforward_predictions.parquet")
GBM_COLS = {q: f"gbm_conf_q{q}" for q in QUANTILES}


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def fit_qra(cal: pd.DataFrame) -> dict[float, QuantileRegressor]:
    X = cal[["gbm_conf_q0.5", "b1_da"]].to_numpy()
    y = cal["y"].to_numpy()
    return {q: QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
            for q in QUANTILES}


def apply_qra(models: dict, test: pd.DataFrame) -> pd.DataFrame:
    X = test[["gbm_conf_q0.5", "b1_da"]].to_numpy()
    out = pd.DataFrame({q: models[q].predict(X) for q in QUANTILES},
                       index=test.index)
    out[:] = np.sort(out.values, axis=1)
    return out


def fit_idr(cal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Binned IDR: x-grid of gbm_q50 bin centers -> per-bin y-quantiles,
    isotonized (cummax) along x within each quantile level."""
    x = cal["gbm_conf_q0.5"].to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1,
                  0, len(edges) - 2)
    centers, table = [], []
    for b in range(len(edges) - 1):
        yb = cal["y"].to_numpy()[idx == b]
        if len(yb) < 30:
            continue
        centers.append(float(np.median(x[idx == b])))
        table.append([float(np.quantile(yb, q)) for q in QUANTILES])
    centers = np.array(centers)
    table = np.maximum.accumulate(np.array(table), axis=0)  # isotone in x
    return centers, table


def apply_idr(fit: tuple, test: pd.DataFrame) -> pd.DataFrame:
    centers, table = fit
    x = test["gbm_conf_q0.5"].to_numpy()
    out = pd.DataFrame(
        {q: np.interp(x, centers, table[:, j])
         for j, q in enumerate(QUANTILES)}, index=test.index)
    out[:] = np.sort(out.values, axis=1)
    return out


def main():
    wf = pd.read_parquet(WF).sort_values("ts").reset_index(drop=True)
    wf = wf.dropna(subset=["y", "gbm_conf_q0.5", "b1_da"])
    wf["week"] = wf["ts"].dt.tz_convert("Europe/Warsaw").dt.to_period("W")

    weeks = wf["week"].unique()
    preds = {m: [] for m in ("QRA", "IDR-b", "Ave-Q")}
    kept = []
    for w in weeks[CAL_WEEKS:]:
        cal = wf[(wf["week"] >= w - CAL_WEEKS) & (wf["week"] < w)]
        test = wf[wf["week"] == w]
        if len(cal) < 2000 or test.empty:
            continue
        qra = apply_qra(fit_qra(cal), test)
        idr = apply_idr(fit_idr(cal), test)
        gbm = test[[GBM_COLS[q] for q in QUANTILES]].set_axis(
            list(QUANTILES), axis=1)
        ave = (qra + idr + gbm) / 3.0
        ave[:] = np.sort(ave.values, axis=1)
        preds["QRA"].append(qra)
        preds["IDR-b"].append(idr)
        preds["Ave-Q"].append(ave)
        kept.append(test)

    test_all = pd.concat(kept)
    y = test_all["y"].to_numpy()
    models = {
        "GBM": test_all[[GBM_COLS[q] for q in QUANTILES]].set_axis(
            list(QUANTILES), axis=1),
        **{m: pd.concat(v) for m, v in preds.items()},
    }

    res = {}
    qtr = test_all["ts"].dt.tz_convert("Europe/Warsaw").dt.to_period("Q").astype(str)
    for name, p in models.items():
        pb = {q: pinball(y, p[q].to_numpy(), q) for q in QUANTILES}
        cov90 = float(np.mean((y >= p[0.10]) & (y <= p[0.90])))
        cov50 = float(np.mean((y >= p[0.25]) & (y <= p[0.75])))
        by_q = {}
        for g in sorted(qtr.unique()):
            m = (qtr == g).to_numpy()
            by_q[g] = round(float(np.mean(
                [pinball(y[m], p[q].to_numpy()[m], q) for q in QUANTILES])), 2)
        res[name] = {
            "mean_pinball": round(float(np.mean(list(pb.values()))), 3),
            "per_q": {str(q): round(v, 2) for q, v in pb.items()},
            "cov_P10_90": round(cov90, 3), "cov_P25_75": round(cov50, 3),
            "pinball_by_quarter": by_q,
        }

    out = {"n_oos": int(len(test_all)),
           "span": f"{test_all['ts'].min()} .. {test_all['ts'].max()}",
           "cal_weeks": CAL_WEEKS, "models": res}
    pathlib.Path("reports/postprocess_walkforward.json").write_text(
        json.dumps(out, indent=2))

    lines = [f"postprocessing, walk-forward OOS n={len(test_all)}  ({out['span']})", ""]
    lines.append(f"{'model':8s} {'pinball':>8s} {'cov10-90':>9s} {'cov25-75':>9s}")
    for name, r in res.items():
        lines.append(f"{name:8s} {r['mean_pinball']:8.2f} "
                     f"{r['cov_P10_90']:9.3f} {r['cov_P25_75']:9.3f}")
    lines.append("\nmean pinball by quarter:")
    qs = sorted(next(iter(res.values()))["pinball_by_quarter"])
    lines.append("model    " + "".join(f"{g:>9s}" for g in qs))
    for name, r in res.items():
        lines.append(f"{name:8s} " + "".join(
            f"{r['pinball_by_quarter'][g]:9.1f}" for g in qs))
    txt = "\n".join(lines)
    pathlib.Path("reports/postprocess_walkforward.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
