"""
LEAR quantile predictions over the walk-forward span, aligned to the rows
of reports/walkforward_predictions.parquet, for use as an independent member
in the postprocessing ensemble (src.postprocess, task #16 / F9).

    python -m src.lear_walkforward

Rolling refit every 7 days on trailing [56, 84]-day windows, price target
(no VST — see F9). Only days present in the walk-forward parquet are
predicted, so the output joins 1:1 on ts. Output: reports/lear_walkforward
.parquet with columns ts + lear_q{0.1..0.9}.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from .lear import load_panel, rolling_predict
from .models import QUANTILES

# light config: single window, monthly refit — the [56,84]/7d version is
# ~8x more fits and sklearn's LP-based QuantileRegressor is O(n_samples) slow
# (2.7 h on the full span). A monthly-refit single window is an adequate
# ensemble member for the F9/#16 diversification test.
WINDOWS = (56,)
REFIT_EVERY = 28
WF = pathlib.Path("reports/walkforward_predictions.parquet")
OUT = pathlib.Path("reports/lear_walkforward.parquet")


def main():
    wf = pd.read_parquet(WF)
    df, feats = load_panel()
    wf_days = set(pd.to_datetime(wf["ts"]).dt.tz_convert("Europe/Warsaw")
                  .dt.normalize().unique())
    days = sorted(d for d in
                  df["ts"].dt.tz_convert("Europe/Warsaw").dt.normalize().unique()
                  if d in wf_days)
    print(f"predicting {len(days)} walk-forward days "
          f"({days[0].date()}..{days[-1].date()})", flush=True)
    preds = rolling_predict(df, feats, days, WINDOWS, REFIT_EVERY, use_vst=False)

    out = df[["ts"]].copy()
    for j, q in enumerate(QUANTILES):
        out[f"lear_q{q}"] = preds[:, j]
    out = out.dropna(subset=[f"lear_q{QUANTILES[0]}"])
    # keep only rows also in the walk-forward bed
    out = out[out["ts"].isin(set(wf["ts"]))].reset_index(drop=True)
    out.to_parquet(OUT)
    print(f"wrote {OUT}  shape={out.shape}  "
          f"span {out['ts'].min()}..{out['ts'].max()}", flush=True)


if __name__ == "__main__":
    main()
