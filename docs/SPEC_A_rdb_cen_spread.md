# SPEC — Project A: Is the intraday↔imbalance (RDB↔CEN) spread predictable in Poland?

Extension of `pl-cen-forecaster`. Read that repo's SPEC.md first; this reuses
its data layer, regime handling, and CV discipline.

## Research question
At decision time `t - H` (default H = 60 min, the SIDC cross-zonal gate), is the
spread

    S(t) = CEN(t) - P_ID(t, t-H)

predictable — in distribution, not just in mean — where P_ID is the last
tradeable intraday price for delivery period t observed at decision time?

The null to beat is the published negative result for other markets: 30-min-ahead
probabilistic imbalance forecasts did NOT substantially outperform the intraday
price index (efficiency between ID and balancing). Poland post-2024-06-14 is a
different design (single-price CEN, central dispatch, thin RDB, PICASSO/MARI
coupling) — the reform *tried* to close this arbitrage; the project measures
whether it succeeded. A clean null result is still a good paper and a good
interview artifact. Do not torture the data for a positive.

## Data (honest constraints)
1. CEN: PSE v2 API (already wired). Target as in the forecaster.
2. P_ID: TGE granular RDB continuous ticks are PAID. Free proxies, in order of
   preference — implement as interchangeable `price_ref` columns and report
   results under each:
   a. IDA auction prices (IDA1/2/3) — discrete but genuinely tradeable prices;
      IDA3 (gate 10:00 D, covers H12–24) is closest to delivery.
   b. TGE published RDB session indices (delayed daily summaries; scrape).
   c. Fallback anchor: Fixing II day-ahead price (weakest — tests DA↔CEN, a
      different but still interesting spread).
   Flag clearly in all outputs which proxy a result uses. If the desk later
   provides real tick data, only the loader changes.
3. Conditioning features: everything already in the forecaster panel (RES
   forecast errors, demand fcst, PL–DE_LU DA spread, PSE surplus/PK5L, reserve
   prices, calendar/ramp flags, regime_15min_da) + aFRR/mFRR activation info
   and, if obtainable, PICASSO cross-border aFRR prices (German aFRR conditions
   as a leading indicator).

## Models (in strictly increasing complexity; stop when gains die)
0. Naive efficiency null: S(t) = 0 (i.e. P_ID is an unbiased CEN predictor).
1. Unconditional per-qh-of-day climatology of S (captures the known midday-
   negative / evening-positive CEN shape).
2. Linear quantile regression on the feature panel (P10/25/50/75/90).
3. LightGBM quantile. Same folds as the forecaster; embargo >= H.
4. Optional: two-part model — P(sign of S) classifier x conditional magnitude —
   because the economically relevant object is tail asymmetry, not the median.

## Evaluation
- Statistical: pinball loss vs models 0/1; Diebold–Mariano per quantile;
  calibration (PIT histograms); results split by qh-of-day and by season —
  the spring/summer midday solar hours are where PL imbalance tails live, so
  report those blocks separately rather than letting them average out.
- Economic: threshold rule — trade only when the predicted P25 > cost (short
  ID / long CEN direction) or P75 < -cost (reverse). One trade per qh, fixed
  clip. Costs: half-spread + fees; run cost sensitivity at 2x and 4x because
  the RDB book is thin and the proxy prices understate impact. Report P&L per
  MWh traded, hit rate, tail drawdown; compare against "always run to CEN" and
  "always close at ID".
- Honesty constraints: no positive claim unless it survives the 4x-cost run
  AND the held-out final 8 weeks. Report the negative clearly if that's what
  it is — the reform working as designed is a finding.

## Repo changes
- `src/spread.py`      — S(t) construction per price_ref; leakage guard.
- `src/models_spread.py` — models 0–4 behind one interface.
- `src/backtest_spread.py` — threshold rule + cost sensitivity + reports.
- `notebooks/spread_report.ipynb` — figures: S distribution by qh-of-day,
  calibration, cumulative P&L per cost scenario, DM test table.

## Timebox
~2–3 weeks of evenings on top of the existing repo. Kill criterion: if model 2
doesn't beat model 1 on pinball at any quantile after the first full run,
write it up as an efficiency result and stop.
