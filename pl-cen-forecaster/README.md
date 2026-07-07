# pl-cen-forecaster

Forecast the Polish single imbalance settlement price (CEN) at a 15-minute
granularity, from a decision time `H` minutes before delivery (default 60).

**Read `SPEC.md` first** — it briefs the design and the hard constraints
(the 2024-06-14 regime break, leakage rules, time-series CV).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data        # pull PSE history + build data/proc/panel_15min.parquet
make eval        # baselines + quantile GBM, holdout report
make eval-ext    # + final-vintage pk5l features (leakage-caveat sensitivity)
```

## Data facts that shape the design (verified on the PSE v2 API, 2026-07)
- CEN settlement (`crb-rozl`) publishes **D+1 ~14:00** — at H=60 the freshest
  visible CEN is 1–2 days old. AR features are built from the *last fully
  published day* under a D+1 15:00 availability rule.
- `csdac_pln` / `rce_pln` publish **D-1 ~13:50** — legal anchors for every
  period of day D.
- The API stores only the **latest vintage** of forecast reports (`pk5l-wp`,
  `price-fcst`, `pdgobpkd`): publication timestamps often post-date delivery.
  PSE's own `cen_fcst` is therefore reported as a reference benchmark, never
  used as a feature; pk5l features live behind `--extended` with an explicit
  leakage caveat.
- Realized actuals (`his-wlk-cal`, `en-rozl`) lag 3–4 days. Fresh RES/load
  actuals need ENTSO-E (token pending) — planned upgrade.

Data: PSE v2 API (free, no auth), ENTSO-E Transparency (free token),
Open-Meteo (free, later iteration for RES nowcast features).

## Results (holdout = last 8 weeks, H=60, strict features)

| model | mean pinball | P10–P90 coverage (nom. 0.80) |
|---|---|---|
| **GBM + hour-conformal** | **68.5** | **0.76** |
| GBM raw | 71.2 | 0.56 |
| B1 day-ahead anchor | 87.1 | — |
| B2 climatology | 89.7 | 0.68 |
| B0 persistence | 140.8 | — |
| *PSE final-vintage forecast* | *27.5* | *(information upper bound)* |

Figures in `reports/figs/`; full numbers in `reports/`. The 68.5 → 27.5 gap
is the value of near-delivery information — the ENTSO-E/intraday upgrade path.

On the 49k-row walk-forward test bed, Weron-style postprocessing
(`python -m src.postprocess`: rolling QRA + binned IDR + quantile
averaging) improves the GBM further: pinball 63.63 → **62.80**, P10–90
coverage 0.775 → 0.795 (`reports/postprocess_walkforward.txt`).

## Beyond the forecaster (same src/, see ../RESEARCH_LOG.md for findings)

| module | what it does |
|---|---|
| `walkforward.py` | expanding-window OOS predictions — the honest test bed |
| `postprocess.py` | QRA / IDR / distribution averaging on those predictions |
| `backtest_spread.py` | DA↔CEN threshold rule (edge existed, died 2025-09-30) |
| `pull_tge_rdb.py` | TGE RDB continuous + IDA1/2/3 auction results scraper |
| `backtest_spread_ida.py` | same rule vs the *tradeable* intraday legs |
| `pull_bpkdbo.py` | PSE balancing ladder → per-period marginal activated price |
| `bess_activation.py` | Project B Layer 1: dispatcher activation curves |
| `bess_cond_model.py` | Layer 2a: P(direction) + conditional CEB_PP quantiles |
| `bess_optimizer.py` | Layer 2b: reservation-price DP + holdout simulation |
| `bess_layer3.py` | Layer 3: capacity (RMB) revenue stack vs energy arbitrage |

Long pulls (`pull_bpkdbo`, `pull_tge_rdb`, `pull_poeb_marginals`) are
day-by-day, checkpointed, and safe to re-run — they skip days already in
their output parquet.
