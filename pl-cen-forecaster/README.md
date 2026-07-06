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
