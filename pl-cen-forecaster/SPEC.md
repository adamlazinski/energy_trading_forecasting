# SPEC — Polish imbalance-price (CEN) forecaster

Brief for future me / Claude Code. Read this first before touching code.

## Goal
Forecast the Polish single imbalance settlement price **CEN** (Cena Energii
Niezbilansowania) for a target 15-minute settlement period, from a decision
time `H` minutes before delivery. Default `H = 60` (the SIDC cross-zonal
intraday gate). Everything is quarter-hourly (96 periods/day).

This is the natural first project because PSE publishes the target series
directly, and because CEN is the involuntary liquidation price for an
asset-less intraday book — so a good CEN forecast is the terminal mark in the
stochastic-control framing of the desk strategy.

## Hard modelling constraints (do not violate)
1. **Regime break: 2024-06-14.** Single-price imbalance settlement started then
   (RB reform Phase 2). Do NOT train across this boundary. Default train start
   is 2024-06-15. Everything before is a different market — ignore it or use it
   only for unsupervised context.
2. **Sub-regime: 2025-09-30.** 15-minute MTU went live on the day-ahead market
   (SDAC). Not enough post-date data to train a separate model (yet), so encode
   it as a binary feature `regime_15min_da` rather than splitting the sample.
3. **No leakage.** A feature for target period `t` may only use information
   published/available at `t - H`. Forecasts (demand, PV, wind, surplus) are
   allowed because they're known ahead; realised values are only allowed with
   the correct publication lag. `targets.py` and `features.py` enforce this;
   keep it that way.
4. **Time-series CV only.** Expanding-window (forward-chaining) splits with an
   embargo of at least `H` around each boundary. Never shuffle. Hold out the
   most recent contiguous ~8 weeks as an untouched test set.

## Data sources
- **PSE v2 API** (`https://api.raporty.pse.pl/api/`): free, no auth, JSON,
  OData filter on `business_date`. 15-min. Primary source. See `config.yaml`
  `pse.resources`. Confirm exact slugs with `python -m src.pse_client discover`.
- **ENTSO-E Transparency** (`entsoe-py`): needs a token (email
  transparency@entsoe.eu, subject "RESTful API access"). Used for
  cross-checks and cross-border (PL vs DE_LU day-ahead, scheduled exchanges,
  imbalance prices as a fallback/validation of the PSE CEN series).
- **Open-Meteo** (free, no key): irradiance (GHI/DNI) + hub-height wind over
  the main PL wind/solar clusters, for the RES-forecast-error features. Add in
  a later iteration; not required for a v0 baseline.

## Target
`CEN` for period `t`. Candidate PSE slug: `rce-pln` is the balancing *energy*
price (RCE); the imbalance settlement price report may be a separate slug —
CONFIRM via `discover`. Set the chosen slug in `config.yaml -> target.resource`
and the value column in `target.value_col`. As a validation, ENTSO-E
`query_imbalance_prices('PL', ...)` should track it closely post-reform.

## Feature groups (all as-of `t - H`)
- Autoregressive: last known CEN/RCE at decision time + rolling stats.
- System forecasts known ahead: demand fcst, PV fcst, wind fcst, surplus
  (nadwyżka mocy) from PSE PK5L.
- RES forecast error: recent realised PV/wind minus their forecasts (the
  imbalance driver). Respect publication lag.
- Cross-border: scheduled PL exchange, PL vs DE_LU day-ahead spread.
- Anchor: RDN (day-ahead) price for `t` — CEN is bounded relative to it.
- Calendar: qh-of-day, hour, dow, holiday (PL), month, sunrise/sunset ramp
  flags; `regime_15min_da` dummy.

## Baselines before ML
1. Persistence: CEN(t) = last published CEN.
2. Day-ahead anchor: CEN(t) = RDN(t).
3. Then gradient boosting (LightGBM/CatBoost) with quantile loss for P10/P50/P90
   — you care about the distribution, not just the point, because the strategy
   trades against the tail.

## Repo layout
- `config.yaml`        — dates, regime breaks, resource slugs, horizon H.
- `src/pse_client.py`  — generic OData fetcher (+ `discover` CLI). Endpoint-agnostic.
- `src/entsoe_client.py` — thin entsoe-py wrappers.
- `src/build_dataset.py` — pull + align everything to a 15-min UTC panel -> parquet.
- `src/features.py`    — leakage-safe feature engineering.
- `src/targets.py`     — CEN target + regime flags + CV split generator.
- `Makefile`           — `make data`, `make features`.

## First session TODO (Claude Code)
1. `pip install -r requirements.txt`
2. `python -m src.pse_client discover` -> confirm target + resource slugs, write
   them into `config.yaml`.
3. `make data` -> raw parquet in `data/raw/`.
4. `make features` -> model-ready panel in `data/proc/`.
5. Fit the two baselines, then the quantile GBM. Report pinball loss on the
   held-out recent block, split out by hour-of-day (sunset ramp is the hard part).
