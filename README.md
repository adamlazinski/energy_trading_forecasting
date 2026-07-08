# Polish Power Markets: Forecasting, Trading & BESS Optimization

Research program on the Polish balancing market after the **2024-06-14 reform**
(single imbalance price "CEN", 15-minute settlement, central dispatch) — one of
the first systematic studies of this market post-reform. Built end-to-end on
free public data: a probabilistic price forecaster, honest backtests of every
tradeable structure around it, a battery revenue/dispatch stack, and live
data-capture infrastructure.

## Headline results

| # | Result | Where |
|---|--------|-------|
| 1 | **Probabilistic CEN forecaster** (quantile GBM + conformal; LEAR benchmark) beats naive/climatology benchmarks on strict walk-forward evaluation | `src/models.py`, `src/lear.py`, F17 |
| 2 | **Market-efficiency map**: every spread among {day-ahead, IDA1/2/3, CEN} tested with publication-timestamp-honest signals, costs on both legs, by-quarter verdicts — all dead. Public scheduled information is priced by the first accessible auction | F2, F24, F25 |
| 3 | **A real signal, honestly killed**: NWP run-to-run forecast revisions (manufactured from archived weather-model runs) predict IDA2→IDA3 moves — Sharpe ~1.9 gross — but the P&L concentrates in 2/7 quarters, so it's a feature, not a book | F26 |
| 4 | **BESS revenue stack**: 1 MW/2 MWh battery nets ~**2.95M PLN/MW/yr**, ~90% from aFRR capacity. Forecast-timed SoC recovery adds **+38k/yr, positive in every quarter** (oracle ceiling +132k — the drag is ~80% structural) | F16, F27 |
| 5 | **aFRR capacity-price forecaster** at the actual D-1 bid gate (a harder information set than the price forecaster's): **16–18% better MAE than persistence**, ~every quarter | F28 |
| 6 | **Commit-vs-trade co-optimization**: capacity fees dominate hourly arbitrage value **15×** — the D-1 commitment decision needs no forecast until fees deflate ~4× (threshold quantified, machinery ready) | F29 |
| 7 | **Live vintage capture**: 15-min snapshots of nine grid feeds with capture timestamps (grid data is silently revised 4–5× intraday — measured, not assumed), plus a daily **shadow forecaster** issuing gate-stamped forecasts before publication | `src/live_collector.py`, `src/shadow_cmbp.py` |

## Why the negative results are the point

Findings F1–F29 (see `RESEARCH_LOG.md`) follow hard rules learned the expensive
way: signals gated on **publication timestamps** (CEN publishes D+1 ~14:00 — the
single biggest leakage trap in this market), no training across the reform
break, transaction costs on both legs of every structure, and **no strategy
verdict without a by-quarter breakdown** (regimes die; we watched one die).
Several once-promising results were retracted by their own audits — the log
keeps the retractions. The efficient-market findings are as load-bearing as the
positive ones: they redirect the forecast's value from financial spreads to
physical dispatch, where it measurably pays.

## Data

All free, ~153M rows, reform → present at 15-min resolution: PSE v2 API
(~20 report types incl. the full balancing **offer ladder** — 67M rows of
merit-order supply curves, the free cousin of the licensed order books the
German intraday literature runs on — see `LITERATURE.md`), TGE day-ahead/IDA
auctions (scraped), ENTSO-E, Open-Meteo incl. **archived past model runs**
(reconstructing forecast revisions Poland doesn't publish). A launchd-scheduled
collector archives the live view every 15 minutes because official history is
revised after the fact.

## Layout

- `RESEARCH_LOG.md` — the working log: rules, findings F1–F29, pipeline map.
- `LITERATURE.md` — survey: what data/code the electricity-trading literature
  actually uses, and where this project sits (free-data side of the EPEX moat).
- `pl-cen-forecaster/` — all code: data layer, feature panel (leakage-honest by
  construction), models, strategy backtests, BESS stack, live collector +
  shadow forecaster (`src/`, one experiment per module, results in `reports/`).
- `docs/` — project specs.

## Setup

```bash
cd pl-cen-forecaster
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# data pulls: see Makefile; every experiment runs as `python -m src.<module>`
```
