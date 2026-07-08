# Interview kit

*Not shipped to recruiters — this is the rehearsal file. Numbers checked
against RESEARCH_LOG.md F1–F29 as of 2026-07-08.*

## CV bullets

**Trading-desk flavor (Danske Commodities & similar):**

> Built a probabilistic forecaster for the post-reform Polish imbalance price
> (15-min single-price settlement, 2024–26) and mapped the efficiency of every
> tradeable structure around it — publication-timestamp-honest signals, costs
> on both legs, by-quarter verdicts. Proved the public-information spread space
> closed; identified and honestly retired a Sharpe ~1.9 weather-revision signal
> on quarter-concentration. Live-shadow infrastructure issues gate-stamped
> forecasts daily and archives grid-data vintages every 15 minutes.

**Asset-optimization flavor (BESS optimizers, Orlen & similar):**

> Quantified the full revenue stack of a grid-scale battery on the Polish
> balancing market (~2.95M PLN/MW/yr, capacity-dominated) from public data:
> activation-probability curves from 67M rows of merit-order ladders, a
> capacity-price forecaster at the real D-1 bid gate (−16–18% MAE vs
> persistence), a forecast-timed SoC-recovery policy worth +38k PLN/MW/yr
> (positive every quarter), and the commit-vs-trade threshold showing when
> forecast-driven dispatch starts to matter (capacity fees currently 15×
> hourly arbitrage value).

## The 90-second pitch (spoken register)

"After Poland's 2024 balancing reform I wanted to know if the new
imbalance price was forecastable and tradeable, so I built the whole chain
myself on public data. Three things came out of it.

First, a probabilistic forecaster — quantiles, not point forecasts — with
strict information gates: the imbalance price publishes at 2 p.m. the next
day, and if you don't respect that timestamp your backtest lies to you.

Second, an efficiency map. I tested every spread between the day-ahead, the
three intraday auctions, and the imbalance price — real entry, real exit,
costs on both legs, results shown quarter by quarter. They're all dead, and
I can show *why*: whatever public data implies is priced into the first
auction you can access. My best signal — weather-model revisions predicting
intraday auction moves, Sharpe about 1.9 — I killed myself, because the
profits lived in two quarters out of seven. I'd rather show you the
discipline than a fragile backtest.

Third — since the value clearly isn't in financial spreads — I measured what
the forecast is worth to a battery. The answer: the revenue is capacity-
dominated, about three million złoty per megawatt-year; my forecaster adds a
measured +38k through smarter charge-management timing, positive in every
quarter; and I know the exact threshold at which falling capacity prices
would make forecast-driven dispatch decisions matter.

And because backtests run on revised data, I have infrastructure capturing
the live view of the grid every 15 minutes and a shadow forecaster that
publishes its prediction every morning before the auction clears — a track
record with timestamps, accumulating as we speak."

## Numbers to know cold

- Reform 2024-06-14; 15-min settlement; SDAC 15-min MTU 2025-09-30.
- Gates: IDA1 15:00 D-1 · IDA2 22:00 D-1 · IDA3 10:00 D · capacity bids
  ~07:30 D-1 (CMBP publishes 09:10 D-1) · CEN publishes D+1 ~14:00.
- Forecaster: quantile GBM + split conformal; LEAR/Ave-style benchmarks;
  ~62 pinball headline (F17); PSE's own forecast beaten.
- BESS: capacity 2.7M gross, −haircut → net stack 2.86M blind / 2.90M with
  F27; activation +864k; recovery drag −668k (only ~⅓ recoverable at the
  oracle bound — decomposition, not hand-waving).
- F28: MAE 62.5/67.6 vs persistence 74/82; coverage flag 0.6 vs 0.8 nominal
  (conformal fix known, not yet applied — say it before they ask).
- F29: fee/EV = 285/19 PLN ≈ 15×; flip at ~4× fee deflation (5% of hours).
- Ladder: ~650–700 offers × 96 periods/day ≈ 64k rows/day; offers D-1
  committed and sticky; publishes ~20 min post-delivery (lag-legal only).
- Liquidity ceiling: IDA1 ~160 MWh/period median → 1–5 MW strategies.

## Hard questions, prepared answers

- **"Why no live trading?"** These markets have no retail access; the honest
  structures are asset-backed. That's a finding, not an excuse — and the
  shadow forecaster is the live test I *can* run.
- **"How do I know there's no leakage?"** Publication-timestamp gating
  everywhere + the retraction story: a carry signal died when re-lagged from
  one to two days (CEN's D+1 publication). I caught it in my own audit and
  logged the retraction.
- **"Isn't the battery number just the capacity market?"** Yes — and I
  proved it at the hourly decision level with gate-honest forecasts on both
  sides, quantified the deflation threshold where that changes, and built
  the machinery that's ready when it does.
- **"What would you do here on day one?"** My models are the public-data
  baseline; your private flow improves on it. I'd start with residual-
  imbalance optimization (carry vs close against my CEN quantiles) because
  it plugs into an existing BRP workflow with zero new market access.
- **"Coverage is 0.6, should be 0.8?"** Correct — GBM quantiles under-cover
  on spiky series; the repo's split-conformal layer fixes exactly this on
  the CEN model and is the known next step on CMBP.

## Register warning

Open every story in plain language (grid pays batteries a retainer to stand
by; the imbalance price is what you pay for being wrong) and only drop into
CEN/CMBP/aFRR vocabulary when the interviewer does first.
