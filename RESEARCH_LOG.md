# Research log & working notes

Living notebook for the project: what we believe, what we built, what we
found, and what's queued. Newest material at the top of each section.
(READMEs describe *how to run things*; this file records *why and what we
learned*.)

## The one-paragraph thesis

Post-reform (2024-06-14) Poland settles imbalance at a single 15-min price
(CEN) under PSE central dispatch. CEN is volatile, partially predictable at
a 60-min gate, and sits at the end of a chain of tradeable prices
(day-ahead SDAC → intraday IDA auctions / RDB continuous → balancing).
Everything here is some version of: **forecast the distribution of CEN (or
of the dispatcher's marginal price) honestly at the gate, then check
whether any spread against an executable leg survives costs.**

## Hard rules (never violate)

1. **Leakage gates.** A feature is usable at decision time `t` only if its
   publication timestamp ≤ `t − H` (H=60 min). Settlement series (crb-rozl,
   eb-rozl, bpkdbo ladders) publish **D+1 ~14:00** → modeled availability is
   D+1 15:00 Warsaw; the freshest CEN history at the gate is 1–2 days old.
   `csdac-pln`/`rce-pln` publish D-1 ~13:50 → legal anchors for all of day D.
2. **Latest-vintage trap.** The PSE API keeps only the last vintage of
   forecast reports (`price-fcst`, `pk5l-wp`, `pdgobpkd`) — publication
   timestamps often post-date delivery. Those series are benchmarks or
   `--extended`-flagged features, never strict inputs.
3. **Regime breaks.** Never train across 2024-06-14 (reform). Encode
   2025-09-30 (SDAC 15-min MTU) as a feature `regime_15min_da`, never split
   on it. Any strategy verdict must be shown **by quarter** — regimes die
   (we watched one die, see Findings).
4. **Verify → commit → only then delete.** (Process rule, learned the hard
   way.)

## Findings so far (chronological)

### F7. BESS Layer 2: settlement rules pinned down; conditional policy ~1,240 PLN/day per 1 MW / 2 MWh on holdout
Rules established from PSE's WDB training deck (Jan 2024, fetched from
pse.pl; local copy of key facts below) and *verified in our data*:
- **Uniform pricing, not pay-as-bid**: freely delivered/withdrawn balancing
  energy settles at **CEB_PP** per 15-min period (forced delivery at
  min/max(CEB_PP, CSDAC)). `crb-rozl` publishes it as `ceb_pp_cost` (100%
  coverage). Consequence: a price-taking battery's optimal offer is its
  **reservation price**, so the DP value function *is* the offer curve.
- **CEN = min/max(CEB_PP, CSDAC) by contracting state SK** — verified:
  72% of periods within 1 PLN (median error 0.00); residual is
  CKOEB/aFRR-platform corrections.
- **Offer gates**: initial OEB bind at the D-1 RBN gate (10:00–14:30;
  csdac publishes 13:50, 40 min before close). Intraday updates (RBB, up
  to 55 min before the hour) may only be LESS aggressive (up-price can
  only rise, down-price only fall). So D-1 features are the right
  conditioning set for the aggressive envelope.
- **Activation approximation**: up-offer at p is in merit iff the period's
  net direction is G and ceb_pp ≥ p (uniform-price logic). The bpkdbo
  ladder crossing stays as diagnostics only — its volume units are
  ambiguous (no ×k scaling reproduces ceb_pp exactly; CEB_PP embeds
  redispatch costs), and the published price needs no reconstruction.
  This also corrects F4's "512 PLN/MWh implied spread": settlement-price
  spread G-vs-D medians are ~543 vs ~306 → **~240 PLN/MWh**, still ample.

Build (`src/bess_cond_model.py` + `src/bess_optimizer.py`):
- Conditional models on the CEN feature panel (strict fx_ set):
  P(dir=G) Brier 0.197 vs 0.214 block-climatology; ceb_pp|G pinball
  **30.6 vs 45.4** climatology; ceb_pp|D **61.1 vs 77.3** (8-wk holdout).
- DP over SoC (piecewise-linear value, efficiency-exact off-grid states),
  5-point quantile approximation of (dir, ceb_pp); reservation prices out.
- Holdout simulation (56 days, 1 MW/2 MWh, η_rt 0.88, c_deg 100, SoC
  carried overnight, terminal 300 PLN/MWh): **conditional 1,244 PLN/day**
  (annualized ≈ 454k PLN/MW), 100% of days positive, 1.15 cycles/day;
  unconditional-climatology policy 1,054 (+18% from the conditional
  model); perfect-foresight bound 2,103 (we capture 59%).
- Degradation sensitivity: c_deg 50/100/200 → 1,347 / 1,244 / 1,112
  PLN/day, ≥96% positive days. The revenue is structural, not tail-luck.

Caveats (in order of expected bite): price-taker assumption; in-merit ⇔
"ceb_pp clears offer" ignores unit-level dispatch/network constraints and
partial activations; counter-direction (forced) activations ignored;
capacity-market (OMB) stacking not yet added — that's Layer 3 and only
adds; grid fees on charged energy not modeled; holdout is one 8-week
window (walk-forward version pending).

### F6. The *tradeable* intraday↔CEN spread: real edge through 2025, dead in 2026
First run of `backtest_spread_ida.py` (walk-forward CEN quantiles × TGE
legs, 49k joint periods Dec 2024–May 2026):
- **IDA1 leg (gate-honest, D-1), cost 10 PLN/MWh:** 9,669 trades, hit
  62.1%, +30 PLN/MWh per trade — but by quarter: +20k/+78k/+100k/+59k/+41k
  (2024Q4–2025Q4) then **~0 in 2026Q1 and negative in 2026Q2**. At cost 20
  the same shape holds (hit 60.3%, +26.5/trade, dead in 2026).
- IDA2 similar but weaker; IDA3 (same-day PM) mostly noise; RDB VWAP proxy
  agrees with IDA1's pattern.
- mean(CEN − IDA1) ≈ −7 PLN/MWh: intraday trades at a small systematic
  premium to eventual imbalance — a risk-premium carry, also fading.
- Nuance vs F2: the DA↔CEN edge died *exactly at* the 2025-09-30 SDAC
  reform; the IDA↔CEN edge survived it by one quarter and faded over
  2026Q1–Q2 — consistent with the intraday market absorbing the balancing
  signal gradually rather than by construction. Caveats: 2026Q2 is partial
  (to May 10), and the walk-forward model refits only every 4 weeks.
- Verdict: **the tradeable Project A edge is historical**. Its decay
  timeline (vs the DA leg's instant death) strengthens the natural-
  experiment write-up; a live strategy would need fresher features
  (ENTSO-E actuals, IDA order flow) to have a claim to any remaining edge.

### F1. The CEN forecaster works and beats its anchors (2026-07)
Quantile LightGBM (P10/25/50/75/90) + per-hour-block split-conformal
recalibration, strict features, 8-week untouched holdout:
mean pinball **68.5**, P10–90 coverage 0.76 — vs DA-anchor 87.1,
climatology 89.7, persistence 140.8. PSE's own final-vintage forecast
scores 27.5: that gap is the value of near-delivery information (ENTSO-E /
intraday features are the upgrade path).

### F2. The DA↔CEN spread edge existed — and died on reform day (flagship)
Walk-forward (expanding windows, 49k OOS predictions, Dec 2024–May 2026),
threshold rule long/short CEN−DA when P25/P75 clears cost: +41..65
PLN/MWh/quarter, 54% hit rate, diversified across days — **until exactly
2025-09-30**, when SDAC moved to 15-min products and the intra-hour shape
arbitrage it was harvesting closed. Negative on the holdout confirms it as
regime death, not overfit. This is a clean natural experiment
(publishable; also exactly Weron-group territory).

### F3. Weron-style postprocessing adds a real, free improvement (2026-07)
Rolling weekly-refit postprocessing of the walk-forward predictions
(`src/postprocess.py`), all OOS: QRA (quantile regression on
[gbm_q50, DA]) 63.16; binned-IDR 66.31; **Vincentized average of
{GBM, QRA, IDR} 62.80 vs GBM-alone 63.63**, coverage 0.775→0.795, better
in 5/6 quarters — the Lipiecki/Uniejewski/Weron (2024) "averaging beats
components" result reproduced on CEN. asinh-VST for the GBM itself was a
wash (68.57 vs 68.47) — kept only for the future LEAR benchmark.

### F4. The dispatcher's ladder is public and crossable (Project B, 2026-07)
- `poeb-rbn` is **per-offer** (~88.5k rows/day): every accepted balancing
  offer with up/down prices. It is the *supply curve*, ~static intraday
  (offers commit D-1); its max is the ladder top, **not** the marginal.
- `oeb-bpkdbo` (~13k rows/day) is the ladder actually taken into the
  balancing plan, with the net activation direction per period.
- **Construction that works:** sort the bpkdbo ladder by price, cumulate
  MW, cross at the activated volume from `eb-rozl` (MWh/15min × 4) → a
  per-period *marginal activated price*. Validated: corr ≈ **0.88 with
  CEN** (0.89 in up-periods) — consistent with CEN being built from
  marginal costs of activated energy plus CKOEB/aFRR components.
- Under central dispatch both directions activate nearly every period, so
  "P(activation)" is only meaningful jointly with price: the object is the
  **marginal-price distribution by direction and hour block**.
- Summer-2024 sample: up-marginal median ~643 PLN/MWh (P90 ~966), down
  median ~62 (P10 −150, i.e. paid to charge). That daily spread is the
  BESS revenue engine for Layers 2–3.
- Caveat: `ceb_sr_*` settlement-price columns only exist from **2025-07-11**
  (PICASSO accession) — the earlier v0 "proxy" curves silently used only
  that late window. v1 (true marginals, full history) supersedes them.
- PICASSO `pzeb_*` semantics still unverified (spread ≈ −461 vs domestic —
  flagged, not interpreted).
- **v1 complete (full history 2024-06→2026-07, 70,368 periods, 0% null):**
  net direction is up ~64% of periods except midday (~50/50, solar);
  up-marginal median 518 (night) → 708 (evening ramp, P90 1100); down-
  marginal midday median 23 with P10 −124 (paid to charge in solar hours).
  Implied daily BESS spread (median up-marg − median down-marg): **median
  512 PLN/MWh, P10 204** across 714 days — and the quarterly table is
  *stable* through both reforms (up-marg p50 ~550–660 every quarter):
  unlike the spread trades (F2/F6), this engine has not decayed. The v0
  proxy curves materially understated the tail: at 600 PLN a discharge
  offer activates 54.6% of evening-ramp periods (v0 said 29%).
  Full tables: `reports/activation_v1_summary.txt`, curves in
  `reports/activation_curves_v1.json`.

### F5. TGE's public results page is scrapable — the intraday leg is free
The WAF rejects bare curl but passes a normal browser header set; the
tables are **server-rendered** (an earlier "JS-loaded" conclusion was an
artifact of a WAF-stripped page). `?dateShow=DD-MM-YYYY` serves history
(verified ≥ 2024-07). Per day: 24 hourly + 96 quarter-hourly instruments
with RDB continuous min/max/VWAP + volumes and IDA1/2/3 uniform-price
auctions (EUR & PLN). This unlocks Project A's *executable* leg — no paid
TGE AIR subscription needed for EOD granularity.

## Current pipeline map (pl-cen-forecaster/src)

Data:
- `pse_client.py` — PSE v2 API client (percent-encoded `$filter`,
  `$first=50000`, `nextLink` pagination, 5xx retry, end→start label shift).
- `build_dataset.py` — raw pulls → 15-min panel; latest-vintage-as-of
  handling; flow pivots; hourly/daily alignment.
- `pull_bpkdbo.py` — per-period marginal activated price + volume-grid
  ladder snapshot from oeb-bpkdbo × eb-rozl (Project B backbone).
- `pull_tge_rdb.py` — TGE RDB/IDA scraper (Project A intraday leg).
- `pull_poeb_marginals.py` — legacy; its output was ladder-top stats, kept
  as `data/raw/pse_poeb_laddertop.parquet` (ladder cap/depth features).

Modeling / evaluation:
- `features.py` — leakage-honest feature builder (fx_ anchors, published
  history with staleness, xt_ extended quarantine).
- `models.py` — baselines + per-quantile LightGBM (optional asinh VST),
  monotone rearrangement.
- `conformal.py` — split-conformal per-quantile offsets, hour-block groups.
- `evaluate.py` — CV + holdout evaluation (`make eval`).
- `walkforward.py` — expanding-window OOS predictions (the honest test bed
  every strategy verdict runs on).
- `postprocess.py` — QRA / binned-IDR / Vincentized averaging (F3).

Strategy / analysis:
- `backtest_spread.py` — DA↔CEN threshold rule (F2).
- `backtest_spread_ida.py` — same rule vs IDA1/2/3 and RDB VWAP legs, with
  per-leg gate-honesty labels (ida1/2: D-1; ida3: same-day afternoon only;
  rdb_vwap: proxy). Runs once the TGE pull lands.
- `bess_activation.py` — Layer 1 v1: direction frequencies, marginal-price
  distributions by block × direction, activation curves
  π_up(p|block) = P(dir=G ∧ marg ≥ p), quarterly regime table.
- `report_figs.py` — report figures.

## In flight right now (2026-07-06 evening)

- `pull_bpkdbo` backfilling 2024-06-15 → today (~2.5 h; checkpointed,
  resumable). Then: rerun `python -m src.bess_activation` → v1 curves.
- `pull_tge_rdb` backfilling the same span (~25 min). Then:
  `python -m src.backtest_spread_ida` → first honest verdict on the
  *tradeable* CEN↔intraday spread.
- ENTSO-E token awaited (user) → fresh RES/load actuals + IDA prices from
  a second source + DE_LU spread features.

## Queue (rough priority)

1. Project A verdict with real legs (F5 data × walk-forward preds).
2. BESS Layer 1 v1 (marginal-price distributions), then a conditional
   quantile model of `marg` on D-1 features; Layers 2–3 per SPEC_B
   (offer-curve DP, revenue stack).
3. Weron continuation: LEAR benchmark (with VST), QRA over a wider model
   pool, calibration-window averaging; possibly CDF-averaging (Ave-P).
4. ENTSO-E integration when the token arrives.
5. Optional: write up F2 as a short note (natural experiment on SDAC
   15-min MTU) — candidate for contact with Weron's group.
