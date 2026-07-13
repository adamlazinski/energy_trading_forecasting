# Probabilistic forecasting of the post-reform Polish balancing price (CEN): the first study of a single-price, 15-minute settlement market

**Working draft — v0.1, 2026-07-12.** Markdown; ports to LaTeX once structure settles.
Source material: `RESEARCH_LOG.md` (findings F1–F31), `LITERATURE.md`, `README.md`.
Bracketed `[TODO]` / `[FIG]` marks are open items, not claims.

---

## Abstract

On 2024-06-14 Poland replaced its dual-price, hourly imbalance settlement with a
single imbalance price (*cena niezbilansowania*, **CEN**) settled in 15-minute
intervals under central dispatch — the same design toward which EU balancing
markets are broadly converging. We present, to our knowledge, the first
systematic forecasting and trading study of this market. Built end-to-end on
free public data (PSE v2 API, TGE auctions, ENTSO-E, Open-Meteo), the study
delivers: (i) a probabilistic CEN forecaster — a quantile gradient-boosting
ensemble with a LEAR benchmark, distribution-averaged over a decorrelated pool
— that beats naive and climatological benchmarks on strict walk-forward
evaluation and is calibrated near nominal (0.795 vs 0.80 coverage); (ii) a
parallel forecaster for the D-1 aFRR balancing-**capacity** price at its actual
bid gate, 16–18% better MAE than persistence, whose intervals we widen to
near-nominal coverage with rolling split-conformal recalibration; (iii)
dedicated tail classifiers that make CEN price spikes 18× more findable than
climatology and negative prices strongly separable (AUC 0.94); and (iv) an
executability-honest map of *where the forecast pays* — not in financial spreads
(every day-ahead/intraday/imbalance structure we test is dead once entry
mechanics and costs are enforced), but in **physical dispatch**, where
forecast-timed battery state-of-charge recovery adds a small but
positive-every-quarter margin. Throughout, signals are gated on real publication
timestamps (the CEN publishes D+1 ≈ 14:00 — the single largest leakage trap in
this market), no model trains across the reform break, and no trading verdict is
issued without a by-quarter stability check. A live collector and a daily
gate-stamped *shadow forecaster*, running since 2026-07, provide prospective
out-of-sample validation.

---

## 1. Introduction

Electricity price forecasting (EPF) is a mature field for day-ahead auction
prices, with an open benchmark and reference models since Lago, Marcjasz, De
Schutter & Weron (2021). The *balancing* (imbalance) price — the price at which a
system operator settles real-time deviations — is a younger target, and the
literature that exists sits mostly on German data (Narajewski 2022). Two
structural shifts make this an unusually good moment to study balancing prices:
Europe is converging on **single-price** imbalance settlement at **15-minute**
resolution, and platform integration (PICASSO for aFRR, MARI for mFRR) is
reshaping how balancing energy is priced.

Poland implemented exactly this design on **2024-06-14**: a single imbalance
price CEN, 15-minute settlement, under central dispatch. That makes the Polish
post-reform market an early, clean instance of the design the rest of Europe is
moving toward — and, as of this writing, an un-studied one. The 2026 European
review of imbalance-price forecasting (arXiv:2605.17054) surveys the field and
notes the EU-wide shift to single-price 15-minute settlement, but no forecasting
or trading study of post-reform Polish CEN has appeared.

This paper fills that gap and makes four contributions:

1. **A calibrated probabilistic CEN forecaster** (§5). A quantile GBM ensemble
   with an added LEAR member, distribution-averaged; walk-forward mean pinball
   62.06 vs 63.63 for the GBM alone, winning in all six evaluated quarters, at
   near-nominal coverage.

2. **A capacity-price forecaster at the true bid gate, with conformal
   calibration** (§6). The D-1 aFRR capacity price — the dominant revenue leg for
   storage — is forecastable 16–18% better than persistence from a *harder*
   information set than the energy price; its too-narrow intervals are fixed to
   near-nominal coverage by rolling split-conformal recalibration, and we report
   honestly where conformal breaks (regime changes violate exchangeability).

3. **Tail classifiers for spikes and negative prices** (§7). The tails are where
   the quantile model family is weakest; dedicated binary classifiers make
   spikes 18× more findable than climatology (71% captured in the top predicted
   decile) and negative prices strongly separable (AUC 0.94), with walk-forward
   isotonic calibration bringing negative-price probabilities to dispatch grade.

4. **An executability-honest value map** (§8). We test every tradeable spread
   around CEN (day-ahead, IDA1–3, imbalance) with publication-timestamp-honest
   signals and costs on both legs, and find them all dead once real entry
   mechanics are enforced — the public scheduled information is priced by the
   first accessible auction. The forecast's value is therefore **physical**:
   forecast-timed battery SoC recovery adds a positive-every-quarter margin. This
   redirection — from financial spread to physical dispatch — is the paper's
   sharpest empirical point.

A recurring methodological theme (§4, and stated as explicit rules) is
*executability honesty*: publication-timestamp gating, no cross-reform training,
costs on both legs, and by-quarter stability as a precondition for any positive
claim. Several once-promising results in the underlying research log were
retracted by their own audits; we keep the retractions, because in this market
the negative results are as load-bearing as the positive ones.

---

## 2. The post-reform Polish balancing market

*[TODO: tighten with 2–3 primary citations to PSE / URE reform documentation.]*

**The reform.** Before 2024-06-14, Poland settled imbalances with a dual-price,
hourly mechanism. The reform introduced:

- a **single imbalance price** (CEN) applied to both directions of imbalance;
- **15-minute** settlement resolution (the imbalance settlement period);
- pricing under **central dispatch**, where the operator (PSE) activates
  balancing energy from a merit-order ladder of offers.

**How CEN is formed.** CEN is built from the marginal cost of activated
balancing energy plus components (CKOEB / aFRR). We verify this construction
empirically: sorting the operator's balancing offer ladder by price, cumulating
MW, and crossing at the period's activated volume yields a *marginal activated
price* that correlates ≈ 0.88 with realized CEN (0.89 in up-regulation periods),
consistent with CEN being a marginal-cost object (F4). Under central dispatch
both directions activate in nearly every period, so the natural forecasting
object is the **marginal-price distribution by direction and hour block**, not a
binary "will it activate" event.

**Publication timing — the central leakage trap.** Settled CEN is published
**D+1 at ≈ 14:00**. Any feature or benchmark that uses a settled CEN value must
respect this: at a decision time *t*, the freshest *settled* CEN day is D-3 for a
D-1 gate. Naively lagging by one day leaks ~24 h of unpublished prices and
inflates every metric. This single fact governs the entire feature panel (§4).

**The information cascade.** The tradeable-price cascade for delivery day D runs:
day-ahead auction (SDAC, gates ≈ noon D-1) → intraday auctions IDA1 (15:00 D-1),
IDA2 (22:00 D-1), IDA3 (10:00 D) → continuous intraday → real-time balancing →
settled CEN (D+1 14:00). The D-1 aFRR balancing-**capacity** auction clears ≈
09:10 D-1 (so its bid gate is ≈ 07:30 D-1). Each forecaster in this paper is
pinned to a specific gate in this cascade, and its information set is exactly
what is public before that gate.

**Why 15-minute single-price matters for storage.** Under single-price
settlement a physical asset that passively balances settles its deviation at CEN
against its own near-zero marginal cost. This is what makes a CEN forecast a
*dispatch* input for batteries and flexible load even when it has no financial
spread value (§8) — the asset does not trade the spread, it settles into it.

---

## 3. Related work

*(Condensed from `LITERATURE.md`; see that file for the full survey and links.)*

The electricity-trading literature splits cleanly by **data access**:

**Day-ahead EPF — free data, open code.** The Weron/Wrocław school runs on free
day-ahead auction data and, since Lago et al. (2021), a shared open benchmark
(`epftoolbox`: 5 markets, LEAR + DNN reference models, GW-test evaluation).
Distributional NN forecasts (Marcjasz, Narajewski, Weron & Ziel 2023),
calibration-window averaging (Serafin, Uniejewski & Weron 2019), and
forecast-averaging results (Lipiecki/Uniejewski/Weron 2024) all sit here. We use
this school's tools directly: a LEAR benchmark and Vincentized distribution
averaging (§5).

**Continuous intraday & order-book — paid EPEX data.** The intraday-microstructure
literature (Narajewski & Ziel; Hirsch & Ziel 2024; Kath & Ziel; OrderFusion and
the recent LOB/RL papers) stands almost entirely on licensed EPEX Spot
transaction / M7 order-book data, which cannot be shared — so that half of the
field has little reproducible code+data. Because Poland's IDA1–3 are **auctions**,
TGE publishes their clearing prices for free; our executable-spread tests
(§8) therefore sit on the *free* side of a line the German intraday literature
cannot cross without a licence.

**Imbalance/balancing — free data (our neighborhood).** Narajewski (2022) is the
nearest methodological neighbor (probabilistic German imbalance-price
forecasting on free data). Conformal prediction has been applied to
balancing-market prices (arXiv:2502.04935); we adopt it in §6. Risk-constrained
trading under single-price balancing (Boomsma/Morales-style, arXiv:1708.02625)
frames the "trade the imbalance exposure" question we test honestly in §8. The
2026 European review (arXiv:2605.17054) confirms the field is young and that
single-price 15-minute settlement is where Europe is heading.

**Our position.** Free-data, executability-guarded, by-quarter-verified, on the
post-2024 single-price Polish CEN — a market with, as far as we can find, no
prior published forecasting or trading study. Methodologically the study is
*stricter* than most of the intraday-trading literature: its trading claims
rarely survive a "two real prints + costs on both legs + quarterly stability"
filter, which we impose throughout.

---

## 4. Data and leakage-honest construction

**Sources (all free).** PSE v2 API (~20 report types incl. the full balancing
offer ladder — merit-order supply curves, ~67M rows, the free cousin of the
licensed order books the German literature uses); TGE day-ahead and IDA1–3
auction results (scraped); ENTSO-E Transparency (load, RES, cross-border);
Open-Meteo forecast weather **including archived past model runs**, from which we
reconstruct forecast revisions Poland does not publish (§8, F26). The full panel
spans reform → present at 15-minute resolution (~153M rows).

**Leakage-honest feature panel.** Features are constructed to be legal at a named
gate by construction, not by after-the-fact filtering:

- **Publication-timestamp gating.** Every settled-CEN feature respects the D+1
  14:00 publication; at a D-1 gate the freshest settled CEN is D-3.
- **No cross-reform training.** Models do not train across 2024-06-14; the price
  process changed, and pooling the two regimes leaks the wrong dynamics.
- **Gate-pinned information sets.** The CEN forecaster (§5) issues at a 60-minute
  horizon; the capacity forecaster (§6) issues at the D-1 07:30 bid gate with no
  DA anchor and only D-3 settled CEN — a strictly harder information set.
- **Costs on both legs; by-quarter stability.** No spread verdict without
  transaction costs on both legs of the structure and a by-quarter breakdown
  (§8).

**Live vintage capture.** Grid data is silently revised intraday (measured, not
assumed — 4–5 revisions per value). A launchd-scheduled collector archives 15-min
snapshots of nine PSE feeds *with capture timestamps* since 2026-07-08, so that
prospective evaluation (§9) uses the data as it actually appeared at decision
time, not the settled series. [FIG: capture-timestamp revision magnitude — from
the live collector once the 2-week audit (≈2026-07-22) completes.]

---

## 5. Probabilistic CEN forecasting

**Target and horizon.** 15-minute CEN, evaluated on strict walk-forward
out-of-sample predictions (expanding windows; no cross-reform training). The
production forecaster issues at a 60-minute horizon, which makes the dispatch
timing window of §8 gate-honest by construction.

**Model.** A quantile gradient-boosting model (GBM) over the leakage-honest
panel, post-processed in the Weron tradition: quantile regression averaging
(QRA), binned isotonic distributional regression (IDR), and a **Vincentized
(quantile-wise) average** of the pool. We add a walk-forward **LEAR** (LASSO
Estimated AutoRegressive) member — worse alone than the GBM but *decorrelated*,
so distribution-averaging turns its diversity into gain (F3, F17).

**Results (walk-forward OOS, 44,335 predictions, 2025-02 … 2026-05; mean
pinball loss, lower is better).**

| Model | Mean pinball | Notes |
|---|---:|---|
| IDR (binned) | 66.30 | |
| LEAR (light) | 65.48 | worse alone, but decorrelated |
| GBM | 63.63 | strong single learner |
| QRA | 63.31 | |
| Ave-Q (GBM+QRA+IDR) | 62.80 | averaging beats components |
| **Ave-Q + LEAR** | **62.06** | best in repo; wins all 6 quarters |

The averaged ensemble improves on the GBM by 1.57 pinball and on Ave-Q by 0.74,
and wins in **all six evaluated quarters** (quarterly pinball 50.6 / 61.0 / 69.6 /
64.5 / 54.9 / 73.9). Coverage of the 10–90 interval is **0.795** against a
nominal 0.80 — near-nominal without recalibration, in contrast to the capacity
forecaster of §6.

**Reading.** The durable forecast edge in this market is *model-averaging a
decorrelated pool*, worth ≈1.5 pinball over the GBM; the base learner is
close to irrelevant, VST (variance-stabilizing transform) is a wash-to-harmful,
and day-ahead fundamentals are already priced in (an oracle with perfect
realized-RES knowledge caps its own pinball improvement at ≈1.2 — the RES→price
channel is efficiently priced). This is a EPF-school result reproduced on a new,
post-reform, 15-minute balancing target.

[FIG 1] prediction intervals over a sample window (`reports/figs/fig1_intervals.png`).
[FIG 2] reliability / calibration diagram (`fig2_calibration.png`).
[FIG 3] pinball by hour of day (`fig3_pinball_by_hour.png`).
[FIG 4] mean CEN day-shape (`fig4_cen_day_shape.png`).

---

## 6. Forecasting the D-1 capacity price, and conformal recalibration

**Why the capacity price.** For a battery, the dominant revenue leg is not energy
arbitrage but **aFRR balancing-capacity** payments (§8; ≈2.6–2.7M PLN/MW/yr vs
tens of thousands elsewhere). The *actual* daily decision behind that leg is the
bid into the **D-1 balancing-capacity auction** (CMBP). We forecast the CMBP
clearing price for both products — aFRR up (`afrr_g`) and aFRR down (`afrr_d`) —
at the true bid gate.

**A harder information set.** CMBP clears ≈ 09:10 D-1, so we assume a **07:30 D-1**
bid gate: no day-ahead anchor (SDAC clears 13:50 D-1, *after* the gate), freshest
settled CEN at D-3, and only lagged capacity-demand publications. Features: CMBP
and demand same-hour lags (1/2/7 d) + D-1 day shape, CEN D-3 history + 7-day
spike count, a 48-h-ahead NWP run (gate-legal by construction), and calendar.
Pooled per-quantile LightGBM, expanding monthly walk-forward 2025-01 … 2026-07
(n = 13,223 hourly, both products).

**Point-forecast accuracy (walk-forward OOS).**

| Product | GBM MAE | naive-1d | Δ | GBM pinball | climatology | Δ | Quarters won |
|---|---:|---:|---:|---:|---:|---:|---:|
| aFRR up (`afrr_g`) | 62.5 | 74.0 | **−16%** | 23.3 | 28.9 | −19% | 6/7 (Q3'26 partial ties) |
| aFRR down (`afrr_d`) | 67.6 | 82.1 | **−18%** | 27.4 | 36.7 | −25% | **7/7** |

The down product is beaten in every quarter including the 2026Q2 blowout (159 vs
naive 166 — everyone is bad, the GBM less so).

**Coverage problem.** The raw quantiles are **too narrow** on this spiky series:
empirical 10–90 coverage 0.59–0.61 against nominal 0.80. This is the opposite of
the CEN energy forecaster (§5, 0.795), and it must be fixed before any
calibration-sensitive use.

**Rolling split-conformal recalibration (F30).** We apply split-conformal
widening on trailing-90-day out-of-sample residuals, grouped by hour-block, with
bid-gate-legal timing:

| Product | Coverage before → after (nom. 0.80) | Pinball before → after |
|---|---|---|
| aFRR up (`afrr_g`) | 0.608 → **0.774** | 23.31 → 23.26 (flat) — textbook |
| aFRR down (`afrr_d`) | 0.594 → **0.728** | 27.4 → 28.4 (+4%) — regime-limited |

For the up product this is textbook: coverage moves to near-nominal at no pinball
cost. For the down product, coverage improves but pinball worsens ~4%, because
the 2026Q2 regime break **violates exchangeability** — trailing calibration
mis-adapts around regime changes. A window sweep {30, 60, 90} days shows coverage
monotone in window length (longer memory averages over regimes), which is why we
choose 90 days; but the honest conclusion is that **regime breaks are conformal's
limit**, not a bug to tune away. [FIG: coverage vs conformal window, both
products.]

---

## 7. Tail classifiers: spikes and negative prices

**Motivation.** The tails are where a quantile model family is structurally
weakest and where a dispatch operator most needs a signal (hold state-of-charge
before a spike, pre-charge before negative prices). We train dedicated binary
classifiers at the 60-minute horizon rather than reading tails off the quantile
forecaster.

**Setup.** Binary LightGBM, walk-forward with monthly refits (2025-01 →), on the
strict H=60 panel plus the reconstructed NWP run-to-run revisions (§8, F26 —
legal at a 60-minute gate even for the freshest run: the signal that failed as a
*trade* finds its home as a *feature*). Benchmark: trailing-90-day
hour-of-day climatology lagged 2 days.

| Target | Base rate | AUC | Avg. precision | Climatology AP | Lift | Quarters won |
|---|---:|---:|---:|---:|---:|---:|
| **Spike** (CEN > 1500) | 0.75% | 0.853 | 0.464 | 0.025 | **18×** | 5/5 (with events) |
| **Negative** (CEN < 0) | 7.6% | 0.935 | 0.542 | 0.269 | 2.0× | 7/7 |

**71% of all spikes fall in the top decile** of predicted risk — a directly
usable operating point for dispatch and risk alerting.

**Calibration.** Walk-forward isotonic calibration (month *m* calibrated only on
OOS pairs from months < *m*):

- **Negatives:** Brier 0.0533 → 0.0475, ECE 0.038 → **0.011** — absolute
  probabilities now dispatch-grade (usable as thresholds, e.g. "pre-charge when
  P(neg) > τ").
- **Spikes:** no improvement (Brier 0.00528 → 0.00543; ECE already 0.0025). With
  only ≈300 spike events in-sample, isotonic cannot beat the raw scale. Use spike
  scores for **ranking** (top-decile rules), negative scores for **absolute
  thresholds**.

**Caveat.** 2026Q1 has a single spike (quarter unevaluable for spikes).

---

## 8. Where the forecast pays: physical dispatch, not financial spreads

This section is the paper's empirical hinge: a good CEN forecast has **no
capturable financial-spread value** in this market, and real **physical-dispatch
value**. We show both sides honestly.

### 8.1 Every tradeable spread is dead once entry mechanics are enforced

We test the full spread lattice around CEN — day-ahead, IDA1/2/3, imbalance —
with signals gated on publication timestamps, costs on both legs, and by-quarter
verdicts.

- **DA↔CEN (F2, the flagship natural experiment).** A threshold rule on the
  CEN−DA spread earned +41…65 PLN/MWh/quarter at a 54% hit rate — **until exactly
  2025-09-30**, when SDAC moved to 15-minute products and the intra-hour shape
  arbitrage it harvested closed. Negative on the holdout: regime death, not
  overfit. A clean natural experiment on market microstructure.

- **The edge is vs day-ahead, but day-ahead is not an available entry (F24).**
  The signals fire ≈2.5 h before delivery; the day-ahead auction closed a day
  earlier. Re-based on the *executable* intraday entry, the edge collapses:
  Sharpe 3.0 (vs DA) → **0.7 (vs intraday) at cost 20**. By the time you can act,
  the intraday market has already priced the same real-time information. A
  cleaner fixed-gate test (enter at the IDA3 clearing price, morning RES surprise
  known before the 10:00 gate, settle at CEN) gives Sharpe **0.9 @cost10, −0.4
  @cost20** — not tradeable — with corr(morning surprise, CEN−IDA3) = −0.026 ≈ 0.

- **Fully-executable auction-to-auction spreads (F25).** Enter one IDA auction,
  exit a later one — two real prints, costs on both legs, signal strictly
  pre-dating the entry gate. Dead across IDA1→2, IDA1→3, IDA2→3.

- **The one real signal, honestly killed (F26).** From archived consecutive NWP
  runs we manufacture the intraday RES-forecast revision Poland does not publish.
  It is real and right-signed (corr(revision, IDA3−IDA2) = −0.072 to −0.109,
  fresher revision → stronger read) and the strict IDA2→IDA3 trade shows Sharpe
  1.68–1.94 at a realistic 5-PLN round-trip. **But the P&L concentrates in 2 of 7
  quarters** — by the by-quarter rule it is a *feature*, not a book. It re-enters
  as a feature in the tail classifiers (§7) and the dispatch tilt below.

**Interpretation.** Public scheduled information is priced by the first accessible
auction; by the time a physically-executable gate opens, the balancing spread
holds nothing public. This is the efficient-market wall, confirmed from several
independent directions — and it is *why* the forecast's value must be physical.

### 8.2 The physical-dispatch value: forecast-timed SoC recovery

A 1 MW / 2 MWh battery nets ≈ **2.95M PLN/MW/yr**, ≈90% of it from aFRR
**capacity** (through-cycle ≈3M gross, range 2.2–4.2M; energy arbitrage a rising
0.2–0.56M floor) — F16. Because a battery passively settles its deviation at CEN
against ≈zero marginal cost, the CEN forecast is a genuine dispatch input even
though it is not a tradeable spread (F24).

**Price-aware SoC recovery (F27).** A price-blind stack pays −668k PLN/MW/yr for
state-of-charge recovery (it refills the instant the band is crossed). The 60-min
forecast makes a recovery *timing window* gate-honest by construction (at
decision time *t*, forecasts for deliveries *t…t+3* were all issued ≤ *t*). The
policy defers recovery to the best forecast-median quarter within the window,
cancels if activation drift re-enters the band, and emergency-executes near
physical limits.

- **Window sweep {2,3,4} picks 30 min** — a *mechanism*, not tuning: waiting
  out-of-band lets activation drift the SoC toward physical limits and forfeits
  capacity offers, and that cost grows faster than the timing gain.
- **Recovery leg −668k → −553k (forecast), −422k (oracle):** the real forecast
  captures **46%** of the oracle's timing value on the leg.
- **Net stack +37.9k PLN/MW/yr (+1.3%)**, positive in **all 5 quarters** — the
  first forecaster monetization in the program that passes the by-quarter rule.
  It is a bid/dispatch decision, not a spread, so the §8.1 executability
  guardrail is satisfied trivially.
- **Honest headline: the −668k drag is ≈80% structural.** Even perfect 1-h-window
  timing recovers only 37% of it; the rest is the unavoidable cost of buying back
  energy at CEN plus degradation. Better forecasts move the needle by at most
  ≈95k/yr more — real but second-order against the 2.7M capacity leg.

**Do the tail classifiers (§7) add recovery-timing value? No (F32).** We wired
the spike/negative classifiers into the recovery selection — discharge into a
predicted spike, charge into a predicted-negative quarter, via an expected-value
blend of the median forecast and the calibrated tail probability. The best
configuration over a window × selection-level sweep beats F27's plain median
timing by **+0.1%** (≈3k PLN/MW/yr), and the sign flips with the
negative-selection level — tuning-fragile, within single-path noise. The reason
is structural: a *forced* recovery rarely coincides with a tail quarter inside a
gate-honest ≤60-minute window, and where the median already picks the best
in-window quarter the tail read adds no ordering. The tail classifiers' dispatch
value therefore stays where §7 put it — *positioning and alerting* (pre-position
SoC ahead of predicted tail windows, a policy-level lever) — not retiming an
already-forced recovery.

**The commitment decision needs no forecast — yet (F29).** Co-optimizing each
hour between committing to aFRR standby (using the §6 capacity forecast) and
staying free for CEN energy arbitrage, **always-commit wins**: capacity fees
dominate hourly arbitrage value **15×** (mean fee 285 vs mean EV 19 PLN/MW/h; fee
p10 sits above EV p99). Only 0.4% of hours would flip even with a *perfect* fee
forecast; fees would need to fall ≈4× before 5% of hours flip. The machinery and
the flip threshold are in place for when platform-driven fee deflation compresses
the margin — a negative result with a quantified expiry date.

**Net value map.** The CEN energy forecaster's battery value lives in *recovery
timing* (§8.2); the capacity forecaster's value lives in *revenue projection and
calibration-sensitive uses* (§6); neither has capturable financial-spread value
(§8.1). This redirection is the study's central practical finding.

---

## 9. Live shadow validation (in progress)

Backtests on settled data can flatter a forecaster that would not survive live
vintage risk. Since 2026-07 we run two prospective checks on this Mac:

- **A live vintage collector** archiving 15-min snapshots of nine PSE feeds with
  capture timestamps (since 2026-07-08), to measure preliminary-vs-settled
  revisions directly and re-score the forecasters on data as it actually appeared.
- **A daily shadow CEN/capacity forecaster** issuing **gate-stamped** forecasts
  before publication. Early track record (first gate-honest issuance 2026-07-09):
  over the first five issued days the shadow's MAE fell to **43.2** against a
  naive-1-day benchmark of 48.5, with gate-honesty (`gate_ok`) climbing toward
  100% as pre-gate-fix rows age out.

A **2-week vintage audit** (≈2026-07-22) will quantify the live-vs-backtest gap
and score PSE's own published `price_fcst` against settled CEN — the benchmark a
live forecaster must beat. A first ~30-day gate-honest shadow read is due ≈
2026-08-10. [TODO: fold both into a "prospective validation" table at revision
time.]

---

## 10. Discussion and limitations

- **Single realized path.** The BESS results (§8.2) are on one realized price
  path; window choices are argued from mechanism, not cross-validated across
  paths. Re-sweep as data accrues.
- **Assumed bid-gate time.** The 07:30 D-1 capacity bid gate (§6) is an
  assumption to verify against PSE balancing-market rules.
- **Conformal and regime breaks.** Split-conformal assumes exchangeability;
  §6 shows it mis-adapts around the 2026Q2 regime change. This is a known,
  reported limit, not a fixable hyperparameter.
- **Spike sample size.** ≈300 spike events in-sample; spike probabilities are
  usable for ranking, not (yet) as calibrated absolute thresholds (§7).
- **Live risk is being measured, not assumed** (§9) — the honest resolution of
  the backtest-vs-live gap awaits the vintage audit.

---

## 11. Conclusion

Poland's 2024-06-14 move to a single, 15-minute imbalance price created an early,
clean instance of the design toward which EU balancing markets are converging —
and, until now, an un-studied forecasting target. On free public data we build a
calibrated probabilistic CEN forecaster, a bid-gate-honest capacity-price
forecaster fixed to near-nominal coverage by conformal recalibration, and
dedicated tail classifiers that make spikes 18× more findable than climatology.
Enforcing real entry mechanics, every financial spread around CEN is dead; the
forecast's value is physical, and forecast-timed battery dispatch monetizes it
with a small but positive-every-quarter margin. Live prospective validation is
under way. The broader methodological point — publication-timestamp gating, no
cross-reform training, costs on both legs, by-quarter stability — is what
separates the survivable claims from the flattering ones in a market this young.

---

## References

*[TODO: convert to BibTeX at LaTeX port. Anchor set from `LITERATURE.md`.]*

- Lago, Marcjasz, De Schutter, Weron (2021). Forecasting day-ahead electricity
  prices: review + open-access benchmark. *Applied Energy*. (`epftoolbox`.)
- Marcjasz, Narajewski, Weron, Ziel (2023). Distributional neural networks for
  EPF. *Energy Economics*. arXiv:2207.02832.
- Uniejewski, Marcjasz, Weron (2019). Understanding intraday electricity markets:
  LASSO.
- Serafin, Uniejewski, Weron (2019). Calibration-window averaging (Ave-*).
- Lipiecki, Uniejewski, Weron (2024). Forecast averaging beats components.
- Narajewski (2022). Probabilistic forecasting of German imbalance prices.
  arXiv:2205.11439.
- Conformal prediction for day-ahead & real-time balancing prices.
  arXiv:2502.04935.
- Risk-constrained trading under single-price balancing. arXiv:1708.02625.
- 2026 European review of imbalance-price forecasting algorithms.
  arXiv:2605.17054.
- Narajewski & Ziel (2019/20); Hirsch & Ziel (2024); Kath & Ziel — continuous
  intraday (licensed EPEX data).

---

## Appendix A. Mapping to the research log

| Paper section | Findings | Code modules |
|---|---|---|
| §2 market / CEN construction | F4 | `activation` curves |
| §5 CEN forecaster | F3, F17 | `models.py`, `lear.py`, `postprocess.py` |
| §6 capacity forecaster + conformal | F28, F30 | `cmbp_forecast.py`, `cmbp_conformal.py` |
| §7 tail classifiers | F31 | `spike_classifier.py` |
| §8.1 spreads dead | F2, F24, F25, F26 | `ida_term.py`, `nwp_revision.py` |
| §8.2 dispatch value | F16, F27, F29, F32 | `bess_soc_policy.py`, `bess_soc_tilt.py`, `bess_commit.py` |
| §9 live validation | — | `live_collector.py`, `shadow_cmbp.py` |
