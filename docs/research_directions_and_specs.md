# Research directions for pl-cen-forecaster — market making / trend / physical

Compiled from recent papers + market news (mid-2026), organized by direction.
Two of these (marked ★) were written up as full Claude-Code-ready specs and
are included in full below.

---

## Market making / stochastic control

**1. AS/GLFT with a stochastic terminal price.** Continuous intraday price
forecasting research is thin compared to day-ahead, and almost all of it is
German — nobody has published the market-making problem where the terminal
inventory penalty is replaced by liquidation at a forecastable imbalance
price with regime structure. The CEN distribution itself supports this: 42%
of settlement periods fall in a concentrated 400–600 PLN/MWh core, but with
pronounced tails in both directions (thousands of negative periods, hundreds
above 1,000 PLN/MWh), and 24% of midday periods are negative vs. 0.3% by
20:00 — a mixture model with time-of-day-dependent mixing weights. Maps
directly onto your thesis's OU-drift/Kalman + regime-switching machinery.

**2. The weak-form efficiency fight, replayed on a thin market.** Live
academic dispute: early work claimed the last traded price already reflects
all available information (weak-form efficient), with fundamentals adding
little; recent order-book-encoding models (OrderFusion) show CID markets do
NOT exhibit perfect weak-form efficiency, and that book-derived features plus
neighboring-product overlap are the most influential predictors — all on
German/Austrian data. A DE → AT → PL comparison (does efficiency weaken as
liquidity thins?) is an open, publishable question.

**3. Queue-position value in a mostly-empty book.** When depth is one or two
orders, queue priority *is* the fill model. An empirical piece measuring the
value of queue position vs. book depth across DE/PL connects your thesis's
zero-profit-equilibrium result to a market where that equilibrium visibly
hasn't arrived yet.

## Trend / systematic / stat-arb

**4. Trade the path, not the point.** Current frontier is trajectory/scenario
forecasting — predicting a full price path and generating scenarios from the
dynamics of fundamental forecast errors — already used to time renewable
volume sales profitably. Your CEN quantile forecaster extends naturally into
a path-scenario generator with a trading overlay.

**5. ★ The RDB↔CEN convergence trade, tested honestly.** — see full spec below.

**6. Cross-market spread: PICASSO leakage.** Poland's integration into the
PICASSO aFRR platform expanded cross-border participation, increasing price
dispersion and less predictable clearing outcomes. German aFRR conditions
leading Polish CEN is a brand-new, unpublished transmission channel.

## Physical / asset-backed

**7. ★ BESS optimization under central dispatch (PSE/ZPG).** — see full spec
below. The hottest commercial question in the market right now: Aurora found
Polish batteries would have earned some of the strongest revenues in Europe
(June 2024–Sept 2025) with ~6–8 month payback, yet as of April 2026 no
utility-scale battery is yet a fully qualified BSP — the first mover
(Nowe Czarnowo/Axpo) is still stuck in qualification. Pipeline: 89 projects,
12.5 GW, heading to 8–9 GW operational by 2030 from 28 MW today.

**8. The revenue-stack decay model.** Consensus direction (AS-led stack will
saturate as BESS capacity grows, shifting value to wholesale arbitrage) but no
good *rate* model exists. A cannibalization model — AS clearing price as a
function of cumulative BESS commissioning, calibrated on PSE's own reserve-
price series — is valuable to any fund or developer and a strong interview
artifact for shops like Respect Energy.

**Recommendation given your positioning:** #7 is the strongest career play
(thesis toolkit + physical-commodity intuition + where Polish capital is
flowing right now). #5 is the fastest to execute on top of the existing repo.
#2 is the most publishable alongside the MSc thesis. #1 is the deepest —
effectively a second thesis chapter.

---

# ★ SPEC — Project A: Is the intraday↔imbalance (RDB↔CEN) spread predictable in Poland?

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

---

# ★ SPEC — Project B: BESS offer-curve design under central dispatch (PSE/ZPG)

Standalone research build (new repo: `pl-bess-zpg`). Reuses the pl-cen-forecaster
data layer as a dependency. This is the deep project: budget ~2 months of
part-time work, structured so each milestone is independently presentable.

## Problem statement
In self-dispatch markets (GB/DE), a battery trades against prices directly, and
the state of the art is rolling-intrinsic / joint ID+FCR bidding solved by DP.
In Poland the battery does NOT choose when it runs: it submits offers into
PSE's Integrated Scheduling Process (ZPG) — typically under limited dispatch
(ZAK=2), exposing chosen capacity per period — and PSE decides activation.
Revenue is therefore a function of the OFFER CURVE, filtered through the
dispatcher's response. The open problem: choose offer curves (and the split of
capacity across CM / FCR / aFRR / mFRR / energy) to maximize expected revenue
net of degradation, subject to SoC dynamics, given a *model of PSE's
activation behavior*.

Formally, per 15-min period t and product m, the asset submits price–quantity
pairs {(p_i, q_i)}. Activation is a random variable A_t^m in [0, q] with
distribution depending on the offer price relative to the marginal system
price and system state x_t (imbalance direction, surplus, RES error, net
xborder). Objective:

    max_{offers}  E[ sum_t sum_m  R_m(A_t^m, prices_t)  -  c_deg(|A_t^energy|) ]
    s.t.  SoC_{t+1} = SoC_t + eta_c * charge_t - discharge_t / eta_d
          SoC bounds, power bounds, product exclusivity/headroom rules,
          CM availability obligations during stress events.

## The three-layer build (each layer is a deliverable)

### Layer 1 — Activation model (empirical, ~3 weeks)
Estimate P(activation | offer price, product, system state) from public PSE
data: CEN, activated balancing energy volumes & prices per direction,
FCR/aFRR/mFRR clearing prices (basic + supplementary), demand/RES forecast
errors, surplus (PK5L), xborder flows. Since individual offers aren't public,
identify the *marginal activated price* per period per product and model the
asset as activated iff offered below it (discriminatory-style approximation;
state the caveat). Deliverable: calibrated activation curves
pi_m(p | x_t) with time-of-day structure, plus a writeup of how PICASSO
coupling changes aFRR activation dispersion. This layer alone is a strong
interview artifact: it is a *statistical model of PSE's dispatcher*.

### Layer 2 — Offer optimizer (stochastic DP, ~3 weeks)
State: (t, SoC, x_t summarized to a small regime variable — e.g. 3–4 system
states from a fitted HMM on imbalance direction/magnitude). Action: offer
prices per product from a discrete grid + capacity allocation across products.
Transition: activation draws from Layer 1; SoC update; degradation cost
(throughput-based, PLN/MWh-cycled — put it in config, sensitivity at 0 and 2x).
Solve by backward induction over the day (96 periods); warm-start next day.
This is the rolling-intrinsic idea transplanted: RI's "trade if spread >
threshold" becomes "offer at price p* where marginal activation probability x
margin = marginal opportunity cost of SoC". Deliverable: policy heatmaps
(offer price vs t, SoC, regime) + expected revenue.

### Layer 3 — Revenue stack & decay scenarios (~2 weeks)
Stack CM (de-rated, PLN/kW from latest auction), AS capacity payments, and
Layer-2 energy/balancing revenue for a reference 1 MW / 2 MWh and 1 MW / 4 MWh
asset. Then the cannibalization overlay: AS clearing price as a declining
function of cumulative commissioned BESS (calibrate the level today; scenario
the slope using the announced pipeline: ~1.3 GW by end-2026, 8–9 GW by 2030).
Deliverable: NPV bands per duration under 3 saturation scenarios; the
crossover date where arbitrage overtakes AS in the stack.

## Validation & honesty
- Backtest Layer 2 policy on held-out months: simulated revenue vs (a) perfect-
  foresight upper bound, (b) naive "always offer at CEN median" lower bound.
- The activation model is the weak link: report revenue sensitivity to +/-20%
  miscalibration of pi. If revenue rankings flip under that perturbation, say
  so prominently.
- Regime discipline as in the forecaster: post-2024-06-14 only; PICASSO
  accession date as a second break for aFRR series.

## Why this project (positioning)
- Methodologically novel: offer design under a dispatcher's response is absent
  from the BESS-trading literature, which assumes self-dispatch.
- Commercially timed: multi-GW Polish pipeline, first assets only now reaching
  BSP qualification, and the trading houses doing offer optimization for
  developers are hiring exactly this skill set.
- Personally leveraged: it is stochastic control + calibration + market
  microstructure — the thesis toolkit pointed at a physical asset.
