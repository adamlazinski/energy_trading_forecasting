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

### F19. RES-surprise persistence: the first tradeable forecast edge (intraday-weather, reframed)
`src/res_surprise.py` — the user's "sudden live-forecast changes" instinct,
made concrete and POSITIVE. F10/F15 killed the day-ahead RES *forecast* (it's
in the DA price); this trades the forecast *error*.
- **Mechanism (ceiling)**: the RES surprise (realized − day-ahead consensus)
  moves CEN vs DA — slope **−56 PLN/MWh per GW**, and gets the *sign* right
  **71%** of the time (more RES than expected ⇒ CEN below DA). Weaker on the
  continuous intraday (55% sign) — balancing is far more imbalance-sensitive.
- **Why it's tradeable without forecasting the surprise**: the surprise is
  strongly persistent — autocorr **0.90 @1h, 0.76 @2h, 0.62 day-level**. So
  the surprise you've *already observed* predicts the upcoming move almost as
  well as the contemporaneous one; you ride the persistent bias.
- **Deployable**: ENTSO-E actual PL wind/solar publishes at **~1.2h latency**
  (verified live), so at the gate (t−60min) you know the surprise ~2–2.5h
  back — the strong regime.
- **Gate-honest P&L** (short CEN−DA when RES over-delivers, deadband 300 MW):
  at **2.5h lag, cost 20 PLN/MWh: +24/trade, 9/10 quarters positive,
  ~+984k** over 2 yr; at 2h lag, +31/trade, 10/10 quarters. Degrades with
  latency (4h ≈ marginal) and cost (dies above ~40 PLN/MWh).
- Caveats, load-bearing: (1) **hit rate ~50%** — profit is payoff-skew, not
  frequency, so tails dominate and realized P&L is lumpy → needs strict
  sizing; (2) the instrument is CEN−DA, i.e. **passive-balancing / imbalance
  positioning** — needs BRP status or spread access, and F6 warns spread
  regimes can shift (though this held every quarter incl. 2026); (3) the
  backtest sourced the surprise from PSE actuals (3–4 d lag) — a live system
  must use the ENTSO-E actual-gen feed (~1.2h) with forecast/actual from the
  *same* provider to avoid the F15 scale bias; (4) cost-sensitive.

First genuine forecast edge in the project, and it *redeems the weather
thread*: not by beating the market's forecast (F15 says we can't), but by
being **fast on the realized error** — the one near-delivery signal the
efficient day-ahead price cannot contain. Live weather (the original idea)
would be an even-earlier, independent read that could push latency below the
1.2h actuals feed — the natural enhancement.

### F18. Co-optimization: the SoC-feasibility haircut is ~6% — F16 gross is broadly achievable
`src/bess_cooptimize.py` — turns F8/F16's GROSS capacity revenue into a NET
one by simulating the SoC coupling that reserving-then-getting-activated
imposes. Single realized-data path, 2025-07→2026-07 (360 days, the aFRR
activation span), 1 MW/2 MWh: offer up-capacity when SoC ≥ 0.4 MWh and down
when SoC ≤ E−0.4; move SoC by realized activation (eb-rozl/zmb utilization,
~13–14% each way); recover to a [0.7,1.3] band via CEN trades.
- **Capacity gross 2.84M → net 2.67M PLN/MW/yr (−6.1%)**: the battery keeps
  its reservation most of the time — only 7.4% of up-hours and **11.1% of
  down-hours** are lost to infeasible SoC.
- **Down-capacity is the binding constraint** — the measured +0.009/period
  net *charging* drift slowly fills the battery, blocking down-offers and
  forcing discharge. The physical asymmetry the gross number missed.
- Activation energy (+864k) ≈ SoC-recovery cost (−668k), so **all-in net
  ≈ 2.86M** — F8's "activation roughly offsets" is now measured, not assumed.
- Robust takeaway: **the feasibility haircut is modest (~6%)**, so F16's
  ~2.9M through-cycle gross capacity is ~2.7M net — the investment
  order-of-magnitude holds. Caveats: activation-settlement sign depends on
  the unverified ceb_sr convention; recovery policy is naive (discharges the
  moment SoC exits the band regardless of price — a price-aware policy would
  recover part of the −668k); single realized path, not stochastic; buffer/
  band heuristic unoptimized. So the energy terms are indicative; the ~6%
  capacity haircut is the defensible number.

### F17. LEAR in the ensemble: best CEN forecaster yet, wins every quarter
Closes #16 / the F9 follow-up. Added the (lightened) walk-forward LEAR as an
independent member of the postprocessing pool (`postprocess.py`, joined on
49,075 rows), with its median also feeding the QRA regressors.
- Walk-forward OOS (44,335 rows, 2025-02..2026-05) mean pinball:
  GBM 63.63 · QRA 63.31 · IDR-b 66.30 · **Ave-Q 62.80 · LEAR 65.48 ·
  Ave-Q+LEAR 62.06**. Coverage: Ave-Q+LEAR 0.795 (best, nominal 0.80).
- **Ave-Q+LEAR is the best forecaster in the repo** — beats Ave-Q by 0.74,
  GBM by 1.57, and wins in **all six quarters** (50.6/61.0/69.6/64.5/54.9/
  73.9). LEAR alone is worse than the GBM (65.48) but decorrelated, so
  distribution-averaging turns its diversity into gain — the F9 hypothesis
  confirmed, and the F3 "averaging is the free lunch" thesis reinforced.
- Caveat: this used the light LEAR (single 56d window, monthly refit — the
  [56,84]/7d version took 2.7 h of LP solves for no material benefit here);
  a fuller LEAR would likely add a touch more. Net verdict on the Weron
  program: base learner ≈ irrelevant (F9), VST harmful (F9), fundamentals
  priced-in (F10/F11/F13/F15) — the only durable forecast edge is
  **model-averaging a decorrelated pool**, worth ~1.5 pinball over the GBM.

### F16. BESS revenue trajectory: cyclical not decaying; F8 caught a high quarter
`src/bess_revenue_history.py` — realized per-quarter revenue for 1 MW/2 MWh
over the full history (capacity = realized CMBP clearing prices, energy = the
causal CEN threshold DP; no forecast needed), the robustness test of F8's
single-8-week headline.
- aFRR capacity (PLN/MW/yr): **5.6M (2024Q2 post-reform spike) → ~2.2–2.5M
  trough (2025) → 4.2M (2026Q2) → 3.4M (2026Q3)**. Volatile and cyclical,
  **not monotone decay** — the feared pipeline-entry erosion is *not yet
  visible*; 2026 rebounded (higher volatility / PICASSO / demand).
- **F8's 3.18M was measured in 2026Q2, a high quarter** → optimistic; the
  through-cycle capacity figure is ~3M gross, range 2.2–4.2M (ex the 5.6M
  reform outlier).
- **Energy arbitrage is the stable, growing floor**: 200–563k PLN/MW/yr,
  highest in 2026Q3 as CEN volatility rises — a natural hedge to capacity's
  cyclicality (energy is best when prices are wild, which is also when
  capacity can wobble).
- Caveats unchanged from F8: gross of SoC-feasibility, fixed/connection
  costs, and assumes dual-direction stacking permitted. The trajectory
  *shape* is the robust takeaway, not the absolute level.

Investment read: revise F8's "large but eroding cream" to **"large,
volatile, cyclical, not yet eroding," with energy arb as a rising floor.**
Through-cycle ~3M gross capacity + ~0.4M energy per MW, minus fixed costs and
feasibility derates — still Europe-leading, but size it on the cycle average,
not the 2026Q2 peak.

### F15. Weather-alpha: consensus already prices the weather; the sliver that's left is untradeable
`src/weather_res.py` — the disciplined gate before any weather trade: can a
weather-driven RES forecast beat the CONSENSUS (published DA forecast, in the
DA price)? Open-Meteo forecast-quality weather at 5 wind + 5 solar PL sites
(national proxies: mean wind_speed_100m & its cube, mean GHI), 2 yr hourly,
vs realized RES (PSE) and consensus (ENTSO-E).
- **Linear: weather adds ~nothing** over consensus (RMSE +0.2% wind,
  −0.2% solar). The eye-catching 0.39 corr of weather with the "residual"
  was a **scale artifact** — consensus runs ~9% high on wind, ~5% on solar
  (likely a definitional ENTSO-E-vs-PSE mismatch); after fitting that slope,
  weather's corr with the proper residual is +0.08 / −0.02, explaining
  1.9% / 0.0% of the error.
- **Nonlinear (GBM): solar −4% (worse), wind +3%** — a small but real wind-
  forecast edge the consensus misses.
- **But it's untradeable**: the oracle (F11) caps *perfect* realized-RES
  knowledge at ~1.2 pinball on CEN (DA price already holds the consensus).
  Capturing 3% of the wind error ⇒ price edge ≲ 0.03×1.2 ≈ 0.04 pinball —
  nil. The RES→price channel is too weak for even a genuine forecast edge to
  pay. (No price backtest needed; the oracle bounds it to zero.)
- Note: Open-Meteo historical-forecast weather is near-analysis quality
  (optimistically good); a strict day-ahead lead would be *more* negative.

Verdict: weather-alpha is dead, and for the deepest reason in the whole
study — the day-ahead information set (weather included) is efficiently
priced. Every fundamentals angle (F10 RES, F11 oracle, F13 tightness, F15
weather) converges on the same wall. **The edge is not in forecasting; it is
in physical participation** (BESS capacity/balancing, F8) and possibly
balancing-process microstructure (F4/Layer 2).

### F14. Intraday sizing: real arb venue, thin MM spread; the value ladder is capacity ≫ balancing > intraday > DA
`src/intraday_mm.py` — battery (1 MW/2 MWh) arbitrage value on each cleared
price curve via a SoC DP (perfect-foresight ceiling + causal P25/P75
threshold policy), plus an MM spread proxy. Grid fees excluded (Poland's
storage reform relieves double-charging; efficiency+degradation are the
modeled throughput cost — see the grid-cost note below).

| curve | PF /day | threshold /day | ann/MW |
|---|---|---|---|
| day-ahead (csdac) | 881 | 384 | 140k |
| intraday (TGE) | 1,084 | 463 | 169k |
| balancing (CEN) | 1,961 | 989 | 361k |

- **Intraday > day-ahead**: +78 PLN/day deployable (~20%), the intraday
  curve is more dispersed — trading intraday is worth it over DA-only.
- **Pure market-making is thin**: RDB intra-period range proxy ≈ 31 PLN/day.
  The continuous book isn't wide/deep enough for bid-ask capture to be the
  edge at these volumes; the intraday money is inter-period *arbitrage*
  (curve shape), not spread capture. (Caveat: true MM P&L needs order-book/
  tick data we don't have — TGE AIR, paid — so this is a lower bound.)
- **Balancing dominates** (989/day, matches F7's conditional policy). The
  CEN column is passive-balancing/imbalance-position value, not a freely
  tradeable venue, so treat it as an upper reference for that leg.

Strategic read: the battery's value ladder is **capacity (F8, ~8,700/day)
≫ balancing energy (~990) > intraday arb (~460) > DA arb (~380)**. "Battery
as intraday market maker" as a *standalone* edge is weaker than hoped
(spreads thin); intraday is best used as an extra arbitrage venue stacked
under the capacity+balancing core. Don't buy the tick-data feed on the MM
thesis alone.

**Grid-cost note (Poland storage reform):** double-charging of network fees
(charge-leg as consumer + discharge-leg as generator) is legally
*prohibited*; storage is defined as neither, connection fee halved. Our BESS
numbers assume that relief holds (right for a transmission-connected
balancing battery) and charge only round-trip efficiency (~12%) + degradation
(~100 PLN/MWh). NOT included and needed for a real IRR: fixed connection/O&M,
the halved-but-nonzero connection fee, balancing-responsibility costs; exact
network-fee treatment varies by voltage/tariff — a due-diligence item, not a
solved constant. So F8's 3.18M PLN/MW is gross of fixed costs.

### F13. Reserve-margin nowcast: the tightness edge is future info, not attainable at the gate
Follow-up to F11's actionable lead ("nowcast the reserve margin"). Added
leakage-safe published-history features of system tightness (rez_under,
rez_over_demand; D+1 availability rule) — `features.build(tight=)`.
- Raw signal is real: lagged `rez_under` corr(CEN)=0.24 (> lagged CEN's
  0.16) with day-lag autocorr 0.69, and a **0.19 partial correlation with
  CEN controlling for lagged CEN** — genuinely independent information.
- **But no forecast gain**: 8-week-holdout A/B pinball 68.54 (tight) vs
  68.46 (base); coverage improves slightly 0.763→0.773. The GBM ranks the
  features (rezo_pubday_min high) but extracts no accuracy — the signal is
  already captured by the *reserve-capacity prices* (fx_afrr_g/d, cleared
  D-1) and the DA anchor, which encode the market's *expected* tightness.
- The resolution of the F11 puzzle: the −6.4 oracle lever is
  *contemporaneous* realized tightness, which at the H=60 gate is **future
  information**. A forecast of it from gate-time features is redundant with
  what the GBM already does; the lagged actual mean-reverts (autocorr 0.69)
  and is subsumed by expected-tightness features already in the panel. So
  there is **no leakage-safe fundamentals feature that materially improves
  CEN** — 68.5 is close to the practical tree-model ceiling for the D-1
  information set. `tight` default OFF, kept for the small coverage gain.

### F12. Temporal hierarchy (THieF): no gain — the GBM is already temporally coherent
`src/temporal_hierarchy.py` — forecast CEN's hourly and 4-hour-block means
with their own GBMs and reconcile the 15-min median onto them (calibration-
tuned weights λ_hour, λ_block ∈ [0,1]), keeping the conformal spread as the
within-level shape.
- **Calibration drove λ_hour = λ_block = 0**: the tuner chose to ignore the
  aggregates entirely; reconciled pinball = base pinball = 68.46 exactly.
- Diagnosed the *why* (not a broken aggregate model): at the hourly level a
  dedicated hourly GBM scores MAE 169.0 vs the bottom GBM's own predictions
  averaged to hourly 168.0 — indistinguishable, and the two forecasts
  correlate 0.98. The aggregate carries no independent information.
- Mechanism: temporal hierarchies pay off when each level is fit by a
  simple model (ARIMA) blind to cross-level structure; reconciliation
  restores coherence. Our GBM already conditions on qh-of-day + smooth
  anchors, so it is coherent by construction and there is nothing to
  reconcile. A negative result specific to feature-rich learners.

### F11. Oracle study: fundamentals are ~worthless for CEN except system tightness; the ceiling is the balancing process itself
`src/oracle_study.py` — a deliberately LEAKAGE-VIOLATING ceiling study:
give the GBM *realized* actuals (known only post-delivery) and measure the
8-week-holdout pinball. Answers "if we knew the exact conditions, how much
better?" and hence whether modelling fundamentals separately is worthwhile.

| perfectly known | pinball | Δ vs strict 68.46 |
|---|---|---|
| realized RES (wind+solar) | 67.23 | −1.2 |
| realized load | 67.41 | −1.1 |
| realized net-load | 67.60 | −0.9 |
| realized cross-border | 67.84 | −0.6 |
| **realized system tightness** | **62.08** | **−6.4** |
| all oracle | 62.48 | −6.0 |

- **Modelling wind/sun/load separately is a dead end.** Even *perfect*
  knowledge of each buys ~1 pinball point — the day-ahead price already
  embeds the fundamentals (F10, now confirmed on actuals, not just
  forecasts). Reconstructing them is reconstructing what `fx_da` contains.
- **System tightness is the only fundamental that matters**: realized
  reserve margins (`rez_under`, `rez_over_demand`) + non-activated
  generation (`gen_not_activ_part`) are worth −6.4 and dominate the whole
  oracle set (all-oracle is *worse* than tight-alone, 62.48 vs 62.08 — the
  useless features add noise). This is the info the DA price can't hold and
  that drives balancing spikes. **Actionable target: nowcast the reserve
  margin** — the single highest-value forecasting object we've identified.
- **The ceiling is not fundamentals.** Even perfect tightness reaches only
  62, vs PSE's final-vintage 27.5. The 62→27.5 residual is the balancing
  *process* — which offers get activated, CKOEB/aFRR corrections — a
  discretionary/mechanical layer no observable fundamental explains. So
  the achievable edge from better fundamentals forecasting is ~6 pinball
  points, not 40.

### F10. The day-ahead price is a near-sufficient statistic for CEN at the gate — ENTSO-E fundamentals don't add
ENTSO-E token is live (all four endpoints verified: PL day-ahead, wind/solar
forecast, PL–DE_LU spread, imbalance prices). Built the leakage-safe RES
feature layer (`src/pull_entsoe.py`, `features.res_features`): day-ahead
wind/solar forecast — published D-1, so genuinely ex-ante, unlike PSE pk5l
PV/wind which the API only serves as a post-delivery latest vintage.
- Raw signal is real: `corr(y, fx_net_load)=+0.38` (residual demand =
  load_fcst − RES), nearly the DA anchor's 0.47 and 2× raw load; RES
  penetration −0.34. The GBM ranks `fx_res_wind` its #6 feature.
- **But it does not improve the forecast.** Clean 8-week-holdout A/B (same
  code, RES parquet hidden): with RES pinball 68.69 / CV 64.17, without RES
  68.47 / CV 63.83 — RES is neutral-to-slightly-worse, well inside the ±11
  CV noise. The tree merely *substitutes* RES for DA-derived features.
  Mechanism: the market already priced the day-ahead RES forecast into
  `fx_da`, so conditional on the anchor it is redundant.
- **Cross-border adds nothing either**: partial corr of the residual
  (CEN − csdac) with the PL–DE spread = −0.003, with the DE price = 0.018.
  The 0.59 raw corr(CEN, DE) is just Europe-wide co-movement in `fx_da`.
- **Intraday RES revision — the one signal that could beat the DA price —
  is not published for PL** (ENTSO-E `NoMatchingDataError`).

Consequence: at the D-1 / H=60 information set the predictable part of CEN
beyond `fx_da` comes from *balancing-system state* (published CEN/imbalance
history, reserve prices, SK contracting — all already in the panel), not
from more day-ahead fundamentals. The 68.5→27.5 gap to PSE's final vintage
is therefore **near-delivery** information (intraday forecast revisions,
real-time frequency/imbalance state, generation nowcasts) — none available
leakage-safe from these free sources. `res` is a `build()` flag, default
OFF; kept for other targets (BESS direction may be less DA-subsumed).
This bounds the ceiling of the day-ahead forecasting approach — a useful
negative result, and it redirects effort away from more fundamentals.

### F9. LEAR benchmark: a linear model matches the GBM — and asinh-VST is a trap on CEN
`src/lear.py` — Weron-style quantile LEAR: per-quantile L1-penalized linear
quantile regression on z-scored strict features, calibration-window
averaging (56 & 84 d), rolling refit every 7 d, same 8-week holdout.
- **LEAR (price target): mean pinball 67.0, coverage 0.74** — essentially
  matches, marginally beats, the tuned **GBM+hour-conformal (68.5 / 0.76)**.
  So on CEN the tree's non-linearity/interactions buy almost nothing; the
  signal is largely **linear** in the anchor + published-history features.
  The GBM keeps only a thin coverage edge, and that comes from its
  conformal step (which LEAR doesn't have yet), not from the base model.
- **asinh-VST is pathological here**: the same LEAR on the VST target
  scores **133** (2x worse), with a tell-tale high-bias per-quantile shape
  (q0.25≈q0.5≈185 falling to q0.9≈110). Cause: CEN's ±45k spikes make the
  `sinh` *inverse* extrapolate and detonate a handful of predictions.
  Invert-before-average vs after barely changed it (135 vs 133) — it's the
  inverse itself, not the averaging. This retro-explains F3 (VST "a wash"
  for the GBM): trees predict inside the training range and clip the
  blow-up; a linear model extrapolates and explodes. **Lesson: VST is for
  bounded heavy tails (day-ahead), not for spiking balancing prices.** The
  `--vst` flag is kept for diagnostics only; default is the price target.
- Implication (queued): the real win isn't picking GBM *or* LEAR but
  averaging them — our postprocess.py QRA/Vincentization pool (F3)
  currently holds only GBM-derived members; adding an independent LEAR
  member is exactly the diversification distribution-averaging rewards.

This closes the Weron thread (task #13): asinh-VST (F3, and F9 negative),
QRA + IDR + distribution averaging (F3), LEAR (F9). Net verdict: **the
free lunch on CEN is postprocessing/averaging, not the base learner or the
VST** — consistent with Lipiecki/Uniejewski/Weron (2024).

### F8. Layer 3: capacity (RMB) dominates the BESS stack ~7:1 — and matches Modo's independent benchmark
`src/bess_layer3.py`, holdout (56 days, 1 MW / 2 MWh):
- **capacity_only ≈ stack ≈ 8,708 PLN/day (ann. 3.18M PLN/MW)** vs
  energy_only 1,244. The honest forecast-based mode chooser picks capacity
  in 100% of hours — the DP's marginal energy value of an hour (~tens of
  PLN) never beats aFRR G+D capacity prices (mean 325 PLN/MW/h on the
  holdout; 250–650 in *every* quarter since the reform — not a window
  fluke).
- **Activation obligation quantified** (eb-rozl × zmb): ~13–14% mean
  utilization both directions → ~3.3 MWh/day cycled each way per procured
  MW, net drift ≈ −0.2 MWh/day. The activation energy itself settles at a
  *positive* margin (up at ceb_sr ~452, down at ~347 → +266 PLN/day)
  against 320–650 PLN/day cycling cost — net −50..−380 PLN/day, noise vs
  capacity revenue. The v1 "excluded, roughly offsetting" assumption is
  now measured.
- **External cross-check**: Modo Energy (May 2026) puts a 2h Polish BESS
  at >€800k/MW/yr annualized (≈3.4M PLN) with aFRR ~€120/MW/h ≈ 500
  PLN/MW/h; our same-window bottom-up numbers are 3.18M PLN/MW/yr and 484
  PLN/MW/h (2026Q2 aFRR G+D mean). Independent agreement to within ~7%.
- LER assumption: 30-min full-power sustain per direction (SUSTAIN_H=0.5,
  bands [0.53, 1.53] MWh for dual-direction provision on 2 MWh).
- **PICASSO pzeb verified as NOT a currency artifact**: ceb_sr/pzeb ratio
  median 13.6, IQR [7, 28] (an EUR quote would give a tight ~4.3).
  Semantics remain undocumented → zeb-rozl stays quarantined.
- Haircuts a real project takes on the 3.18M: dual-direction co-provision
  from one inverter must be permitted (else ~halve); prequalification/LER
  derating; and above all **entry compression** — these prices exist
  because prequalified supply is scarce; the 4 GWh+ Polish BESS pipeline
  is aimed straight at them (they already fell ~2x from 2024Q2 to 2025).
  The energy-arbitrage layer (F7) is the durable floor under that decay;
  the capacity layer is the (large, eroding) cream.

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
- `pull_entsoe.py` / `entsoe_client.py` — ENTSO-E day-ahead RES forecast and
  cross-border prices (token in gitignored `.env`). Feature layer
  `features.res_features` is `build(res=True)`, default OFF — neutral for
  CEN (F10), kept for other targets.
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
