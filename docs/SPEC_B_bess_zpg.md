# SPEC — Project B: BESS offer-curve design under central dispatch (PSE/ZPG)

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
