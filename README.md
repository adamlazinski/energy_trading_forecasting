# Energy Trading Forecasting

Research monorepo for Polish power-market forecasting and trading, centered on
the post-2024-06-14 balancing reform (single-price CEN, central dispatch).

## Contents

- `RESEARCH_LOG.md` — **start here**: working notes — thesis, hard rules
  (leakage gates, regime breaks), findings to date, pipeline map, queue.
- `pl-cen-forecaster/` — probabilistic forecaster for the Polish imbalance
  price (CEN) on 15-min settlement periods, plus the strategy backtests and
  the Project B (BESS) data layer that grew around it. PSE v2 + TGE +
  ENTSO-E data layers, feature panel, quantile targets. See its `SPEC.md`.
- `docs/research_directions_and_specs.md` — survey of research directions
  (market making, trend/stat-arb, physical) built on top of the forecaster.
- `docs/SPEC_A_rdb_cen_spread.md` — Project A: is the intraday↔imbalance
  (RDB↔CEN) spread predictable? Extension of pl-cen-forecaster.
- `docs/SPEC_B_bess_zpg.md` — Project B: BESS offer-curve design under PSE
  central dispatch (ZPG). Deep three-layer build; planned as its own repo
  (`pl-bess-zpg`) reusing the forecaster's data layer.

## Setup (pl-cen-forecaster)

```bash
cd pl-cen-forecaster
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See `pl-cen-forecaster/Makefile` for data-build targets.
