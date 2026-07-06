# pl-cen-forecaster

Forecast the Polish single imbalance settlement price (CEN) at a 15-minute
granularity, from a decision time `H` minutes before delivery.

**Read `SPEC.md` first** — it briefs the whole design and the hard constraints
(the 2024-06-14 regime break, leakage rules, time-series CV).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pse_client discover        # confirm resource slugs -> config.yaml
export ENTSOE_API_TOKEN=...              # optional: cross-border + validation
make data                                # -> data/proc/panel_15min.parquet
```

Data: PSE v2 API (free, no auth), ENTSO-E Transparency (free token),
Open-Meteo (free, later iteration for RES nowcast features).
