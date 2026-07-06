"""
Orchestrate: pull PSE reports, align to a 15-min UTC panel, attach the CEN
target and regime flags, and write parquet.

    python -m src.build_dataset            # uses config.yaml
    python -m src.build_dataset --raw-only # just cache raw pulls
    python -m src.build_dataset --from-raw # skip pulling, rebuild from cache

Report semantics (see docs/pse_api_EndpointsMap.pdf and config.yaml):
- Multi-version reports (pk5l-wp, price-fcst, pdgobpkd) publish several rows
  per delivery period; the leakage-safe pick is the latest version whose
  publication_ts <= decision time (t - H). That selection happens here, so
  downstream features can trust one row per 15-min period.
- Settlement reports (crb-rozl, en-rozl, his-wlk-cal) publish D+1; we keep
  their pub_ts so features.py can lag them correctly.
- przeplywy-mocy is long (section_code, value) -> pivoted to one column per
  border section.
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import yaml

from . import targets
from .pse_client import PSEClient

# columns to carry into the panel, per config resource key.
# None -> keep every non-time column the report has.
KEEP: dict[str, list[str] | None] = {
    "imbalance":   ["cen_cost", "ckoeb_cost", "ceb_sr_cost",
                    "ceb_sr_afrrd_cost", "ceb_sr_afrrg_cost"],
    "price_fcst":  ["cen_fcst", "ckoeb_fcst", "cor_fcst", "imb_energy", "contracting"],
    "rce":         ["rce_pln"],
    "da_price":    ["csdac_pln"],
    "plan_pk5l":   ["grid_demand_fcst", "fcst_pv_tot_gen", "fcst_wi_tot_gen",
                    "planned_exchange", "surplus_cap_avail_tso", "req_pow_res",
                    "gen_surplus_avail_tso_above"],
    "kse_load":    ["load_fcst", "load_actual"],
    "kse_snapshot": ["gen_fv", "gen_wi", "kse_pow_dem",
                     "dom_balance_exchange_par", "dom_balance_exchange_non_par",
                     "rez_over_demand", "rez_under", "tot_jgm_char_pow"],
    "kse_actuals": ["demand", "pv", "wi", "swm_p", "swm_np"],
    "imb_energy":  ["balance", "en_d", "en_w"],
    "bal_energy":  ["eb_afrrd", "eb_afrrg", "eb_d_pp", "eb_w_pp"],
    "reserve_prices_basic": ["fcr_d", "fcr_g", "afrr_d", "afrr_g",
                             "mfrrd_d", "mfrrd_g", "rr_d", "rr_g"],
    "reserve_prices_suppl": ["fcr_d", "fcr_g", "afrr_d", "afrr_g",
                             "mfrrd_d", "mfrrd_g"],
    "reserve_req": ["zmb_fcrd", "zmb_fcrg", "zmb_afrrd", "zmb_afrrg",
                    "zmb_frrd", "zmb_frrg", "zmb_rrd", "zmb_rrg"],
    "contracting": ["sk_cost", "sk_d1_fcst", "sk_d_fcst"],
    "co2":         ["rcco2_pln"],
}

# reports that publish multiple versions per delivery period: keep the latest
# version visible at decision time t - H (falls back to earliest otherwise,
# which only happens if a period was never forecast before the gate).
MULTI_VERSION = {"price_fcst", "plan_pk5l", "kse_snapshot"}


def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())


def _pull_pse(cfg: dict) -> dict[str, pd.DataFrame]:
    cli = PSEClient(base=cfg["pse"]["base"])
    start = cfg["regime"]["train_start"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = {}
    for name, slug in cfg["pse"]["resources"].items():
        print(f"[pull] {name} <- {slug}  {start}..{end}", flush=True)
        try:
            out[name] = cli.fetch(slug, start, end)
        except Exception as e:                       # keep going; report at end
            print(f"[FAIL] {name}: {e}", flush=True)
    return out


def _latest_version_asof(df: pd.DataFrame, h_minutes: int) -> pd.DataFrame:
    """One row per ts: the newest version with pub_ts <= ts - H."""
    if "pub_ts" not in df.columns:
        return df.drop_duplicates("ts", keep="last")
    df = df.sort_values(["ts", "pub_ts"])
    gate = df["ts"] - pd.Timedelta(minutes=h_minutes)
    ok = df[df["pub_ts"] <= gate]
    picked = ok.drop_duplicates("ts", keep="last")
    # periods with no version before the gate: fall back to earliest version,
    # flagged so features can drop them if strictness demands
    missing = df[~df["ts"].isin(picked["ts"])].drop_duplicates("ts", keep="first")
    if len(missing):
        missing = missing.copy()
        missing["version_after_gate"] = 1
    out = pd.concat([picked, missing]).sort_values("ts")
    out["version_after_gate"] = out.get("version_after_gate", pd.Series(dtype="float64")).fillna(0).astype("int8")
    return out


def _pivot_flows(df: pd.DataFrame) -> pd.DataFrame:
    """przeplywy-mocy: long (section_code, value) -> wide, plus net total."""
    df = df.dropna(subset=["ts"])
    wide = (df.pivot_table(index="ts", columns="section_code",
                           values="value", aggfunc="last"))
    wide.columns = [f"flow_{str(c).strip().lower().replace(' ', '_')}" for c in wide.columns]
    wide["flow_net_total"] = wide.sum(axis=1)
    return wide.reset_index()


def _prep(name: str, df: pd.DataFrame, h_minutes: int) -> pd.DataFrame | None:
    """Normalize one raw report to columns [ts, <name>__x, ...] (+ pub_ts)."""
    if df is None or df.empty or "ts" not in df.columns:
        return None
    if name == "xborder_flows":
        out = _pivot_flows(df)
    else:
        if name in MULTI_VERSION:
            df = _latest_version_asof(df, h_minutes)
        else:
            df = df.drop_duplicates("ts", keep="last")
        keep = KEEP.get(name)
        cols = [c for c in (keep or df.columns) if c in df.columns
                and c not in ("ts", "pub_ts")]
        extra = [c for c in ("pub_ts", "version_after_gate") if c in df.columns]
        out = df[["ts"] + cols + extra]
    out = out.dropna(subset=["ts"])
    return out.rename(columns={c: f"{name}__{c}" for c in out.columns if c != "ts"})


def _align_15min(frames: dict[str, pd.DataFrame], h_minutes: int) -> pd.DataFrame:
    base = None
    for name, df in frames.items():
        sub = _prep(name, df, h_minutes)
        if sub is None:
            print(f"[skip] {name}: empty or no time axis")
            continue
        sub = sub.set_index("ts")
        base = sub if base is None else base.join(sub, how="outer")
    if base is None:
        return pd.DataFrame()
    full = base.sort_index()
    grid = pd.date_range(full.index.min(), full.index.max(), freq="15min", tz="UTC")
    out = full.reindex(grid).rename_axis("ts").reset_index()
    # coarser-than-15min series: forward-fill within their native step
    HOURLY = ("plan_pk5l__", "reserve_prices_basic__", "reserve_req__")
    DAILY = ("co2__",)
    for c in out.columns:
        if c.startswith(HOURLY):
            out[c] = out[c].ffill(limit=3)
        elif c.startswith(DAILY):
            out[c] = out[c].ffill(limit=96)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw-only", action="store_true")
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild the panel from cached raw parquet, no pulling")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    raw_dir = pathlib.Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = pathlib.Path(cfg["paths"]["proc"]); proc_dir.mkdir(parents=True, exist_ok=True)

    if a.from_raw:
        frames = {}
        for name in cfg["pse"]["resources"]:
            p = raw_dir / f"pse_{name}.parquet"
            if p.exists():
                frames[name] = pd.read_parquet(p)
    else:
        frames = _pull_pse(cfg)
        for name, df in frames.items():
            df.to_parquet(raw_dir / f"pse_{name}.parquet")
            print(f"[raw] {name}: {len(df)} rows cached")
        if a.raw_only:
            print("raw cached; done."); return

    panel = _align_15min(frames, cfg["horizon_minutes"])
    if panel.empty:
        print("no aligned data — check raw cache / API."); return

    panel = targets.add_regime_flags(panel, cfg)
    panel = targets.clip_to_regime(panel, cfg)

    tgt = f"{cfg['target']['resource']}__{cfg['target']['value_col']}"
    if tgt in panel:
        panel["y"] = pd.to_numeric(panel[tgt], errors="coerce")
    else:
        print(f"[warn] target col {tgt} not found in panel")

    out = proc_dir / "panel_15min.parquet"
    panel.to_parquet(out)
    print(f"wrote {out}  shape={panel.shape}  {panel['ts'].min()}..{panel['ts'].max()}")


if __name__ == "__main__":
    main()
