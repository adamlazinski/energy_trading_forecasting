"""
Orchestrate: pull PSE (+ optional ENTSO-E) reports, align to a 15-min UTC panel,
attach the CEN target and regime flags, and write parquet.

    python -m src.build_dataset            # uses config.yaml
    python -m src.build_dataset --raw-only # just cache raw pulls
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import yaml

from . import targets
from .pse_client import PSEClient


def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())


def _pull_pse(cfg: dict) -> dict[str, pd.DataFrame]:
    cli = PSEClient(base=cfg["pse"]["base"])
    start = cfg["regime"]["train_start"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = {}
    for name, slug in cfg["pse"]["resources"].items():
        if slug in (None, "", "CONFIRM"):
            print(f"[skip] {name}: slug not confirmed yet")
            continue
        print(f"[pull] {name} <- {slug}  {start}..{end}")
        out[name] = cli.fetch(slug, start, end)
    return out


def _align_15min(frames: dict[str, pd.DataFrame], value_cols: dict[str, str]) -> pd.DataFrame:
    """Left-join every report onto a common 15-min UTC index by 'ts'."""
    base = None
    for name, df in frames.items():
        if "ts" not in df or df.empty:
            continue
        col = value_cols.get(name)
        keep = ["ts"] + ([col] if col and col in df else
                         [c for c in df.columns if c != "ts"])
        sub = df[keep].copy()
        sub = sub.rename(columns={c: f"{name}__{c}" for c in sub.columns if c != "ts"})
        sub = sub.drop_duplicates("ts").set_index("ts")
        base = sub if base is None else base.join(sub, how="outer")
    if base is None:
        return pd.DataFrame()
    full = base.sort_index()
    grid = pd.date_range(full.index.min(), full.index.max(), freq="15min", tz="UTC")
    return full.reindex(grid).rename_axis("ts").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw-only", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    raw_dir = pathlib.Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = pathlib.Path(cfg["paths"]["proc"]); proc_dir.mkdir(parents=True, exist_ok=True)

    frames = _pull_pse(cfg)
    for name, df in frames.items():
        df.to_parquet(raw_dir / f"pse_{name}.parquet")
    if a.raw_only:
        print("raw cached; done."); return

    # value-column hints per report (extend as you confirm slugs/columns)
    value_cols = {"rce": "rce_pln", "target": cfg["target"]["value_col"]}
    panel = _align_15min(frames, value_cols)
    if panel.empty:
        print("no aligned data — confirm slugs with `pse_client discover`."); return

    panel = targets.add_regime_flags(panel, cfg)
    panel = targets.clip_to_regime(panel, cfg)

    # attach target y from whichever report holds CEN/RCE
    tgt_key = "imbalance" if "imbalance" in frames else "rce"
    tgt_val = f"{tgt_key}__{cfg['target']['value_col']}"
    if tgt_val in panel:
        panel["y"] = pd.to_numeric(panel[tgt_val], errors="coerce")
    else:
        print(f"[warn] target col {tgt_val} not found; set it once slugs confirmed")

    out = proc_dir / "panel_15min.parquet"
    panel.to_parquet(out)
    print(f"wrote {out}  shape={panel.shape}  {panel['ts'].min()}..{panel['ts'].max()}")


if __name__ == "__main__":
    main()
