"""Config: defaults, YAML loading, and strict validation (typos are errors)."""
from __future__ import annotations

import copy
import difflib
import pathlib

import yaml

from . import UserError

DEFAULTS: dict = {
    "workdir": "work",               # cache, models, reports (relative to the config file)
    "start": "2024-06-14",           # single-price regime start; do not go earlier
    "horizon_min": 60,               # forecast made H minutes before delivery start
    "cor_min": {2024: 10.00, 2025: 10.50, 2026: 10.82},   # CORMin [zl/MWh], WDB 12.4(3)
    "own_data": [],
    "model": {
        "final": "blend",            # structural | direct | blend
        "n_scenarios": 200,
        "conformal_days": 14,
        "conformal_lag_days": 2,     # CEN publishes D+1 ~14:00
        "holdout_days": 16,          # last days left out of `train` (out-of-sample calibration)
        "train_start": "auto",       # auto = max(start, first own-data timestamp)
        "seed": 0,
    },
}

SOURCE_DEFAULTS: dict = {
    "name": None,                    # feature prefix (letters, digits, _)
    "path": None,
    "kind": "realtime",              # realtime | forecast
    "time_col": "ts",                # realtime: when the value became KNOWN to you
    "issued_col": "issued_at",       # forecast: when the forecast was issued
    "target_col": "target_ts",       # forecast: START of the delivery period it is for
    "target_resolution_min": 15,     # forecast: 15 or 60 (hourly rows are spread to 4 quarters)
    "timezone": None,                # only for timestamps without an offset, e.g. Europe/Warsaw
    "columns": None,                 # default: every numeric column
    "known_delay_min": 0,            # realtime: added to time_col (safety margin)
    "max_staleness_min": 60,         # realtime: older values count as missing
    "sep": ",",
    "decimal": ".",
}


def _check_keys(d: dict, allowed: dict, where: str) -> None:
    for k in d:
        if k not in allowed:
            hint = difflib.get_close_matches(str(k), list(allowed), n=1)
            raise UserError(f"Nieznany klucz '{k}' w {where}."
                            + (f" Czy chodziło o '{hint[0]}'?" if hint else "")
                            + f" Dozwolone: {', '.join(allowed)}")


def load(path: str | None) -> dict:
    """Load config.yaml (or defaults when no file), resolve paths, validate."""
    cfg = copy.deepcopy(DEFAULTS)
    base = pathlib.Path.cwd()
    if path is None and pathlib.Path("config.yaml").exists():
        path = "config.yaml"
    if path is not None:
        p = pathlib.Path(path)
        if not p.exists():
            raise UserError(f"Nie ma pliku konfiguracji {p.resolve()}. "
                            "Utwórz go: python -m cenmodel init")
        try:
            user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise UserError(f"Błąd składni YAML w {p}: {e}") from None
        if not isinstance(user, dict):
            raise UserError(f"{p} musi zawierać słownik klucz: wartość.")
        _check_keys(user, DEFAULTS, str(p))
        _check_keys(user.get("model") or {}, DEFAULTS["model"], f"{p} -> model")
        cfg.update({k: v for k, v in user.items() if k != "model"})
        cfg["model"].update(user.get("model") or {})
        base = p.resolve().parent
    cfg["_base"] = base
    cfg["workdir"] = (base / cfg["workdir"]).resolve()
    cfg["cor_min"] = {int(k): float(v) for k, v in (cfg["cor_min"] or {}).items()}

    h = cfg["horizon_min"]
    if not isinstance(h, int) or h < 15 or h % 15:
        raise UserError(f"horizon_min musi być wielokrotnością 15 i >= 15 (jest {h!r}).")
    if cfg["model"]["final"] not in ("structural", "direct", "blend"):
        raise UserError("model.final musi być jednym z: structural, direct, blend.")

    srcs, names = [], set()
    for i, s in enumerate(cfg["own_data"] or []):
        if not isinstance(s, dict):
            raise UserError(f"own_data[{i}] musi być słownikiem (name, path, kind, ...).")
        _check_keys(s, SOURCE_DEFAULTS, f"own_data[{i}]")
        s = {**SOURCE_DEFAULTS, **s}
        if not s["name"] or not str(s["name"]).replace("_", "").isalnum():
            raise UserError(f"own_data[{i}].name jest wymagane: litery, cyfry, _ (np. 'imb').")
        if s["name"] in names:
            raise UserError(f"own_data: nazwa '{s['name']}' występuje dwa razy.")
        names.add(s["name"])
        if not s["path"]:
            raise UserError(f"own_data '{s['name']}': brak 'path'.")
        s["path"] = (base / s["path"]).resolve()
        if s["kind"] not in ("realtime", "forecast"):
            raise UserError(f"own_data '{s['name']}': kind musi być realtime albo forecast.")
        if s["target_resolution_min"] not in (15, 30, 60):
            raise UserError(f"own_data '{s['name']}': target_resolution_min to 15, 30 albo 60.")
        srcs.append(s)
    cfg["own_data"] = srcs
    return cfg


def cor_min_for(cfg: dict, years) -> list[float]:
    """CORMin per year; unknown future years reuse the latest known value."""
    table = cfg["cor_min"]
    last = table[max(table)]
    return [table.get(int(y), last) for y in years]


TEMPLATE = """\
# Konfiguracja cenmodel. Ścieżki są względne względem TEGO pliku.
workdir: work            # tu trafia cache z API PSE, modele i raporty
start: "2024-06-14"      # początek jednolitej ceny niezbilansowania — nie zmieniaj na wcześniej
horizon_min: 60          # prognoza robiona H minut przed początkiem okresu dostawy

# CORMin [zł/MWh] (WDB 12.4(3), indeksowany CPI). Dopisz nowy rok, gdy PSE go ogłosi;
# do tego czasu używana jest ostatnia znana wartość.
cor_min: {2024: 10.00, 2025: 10.50, 2026: 10.82}

# Twoje dane. Usuń [] i odkomentuj przykłady. Dwa rodzaje plików (CSV lub parquet):
#  realtime: pomiar znany w chwili `time_col` (np. niezbilansowanie, aktywacja aFRR, ACE)
#  forecast: prognoza dla okresu `target_col`, wydana w chwili `issued_col` (np. wiatr, PV)
# Model sam bierze tylko to, co było znane w chwili bramki (początek okresu - horizon_min).
own_data: []
#  - name: imb                        # prefiks cech, krótki
#    path: my_data/imbalance.csv
#    kind: realtime
#    time_col: ts                     # KIEDY wartość była znana (nie początek pomiaru!)
#    timezone: Europe/Warsaw          # tylko gdy daty są bez strefy (bez +01:00 / Z)
#    columns: [imbalance_mw, afrr_up_mw, afrr_down_mw]   # pomiń = wszystkie liczbowe
#    known_delay_min: 0               # dodaj zapas, jeśli znacznik to początek uśredniania
#    max_staleness_min: 60            # starsze wartości = brak danych
#    sep: ";"                         # CSV z polskiego Excela
#    decimal: ","
#  - name: wind
#    path: my_data/wind_forecast.parquet
#    kind: forecast
#    issued_col: issued_at
#    target_col: target_ts            # POCZĄTEK okresu dostawy
#    target_resolution_min: 15        # 60 jeśli prognoza godzinowa
#    timezone: UTC

model:
  final: blend           # structural | direct | blend — co trafia do kolumn q05..q95
  n_scenarios: 200
  conformal_days: 14     # okno kalibracji przedziałów
  conformal_lag_days: 2  # CEN publikuje się D+1 ~14:00
  holdout_days: 16       # tyle ostatnich dni `train` zostawia poza treningiem (kalibracja)
  train_start: auto      # auto = od najwcześniejszych Twoich danych (lub od `start`)
  seed: 0
"""
