"""Command line: python -m cenmodel <command> [options]. See README.md."""
from __future__ import annotations

import argparse
import pickle
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

from . import UserError, __version__
from . import config as config_mod
from . import model as M
from . import owndata, pse, wdb
from .panel import last_published, load_public

WAW = "Europe/Warsaw"


def _t(ts) -> str:
    return pd.Timestamp(ts).tz_convert(WAW).strftime("%Y-%m-%d %H:%M")


def _say(*a):
    print(*a, flush=True)


def _own_cache(cfg) -> dict:
    return {s["name"]: owndata.load_source(s) for s in cfg["own_data"]}


def _own_features(cfg, idx, cache) -> pd.DataFrame:
    return owndata.all_features(cfg["own_data"], idx, cfg["horizon_min"], cache)


def _train_start(cfg, cache) -> pd.Timestamp:
    t = pd.Timestamp(cfg["start"], tz=WAW).tz_convert("UTC")
    ts = cfg["model"]["train_start"]
    if ts not in (None, "auto"):
        return max(t, pd.Timestamp(str(ts), tz=WAW).tz_convert("UTC"))
    for s in cfg["own_data"]:
        df = cache[s["name"]]
        first = df.index.min() if s["kind"] == "realtime" else df.target.min()
        t = max(t, first.floor("15min"))
    return t


def _model_path(cfg):
    return cfg["workdir"] / "models" / "cenmodel.pkl"


# --- commands --------------------------------------------------------------
def cmd_init(cfg, a):
    p = cfg["_base"] / "config.yaml"
    if p.exists():
        _say(f"{p} już istnieje — nie nadpisuję.")
    else:
        p.write_text(config_mod.TEMPLATE, encoding="utf-8")
        _say(f"Utworzono {p}. Uzupełnij sekcję own_data, potem: python -m cenmodel doctor")
    (cfg["_base"] / "my_data").mkdir(exist_ok=True)


def cmd_doctor(cfg, a):
    ok = True
    _say(f"cenmodel {__version__}, Python {platform.python_version()} ({platform.system()})")
    if sys.version_info < (3, 10):
        _say("  BŁĄD: potrzebny Python >= 3.10"); ok = False
    for m in ("pandas", "numpy", "lightgbm", "sklearn", "pyarrow", "requests", "yaml"):
        try:
            __import__(m)
        except Exception as e:                                  # noqa: BLE001
            _say(f"  BŁĄD: brak pakietu {m} ({e}). Uruchom: pip install -r requirements.txt")
            ok = False
    for m in ("pandas", "numpy", "lightgbm", "sklearn", "pyarrow"):
        if m not in sys.modules:
            continue
        _say(f"  {m:9s} {sys.modules[m].__version__}")
    _say(f"workdir: {cfg['workdir']}")
    _say("API PSE: " + ("OK" if pse.PSE().ping() else "BRAK POŁĄCZENIA"))
    try:
        df = load_public(cfg["workdir"], cfg["start"])
        _say(f"cache PSE: {_t(df.index.min())} .. {_t(df.index.max())}, "
             f"ostatnie opublikowane CEN: {_t(last_published(df))}")
    except UserError as e:
        _say(f"cache PSE: {e}"); ok = False
    for s in cfg["own_data"]:
        try:
            for line in owndata.describe(s, owndata.load_source(s)):
                _say(line)
        except UserError as e:
            _say(f"  BŁĄD: {e}"); ok = False
    mp = _model_path(cfg)
    if mp.exists():
        b = pickle.loads(mp.read_bytes())
        _say(f"model: trenowany do {_t(b['train_end'])}, {b['n_train']} okresów, cechy own: {b['own']}")
    else:
        _say("model: brak (python -m cenmodel train)")
    _say("\nWszystko gotowe." if ok else "\nSą problemy — popraw je i uruchom doctor ponownie.")


def _month_range(spec: str) -> tuple[str, str]:
    """'RRRR-MM' or 'RRRR-MM..RRRR-MM' -> (first day, last day), capped at today."""
    lo, _, hi = spec.partition("..")
    try:
        m0, m1 = pd.Period(lo, freq="M"), pd.Period(hi or lo, freq="M")
    except ValueError:
        raise UserError(f"Zły format miesiąca '{spec}'. Użyj RRRR-MM albo RRRR-MM..RRRR-MM.") from None
    if m1 < m0:
        raise UserError(f"Zakres '{spec}' jest odwrócony.")
    end = min(m1.end_time.normalize(), pd.Timestamp.now(WAW).tz_localize(None).normalize())
    return m0.start_time.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def cmd_fetch(cfg, a):
    wd = cfg["workdir"]
    for spec, slug, label in ((a.ladder, "oeb-bpkdbo", "Drabinka BPKD/BO (oeb-bpkdbo)"),
                              (a.offers, "poeb-rbn", "Oferty RB (poeb-rbn)")):
        if spec:
            lo, hi = _month_range(spec)
            _say(f"{label} {lo} .. {hi} (dzień po dniu, ~10 s/dzień)...")
            pse.fetch_ladder(wd, lo, hi, log=_say, slug=slug)
    if a.ladder or a.offers:
        return
    start = a.start or cfg["start"]
    _say(f"Pobieram dane rozliczeniowe PSE od {start} do {a.end or 'jutra'} -> {wd / 'raw'}")
    pse.fetch_public(wd, start, a.end, log=_say)
    df = load_public(wd, cfg["start"])
    _say(f"OK. Ostatnie opublikowane CEN: {_t(last_published(df))}; CSDAC do {_t(df.csdac.dropna().index.max())}")


def cmd_verify(cfg, a):
    df = load_public(cfg["workdir"], cfg["start"])
    lad = None
    if a.month:
        m = pd.Period(a.month, freq="M")
        df = df[(df.index.tz_convert(WAW) >= m.start_time.tz_localize(WAW))
                & (df.index.tz_convert(WAW) < (m.end_time + pd.Timedelta(1)).tz_localize(WAW))]
        try:
            lad = pse.read_ladder(cfg["workdir"], m.start_time.strftime("%Y-%m-%d"),
                                  m.end_time.strftime("%Y-%m-%d"))
        except UserError as e:
            _say(f"({e})")
    lines = wdb.verify(df, lad, lambda y: config_mod.cor_min_for(cfg, y))
    _say("\n".join(lines))
    _write_report(cfg, "verify.txt", lines)


def cmd_check_data(cfg, a):
    if not cfg["own_data"]:
        raise UserError("W konfiguracji nie ma own_data — nie ma czego sprawdzać.")
    cache = _own_cache(cfg)
    df = load_public(cfg["workdir"], cfg["start"])
    t0 = _train_start(cfg, cache)
    idx = df.index[(df.index >= t0) & df.sk.notna()]
    F = _own_features(cfg, idx, cache)
    lines = []
    for s in cfg["own_data"]:
        lines += owndata.describe(s, cache[s["name"]])
    lines += ["", f"Pokrycie cech w okresach z opublikowanym SK od {_t(t0)} ({len(idx)} okresów):"]
    for c in F.columns:
        cov = F[c].notna().mean()
        cor = F[c].corr(df.sk.loc[idx]) if F[c].std() > 0 else np.nan
        flag = "  <- PODEJRZANIE WYSOKA: sprawdź znacznik czasu (wyciek?)" if abs(cor) > 0.95 else ""
        flag = "  <- mało danych" if cov < 0.5 else flag
        lines.append(f"   {c:40s} pokrycie {cov:6.1%}   korelacja z SK {cor:+.2f}{flag}")
    lines += ["", "Korelacja z SK mierzona na bramce (H min przed okresem). Typowo |r| 0.3-0.8 dla",
              "dobrego sygnału niezbilansowania; ~1.0 oznacza zwykle, że dane widzą przyszłość."]
    _say("\n".join(lines))
    _write_report(cfg, "check_data.txt", lines)


def cmd_backtest(cfg, a):
    t_start = time.time()
    df = load_public(cfg["workdir"], cfg["start"])
    cache = _own_cache(cfg)
    lp = last_published(df)
    test_end = pd.Timestamp(a.test_end, tz=WAW).tz_convert("UTC") if a.test_end else lp
    if a.test_start:
        test_start = pd.Timestamp(a.test_start, tz=WAW).tz_convert("UTC")
    else:
        test_start = (test_end - pd.Timedelta(days=a.test_days)).tz_convert(WAW).normalize().tz_convert("UTC")
    mc = cfg["model"]
    warm = pd.Timedelta(days=mc["conformal_days"] + mc["conformal_lag_days"])
    pred_start = test_start - warm
    t0 = _train_start(cfg, cache)
    ok = M.trainable(df)
    tr_rows = df.index[ok & (df.index >= t0) & (df.index < pred_start)]
    pr_rows = df.index[df.csdac.notna() & (df.index >= pred_start) & (df.index <= test_end)]
    if len(pr_rows) == 0 or test_start >= test_end:
        raise UserError(f"Pusty okres testu {_t(test_start)} .. {_t(test_end)}.")
    _say(f"trening {_t(t0)} .. {_t(pred_start)}  ({len(tr_rows) / 96:.0f} dni)")
    _say(f"test    {_t(test_start)} .. {_t(test_end)}  (+{warm.days} dni rozgrzewki kalibracji)")

    own = _own_features(cfg, df.index, cache) if cfg["own_data"] else None
    variants = [("publiczne", M.make_features(df, cfg))]
    if own is not None:
        variants.append(("publiczne+Twoje", M.make_features(df, cfg, own)))
    if a.simulate_rt_sk:
        variants.append(("publiczne+Twoje+idealne SK" if own is not None else "publiczne+idealne SK",
                         M.make_features(df, cfg, own, sim_rt=True)))

    ev = df.index[(df.index >= test_start) & (df.index <= test_end) & df.cen.notna()]
    y = df.cen
    lines = [f"BACKTEST  H={cfg['horizon_min']} min  test {_t(test_start)} .. {_t(test_end)}  "
             f"({len(ev)} okresów)", "",
             f"{'zestaw cech':28s} {'model':11s} {'pinball':>8s} {'pokr.P10-90':>11s} "
             f"{'pokr.P5-95':>10s} {'szer.P10-90':>11s}  {'pinball surowy':>14s}"]
    store = {}
    for name, X in variants:
        _say(f"\n[{name}] {X.shape[1]} cech — trening...")
        rng = np.random.default_rng(mc["seed"])
        b = M.fit(df, X, tr_rows, cfg, log=_say)
        raw = M.predict_raw(b, df, X, pr_rows, cfg, rng)
        _say(f"  P(short) Brier {np.mean((raw['p_short'][ev] - (df.sk[ev] < 0)) ** 2):.3f}, "
             f"MAE mediany SK {np.abs(raw['sk_q50'][ev] - df.sk[ev]).mean():.1f} MWh")
        for model in ("structural", "direct", "blend"):
            rc = M.rolling_conformal(y, raw[model], mc["conformal_days"], mc["conformal_lag_days"])
            r, e = rc.loc[ev], y.loc[ev]
            lines.append(f"{name:28s} {model:11s} {M.pinball(e, r):8.1f} {M.coverage(e, r):11.3f} "
                         f"{M.coverage(e, r, 'q05', 'q95'):10.3f} {(r.q90 - r.q10).median():11.0f}  "
                         f"{M.pinball(e, raw[model].loc[ev]):14.1f}")
            store[(name, model)] = rc.loc[ev]
        lines.append("")
    lines += ["Pinball: średnia strata kwantylowa [zł/MWh], mniej = lepiej. Pokrycie nominalne:",
              "P10-90 = 0.80, P5-95 = 0.90. 'surowy' = przed kalibracją konformalną.", ""]
    best = min(store, key=lambda k: M.pinball(y.loc[ev], store[k]))
    lines.append(f"Najlepszy: {best[0]} / {best[1]}. Wg miesięcy (pokrycie P10-90 / pinball):")
    for mth, g in pd.Series(ev, index=ev).groupby(ev.tz_convert(WAW).strftime("%Y-%m")):
        r = store[best].loc[g.index]
        lines.append(f"   {mth}  {M.coverage(y.loc[g.index], r):.2f} / {M.pinball(y.loc[g.index], r):.0f}")
    lines.append(f"\n(czas: {(time.time() - t_start) / 60:.1f} min)")
    _say("\n" + "\n".join(lines))
    _write_report(cfg, "backtest.txt", lines)
    out = pd.concat({f"{k[0]}|{k[1]}": v for k, v in store.items()}, axis=1)
    out[("y", "cen")] = y.loc[ev]
    out.columns = ["|".join(c) for c in out.columns]
    p = cfg["workdir"] / "reports" / "backtest_preds.parquet"
    out.to_parquet(p)
    _say(f"Raport: {cfg['workdir'] / 'reports' / 'backtest.txt'}\nPredykcje: {p}")


def cmd_train(cfg, a):
    df = load_public(cfg["workdir"], cfg["start"])
    cache = _own_cache(cfg)
    lp = last_published(df)
    end = lp - pd.Timedelta(days=cfg["model"]["holdout_days"])
    if a.end:
        end = min(end, pd.Timestamp(a.end, tz=WAW).tz_convert("UTC"))
    t0 = _train_start(cfg, cache)
    own = _own_features(cfg, df.index, cache) if cfg["own_data"] else None
    X = M.make_features(df, cfg, own)
    rows = df.index[M.trainable(df) & (df.index >= t0) & (df.index <= end)]
    _say(f"Trening na {_t(t0)} .. {_t(end)} ({len(rows) / 96:.0f} dni, {X.shape[1]} cech).")
    _say(f"Ostatnie {cfg['model']['holdout_days']} dni ({_t(end)} .. {_t(lp)}) zostaje do kalibracji.")
    b = M.fit(df, X, rows, cfg, log=_say)
    p = _model_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(b))
    _say(f"Zapisano model: {p}\nTeraz: python -m cenmodel predict")


def cmd_predict(cfg, a):
    p = _model_path(cfg)
    if not p.exists():
        raise UserError("Brak modelu. Najpierw: python -m cenmodel train")
    b = pickle.loads(p.read_bytes())
    if b["horizon_min"] != cfg["horizon_min"]:
        raise UserError(f"Model ma horizon_min={b['horizon_min']}, konfiguracja {cfg['horizon_min']}. "
                        "Uruchom train ponownie.")
    if [tuple(o) for o in b["own"]] != [(s["name"], s["kind"]) for s in cfg["own_data"]]:
        raise UserError("own_data w konfiguracji różni się od tego, z czym trenowano model. "
                        "Uruchom train ponownie.")
    if a.refresh:
        _say("Odświeżam dane PSE (ostatnie 10 dni)...")
        pse.fetch_public(cfg["workdir"], (pd.Timestamp.now(WAW) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                         log=lambda *_: None)
    df = load_public(cfg["workdir"], cfg["start"])
    lp = last_published(df)
    h = cfg["horizon_min"]
    now = pd.Timestamp.now("UTC")
    if a.start:
        t_lo = pd.Timestamp(a.start, tz=WAW).tz_convert("UTC")
        t_hi = pd.Timestamp(a.end, tz=WAW).tz_convert("UTC") if a.end else t_lo + pd.Timedelta(hours=1)
    else:                                   # periods whose gate has passed but not yet delivered
        t_lo = now.floor("15min") + pd.Timedelta(minutes=15)
        t_hi = now.floor("15min") + pd.Timedelta(minutes=h)
    target = df.index[(df.index >= t_lo) & (df.index <= t_hi)]
    if len(target) == 0:
        raise UserError(f"Brak okresów {_t(t_lo)} .. {_t(t_hi)} w danych PSE. "
                        "Uruchom z --refresh albo fetch (CSDAC na jutro jest od ~14:00).")
    nocsdac = target[df.csdac.loc[target].isna()]
    if len(nocsdac):
        _say(f"UWAGA: brak CSDAC dla {len(nocsdac)} okresów (od {_t(nocsdac[0])}) — pomijam. "
             "CSDAC na dobę D publikuje się D-1 ~14:00; uruchom z --refresh.")
        target = target.difference(nocsdac)
        if len(target) == 0:
            raise UserError("Nie zostało nic do prognozy.")

    cache = _own_cache(cfg)
    mc = cfg["model"]
    res_idx = df.index[(df.index > b["train_end"]) & df.cen.notna() & df.csdac.notna()
                       & (df.index > lp - pd.Timedelta(days=mc["conformal_days"]))]
    idx = target.union(res_idx)
    own = _own_features(cfg, idx, cache) if cfg["own_data"] else None
    X = M.make_features(df, cfg).loc[idx]
    if own is not None:
        X = X.join(own)
    rng = np.random.default_rng(mc["seed"])
    raw = M.predict_raw(b, df, X, idx, cfg, rng, keep_scen=a.scenarios)
    pred = raw[mc["final"]]
    if len(res_idx) >= 96 * 2:
        off = M.live_offsets(df.cen.loc[res_idx], pred.loc[res_idx])
        cal = f"kalibracja na {len(res_idx) / 96:.1f} dniach poza treningiem"
    else:
        off = pd.Series(0.0, index=M.QCOLS)
        cal = "BEZ KALIBRACJI (model za świeży lub stare dane PSE) — przedziały będą za wąskie"
    fin = M.apply_offsets(pred.loc[target], off)

    out = fin.round(2)
    out.insert(0, "gate_warsaw", (target - pd.Timedelta(minutes=h)).tz_convert(WAW).strftime("%Y-%m-%d %H:%M"))
    out.insert(0, "period_start_warsaw", target.tz_convert(WAW).strftime("%Y-%m-%d %H:%M"))
    out["p_short"] = raw["p_short"].loc[target].round(3)
    out["sk_q50_mwh"] = raw["sk_q50"].loc[target].round(1)
    out["csdac"] = df.csdac.loc[target]
    out["cen_actual"] = df.cen.loc[target]
    out.index = out.index.rename("period_start_utc")

    for s in cfg["own_data"]:
        col = f"{s['name']}_age_min"
        if col in X.columns:
            age = X.loc[target, col]
            stale = age.isna() | (age > s["max_staleness_min"])
            if stale.any():
                _say(f"UWAGA: dane '{s['name']}' nieaktualne na bramce dla {stale.sum()} okresów "
                     f"(wiek {age.max():.0f} min) — prognoza bez Twojego sygnału.")
    _say(f"Model: {mc['final']}, trenowany do {_t(b['train_end'])}; {cal}.")
    _say(f"Ostatnie opublikowane CEN: {_t(lp)}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        _say(out.drop(columns=["cen_actual"] if out.cen_actual.isna().all() else []).to_string(index=False))
    path = (cfg["_base"] / a.out) if a.out else cfg["workdir"] / "predictions" / f"cen_{now:%Y%m%d_%H%M}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    _say(f"\nZapisano: {path}")
    if a.scenarios:
        sp = path.with_suffix(".scenarios.parquet")
        raw["scenarios"].loc[target].to_parquet(sp)
        _say(f"Scenariusze ({mc['n_scenarios']} na okres, przed kalibracją): {sp}")


def cmd_ladder(cfg, a):
    from . import ladder as L
    wd = cfg["workdir"]
    pub = load_public(wd, cfg["start"])
    lo = a.start or None
    hi = a.end or None
    _say("Buduję krzywe cena-wolumen z drabinki BPKD/BO...")
    c = L.build(wd, "bpkdbo", lo, hi, zeb_pp=pub.zeb_pp, log=_say)
    rbn = None
    if (wd / "raw" / "offers").exists() and any((wd / "raw" / "offers").glob("*.parquet")):
        _say("Buduję krzywe z ofert poeb-rbn...")
        rbn = L.build(wd, "rbn", lo, hi, log=_say)
    rep = wd / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    c.to_parquet(rep / "ladder_curves.parquet", index=False)
    if rbn is not None:
        rbn.to_parquet(rep / "ladder_curves_rbn.parquet", index=False)
    lines = [f"DRABINKI  {_t(c.ts.min())} .. {_t(c.ts.max())}   siatka MW: {', '.join(map(str, L.GRID))}"]
    if "ckoeb_hat" in c:
        j = c.set_index("ts").join(pub.ckoeb).dropna(subset=["ckoeb_hat", "ckoeb"])
        lines.append(f"kontrola: CKOEB z drabinki == opublikowane w {np.mean(np.abs(j.ckoeb_hat - j.ckoeb) < 0.01):.2%}"
                     f" z {len(j)} okresów (powinno być ~100%)")
    lines += ["", "BPKD/BO (cena CDO, kierunek aktywacji):"] + L.describe(c, pub) + L.pca_lines(c)
    if rbn is not None:
        lines += ["", "poeb-rbn (wszystkie oferty, oba kierunki):"] + L.describe(rbn, pub)
    ev_lines, P = L.evaluate(c, pub, a.test_days, cfg["horizon_min"], rbn)
    lines += ev_lines
    if not P.empty:
        P.to_parquet(rep / "ladder_eval.parquet", index=False)
    _say("\n" + "\n".join(lines))
    _write_report(cfg, "ladder.txt", lines)
    _say(f"\nRaport: {rep / 'ladder.txt'}\nKrzywe: {rep / 'ladder_curves.parquet'}"
         + (f", {rep / 'ladder_curves_rbn.parquet'}" if rbn is not None else "")
         + (f"\nPrognozy krzywej: {rep / 'ladder_eval.parquet'}" if not P.empty else ""))


def _write_report(cfg, name, lines):
    d = cfg["workdir"] / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


STEPS = """
Kolejność (pierwszy raz):
  1. python -m cenmodel init          utwórz config.yaml
  2. python -m cenmodel fetch         pobierz dane PSE (~10 min za pierwszym razem)
  3. python -m cenmodel verify        sprawdź replikację CEN (reguła WDB)
  4. python -m cenmodel check-data    sprawdź swoje dane (own_data w config.yaml)
  5. python -m cenmodel backtest      ile dają Twoje dane, uczciwy test poza próbą
  6. python -m cenmodel train         trenuj model do użycia
  7. python -m cenmodel predict --refresh   prognoza na najbliższe okresy
  python -m cenmodel doctor           diagnoza, gdy coś nie działa

Drabinki ofert (osobno):
  python -m cenmodel fetch --ladder 2026-01..2026-09   drabinka BPKD/BO, dzień po dniu
  python -m cenmodel fetch --offers 2026-01..2026-09   oferty poeb-rbn (opcjonalnie)
  python -m cenmodel ladder                            krzywe, PCA, prognoza krzywej -> reports/ladder.txt
"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cenmodel", description="Prognoza i replikacja CEN (PSE).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=STEPS)
    ap.add_argument("-c", "--config", help="ścieżka do config.yaml (domyślnie ./config.yaml)")
    ap.add_argument("--debug", action="store_true", help="pełny traceback przy błędach")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("init", help="utwórz config.yaml z szablonu")
    sub.add_parser("doctor", help="sprawdź instalację, API, cache, dane i model")
    f = sub.add_parser("fetch", help="pobierz/odśwież dane PSE")
    f.add_argument("--start", help="RRRR-MM-DD (domyślnie `start` z config)")
    f.add_argument("--end", help="RRRR-MM-DD (domyślnie jutro)")
    f.add_argument("--ladder", metavar="RRRR-MM[..RRRR-MM]", help="zamiast tego: drabinka BPKD/BO (oeb-bpkdbo)")
    f.add_argument("--offers", metavar="RRRR-MM[..RRRR-MM]", help="zamiast tego: wszystkie oferty RB (poeb-rbn)")
    v = sub.add_parser("verify", help="sprawdź replikację CEN z danych PSE")
    v.add_argument("--month", metavar="RRRR-MM", help="tylko ten miesiąc (+ drabinka, jeśli pobrana)")
    sub.add_parser("check-data", help="sprawdź Twoje dane (own_data)")
    b = sub.add_parser("backtest", help="test poza próbą: publiczne vs publiczne+Twoje dane")
    b.add_argument("--test-start", help="RRRR-MM-DD (domyślnie ostatnie --test-days dni)")
    b.add_argument("--test-end", help="RRRR-MM-DD (domyślnie ostatnie opublikowane CEN)")
    b.add_argument("--test-days", type=int, default=60)
    b.add_argument("--simulate-rt-sk", action="store_true",
                   help="dodaj wariant z idealnym SK w czasie rzeczywistym (górna granica)")
    t = sub.add_parser("train", help="trenuj model do predict")
    t.add_argument("--end", help="RRRR-MM-DD: trenuj tylko do tej daty")
    lp = sub.add_parser("ladder", help="krzywe cena-wolumen z drabinek: opis, PCA, prognoza krzywej")
    lp.add_argument("--start", help="RRRR-MM-DD (domyślnie cała pobrana drabinka)")
    lp.add_argument("--end", help="RRRR-MM-DD")
    lp.add_argument("--test-days", type=int, default=60, help="dni testu prognozy krzywej (domyślnie 60)")
    p = sub.add_parser("predict", help="prognoza kwantyli CEN")
    p.add_argument("--start", help="'RRRR-MM-DD HH:MM' czasu polskiego, początek pierwszego okresu")
    p.add_argument("--end", help="'RRRR-MM-DD HH:MM' czasu polskiego (domyślnie start + 1 h)")
    p.add_argument("--refresh", action="store_true", help="najpierw dociągnij świeże dane PSE")
    p.add_argument("--scenarios", action="store_true", help="zapisz też scenariusze CEN")
    p.add_argument("--out", help="plik CSV wyniku")
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 0
    try:
        cfg = config_mod.load(a.config)
        cfg["workdir"].mkdir(parents=True, exist_ok=True)
        globals()["cmd_" + a.cmd.replace("-", "_")](cfg, a)
        return 0
    except UserError as e:
        _say(f"\nBŁĄD: {e}")
        return 2
    except KeyboardInterrupt:
        _say("\nPrzerwano. Pobrane dane zostają w cache — można wznowić tą samą komendą.")
        return 130
    except Exception:                                         # noqa: BLE001
        if a.debug:
            raise
        _say("\nNieoczekiwany błąd (to bug, nie Twoja wina). Szczegóły:\n")
        traceback.print_exc()
        _say("\nWyślij powyższy tekst razem z wynikiem `python -m cenmodel doctor`.")
        return 1
