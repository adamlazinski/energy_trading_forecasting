"""End-to-end through the CLI on a synthetic PSE cache (no network)."""
import numpy as np
import pandas as pd

from cenmodel import cli, pse

DAYS = 110


def _fake_cache(work, rng):
    ts = pd.date_range("2026-06-01", periods=96 * DAYS, freq="15min", tz="Europe/Warsaw").tz_convert("UTC")
    n = len(ts)
    sk = np.cumsum(rng.normal(0, 30, n)) * 0.2 + rng.normal(0, 80, n)
    csdac = 400 + 150 * np.sin(np.arange(n) * 2 * np.pi / 96) + rng.normal(0, 20, n)
    zeb = -sk / 4 + rng.normal(0, 5, n)
    ckoeb = csdac + 2.0 * zeb + rng.normal(0, 30, n)
    pzeb_g, pzeb_d = np.clip(-sk / 8, 0, None), np.clip(-sk / 8, None, 0)
    ceb_pp = ckoeb + 10.82
    ceb_sr = ceb_pp + rng.normal(0, 20, n)
    cen = np.where(sk < 0, np.maximum(ceb_sr, csdac), np.minimum(ceb_sr, csdac))
    cols = {
        "crb-rozl": {"cen_cost": cen, "ckoeb_cost": ckoeb, "ceb_pp_cost": ceb_pp, "ceb_sr_cost": ceb_sr,
                     "ceb_sr_afrrg_cost": ceb_pp + 50, "ceb_sr_afrrd_cost": ceb_pp - 50},
        "price-cost": {"cor_pp_cost": np.full(n, 10.82), "contracting": np.where(sk < 0, "short", "long")},
        "zeb-rozl": {"zebpp": zeb, "pzeb_afrrg_cost": pzeb_g, "pzeb_afrrd_cost": pzeb_d},
        "sk": {"sk_cost": sk}, "csdac-pln": {"csdac_pln": csdac},
        "lolp": {f"b{i}": np.full(n, 500.0 * (i + 1)) for i in range(10)}
                | {f"p{i}": np.full(n, 1 - i / 9) for i in range(10)},
        "ro-rozl": {"ro_cost": np.full(n, 6000.0)},
    }
    hide = ts >= ts[-96 * 2]                       # last 2 days: settlement not yet published
    for slug, c in cols.items():
        df = pd.DataFrame({"ts": ts, **c})
        if slug != "csdac-pln":
            df = df[~hide]
        (work / "raw" / slug).mkdir(parents=True)
        df.to_parquet(work / "raw" / slug / "all.parquet", index=False)
    (work / "raw" / "cordmax").mkdir()
    days = pd.date_range("2026-06-01", periods=DAYS, freq="D").strftime("%Y-%m-%d")
    pd.DataFrame({"business_date": days, "cordmax_cost": 1000.0}).to_parquet(
        work / "raw" / "cordmax" / "all.parquet", index=False)
    return ts, sk


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(1)
    work = tmp_path / "work"
    ts, sk = _fake_cache(work, rng)
    # own realtime signal: noisy SK, known 2 min after each quarter ends
    known = ts + pd.Timedelta(minutes=17)
    pd.DataFrame({"time": known.tz_convert("Europe/Warsaw").strftime("%Y-%m-%d %H:%M:%S%z"),
                  "imb": sk + rng.normal(0, 20, len(sk))}).to_csv(tmp_path / "imb.csv", index=False)
    (tmp_path / "config.yaml").write_text(
        'workdir: work\nstart: "2026-06-01"\n'
        "own_data:\n  - {name: rt, path: imb.csv, kind: realtime, time_col: time}\n"
        "model: {n_scenarios: 40, holdout_days: 16}\n")
    c = ["-c", str(tmp_path / "config.yaml")]
    assert cli.main(c + ["verify"]) == 0
    assert "100.0%" in capsys.readouterr().out          # L1 rule on published CEBsr
    assert cli.main(c + ["check-data"]) == 0
    out = capsys.readouterr().out
    assert "PODEJRZANIE" not in out                       # signal is lagged, no leakage flag
    assert cli.main(c + ["backtest", "--test-days", "20"]) == 0
    out = capsys.readouterr().out
    assert "publiczne+Twoje" in out
    res = (work / "reports" / "backtest.txt").read_text()
    pub = float(next(l for l in res.splitlines() if l.startswith("publiczne ") and "structural" in l).split()[2])
    own = float(next(l for l in res.splitlines() if l.startswith("publiczne+Twoje") and "structural" in l).split()[2])
    assert own < pub                                      # the own signal must help
    assert cli.main(c + ["train"]) == 0
    start = (ts[-96 * 3]).tz_convert("Europe/Warsaw").strftime("%Y-%m-%d %H:%M")
    assert cli.main(c + ["predict", "--start", start, "--out", "p.csv", "--scenarios"]) == 0
    p = pd.read_csv(tmp_path / "p.csv")
    q = p[["q05", "q10", "q25", "q50", "q75", "q90", "q95"]].to_numpy()
    assert len(p) == 5 and (np.diff(q, axis=1) >= 0).all()
    assert (tmp_path / "p.scenarios.parquet").exists()
    assert "kalibracja na" in capsys.readouterr().out


def test_user_errors_are_friendly(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("workdir: work\n")
    assert cli.main(["-c", str(tmp_path / "config.yaml"), "train"]) == 2
    assert "python -m cenmodel fetch" in capsys.readouterr().out


def test_ts_is_period_start():
    df = pd.DataFrame({"dtime_utc": ["2026-10-14 22:15:00"]})
    assert pse.period_start_utc(df).iloc[0] == pd.Timestamp("2026-10-14 22:00", tz="UTC")


def test_refresh_fetches_whole_months():
    months = list(pse._months("2026-10-09", "2026-11-03"))
    assert [(t, lo, hi) for t, lo, hi, _ in months] == [
        ("2026-10", "2026-10-01", "2026-10-31"), ("2026-11", "2026-11-01", "2026-11-03")]
