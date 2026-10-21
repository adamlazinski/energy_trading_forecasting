import numpy as np
import pandas as pd
import pytest

from cenmodel import UserError, cli, ladder as L, wdb
from tests.test_pipeline import _fake_cache

T0 = pd.Timestamp("2026-09-01 10:00", tz="UTC")


def _bands():
    # G: 100 MW @ 300, 200 MW @ 500 (start-up corrected), 50 MW @ 900; D: 100 MW @ 200, 300 MW @ -50
    return pd.DataFrame({
        "ts": [T0] * 5, "dir": ["G", "G", "G", "D", "D"],
        "price": [500.0, 300.0, 900.0, -50.0, 200.0], "mw": [200.0, 100.0, 50.0, 300.0, 100.0],
        "startup": [True, False, False, False, False]})


def test_curves_merit_order_and_exhaustion():
    c = L.curves(_bands()).set_index("dir")
    assert c.loc["G", "p100"] == 300 and c.loc["G", "p150"] == 500 and c.loc["G", "p300"] == 500
    assert np.isnan(c.loc["G", "p400"])                    # 350 MW only -> exhausted
    assert c.loc["G", "ladder_mw"] == 350 and c.loc["G", "p_top"] == 900
    assert c.loc["D", "p100"] == 200 and c.loc["D", "p400"] == -50   # D: highest price first


def test_cross_is_ckoeb_with_startup_flag():
    x = L.cross(_bands(), pd.Series([50.0], index=[T0]))  # 50 MWh -> 200 MW
    g = x.loc[(T0, "G")]
    assert g.ckoeb_hat == 500 and g.need_mw == 200 and g.startup_marginal


def test_at_volume_interpolates_and_clamps():
    curve = np.array([[np.nan] * len(L.GRID)])
    curve[0, :3] = [100, 100, 400]                       # 25, 50, 100 MW
    assert L.at_volume(curve, np.array([25.0]))[0] == pytest.approx(100)
    assert 100 < L.at_volume(curve, np.array([75.0]))[0] < 400
    assert L.at_volume(curve, np.array([5000.0]))[0] == pytest.approx(400)


def test_replication_fix_keeps_ofc():
    lad = pd.DataFrame({"ts": [T0] * 20, "cdo": [1.0, 2.0] * 10, "ofc": [1.0, 1.5] * 10,
                        "ofp": [5.0] * 20, "activ_direction": ["G"] * 20})
    out = wdb.undo_api_replication(lad)
    assert len(out) == 2 and "ofc" in out


def test_month_range():
    assert cli._month_range("2026-01..2026-03")[0] == "2026-01-01"
    with pytest.raises(UserError):
        cli._month_range("2026-13")


def test_ladder_command_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(3)
    work = tmp_path / "work"
    ts, sk = _fake_cache(work, rng)
    zeb = pd.read_parquet(work / "raw" / "zeb-rozl" / "all.parquet").set_index("ts").zebpp
    d = work / "raw" / "ladder"
    d.mkdir()
    rows = []
    for t, z in zeb.items():                             # ladder whose crossing defines CKOEB
        up = z > 0
        p = np.sort(rng.normal(500 if up else 200, 80, 12))
        p = p if up else p[::-1]
        rows.append(pd.DataFrame({"ts": t, "cdo": p.round(2), "ofc": p.round(2), "ofp": np.full(12, 150.0),
                                  "activ_direction": "G" if up else "D"}))
    lad = pd.concat(rows)
    for day, g in lad.groupby(lad.ts.dt.tz_convert("Europe/Warsaw").dt.strftime("%Y-%m-%d")):
        g.to_parquet(d / f"{day}.parquet", index=False)
    ck = L.cross(L.bands_bpkdbo(lad), zeb).reset_index(level=1, drop=True).ckoeb_hat
    crb = pd.read_parquet(work / "raw" / "crb-rozl" / "all.parquet")
    crb["ckoeb_cost"] = crb.ts.map(ck)
    crb.to_parquet(work / "raw" / "crb-rozl" / "all.parquet", index=False)
    (tmp_path / "config.yaml").write_text('workdir: work\nstart: "2026-06-01"\n')
    assert cli.main(["-c", str(tmp_path / "config.yaml"), "ladder", "--test-days", "20"]) == 0
    out = capsys.readouterr().out
    assert "opublikowane w 100.00%" in out
    assert "PCA" in out and "regresja" in out
    assert (work / "reports" / "ladder_curves.parquet").exists()
    P = pd.read_parquet(work / "reports" / "ladder_eval.parquet")
    oracle = P[P.model.str.startswith("[wyrocznia")]
    assert (oracle.pred - oracle.ckoeb).abs().median() < 1e-6   # true curve reproduces CKOEB
