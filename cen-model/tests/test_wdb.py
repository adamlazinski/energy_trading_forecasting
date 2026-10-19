import numpy as np
import pandas as pd

from cenmodel import wdb


def test_cen_rule_cases():
    sk = [100, 100, -100, -100, 0, 100, -100]
    ceb = [500, 300, 300, 500, 123, 5, -5]
    csdac = [400, 400, 400, 400, 400, 0, 0]
    assert list(wdb.cen_rule(sk, ceb, csdac)) == [400, 300, 400, 500, 123, -0.01, 0.01]


def _lolp_frame(ro):
    b = np.linspace(500, 5000, 10)
    p = np.linspace(1, 0, 10)
    df = pd.DataFrame({f"b{i}": b[i] for i in range(10)} | {f"p{i}": p[i] for i in range(10)},
                      index=range(len(ro)))
    return df.assign(ro=ro, cordmax=1000.0)


def test_lolp_and_cor():
    df = _lolp_frame([100, 500, 750, 5000, 9000])
    lp = wdb.lolp(df)
    assert lp[0] == 1 and lp[-1] == 0 and lp[-2] == 0
    assert np.isclose(lp[2], 1 - 0.5 / 9)               # halfway in the first segment
    cor = wdb.cor_from_lolp(df, np.zeros(5), np.full(5, 10.82))
    assert cor[0] == 1000 and cor[-1] == 0
    df.cordmax = 1.0                                     # tiny CORDMax -> CORMin binds
    assert wdb.cor_from_lolp(df, np.zeros(5), np.full(5, 10.82))[1] == 10.82


def test_ckoeb_crossing_and_replication_fix():
    ts = pd.Timestamp("2026-09-01", tz="UTC")
    lad = pd.DataFrame({"ts": ts, "cdo": [300.0, 100.0, 200.0], "ofp": [40.0, 20.0, 20.0],
                        "activ_direction": "G"})
    zeb = pd.Series([10.0], index=[ts])                  # 10 MWh -> 40 MW
    assert wdb.ckoeb_from_ladder(lad, zeb).ckoeb_hat.iloc[0] == 200.0
    rep = pd.concat([lad] * 10, ignore_index=True)       # API k-fold replication
    assert wdb.ckoeb_from_ladder(rep, zeb).ckoeb_hat.iloc[0] == 200.0
    down = lad.assign(activ_direction="D", ofp=-lad.ofp)
    assert wdb.ckoeb_from_ladder(down, -zeb).ckoeb_hat.iloc[0] == 300.0   # D: highest CDO first


def test_ceb_sr_weighting_and_cua():
    df = pd.DataFrame({"sk": [-50, -50, 50, 0], "zeb_pp": [10, 10, 5, 3],
                       "pzeb_g": [0, 10, 0, 0], "pzeb_d": [0, 0, 0, 0],
                       "ceb_afrr_g": [0, 600, 0, 0], "ceb_afrr_d": [0, 0, 0, 0]})
    est, cua = wdb.ceb_sr_from_parts(df, np.array([400.0, 400.0, 400.0, 400.0]))
    assert est[0] == 400 and est[1] == 500
    assert list(cua) == [False, False, True, True]     # long but only up activated; SK=0
