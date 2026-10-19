import numpy as np
import pandas as pd
import pytest

from cenmodel import UserError, config, owndata


def _spec(tmp_path, name="x", **kw):
    return {**config.SOURCE_DEFAULTS, "name": name, "path": tmp_path / "f.csv", **kw}


def test_realtime_never_uses_values_after_gate(tmp_path):
    t = pd.date_range("2026-10-01", periods=600, freq="1min", tz="UTC")
    pd.DataFrame({"ts": t, "v": np.arange(600.0)}).to_csv(tmp_path / "f.csv", index=False)
    s = _spec(tmp_path)
    df = owndata.load_source(s)
    idx = pd.date_range("2026-10-01 03:00", periods=12, freq="15min", tz="UTC")
    f = owndata.gate_features(s, df, idx, h=60)
    gate_minute = ((idx - pd.Timedelta(minutes=60)) - t[0]).total_seconds() / 60
    assert (f.x_v_last.to_numpy() <= gate_minute).all()
    assert np.allclose(f.x_v_last, gate_minute)
    assert np.allclose(f.x_v_d60, 60)


def test_known_delay_and_staleness(tmp_path):
    t = pd.date_range("2026-10-01", periods=10, freq="15min", tz="UTC")
    pd.DataFrame({"ts": t, "v": np.arange(10.0)}).to_csv(tmp_path / "f.csv", index=False)
    s = _spec(tmp_path, known_delay_min=5, max_staleness_min=30)
    df = owndata.load_source(s)
    idx = pd.DatetimeIndex([pd.Timestamp("2026-10-01 01:15", tz="UTC"),    # gate 00:15
                            pd.Timestamp("2026-10-01 05:00", tz="UTC")])   # gate 04:00: stale
    f = owndata.gate_features(s, df, idx, h=60)
    assert f.x_v_last.iloc[0] == 0          # the 00:15 value is known only at 00:20
    assert np.isnan(f.x_v_last.iloc[1])


def test_naive_times_need_timezone_and_polish_csv(tmp_path):
    (tmp_path / "f.csv").write_text("ts;v\n01.10.2026 12:00;1,5\n01.10.2026 12:15;2,5\n")
    with pytest.raises(UserError, match="strefy"):
        owndata.load_source(_spec(tmp_path, sep=";", decimal=","))
    df = owndata.load_source(_spec(tmp_path, sep=";", decimal=",", timezone="Europe/Warsaw"))
    assert df.index[0] == pd.Timestamp("2026-10-01 10:00", tz="UTC")   # day-first, CEST
    assert df.v.tolist() == [1.5, 2.5]


def test_forecast_latest_issue_before_gate_and_hourly_spread(tmp_path):
    rows = [("2026-10-01 00:00Z", "2026-10-01 05:00Z", 1.0),
            ("2026-10-01 03:50Z", "2026-10-01 05:00Z", 2.0),   # issued before gate 04:00
            ("2026-10-01 04:10Z", "2026-10-01 05:00Z", 3.0)]   # after gate 04:00, before 04:15
    pd.DataFrame(rows, columns=["issued_at", "target_ts", "w"]).to_csv(tmp_path / "f.csv", index=False)
    s = _spec(tmp_path, kind="forecast", target_resolution_min=60)
    df = owndata.load_source(s)
    idx = pd.date_range("2026-10-01 05:00", periods=4, freq="15min", tz="UTC")
    f = owndata.gate_features(s, df, idx, h=60)
    assert f.x_w.iloc[0] == 2.0
    assert f.x_w.iloc[1:].tolist() == [3.0, 3.0, 3.0]   # later quarters: later gates


def test_config_typo_is_caught(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("horizon_mins: 60\n")
    with pytest.raises(UserError, match="horizon_min"):
        config.load(str(p))
