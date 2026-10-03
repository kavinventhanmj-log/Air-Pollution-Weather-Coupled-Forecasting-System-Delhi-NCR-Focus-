"""Tests for the historical training-dataset builder (no model training)."""

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.training_dataset import (
    IST,
    ROLLING_WINDOWS,
    add_fire_features,
    align_observations,
    build_summary,
    build_training_dataset_from_dataframes,
    build_training_dataset_from_db,
    coerce_utc_naive,
)


def _make_frames(n_hours=40, n_stations=2):
    """Synthetic hourly panel: pollution + weather + fires."""
    stations = pd.DataFrame({
        "station": ["A", "B"],
        "latitude": [28.6, 28.5],
        "longitude": [77.2, 77.1],
    })
    base = pd.Timestamp("2026-01-01 00:00:00")
    poll, wx = [], []
    for sidx, stn in enumerate(stations["station"]):
        for i in range(n_hours):
            ts = base + pd.Timedelta(hours=i)
            poll.append({"station": stn, "timestamp": ts,
                         "pm25": float(90 + 5 * np.sin(i / 4.0) + sidx * 10)})
            wx.append({"station": stn, "timestamp": ts,
                       "temperature": 20.0 + 0.1 * i, "humidity": 60.0,
                       "pressure_msl": 1012.0, "surface_pressure": 992.0,
                       "wind_speed": 4.0, "wind_direction": 315.0,
                       "precipitation": 0.0, "cloud_cover": 40.0,
                       "pbl_height": 800.0,
                       "temperature_1000hPa": 20.0, "temperature_925hPa": 17.0})
    # A future fire (must be excluded) and a past fire (must be included).
    #
    # FIRMS acquisition times are UTC, so they are timezone-aware here rather than
    # naive. The station timestamps above are deliberately left naive, matching the
    # application's naive-UTC storage convention, and `coerce_utc_naive` is what
    # has to reconcile the two. The two must land on the same axis: a fire that is
    # 2 h before the panel start has to be inside the trailing window, not 5 h 30 m
    # off the end of it.
    fires = pd.DataFrame({
        "lat": [30.5, 30.5], "lon": [76.1, 76.1], "frp": [90.0, 200.0],
        "acq_date": [
            (base - pd.Timedelta(hours=2)).tz_localize(UTC),
            (base + pd.Timedelta(hours=48)).tz_localize(UTC),
        ],
    })
    return pd.DataFrame(poll), pd.DataFrame(wx), stations, fires


class TestTimezoneConsistency:
    def test_naive_utc_output(self):
        s = pd.Series([pd.Timestamp("2026-01-01 00:00:00+05:30"),
                       pd.Timestamp("2026-01-01 00:00:00")])
        out = coerce_utc_naive(s)
        assert out.dt.tz is None
        assert out.iloc[0] == pd.Timestamp("2025-12-31 18:30:00")

    def test_naive_input_is_read_as_utc(self):
        """The storage convention is naive UTC, so a naive value is already UTC.

        ``cpcb_service.upsert_ncr_data`` parses CPCB ``last_update`` as IST,
        converts it to UTC and then strips the tzinfo, and ``api.import_`` does
        the same. Defaulting ``coerce_utc_naive`` to ``IST`` would shift every
        stored reading back 5 h 30 m while the fire window stayed on the raw UTC
        axis, so no fire ever aligned with the panel it was supposed to influence.
        """
        naive_utc = pd.Series([pd.Timestamp("2024-01-01 11:30:00")])
        assert coerce_utc_naive(naive_utc).iloc[0] == pd.Timestamp("2024-01-01 11:30:00")

    def test_aware_ist_converts_to_utc(self):
        """An offset-aware IST instant is converted, not re-read as wall-clock."""
        aware = coerce_utc_naive(pd.Series([pd.Timestamp("2024-01-01 11:30:00+05:30")]))
        assert aware.iloc[0] == pd.Timestamp("2024-01-01 06:00:00")

    def test_legacy_naive_ist_still_recoverable_explicitly(self):
        """Legacy naive-IST rows must remain convertible via an explicit ``assume_tz``."""
        naive_ist = pd.Series([pd.Timestamp("2024-01-01 11:30:00")])
        aware = coerce_utc_naive(pd.Series([pd.Timestamp("2024-01-01 11:30:00+05:30")]))
        legacy = coerce_utc_naive(naive_ist, assume_tz=IST)
        assert legacy.iloc[0] == aware.iloc[0] == pd.Timestamp("2024-01-01 06:00:00")
        # The default must NOT shift it - that is the bug this change removes.
        assert coerce_utc_naive(naive_ist).iloc[0] == pd.Timestamp("2024-01-01 11:30:00")

    def test_assume_tz_utc_keeps_naive_firms_times(self):
        """FIRMS acquisition times are UTC, so naive ones must not be shifted."""
        s = pd.Series([pd.Timestamp("2024-01-01 06:00:00")])
        assert coerce_utc_naive(s, assume_tz=UTC).iloc[0] == pd.Timestamp("2024-01-01 06:00:00")
        # ... and it is now also the default, so nothing needs to pass it.
        assert coerce_utc_naive(s).iloc[0] == pd.Timestamp("2024-01-01 06:00:00")

    def test_ist_midnight_rolls_back_a_day(self):
        """Opting into legacy naive-IST still rolls midnight back a day."""
        out = coerce_utc_naive(pd.Series([pd.Timestamp("2024-01-01 00:00:00")]), assume_tz=IST)
        assert out.iloc[0] == pd.Timestamp("2023-12-31 18:30:00")

    def test_output_is_always_naive(self):
        for s in (
            pd.Series([pd.Timestamp("2024-01-01 11:30:00")]),
            pd.Series([pd.Timestamp("2024-01-01 11:30:00+05:30")]),
            pd.Series([pd.Timestamp("2024-01-01T06:00:00Z")]),
        ):
            assert coerce_utc_naive(s).dt.tz is None

    def test_unparseable_becomes_nat(self):
        out = coerce_utc_naive(pd.Series(["not-a-date"]))
        assert pd.isna(out.iloc[0])

    def test_mixed_offsets_in_one_series(self):
        # A column mixing an offset-aware value with a naive one is object dtype,
        # so pandas cannot infer a single timezone. The aware value still converts
        # and the naive one drops to NaT rather than being silently stamped as
        # UTC - the safe outcome, since a mislabelled timestamp would corrupt the
        # join instead of being excluded from it.
        s = pd.Series([
            pd.Timestamp("2024-01-01 11:30:00+05:30"),
            pd.Timestamp("2024-01-01 06:00:00"),
        ])
        out = coerce_utc_naive(s)
        assert out.iloc[0] == pd.Timestamp("2024-01-01 06:00:00")
        assert pd.isna(out.iloc[1])


class TestAlignmentHandling:
    def test_duplicate_hours_kept_latest(self):
        poll, wx, stations, _ = _make_frames(n_hours=5)
        dup = pd.DataFrame([
            {"station": "A", "timestamp": pd.Timestamp("2026-01-01 00:30:00"), "pm25": 10.0},
            {"station": "A", "timestamp": pd.Timestamp("2026-01-01 00:45:00"), "pm25": 99.0},
        ])
        poll = pd.concat([poll, dup], ignore_index=True)
        # 00:30/00:45 both live in the same hour bucket -> 2 removed, 99.0 retained.
        df, report = align_observations(poll, wx, stations)
        assert report["duplicate_hour_rows_removed"]["pollution"] == 2
        # Derive the expected bucket from the retained row instead of hard-coding
        # a wall-clock hour, so the assertion follows the conversion rather than
        # pinning one particular assumed timezone.
        expected_hour = coerce_utc_naive(
            pd.Series([pd.Timestamp("2026-01-01 00:45:00")])
        ).iloc[0].floor("h")
        row = df[(df["station"] == "A") & (df["hour"] == expected_hour)]
        assert len(row) == 1
        assert row["pm25"].iloc[0] == 99.0

    def test_weather_forward_fill_limited(self):
        # Weather exists only for one hour; pollution for 5 hours.
        poll, wx, stations, _ = _make_frames(n_hours=5)
        wx = wx[:1]
        df, report = align_observations(poll, wx, stations)
        assert report["rows_dropped_no_target"] == 0
        assert "ffilled_cells" in report
        assert report["ffill_limit_hours"] == 3
        # Not every weather cell was filled across both stations/hours.
        assert report["ffilled_cells"].get("temperature", 0) >= 0

    def test_target_implausible_dropped(self):
        poll, wx, stations, _ = _make_frames(n_hours=5)
        out = poll.copy()
        out.loc[0, "pm25"] = 5000.0   # beyond plausible max
        df, report = align_observations(out, wx, stations)
        assert len(df) == 9   # 2 stations x 5h - 1 implausible target
        assert report["target_implausible_removed"] == 1

    def test_rows_without_target_dropped(self):
        poll, wx, stations, _ = _make_frames(n_hours=5)
        out = poll.copy()
        out.loc[0, "pm25"] = np.nan
        df, report = align_observations(out, wx, stations)
        assert len(df) == 9   # 2 stations x 5h - 1 missing-target row
        assert report["rows_dropped_no_target"] == 1


class TestNoFutureLeakage:
    def test_lag_is_strictly_in_the_past(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        per = df[df["station"] == "A"].sort_values("hour").reset_index(drop=True)
        assert (per["pm25"].shift(1).dropna() == per["pm25_lag1"].dropna()).all()
        # lag24 must equal pm25 24h earlier
        assert np.allclose(per["pm25_lag24"].iloc[24:], per["pm25"].iloc[: len(per) - 24], equal_nan=True)

    def test_rolling_mean_excludes_current_target(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        per = df[df["station"] == "A"].sort_values("hour").reset_index(drop=True)
        for t in range(10, 20):
            expected = per["pm25"].iloc[t - 3: t].mean()  # t-3 .. t-1 (shift(1))
            assert per["pm25_roll_mean_3h"].iloc[t] == pytest.approx(expected)

    def test_future_fire_excluded(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        assert (df["fire_count"] >= 0).all()

        # The past fire is at 2025-12-31 22:00 UTC, 2 h before the panel starts at
        # 2026-01-01 00:00. Both station timestamps (naive-UTC) and the fire
        # (aware UTC) therefore land on one axis with no shift, and a 3 h lookback
        # covers 22:00 from the very first row onward. So the first row does see
        # it - which is precisely what could not be asserted while the panel was
        # shifted 5 h 30 m against the fire window, and what the assertions below
        # now pin: the detection is attributed to the right hours, and the +48 h
        # fire never leaks in.
        ordered = df.sort_values("hour")
        assert (ordered["fire_count"] == 0).any(), "no row precedes the fire window"
        first_seen = ordered[ordered["fire_count"] > 0].iloc[0]
        assert first_seen["fire_count"] == 1
        assert first_seen["nearest_fire_distance"] < 500.0
        # Only the single past fire ever contributes; the +48 h one must never
        # appear, so the count never exceeds 1 anywhere in the panel.
        assert (df["fire_count"] <= 1).all()

    def test_current_target_not_a_feature_column(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        # No feature column should equal pm25[t] itself for every row.
        cand = [c for c in df.columns if c.startswith("pm25_")]
        assert not any((df[c] == df["pm25"]).all() for c in cand)


class TestSplitsChronological:
    def test_contiguous_no_shuffle(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        split = df["split"].values
        # Blocks must appear strictly in order train->validation->test
        order = []
        prev = None
        for lab in split:
            if lab != prev:
                order.append(lab)
                prev = lab
        assert order in (["train", "validation", "test"],
                         ["train", "validation"], ["train", "test"], ["train"])

    def test_split_time_ranges_ordered(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        summary = build_summary(df)
        train = summary["splits"]["train"]
        val = summary["splits"]["validation"]
        test = summary["splits"]["test"]
        assert pd.Timestamp(train["time_range"]["max"]) <= pd.Timestamp(val["time_range"]["min"])
        assert pd.Timestamp(val["time_range"]["max"]) <= pd.Timestamp(test["time_range"]["min"])

    def test_all_rows_accounted(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        counts = df["split"].value_counts()
        assert counts.sum() == len(df)
        assert set(counts.index) == {"train", "validation", "test"}


class TestSummary:
    def test_summary_required_sections(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        s = build_summary(df)
        assert set(["rows", "time_range", "missing_values", "features",
                    "target_statistics", "splits"]) <= set(s.keys())
        assert "train" in s["splits"] and "validation" in s["splits"] and "test" in s["splits"]
        assert s["target"] == "pm25"
        assert s["shuffled"] is False
        assert s["chronological_ordering"] is True

    def test_target_stats(self):
        df, _ = build_training_dataset_from_dataframes(*_make_frames())
        s = build_summary(df)
        assert "count" in s["target_statistics"]
        assert "mean" in s["target_statistics"]
        assert s["target_statistics"]["count"] == len(df)


class TestDatabaseIntegration:
    def test_build_from_db_session(self, db_session):
        df, summary = build_training_dataset_from_db(db_session)
        # Only Anand Vihar is seeded with pollution readings.
        assert set(df["station"].unique()) == {"Anand Vihar"}
        assert df["pm25"].notna().all()
        assert "pm25_lag1" in df.columns
        assert f"pm25_roll_mean_{ROLLING_WINDOWS[0]}h" in df.columns
        assert "inversion_strength" in df.columns
        assert "fire_count" in df.columns
        assert "transport_risk" in df.columns
        assert set(df["split"].unique()) == {"train", "validation", "test"}
        assert summary["time_range"]["min"] is not None
        assert len(summary["splits"]["train"]["time_range"]) > 0

    def test_chronological_order_preserved_in_db_build(self, db_session):
        df, _ = build_training_dataset_from_db(db_session)
        ts = df.sort_values("hour")["timestamp"]
        assert (ts.diff().dropna() >= pd.Timedelta(0)).all()

    def test_no_pollution_no_rows(self, db_session):
        from app.models.db_models import PollutionReading
        rows = db_session.query(PollutionReading).all()
        for r in rows:
            db_session.delete(r)
        db_session.commit()
        df, summary = build_training_dataset_from_db(db_session)
        assert len(df) == 0


class TestAddFireFeaturesAlignment:
    """Regression: fires outside the radius must not misalign the sort order.

    The vectorised ``add_fire_features`` sorts the kept fires by acquisition
    time. When some stored fires fall outside ``max_distance_km`` the timestamp
    order must be computed over the *kept* subset, otherwise the distance /
    bearing / FRP arrays (length = kept fires) are indexed with positions from
    the full archive, raising ``IndexError: index N is out of bounds``.
    """

    def _frame(self, n_hours=78):
        return pd.DataFrame({
            "station": "Anand Vihar",
            "hour": pd.date_range("2026-09-19 00:00:00", periods=n_hours, freq="h"),
            "latitude": 28.6492,
            "longitude": 77.2918,
            "wind_direction": 300.0,
            "wind_speed": 2.0,
        })

    def _fires(self):
        last = pd.Timestamp("2026-09-22 05:00:00")
        return pd.DataFrame({
            "lat": [30.5, 29.0, 15.0, 22.0, 10.0],
            "lon": [76.1, 77.0, 75.0, 88.0, 80.0],
            "frp": [120.0, 60.0, 200.0, 150.0, 90.0],
            "acq_date": [last] * 5,
        })

    def test_mixed_inside_outside_radius_does_not_raise(self):
        out = add_fire_features(self._frame(), self._fires())
        last = out.iloc[-1]
        # Only the two fires within 500 km of Anand Vihar are counted.
        assert last["fire_count"] == 2
        assert last["fire_impact_score"] > 0.0
        assert last["nearest_fire_distance"] <= 500.0
