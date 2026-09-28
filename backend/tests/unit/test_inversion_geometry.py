"""Unit tests for inversion layer geometry and episode statistics (SIH26082).

The audit recorded inversion base/top *height*, layer thickness, and
duration/persistence as unimplemented because only the pressures were
reported. These tests pin the geometry derived from the pressure-level
profile and the temporal behaviour measured over the hourly archive.
"""

import pandas as pd
import pytest

from ml.features.atmospheric_profile import (
    GRAVITY_M_S2,
    HEIGHT_REFERENCE_HPA,
    R_DRY_AIR_J_KG_K,
    classify_gradient,
    combine_inversion,
    compute_lapse_rates,
    layer_heights_m,
    pressure_to_height_m,
)
from ml.features.inversion import (
    EPISODE_GAP_TOLERANCE_H,
    add_lapse_rate_inversion_features,
    summarise_inversion_episode,
)

INVERTED = {1000: 12.0, 925: 10.0, 850: 14.0, 700: 8.0}


class TestPressureToHeight:
    def test_reference_level_is_zero(self):
        assert pressure_to_height_m(HEIGHT_REFERENCE_HPA, 20.0) == 0.0

    def test_matches_hypsometric_equation(self):
        # dz = (R_d * T / g) * ln(p_ref / p)
        import math

        expected = (R_DRY_AIR_J_KG_K * (15.0 + 273.15) / GRAVITY_M_S2) * math.log(1000.0 / 925.0)
        assert pressure_to_height_m(925, 15.0) == pytest.approx(expected, rel=1e-9)

    def test_height_increases_upward(self):
        assert pressure_to_height_m(700, 0.0) > pressure_to_height_m(850, 5.0) > pressure_to_height_m(925, 10.0)

    def test_non_physical_input_returns_none(self):
        assert pressure_to_height_m(0, 20.0) is None
        assert pressure_to_height_m(-10, 20.0) is None
        assert pressure_to_height_m(None, 20.0) is None
        assert pressure_to_height_m(925, None) is None
        assert pressure_to_height_m(925, -400.0) is None

    def test_above_reference_returns_none(self):
        # Cannot be extrapolated upward from a 1000 hPa reference.
        assert pressure_to_height_m(1050, 20.0) is None


class TestLayerHeights:
    def test_integrates_upward_from_reference(self):
        heights = layer_heights_m({1000: 20.0, 925: 10.0, 850: 5.0, 700: 0.0})
        assert heights[1000] == 0.0
        assert heights[925] > 0
        assert heights[850] > heights[925]
        assert heights[700] > heights[850]

    def test_geopotential_supplied_wins(self):
        heights = layer_heights_m({925: 10.0, 850: 5.0}, {925: 810.0, 850: 1500.0})
        assert heights[925] == 810.0
        assert heights[850] == 1500.0

    def test_mixed_supplied_and_derived(self):
        heights = layer_heights_m({1000: 20.0, 925: 10.0, 850: 5.0}, {925: 800.0})
        assert heights[925] == 800.0
        # 850 is derived upward from the supplied 925 hPa height.
        assert heights[850] > 800.0

    def test_single_level_is_not_enough(self):
        assert layer_heights_m({1000: 20.0}) == {}

    def test_empty_profile(self):
        assert layer_heights_m({}) == {}
        assert layer_heights_m(None) == {}


class TestGradientGeometry:
    def test_base_top_height_and_thickness(self):
        res = classify_gradient(compute_lapse_rates(INVERTED), INVERTED)
        assert res["inversion_base_pressure"] == 925
        assert res["inversion_top_pressure"] == 850
        assert res["inversion_base_height_m"] is not None
        assert res["inversion_top_height_m"] > res["inversion_base_height_m"]
        assert res["inversion_thickness_m"] == pytest.approx(
            res["inversion_top_height_m"] - res["inversion_base_height_m"], abs=0.05
        )
        assert res["inversion_thickness_hpa"] == 75.0

    def test_thickness_matches_pressure_span(self):
        """Thickness in hPa is exactly the pressure span of the layer."""
        res = classify_gradient(compute_lapse_rates(INVERTED), INVERTED)
        assert res["inversion_thickness_hpa"] == pytest.approx(
            res["inversion_base_pressure"] - res["inversion_top_pressure"], abs=0.05
        )

    def test_thickness_tracks_layer_depth(self):
        """A deeper layer (more pressure extent) is thicker in metres."""
        # 850->700 spans 150 hPa, so it is thicker than the 925->850 layer.
        deep = {1000: 25.0, 925: 20.0, 850: 15.0, 700: 25.0}
        deep_res = classify_gradient(compute_lapse_rates(deep), deep)
        shallow_res = classify_gradient(compute_lapse_rates(INVERTED), INVERTED)
        assert deep_res["inversion_base_pressure"] == 850
        assert deep_res["inversion_thickness_hpa"] == 150.0
        assert deep_res["inversion_thickness_m"] > shallow_res["inversion_thickness_m"]

    def test_no_profile_reports_no_geometry(self):
        res = classify_gradient({})
        assert res["inversion_base_height_m"] is None
        assert res["inversion_thickness_m"] is None
        assert res["inversion_thickness_hpa"] is None
        assert res["profile_available"] is False

    def test_geometry_reported_even_without_inversion(self):
        """Height/thickness describe the strongest layer, not only inversions."""
        stable = {1000: 20.0, 925: 15.0, 850: 10.0, 700: 5.0}
        res = classify_gradient(compute_lapse_rates(stable), stable)
        assert res["inversion_detected"] is False
        assert res["inversion_thickness_m"] is not None
        assert res["inversion_thickness_m"] > 0


class TestProxyHasNoFabricatedGeometry:
    def test_pbl_proxy_reports_no_thickness(self):
        res = combine_inversion(120.0, None)
        assert res["inversion_source"] == "pbl_proxy"
        assert res["inversion_thickness_m"] is None
        assert res["inversion_base_height_m"] is None
        assert res["inversion_thickness_hpa"] is None

    def test_lapse_rate_source_reports_geometry(self):
        res = combine_inversion(120.0, INVERTED)
        assert res["inversion_source"] == "lapse_rate"
        assert res["inversion_thickness_m"] is not None


def _hourly(count, start="2026-09-26T00:00:00Z"):
    base = pd.Timestamp(start)
    return [base + pd.Timedelta(hours=h) for h in range(count)]


class TestEpisodeDuration:
    def test_current_run_duration(self):
        ts = _hourly(25)
        flags = [False] * 6 + [True] * 19
        ep = summarise_inversion_episode(ts, flags, window_h=24)
        # 19 consecutive hourly samples = 19 h of observed inversion.
        assert ep["current_duration_h"] == pytest.approx(19.0, abs=0.01)
        assert ep["onset"] == pd.Timestamp("2026-09-26T06:00:00Z")
        assert ep["samples"] == 25

    def test_no_inversion_now(self):
        ts = _hourly(10)
        ep = summarise_inversion_episode(ts, [True] * 5 + [False] * 5, window_h=24)
        assert ep["current_duration_h"] == 0.0
        assert ep["onset"] is None

    def test_gap_breaks_the_run(self):
        """A sampling gap must not be reported as observed persistence."""
        base = pd.Timestamp("2026-09-26T00:00:00Z")
        ts = [base + pd.Timedelta(hours=h) for h in [0, 1, 2, 20, 21, 22]]
        flags = [False, False, False, True, True, True]
        ep = summarise_inversion_episode(ts, flags, window_h=24)
        # 3 hourly samples, not 22 h -- the 18 h gap is not observation.
        assert ep["current_duration_h"] == pytest.approx(3.0, abs=0.01)
        assert ep["onset"] == base + pd.Timedelta(hours=20)

    def test_small_gap_tolerated(self):
        base = pd.Timestamp("2026-09-26T00:00:00Z")
        ts = [base + pd.Timedelta(hours=h) for h in [0, 1, 2, 3, 4]]
        ep = summarise_inversion_episode(ts, [True] * 5, window_h=24)
        # 5 consecutive hourly samples = 5 h observed.
        assert ep["current_duration_h"] == pytest.approx(5.0, abs=0.01)
        assert ep["onset"] == base

    def test_unsorted_input_is_sorted(self):
        ts = _hourly(12)
        shuffled = list(reversed(ts))
        flags = [True] * 12
        ep = summarise_inversion_episode(shuffled, flags, window_h=24)
        assert ep["onset"] == pd.Timestamp("2026-09-26T00:00:00Z")
        assert ep["current_duration_h"] == pytest.approx(12.0, abs=0.01)

    def test_single_sample(self):
        ep = summarise_inversion_episode([pd.Timestamp("2026-09-26T00:00:00Z")], [True])
        assert ep["samples"] == 1
        assert ep["sufficient_history"] is False


class TestEpisodePersistence:
    def test_full_persistence(self):
        ep = summarise_inversion_episode(_hourly(25), [True] * 25, window_h=24)
        assert ep["persistence_fraction"] == pytest.approx(1.0)
        assert ep["sufficient_history"] is True
        assert ep["complete"] is True

    def test_fraction_of_window(self):
        ts = _hourly(25)
        flags = [False] + [True] * 24
        ep = summarise_inversion_episode(ts, flags, window_h=24)
        assert ep["persistence_fraction"] == pytest.approx(24 / 25, abs=0.01)

    def test_short_history_flagged(self):
        """A 6 h archive must not present a confident 24 h persistence claim."""
        ep = summarise_inversion_episode(_hourly(6), [True] * 6, window_h=24)
        assert ep["sufficient_history"] is False
        assert ep["complete"] is False
        assert ep["measured_window_h"] == pytest.approx(5.0, abs=0.01)
        assert ep["samples"] == 6

    def test_measured_window_disclosed(self):
        ep = summarise_inversion_episode(_hourly(49), [True] * 49, window_h=24)
        assert ep["measured_window_h"] == pytest.approx(48.0, abs=0.01)
        assert ep["persistence_window_h"] == 24


class TestEpisodeEdgeCases:
    def test_empty_input(self):
        ep = summarise_inversion_episode([], [])
        assert ep["current_duration_h"] == 0.0
        assert ep["persistence_fraction"] is None
        assert ep["samples"] == 0
        assert ep["sufficient_history"] is False

    def test_all_unparseable_timestamps(self):
        ep = summarise_inversion_episode(["not-a-date", "also-bad"], [True, True])
        assert ep["samples"] == 0

    def test_unparseable_timestamps_dropped(self):
        ts = [None, "not-a-date"] + _hourly(3)
        ep = summarise_inversion_episode(ts, [True, True, True, True, True])
        assert ep["samples"] == 3

    def test_gap_tolerance_is_documented_constant(self):
        assert EPISODE_GAP_TOLERANCE_H == 3.0

    @pytest.mark.parametrize("count", [2, 3, 12, 19, 25])
    def test_duration_is_resolution_independent(self, count):
        """N hourly samples must report N hours on every supported pandas.

        The duration is the run's clock span plus one sampling interval, so it
        depends on the *typical* interval being derived correctly. That was
        previously computed as ``np.diff(series.astype("int64")) / 3.6e12``,
        which hard-codes a nanosecond divisor. pandas 3 resolves datetime64 to
        microseconds, so every interval became 0.001 h and each episode was
        reported an hour short - 19 h of inversion as 18.0. These tests pass on
        pandas 2.2 locally and failed on the pandas 3 that CI installs, which is
        exactly the divergence this pins shut.
        """
        flags = [True] * count
        ep = summarise_inversion_episode(_hourly(count), flags, window_h=24)
        assert ep["current_duration_h"] == pytest.approx(float(count), abs=0.01)

    def test_typical_interval_matches_the_sampling_cadence(self):
        """A 15-minute feed must add 0.25 h per sample, not a hard-coded 1 h.

        The added term is the *observed* interval, not a fixed hour, so a
        sub-hourly archive reports its own cadence instead of being inflated.
        8 samples at 15 min span 7 intervals (1.75 h) and report 8 (2.0 h),
        matching the "N samples cover N intervals" rule used above.
        """
        base = pd.Timestamp("2026-09-26T00:00:00Z")
        ts = [base + pd.Timedelta(minutes=15 * i) for i in range(8)]
        ep = summarise_inversion_episode(ts, [True] * 8, window_h=24)
        assert ep["current_duration_h"] == pytest.approx(2.0, abs=0.01)
        # measured_window_h is the raw 1.75 h span rounded to one decimal.
        assert ep["measured_window_h"] == pytest.approx(1.8, abs=0.05)


class TestLapseRateFeatureFrameColumns:
    def test_geometry_columns_added(self):
        df = pd.DataFrame(
            {
                "pbl_height": [500.0, 500.0],
                "temperature_1000hPa": [12.0, 20.0],
                "temperature_925hPa": [10.0, 15.0],
                "temperature_850hPa": [14.0, 10.0],
                "geopotential_height_925hPa": [780.0, 790.0],
                "geopotential_height_850hPa": [1490.0, 1500.0],
            }
        )
        out = add_lapse_rate_inversion_features(df)
        for col in (
            "inversion_base_height_m",
            "inversion_top_height_m",
            "inversion_thickness_m",
            "inversion_thickness_hpa",
        ):
            assert col in out.columns
        # Real geopotential supplied for the inversion layer.
        assert out["inversion_base_height_m"].iloc[0] == pytest.approx(780.0, abs=0.01)
        assert out["inversion_thickness_m"].iloc[0] == pytest.approx(1490.0 - 780.0, abs=0.01)
        # Second row: normal lapse, no inversion, but geometry still measured.
        assert out["inversion_thickness_m"].iloc[1] is not None
