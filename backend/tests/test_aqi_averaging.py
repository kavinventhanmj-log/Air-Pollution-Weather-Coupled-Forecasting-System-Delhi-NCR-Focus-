"""CPCB averaging-window aggregation tests.

Covers requirement D: trailing windows, 24-hour PM, 8-hour O3/CO, no future-data
leakage, safe handling of missing observations, and returning unavailable rather
than fabricating a mean.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.services.aqi_averaging import (
    MIN_WINDOW_COVERAGE,
    O3_BASIS_1H_FALLBACK,
    O3_BASIS_8H,
    O3_BASIS_8H_FALLBACK_UNAVAILABLE,
    O3_BASIS_UNAVAILABLE,
    O3_FALLBACK_THRESHOLD_UG_M3,
    latest_valid_observation,
    min_observations_for,
    normalise_timestamp,
    rolling_means,
    rolling_means_by_station,
    select_o3_concentration,
    truncate_concentration,
    window_metadata,
)
from app.services.aqi_calculator import AVERAGING_WINDOW_HOURS, calculate_iaqi, evaluate_aqi

BASE = datetime(2026, 9, 17, 12, 0, 0)  # naive UTC, matching stored rows


def _series(values, *, pollutant="pm25", start=BASE, step_hours=1):
    """Build ``count`` readings ending at BASE with the given hourly values."""
    n = len(values)
    out = []
    for i, value in enumerate(values):
        ts = start - timedelta(hours=step_hours * (n - 1 - i))
        out.append({"timestamp": ts, pollutant: value})
    return out


# ---------------------------------------------------------------------------
# Window sizing
# ---------------------------------------------------------------------------


def test_window_lengths_match_cpcb():
    assert min_observations_for(24) == 6   # 24 * 0.25
    assert min_observations_for(8) == 2    # max(2, 8 * 0.25)
    assert MIN_WINDOW_COVERAGE == 0.25
    assert window_metadata()["co"] == {"unit": "mg/m3", "window_hours": 8}
    assert window_metadata()["pm25"] == {"unit": "ug/m3", "window_hours": 24}
    assert set(AVERAGING_WINDOW_HOURS) == {
        "pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb"
    }


def test_timestamp_normalisation():
    assert normalise_timestamp(BASE) == BASE
    aware = BASE.replace(tzinfo=UTC)
    assert normalise_timestamp(aware) == BASE
    assert normalise_timestamp(None) is None
    assert normalise_timestamp("nonsense") is None
    assert normalise_timestamp(float("nan")) is None


# ---------------------------------------------------------------------------
# 15. 24-hour PM averaging
# ---------------------------------------------------------------------------


def test_pm25_24h_mean_over_trailing_day():
    """15. 24-hour PM2.5 mean, trailing window ending at BASE."""
    readings = _series([10.0] * 24)  # 24 hourly readings ending at BASE
    means = rolling_means(readings, BASE)
    assert means["pm25"] == pytest.approx(10.0)
    # pm10 is simply absent from this series, so it is unavailable, not 0.
    assert means["pm10"] is None


def test_pm25_24h_mean_is_the_plain_average():
    values = [float(i) for i in range(1, 25)]  # 1..24, mean 12.5
    means = rolling_means(_series(values), BASE)
    assert means["pm25"] == pytest.approx(12.5)


def test_pm25_excludes_readings_older_than_24h():
    """A reading 25 h old must not drag the 24-hour mean down."""
    readings = _series([0.0] * 20) + [
        {"timestamp": BASE - timedelta(hours=25), "pm25": 1000.0}
    ]
    means = rolling_means(readings, BASE)
    assert means["pm25"] == pytest.approx(0.0)


def test_pm25_window_boundary_is_inclusive():
    """Exactly 24 h old is inside the window; 24 h + 1 s is not."""
    on_edge = _series([5.0] * 23) + [{"timestamp": BASE - timedelta(hours=24), "pm25": 9.0}]
    # 24 values inside the window: 23 x 5.0 plus the 9.0 sitting on the edge.
    assert rolling_means(on_edge, BASE)["pm25"] == pytest.approx((23 * 5.0 + 9.0) / 24)

    just_outside = _series([5.0] * 23) + [
        {"timestamp": BASE - timedelta(hours=24, seconds=1), "pm25": 9.0}
    ]
    # The 9.0 drops out, leaving the 23 readings that are safely inside.
    assert rolling_means(just_outside, BASE)["pm25"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 16/17. 8-hour O3 and CO averaging
# ---------------------------------------------------------------------------


def test_o3_8h_mean():
    """16. O3 uses an 8-hour window."""
    readings = _series([40.0] * 12, pollutant="o3")
    means = rolling_means(readings, BASE)
    assert means["o3"] == pytest.approx(40.0)


def test_o3_8h_window_excludes_older_data():
    """An O3 reading 9 h old is outside the 8-hour window."""
    readings = _series([10.0] * 10, pollutant="o3") + [
        {"timestamp": BASE - timedelta(hours=9), "o3": 900.0}
    ]
    assert rolling_means(readings, BASE)["o3"] == pytest.approx(10.0)


def test_co_8h_mean_in_mg_m3():
    """17. CO 8-hour mean, values already in mg/m3 - no conversion here."""
    readings = _series([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], pollutant="co")
    means = rolling_means(readings, BASE)
    assert means["co"] == pytest.approx(2.75)
    # Scoring the mean gives the expected band, and the unit is unchanged.
    # CO is published to 1 decimal, so the mean truncates to 2.7 and then
    # 101 + (200-101)/(10-2.1) * (2.7-2.1) = 108.518987
    result = evaluate_aqi({"co": means["co"]})
    assert result.sub_indices["co"] == pytest.approx(108.518987, abs=1e-6)
    assert result.as_dict()["pollutant_units"]["co"] == "mg/m3"


def test_pm_and_o3_use_different_window_lengths():
    """The same series must yield different means for a 24 h vs 8 h window."""
    readings = _series([0.0] * 20 + [100.0] * 4, pollutant="pm25")
    readings += [
        {"timestamp": r["timestamp"], "o3": r["pm25"]} for r in readings
    ]
    means = rolling_means(readings, BASE)
    # The 24 h window is inclusive at both ends, so it holds all 24 readings.
    assert means["pm25"] == pytest.approx(100.0 * 4 / 24)
    # The 8 h window spans hours 0..8 inclusive: 5 clean + 4 elevated.
    assert means["o3"] == pytest.approx(100.0 * 4 / 9)


# ---------------------------------------------------------------------------
# 18. No future-data leakage
# ---------------------------------------------------------------------------


def test_future_observations_are_ignored():
    """18. Evaluating as of BASE must never see a reading after BASE."""
    past = _series([20.0] * 8)
    future = [
        {"timestamp": BASE + timedelta(hours=h), "pm25": 900.0} for h in range(1, 6)
    ]
    assert rolling_means(past + future, BASE)["pm25"] == pytest.approx(20.0)


def test_leakage_check_across_every_pollutant():
    past = [
        {
            "timestamp": BASE - timedelta(hours=h),
            "pm25": 10.0, "pm10": 20.0, "o3": 30.0,
            "no2": 40.0, "so2": 5.0, "co": 0.5,
        }
        for h in range(24)
    ]
    future = [
        {
            "timestamp": BASE + timedelta(hours=h),
            "pm25": 1e6, "pm10": 1e6, "o3": 1e6,
            "no2": 1e6, "so2": 1e6, "co": 1e6,
        }
        for h in range(1, 10)
    ]
    means = rolling_means(past + future, BASE)
    assert means["pm25"] == pytest.approx(10.0)
    assert means["pm10"] == pytest.approx(20.0)
    assert means["o3"] == pytest.approx(30.0)
    assert means["no2"] == pytest.approx(40.0)
    assert means["so2"] == pytest.approx(5.0)
    assert means["co"] == pytest.approx(0.5)


def test_results_are_order_independent():
    readings = _series([float(i) for i in range(1, 13)])
    forward = rolling_means(readings, BASE)
    backward = rolling_means(list(reversed(readings)), BASE)
    assert forward == backward


# ---------------------------------------------------------------------------
# 19. Missing observations inside a window
# ---------------------------------------------------------------------------


def test_none_values_inside_window_are_skipped():
    """19. Gaps are skipped, not treated as zero."""
    readings = [
        {"timestamp": BASE - timedelta(hours=h), "pm25": (None if h % 2 else 40.0)}
        for h in range(12)
    ]
    means = rolling_means(readings, BASE)
    assert means["pm25"] == pytest.approx(40.0)


def test_nan_and_negative_values_are_skipped():
    readings = [
        {"timestamp": BASE - timedelta(hours=h), "pm25": v}
        for h, v in enumerate([40.0, float("nan"), -5.0, 40.0, 40.0, 40.0,
                               40.0, 40.0, 40.0])
    ]
    means = rolling_means(readings, BASE)
    assert means["pm25"] == pytest.approx(40.0)


def test_insufficient_history_returns_none_not_a_fabricated_mean():
    """A single reading is not a 24-hour average, so it must be unavailable."""
    means = rolling_means(_series([50.0]), BASE)
    assert means["pm25"] is None
    assert means["pm10"] is None


def test_five_readings_short_of_the_24h_requirement():
    """24 h needs 6 observations; 5 is not enough."""
    assert rolling_means(_series([50.0] * 5), BASE)["pm25"] is None
    assert rolling_means(_series([50.0] * 6), BASE)["pm25"] == pytest.approx(50.0)


def test_eight_hour_window_needs_only_two_observations():
    assert rolling_means(_series([50.0], pollutant="o3"), BASE)["o3"] is None
    assert rolling_means(_series([50.0] * 2, pollutant="o3"), BASE)["o3"] == pytest.approx(50.0)


def test_empty_input_returns_all_none():
    means = rolling_means([], BASE)
    assert set(means) >= {"pm25", "pm10", "o3", "no2", "so2", "co"}
    assert all(v is None for v in means.values())


def test_unavailable_pollutant_is_excluded_from_aqi():
    """A None mean must drop the pollutant, not contribute AQI 0."""
    readings = [
        {"timestamp": BASE - timedelta(hours=h), "pm25": 45.0, "o3": None}
        for h in range(24)
    ]
    means = rolling_means(readings, BASE)
    assert means["o3"] is None
    result = evaluate_aqi(means)
    assert "o3" not in result.sub_indices
    assert result.data_availability["o3"] == "no_valid_observation"
    assert result.dominant_pollutant == "pm25"
    assert result.aqi == 75


def test_nh3_pb_none_never_scored():
    means = rolling_means(
        [{"timestamp": BASE - timedelta(hours=h), "nh3": 30.0, "pb": 0.4}
         for h in range(24)],
        BASE,
    )
    assert means["nh3"] == pytest.approx(30.0)
    assert means["pb"] == pytest.approx(0.4)
    result = evaluate_aqi(means)
    assert "nh3" not in result.sub_indices
    assert "pb" not in result.sub_indices


# ---------------------------------------------------------------------------
# Grouping by station
# ---------------------------------------------------------------------------


def test_rolling_means_by_station_groups_correctly():
    readings = [
        {"station_id": 1, "timestamp": BASE - timedelta(hours=h), "pm25": 10.0}
        for h in range(24)
    ] + [
        {"station_id": 2, "timestamp": BASE - timedelta(hours=h), "pm25": 300.0}
        for h in range(24)
    ]
    grouped = rolling_means_by_station(readings, BASE)
    assert set(grouped) == {1, 2}
    assert grouped[1]["pm25"] == pytest.approx(10.0)
    assert grouped[2]["pm25"] == pytest.approx(300.0)


def test_rolling_means_by_station_shares_one_trailing_edge():
    """Stations must be evaluated as of the same instant."""
    readings = [
        {"station_id": 1, "timestamp": BASE - timedelta(hours=h), "pm25": 10.0}
        for h in range(24)
    ] + [
        # Station 2 is stale - nothing in the last 24 h.
        {"station_id": 2, "timestamp": BASE - timedelta(hours=48), "pm25": 300.0}
    ]
    grouped = rolling_means_by_station(readings, BASE)
    assert grouped[1]["pm25"] == pytest.approx(10.0)
    assert grouped[2]["pm25"] is None


def test_accepts_orm_style_objects():
    class Row:
        def __init__(self, ts, value):
            self.timestamp = ts
            self.pm25 = value

    rows = [Row(BASE - timedelta(hours=h), 40.0) for h in range(12)]
    assert rolling_means(rows, BASE)["pm25"] == pytest.approx(40.0)


def test_pollutant_subset_selection():
    readings = _series([40.0] * 12)
    means = rolling_means(readings, BASE, pollutants=["pm25"])
    assert set(means) == {"pm25"}


def test_explicit_min_observations_override():
    readings = _series([40.0] * 3)
    assert rolling_means(readings, BASE, min_observations={"pm25": 3})["pm25"] == pytest.approx(40.0)
    assert rolling_means(readings, BASE, min_observations={"pm25": 9})["pm25"] is None


def test_custom_window_override():
    readings = _series([10.0] * 6)
    means = rolling_means(readings, BASE, pollutants=["pm25"], windows={"pm25": 4})
    assert means["pm25"] == pytest.approx(10.0)
    # With a 4 h window only the last 4 readings qualify, still a valid mean.
    assert rolling_means(readings, BASE, pollutants=["pm25"], windows={"pm25": 2},
                         min_observations={"pm25": 9})["pm25"] is None


# ---------------------------------------------------------------------------
# CPCB O3 rule: 8-hour mean above 208 ug/m3 -> use the 1-hour value instead
# ---------------------------------------------------------------------------

#: Nine hourly readings ending at BASE all fall inside the inclusive 8 h
#: window, so the window mean is the plain arithmetic mean of the series.
_O3_WINDOW_SLOTS = 9


def _o3_episode(mean_8h, latest, slots=_O3_WINDOW_SLOTS):
    """Hourly O3 series with 8 h mean `mean_8h` and newest reading `latest`.

    The background is solved rather than hard-coded so the 8-hour mean and the
    1-hour value can be stated independently - which is the whole point of the
    substitution tests.
    """
    background = [(mean_8h * slots - latest) / (slots - 1)] * (slots - 1)
    return _series(background + [latest], pollutant="o3")


def test_o3_fallback_threshold_matches_cpcb():
    assert O3_FALLBACK_THRESHOLD_UG_M3 == 208
    assert truncate_concentration("o3", 207.9) == 207.0
    assert truncate_concentration("o3", 208.9) == 208.0
    assert truncate_concentration("o3", 209.0) == 209.0


# --- Trigger boundary ------------------------------------------------------


@pytest.mark.parametrize("mean_8h", [207.9, 208.0, 208.9])
def test_o3_at_or_below_208_uses_the_8h_value(mean_8h):
    """207.9 -> 207, and 208.9 -> 208 after truncation: the trigger is strict."""
    readings = _series([mean_8h] * 8, pollutant="o3")
    assert rolling_means(readings, BASE)["o3"] == pytest.approx(mean_8h)
    value, basis = select_o3_concentration(readings, BASE)
    assert basis == O3_BASIS_8H
    assert value == pytest.approx(mean_8h)


def test_o3_above_208_activates_the_fallback():
    value, basis = select_o3_concentration(_series([209.0] * 8, pollutant="o3"), BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(209.0)


# --- Substitution, including downwards -------------------------------------


def test_1h_value_substitutes_for_a_higher_8h_mean():
    """8 h 280, 1 h 340: the 1 h value is used, not max(280, 340)."""
    readings = _o3_episode(280.0, 340.0)
    assert rolling_means(readings, BASE)["o3"] == pytest.approx(280.0)
    value, basis = select_o3_concentration(readings, BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(340.0)
    # Not the 8-hour mean. (This pair cannot separate substitution from max,
    # since 340 is also the larger value - the reversed case below does that.)


def test_1h_value_substitutes_for_a_lower_8h_mean():
    """8 h 300, 1 h 220: the rule lowers the sub-index, it does not take a max."""
    readings = _o3_episode(300.0, 220.0)
    assert rolling_means(readings, BASE)["o3"] == pytest.approx(300.0)
    value, basis = select_o3_concentration(readings, BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(220.0)
    # This pair is what rules out max(8h, 1h): a max would have returned 300.
    assert value < 300.0


def test_iaqi_is_computed_from_the_selected_concentration():
    value, basis = select_o3_concentration(_o3_episode(280.0, 340.0), BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert calculate_iaqi("o3", value) == calculate_iaqi("o3", 340.0)
    assert calculate_iaqi("o3", value) != calculate_iaqi("o3", 280.0)


def test_evaluate_aqi_scores_the_substituted_o3_value():
    rows = [dict(r, pm25=10.0) for r in _o3_episode(280.0, 340.0)]
    value, basis = select_o3_concentration(rows, BASE)
    result = evaluate_aqi({**rolling_means(rows, BASE), "o3": value})
    assert basis == O3_BASIS_1H_FALLBACK
    assert result.sub_indices["o3"] == calculate_iaqi("o3", 340.0)
    # PM2.5 10.0 -> IAQI 16.7, so the substituted O3 value takes dominance.
    assert result.dominant_pollutant == "o3"


# --- Missing data ----------------------------------------------------------


def test_triggered_fallback_without_a_1h_value_retains_the_8h_value(monkeypatch):
    """Documented project policy for a case CPCB does not specify."""
    import app.services.aqi_averaging as averaging

    monkeypatch.setattr(averaging, "latest_valid_observation", lambda *a, **k: None)
    value, basis = select_o3_concentration(_o3_episode(300.0, 220.0), BASE)
    assert value == pytest.approx(300.0)
    assert basis == O3_BASIS_8H_FALLBACK_UNAVAILABLE


def test_8h_unavailable_never_borrows_the_1h_value():
    """One valid reading cannot make an 8 h mean, so O3 is simply not scored."""
    readings = _series([None] * 8, pollutant="o3")
    readings[-1]["o3"] = 300.0
    assert rolling_means(readings, BASE)["o3"] is None
    assert latest_valid_observation(readings, BASE, pollutant="o3") == pytest.approx(300.0)
    value, basis = select_o3_concentration(readings, BASE)
    assert value is None
    assert basis == O3_BASIS_UNAVAILABLE


def test_both_periods_unavailable():
    value, basis = select_o3_concentration(_series([None] * 8, pollutant="o3"), BASE)
    assert value is None
    assert basis == O3_BASIS_UNAVAILABLE


def test_no_history_at_all():
    assert select_o3_concentration([], BASE) == (None, O3_BASIS_UNAVAILABLE)


def test_valid_zero_is_a_value_not_a_gap():
    """8 h 280 triggered by a 1 h value of exactly 0 - that is clean air."""
    value, basis = select_o3_concentration(_o3_episode(280.0, 0.0), BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == 0.0
    assert calculate_iaqi("o3", value) == 0.0


# --- Invalid values --------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [None, float("nan"), float("inf"), float("-inf"), -5.0, "abc", True, [1.0]],
)
def test_invalid_latest_value_is_skipped_for_the_previous_valid_one(bad):
    readings = [
        {"timestamp": BASE - timedelta(hours=1), "o3": 300.0},
        {"timestamp": BASE, "o3": bad},
    ]
    assert latest_valid_observation(readings, BASE, pollutant="o3") == pytest.approx(300.0)


def test_latest_valid_observation_is_none_when_nothing_is_usable():
    readings = [{"timestamp": BASE, "o3": v} for v in (None, float("nan"), -1.0, "x")]
    assert latest_valid_observation(readings, BASE, pollutant="o3") is None


def test_latest_valid_observation_keeps_existing_string_coercion():
    """Numeric strings are coerced, as `rolling_means` already did."""
    readings = [{"timestamp": BASE, "o3": " 340.0 "}]
    assert latest_valid_observation(readings, BASE, pollutant="o3") == pytest.approx(340.0)


# --- Leakage and the as_of boundary ---------------------------------------


def test_future_o3_is_never_used_for_the_fallback_or_the_mean():
    readings = _o3_episode(280.0, 340.0) + [
        {"timestamp": BASE + timedelta(hours=1), "o3": 9999.0}
    ]
    value, basis = select_o3_concentration(readings, BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(340.0)
    assert rolling_means(readings, BASE, pollutants=["o3"])["o3"] == pytest.approx(280.0)


def test_row_exactly_at_as_of_is_included():
    background = (280.0 * _O3_WINDOW_SLOTS - 340.0) / (_O3_WINDOW_SLOTS - 1)
    readings = _o3_episode(280.0, 340.0)
    assert latest_valid_observation(readings, BASE, pollutant="o3") == pytest.approx(340.0)
    # One second earlier the 340 reading has not happened yet.
    assert latest_valid_observation(
        readings, BASE - timedelta(seconds=1), pollutant="o3"
    ) == pytest.approx(background)


def test_select_accepts_a_one_shot_iterator():
    """Both passes read the same rows, so an exhausted iterator must not
    silently degrade to "1-hour value unavailable"."""
    value, basis = select_o3_concentration(iter(_o3_episode(280.0, 340.0)), BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(340.0)


def test_averaging_windows_still_report_eight_hours_for_o3():
    assert AVERAGING_WINDOW_HOURS["o3"] == 8
    assert window_metadata()["o3"]["window_hours"] == 8


# --- Non-interference ------------------------------------------------------


def test_other_pollutants_are_untouched_by_the_o3_rule():
    rows = [
        {
            "timestamp": BASE - timedelta(hours=h),
            "pm25": 40.0 + h, "pm10": 90.0, "no2": 30.0, "so2": 8.0, "co": 0.9,
            "o3": 300.0,
        }
        for h in range(24)
    ]
    means = rolling_means(rows, BASE)
    assert means["pm25"] == pytest.approx(sum(40.0 + h for h in range(24)) / 24)
    assert means["pm10"] == pytest.approx(90.0)
    assert means["no2"] == pytest.approx(30.0)
    assert means["so2"] == pytest.approx(8.0)
    assert means["co"] == pytest.approx(0.9)
    # O3 triggers the substitution, yet the reported window mean is unchanged.
    value, basis = select_o3_concentration(rows, BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(300.0)
    assert means["o3"] == pytest.approx(300.0)


def test_real_observed_episode_scores_the_collapsed_1h_value():
    """Anand Vihar 2024-10-21 08:00: 8 h mean 251 while the hour had fallen to 38.

    Truncating to 251 -> IAQI 309 would keep a station in "Very Poor" hours
    after the episode had broken; the 1-hour value scores 38 instead.
    """
    value, basis = select_o3_concentration(_o3_episode(251.0, 38.0), BASE)
    assert basis == O3_BASIS_1H_FALLBACK
    assert value == pytest.approx(38.0)
    assert calculate_iaqi("o3", value) == pytest.approx(38.0)
    # The stale 8-hour mean would keep the station in the 301-400 "Very Poor"
    # band; the 1-hour value collapses it to the bottom of the table.
    assert calculate_iaqi("o3", 251.0) > 300.0
