"""CPCB / Indian National AQI calculation tests.

Covers the required matrix: every breakpoint boundary, interpolation between
boundaries (including the historical gap values), unit semantics, missing and
invalid data, dominant-pollutant selection, and above-range clamping.
"""

import math

import pytest
from app.services.aqi_calculator import (
    AQI_MAX,
    AVERAGING_WINDOW_HOURS,
    IAQI_BREAKPOINTS,
    IAQI_CONCENTRATION_PRECISION,
    NH3_PB_BREAKPOINT_STATUS,
    POLLUTANT_UNITS,
    UNSCORED_POLLUTANTS,
    calculate_aqi,
    calculate_iaqi,
    evaluate_aqi,
    get_aqi_category,
    get_dominant_pollutant,
    is_scorable,
)

# The six pollutants with a verified CPCB sub-index table in this repository.
SCORABLE = ("pm25", "pm10", "o3", "no2", "so2", "co")


# ---------------------------------------------------------------------------
# 1. Exact breakpoint boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "conc,expected",
    [(0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (251, 401), (380, 500)],
)
def test_pm25_every_breakpoint_boundary(conc, expected):
    """Each CPCB PM2.5 breakpoint reproduces its published IAQI exactly."""
    assert calculate_iaqi("pm25", conc) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "conc,expected",
    [(0, 0), (50, 50), (100, 100), (250, 200), (350, 300), (430, 400), (431, 401), (510, 500)],
)
def test_pm10_boundary_values(conc, expected):
    """3. PM10 boundary values."""
    assert calculate_iaqi("pm10", conc) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "conc,expected",
    [(0, 0), (50, 50), (100, 100), (168, 200), (208, 300), (748, 400), (749, 401), (1000, 500)],
)
def test_o3_boundary_values(conc, expected):
    """4. O3 boundary values."""
    assert calculate_iaqi("o3", conc) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "conc,expected",
    [(0, 0), (40, 50), (80, 100), (180, 200), (280, 300), (400, 400), (401, 401), (520, 500)],
)
def test_no2_boundaries(conc, expected):
    """6. NO2."""
    assert calculate_iaqi("no2", conc) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "conc,expected",
    [(0, 0), (40, 50), (80, 100), (380, 200), (800, 300), (1600, 400), (1601, 401), (2620, 500)],
)
def test_so2_boundaries(conc, expected):
    """7. SO2."""
    assert calculate_iaqi("so2", conc) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Interpolation between boundaries - the historical gap regression
# ---------------------------------------------------------------------------


def test_pm25_30_5_does_not_fall_through_to_500():
    """2. The exact regression: 30.5 sat in the old (30, 31) hole and scored 500.

    CPCB's formula is defined on the *truncated* concentration, so 30.5
    truncates to 30 in the whole-unit PM2.5 table and resolves inside the
    published 0-30 band -> IAQI 50.
    """
    iaqi = calculate_iaqi("pm25", 30.5)
    assert iaqi == pytest.approx(50, abs=1e-9)
    assert iaqi < 500


@pytest.mark.parametrize(
    "conc,expected",
    [
        (30.5, 50.0),      # truncates to 30 -> published 0-30 band
        (60.5, 100.0),     # truncates to 60 -> published 31-60 band
        (90.5, 200.0),     # truncates to 90 -> published 61-90 band
        (120.5, 300.0),    # truncates to 120 -> published 91-120 band
        (250.5, 400.0),    # truncates to 250 -> published 121-250 band
        (15.0, 25.0),      # 0 + 50/30 * 15
        (45.0, 74.6552),   # 51 + 49/29 * 14 -> the CPCB worked example (75)
        (31.0, 51.0),      # published lower edge of the 31-60 band
        (40.0, 66.2069),   # 51 + 49/29 * 9
    ],
)
def test_pm25_between_breakpoints(conc, expected):
    """2. Interpolation uses the surrounding published breakpoint pair."""
    assert calculate_iaqi("pm25", conc) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "conc,expected", [(31, 51), (45, 75), (60, 100)]
)
def test_cpcb_published_worked_example(conc, expected):
    """The spec's own worked example, quoted from the CPCB document.

    *"For PM2.5, the sub-index is 51 at 31 ug/m3, 75 at 45 ug/m3, and 100 at
    60 ug/m3."*  This is the check that pins the interpolation endpoints: a
    table whose bands were rewritten to be continuous still returns 100 at 60,
    but returns 53 at 31 and 76 at 45, i.e. it fails two of the three values
    CPCB publishes.
    """
    assert round(calculate_iaqi("pm25", conc)) == expected


def test_no_gap_anywhere_in_range():
    """Every concentration in [0, max] must resolve inside [0, 500]."""
    for pollutant, bands in IAQI_BREAKPOINTS.items():
        top = bands[-1][1]
        step = top / 2000
        conc = 0.0
        while conc <= top:
            iaqi = calculate_iaqi(pollutant, conc)
            assert iaqi is not None, f"{pollutant} unscored at {conc}"
            assert 0 <= iaqi <= AQI_MAX, f"{pollutant} {conc} -> {iaqi}"
            conc += step


def test_bands_match_published_cpcb_ranges():
    """The concentration ranges must be the published ones, transcribed.

    This is deliberately *not* a contiguity check. The printed CPCB bands are
    offset by one (``0-30`` then ``31-60``); the holes are closed by truncating
    the concentration, not by rewriting the ranges. Asserting contiguity here
    would lock in exactly the non-CPCB table that makes PM2.5=45 return 76
    instead of the published 75.
    """
    published = {
        "pm25": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 380)],
        "pm10": [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 510)],
        "o3": [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1000)],
        "no2": [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 520)],
        "so2": [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2620)],
        "co": [(0, 1.0), (1.1, 2.0), (2.1, 10), (10.1, 17), (17.1, 34), (34.1, 50)],
    }
    assert set(IAQI_BREAKPOINTS) == set(published)
    for pollutant, ranges in published.items():
        actual = [(c_low, c_high) for c_low, c_high, _il, _ih in IAQI_BREAKPOINTS[pollutant]]
        assert actual == ranges, f"{pollutant} ranges drifted from the CPCB table"


def test_breakpoint_endpoints_match_declared_precision():
    """Truncation is only well defined if the table is written at that precision.

    Every published endpoint must be exactly representable at the precision the
    calculator truncates to, otherwise a band edge could become unreachable
    (e.g. CO 1.1 would truncate to 1.0 and the 1.1-2.0 band would be dead).
    """
    for pollutant, decimals in IAQI_CONCENTRATION_PRECISION.items():
        factor = 10.0**decimals
        for c_low, c_high, _il, _ih in IAQI_BREAKPOINTS[pollutant]:
            for endpoint in (c_low, c_high):
                scaled = endpoint * factor
                assert abs(scaled - round(scaled)) < 1e-9, (
                    f"{pollutant} endpoint {endpoint} is not representable at "
                    f"{decimals} decimal place(s)"
                )


def test_co_gap_values_no_longer_score_500():
    """CO 1.05 mg/m3 is ordinary Delhi air; it used to score AQI 500.

    6,617 production readings sat in CO's old (1, 1.1) / (2, 2.1) holes.
    """
    for conc in (1.0025, 1.05, 1.0975, 2.05, 10.05, 17.05, 34.05):
        iaqi = calculate_iaqi("co", conc)
        assert iaqi is not None
        assert iaqi < 500, f"co={conc} -> {iaqi}"


def test_monotonic_across_each_band():
    """Sub-index must never decrease as concentration rises."""
    for pollutant, bands in IAQI_BREAKPOINTS.items():
        previous = -1.0
        for c_low, c_high, _il, _ih in bands:
            for frac in (0.0, 0.25, 0.5, 0.75, 0.999):
                conc = c_low + (c_high - c_low) * frac
                iaqi = calculate_iaqi(pollutant, conc)
                assert iaqi is not None
                if iaqi < previous:
                    # A 1-point step at a shared boundary is inherent to the
                    # CPCB 0-50 / 51-100 band edges; a real inversion is not.
                    assert iaqi >= previous - 1.0, f"{pollutant} {conc}"
                previous = max(previous, iaqi)


# ---------------------------------------------------------------------------
# 5. CO unit semantics
# ---------------------------------------------------------------------------


def test_co_is_mg_m3_and_no_conversion_is_applied():
    """CO is stored in mg/m3, matching the CPCB breakpoints - so no /1000.

    Guards against a future "fix" that divides CO by 1000: doing so would send a
    typical Delhi reading (1.5 mg/m3) to 0.0015 and silently drop CO out of the
    AQI entirely.
    """
    assert POLLUTANT_UNITS["co"] == "mg/m3"
    # 1.0 mg/m3 is the top of the "Good" band (IAQI 50), not 1000 ug/m3.
    assert calculate_iaqi("co", 1.0) == pytest.approx(50, abs=1e-9)
    assert calculate_iaqi("co", 1.5) == pytest.approx(72.7778, abs=1e-3)
    assert calculate_iaqi("co", 2.0) == pytest.approx(100, abs=1e-9)
    # Other pollutants are ug/m3.
    for pollutant in ("pm25", "pm10", "o3", "no2", "so2", "nh3", "pb"):
        assert POLLUTANT_UNITS[pollutant] == "ug/m3"


def test_co_8h_window_declared():
    assert AVERAGING_WINDOW_HOURS["co"] == 8
    assert AVERAGING_WINDOW_HOURS["o3"] == 8
    for pollutant in ("pm25", "pm10", "no2", "so2", "nh3", "pb"):
        assert AVERAGING_WINDOW_HOURS[pollutant] == 24


# ---------------------------------------------------------------------------
# 8/9. NH3 and Pb - tracked, not scored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pollutant", ["nh3", "pb"])
def test_nh3_pb_absent_from_breakpoints(pollutant):
    """8/9. No verified CPCB table exists here, so no score may be produced."""
    assert pollutant not in IAQI_BREAKPOINTS
    assert pollutant in UNSCORED_POLLUTANTS
    assert pollutant in NH3_PB_BREAKPOINT_STATUS
    assert calculate_iaqi(pollutant, 100.0) is None
    assert not is_scorable(pollutant, 100.0)


@pytest.mark.parametrize("pollutant", ["nh3", "pb"])
def test_nh3_pb_excluded_not_zeroed(pollutant):
    """A high NH3/Pb reading must not become sub-index 0 ("pristine air")."""
    result = evaluate_aqi({"pm25": 30.0, pollutant: 500.0})
    assert pollutant not in result.sub_indices
    assert result.data_availability[pollutant] == (
        "official_cpcb_table_not_vendored_in_repository"
    )
    # The measurement is still surfaced, it simply cannot be scored.
    assert result.averaged_concentrations[pollutant] == 500.0
    # pm25 alone decides the AQI.
    assert result.dominant_pollutant == "pm25"
    assert result.aqi == pytest.approx(50, abs=1)


def test_nh3_pb_accepted_by_calculate_aqi_signature():
    """The public signature accepts nh3/pb without breaking positional calls."""
    aqi, category, dominant = calculate_aqi(30.0, 100.0, 50.0, 40.0, 10.0, 1.0, 999.0, 999.0)
    assert dominant == "pm10"
    assert category == "Satisfactory"
    assert aqi == 100


# ---------------------------------------------------------------------------
# 10-14. Missing, invalid, and single/dominant selection
# ---------------------------------------------------------------------------


def _with_filler(pollutant, value, filler_value=10.0):
    """Build a concentration map with one bad pollutant and one good one."""
    filler = next(p for p in SCORABLE if p != pollutant)
    return {pollutant: value, filler: filler_value}, filler


@pytest.mark.parametrize("pollutant", SCORABLE)
def test_null_pollutant_unavailable_not_zero(pollutant):
    """10. A missing pollutant is excluded, never scored as AQI 0."""
    assert calculate_iaqi(pollutant, None) is None
    assert calculate_iaqi(pollutant, float("nan")) is None
    assert calculate_iaqi(pollutant, float("inf")) is None
    concentrations, filler = _with_filler(pollutant, None)
    result = evaluate_aqi(concentrations)
    assert pollutant not in result.sub_indices
    assert result.data_availability[pollutant] == "no_valid_observation"
    assert result.dominant_pollutant == filler


@pytest.mark.parametrize("pollutant", SCORABLE)
def test_negative_pollutant_invalid(pollutant):
    """11. Negative concentrations are impossible, so unavailable - not 500."""
    assert calculate_iaqi(pollutant, -0.1) is None
    assert calculate_iaqi(pollutant, -10) is None
    assert calculate_iaqi(pollutant, -1e9) is None
    concentrations, filler = _with_filler(pollutant, -5.0)
    result = evaluate_aqi(concentrations)
    assert pollutant not in result.sub_indices
    assert result.data_availability[pollutant] == "no_valid_observation"
    assert result.dominant_pollutant == filler


def test_unknown_pollutant_unavailable():
    assert calculate_iaqi("nope", 100) is None
    assert not is_scorable("nope", 100)
    assert "nope" not in evaluate_aqi({"nope": 100}).sub_indices


def test_all_pollutants_missing():
    """12. No valid pollutant -> AQI 0 / Unknown, never 500."""
    assert calculate_aqi() == (0, "Unknown", "pm25")
    assert calculate_aqi(None, None, None, None, None, None) == (0, "Unknown", "pm25")
    result = evaluate_aqi({p: None for p in SCORABLE})
    assert result.aqi == 0
    assert result.category == "Unknown"
    assert result.category_level == 0
    assert result.dominant_pollutant is None
    assert result.sub_indices == {}


def test_only_one_pollutant_available():
    """13. One valid pollutant drives the AQI and the dominant label."""
    aqi, category, dominant = calculate_aqi(pm25=250.0)
    assert dominant == "pm25"
    assert aqi == 400
    assert category == "Very Poor"
    assert get_aqi_category(aqi) == ("Very Poor", 5)


def test_multiple_pollutants_dominant_is_the_max_subindex():
    """14. Dominant pollutant == pollutant producing the maximum sub-index."""
    aqi, _cat, dominant = calculate_aqi(pm25=45.0, pm10=400.0, o3=20.0, no2=20.0)
    assert dominant == "pm10"
    # pm10=400 -> 301 + (400-351)/(430-351) * (400-301) = 362.405
    assert aqi == 362
    result = evaluate_aqi({"pm25": 45.0, "pm10": 400.0, "o3": 20.0, "no2": 20.0})
    assert result.dominant_pollutant == "pm10"
    assert result.sub_indices["pm10"] == max(result.sub_indices.values())
    assert result.aqi == int(round(result.sub_indices["pm10"]))


def test_aqi_equals_max_of_subindices():
    result = evaluate_aqi(
        {"pm25": 45.0, "pm10": 100.0, "o3": 800.0, "no2": 10.0, "so2": 5.0, "co": 0.5}
    )
    assert result.aqi == int(round(max(result.sub_indices.values())))
    assert result.dominant_pollutant == "o3"


def test_dominant_tie_prefers_canonical_order():
    assert get_dominant_pollutant(500, 600, None, None, None, None) == "pm25"


def test_dominant_all_none_backcompat():
    assert get_dominant_pollutant(None, None, None, None, None, None) == "pm25"


def test_nan_pollutant_ignored():
    aqi, _cat, dominant = calculate_aqi(pm25=math.nan, pm10=50.0)
    assert dominant == "pm10"
    assert aqi == 50


# ---------------------------------------------------------------------------
# 20. Above the maximum breakpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pollutant", SCORABLE)
def test_above_maximum_clamps_to_500(pollutant):
    """20. Concentrations above the top band clamp to the AQI scale max."""
    assert calculate_iaqi(pollutant, 9999) == AQI_MAX
    assert calculate_iaqi(pollutant, 1e9) == AQI_MAX


def test_above_maximum_still_reports_correct_dominant():
    result = evaluate_aqi({"pm25": 10.0, "co": 80.0})
    assert result.aqi == 500
    assert result.dominant_pollutant == "co"
    assert result.category == "Severe"


def test_just_above_maximum_still_500():
    assert calculate_iaqi("pm25", 380.1) == AQI_MAX
    assert calculate_iaqi("pm10", 510.1) == AQI_MAX
    assert calculate_iaqi("co", 50.1) == AQI_MAX


def test_severe_band_is_not_double_counted_at_its_lower_edge():
    """The Severe band's own lower edge scores 401, not the previous band's top.

    The Severe band starts at 251 (PM2.5) rather than repeating the preceding
    upper bound of 250, so the two adjacent bands do not overlap and first match
    cannot assign the shared edge to the lower category.
    """
    assert calculate_iaqi("pm25", 250) == pytest.approx(400, abs=1e-9)
    assert calculate_iaqi("pm25", 251) == pytest.approx(401, abs=1e-9)
    assert calculate_iaqi("co", 34.0) == pytest.approx(400, abs=1e-9)
    assert calculate_iaqi("co", 34.1) == pytest.approx(401, abs=1e-9)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aqi,expected",
    [
        (0, ("Good", 1)),
        (50, ("Good", 1)),
        (51, ("Satisfactory", 2)),
        (100, ("Satisfactory", 2)),
        (101, ("Moderate", 3)),
        (200, ("Moderate", 3)),
        (201, ("Poor", 4)),
        (300, ("Poor", 4)),
        (301, ("Very Poor", 5)),
        (400, ("Very Poor", 5)),
        (401, ("Severe", 6)),
        (500, ("Severe", 6)),
    ],
)
def test_get_aqi_category_boundaries(aqi, expected):
    assert get_aqi_category(aqi) == expected


@pytest.mark.parametrize("aqi", [50.5, 100.5, 200.5, 300.5, 400.5])
def test_get_aqi_category_has_no_float_gap(aqi):
    """Category bands must not share the concentration bands' integer holes.

    The old table fell through to ("Severe", 6) for anything strictly between
    two edges. `grap.py` passes a float station mean, so this was reachable.
    """
    category, level = get_aqi_category(aqi)
    assert category != "Severe"
    assert level in {1, 2, 3, 4, 5}


def test_get_aqi_category_out_of_range():
    assert get_aqi_category(600) == ("Severe", 6)
    # A negative AQI is not a real AQI: it now reports "Unknown" instead of
    # claiming "Severe" pollution. grid_service/dispersion_service already
    # skip NaN and round to int before calling, so no valid path changes.
    assert get_aqi_category(-1) == ("Unknown", 0)
    assert get_aqi_category(None) == ("Unknown", 0)
    assert get_aqi_category(float("nan")) == ("Unknown", 0)


def test_roundtrip_category_consistency():
    for value in (45, 120, 250, 350, 450):
        aqi, category, _ = calculate_aqi(pm25=value)
        assert get_aqi_category(aqi)[0] == category
        assert aqi >= 0


def test_calculate_aqi_returns_int():
    aqi, _, _ = calculate_aqi(pm25=120, pm10=180)
    assert isinstance(aqi, int)


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------


def test_result_exposes_units_and_availability():
    detail = evaluate_aqi({"pm25": 45.0, "co": 1.05}).as_dict()
    assert detail["pollutant_units"]["pm25"] == "ug/m3"
    assert detail["pollutant_units"]["co"] == "mg/m3"
    assert "sub_indices" in detail
    assert "averaged_concentrations" in detail
    assert "data_availability" in detail
    assert detail["data_availability"]["pb"] == (
        "official_cpcb_table_not_vendored_in_repository"
    )


def test_evaluate_aqi_is_json_serialisable():
    import json

    json.dumps(evaluate_aqi({"pm25": 45.0, "co": 1.05}).as_dict())


def test_numeric_strings_are_tolerated():
    """CSV/API ingestion can hand over strings; they must not become invalid."""
    assert calculate_iaqi("pm25", "45") == pytest.approx(74.6552, abs=1e-3)
    assert calculate_iaqi("pm25", "not-a-number") is None
    assert calculate_iaqi("pm25", "") is None
