"""Unit tests for the NCR spatial IDW grid layer."""

from types import SimpleNamespace

import numpy as np
import pytest
from app.services.grid_service import (
    GRID_STEP,
    NCR_BOUNDS,
    advective_shift,
    build_grid,
    compute_ncr_grid,
    idw_interpolate,
)


def _station(name, lat, lon):
    return SimpleNamespace(name=name, latitude=lat, longitude=lon)


def _forecast(station_name, horizon, aqi):
    return {station_name: [{"horizon_hours": horizon, "aqi_pred": aqi}]}


class TestBuildGrid:
    def test_dimensions_match_domain(self):
        lats, lons = build_grid()
        assert lats[0] == pytest.approx(NCR_BOUNDS["lat_min"])
        assert lats[-1] == pytest.approx(NCR_BOUNDS["lat_max"])
        expected_nlat = int(round((NCR_BOUNDS["lat_max"] - NCR_BOUNDS["lat_min"]) / GRID_STEP)) + 1
        expected_nlon = int(round((NCR_BOUNDS["lon_max"] - NCR_BOUNDS["lon_min"]) / GRID_STEP)) + 1
        assert lats.size == expected_nlat
        assert lons.size == expected_nlon


class TestIdw:
    def test_single_station_fills_constant_field(self):
        lats, lons = build_grid()
        field = idw_interpolate(
            np.array([28.6]), np.array([77.2]), np.array([123.0]),
            lats, lons,
        )
        assert np.allclose(field, 123.0, atol=0.01)

    def test_exact_station_collocation_wins(self):
        lats, lons = build_grid()
        i, j = 20, 25
        field = idw_interpolate(
            np.array([lats[i], 28.4]), np.array([lons[j], 76.9]),
            np.array([77.0, 50.0]), lats, lons,
        )
        assert field[i, j] == pytest.approx(77.0)


class TestAdvectiveShift:
    """The transport vector must follow the meteorological from-convention.

    ``wind_dir`` is the compass bearing the wind blows *from*, so the downwind
    direction is the negation of it. This is the same conversion the ML pipeline
    uses in ``ml/preprocessing/weather_processor.py::compute_wind_components``.
    Dropping the negation advected the pollutant field *upwind*, so a westerly
    plume was drawn on the eastern side of the domain.
    """

    # (bearing, from-direction, axis that must change, sign of that change)
    COMPASS_CASES = [
        (0.0, "north", "lat", -1.0),    # from N blows south
        (90.0, "east", "lon", -1.0),    # from E blows west
        (180.0, "south", "lat", +1.0),  # from S blows north
        (270.0, "west", "lon", +1.0),   # from W blows east
    ]

    @pytest.mark.parametrize(("bearing", "name", "axis", "sign"), COMPASS_CASES)
    def test_downwind_axis_and_sign(self, bearing, name, axis, sign):
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 10.0])
        # The function returns (shifted lon index, shifted lat index).
        shifted_lon, shifted_lat = advective_shift(lon, lat, bearing, 5.0)

        moved, still = (
            (shifted_lon.mean() - lon.mean(), shifted_lat.mean() - lat.mean())
            if axis == "lon"
            else (shifted_lat.mean() - lat.mean(), shifted_lon.mean() - lon.mean())
        )
        assert moved * sign > 0, f"wind from {name} should advect {axis} {sign:+.0f}"
        assert still == pytest.approx(0.0, abs=1e-9), f"wind from {name} must not move the other axis"

    def test_westerly_shifts_eastward(self):
        """Explicit guard on the sign regression: from-west blows east."""
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 10.0])
        shifted_lon, _ = advective_shift(lon, lat, wind_dir=270.0, wind_speed=4.0)
        assert shifted_lon.mean() > lon.mean()

    def test_easterly_moves_west_cells(self):
        """Explicit guard: from-east blows west (previously moved east)."""
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 10.0])
        shifted_lon, _ = advective_shift(lon, lat, wind_dir=90.0, wind_speed=4.0)
        assert shifted_lon.mean() < lon.mean()
        assert shifted_lat_is_unchanged(lat, 90.0, 4.0)

    def test_northerly_moves_south_cells(self):
        """Explicit guard: from-north blows south (previously moved north)."""
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 10.0])
        _, shifted_lat = advective_shift(lon, lat, wind_dir=0.0, wind_speed=4.0)
        assert shifted_lat.mean() < lat.mean()

    def test_matches_the_ml_pipeline_conversion(self):
        """`advective_shift` must agree with the project's reference conversion."""
        from ml.preprocessing.weather_processor import compute_wind_components

        import pandas as pd

        for bearing in (0.0, 90.0, 180.0, 270.0, 315.0):
            speed = 4.0
            ref = compute_wind_components(
                pd.DataFrame({"wind_speed": [speed], "wind_direction": [bearing]})
            )
            # Same sign convention: u east+, v north+, both negated for a from-bearing.
            lon = np.array([0.0, 10.0])
            lat = np.array([0.0, 10.0])
            shifted_lon, shifted_lat = advective_shift(lon, lat, bearing, speed)
            d_lon = (shifted_lon.mean() - lon.mean()) / lon.size
            d_lat = (shifted_lat.mean() - lat.mean()) / lat.size
            # The grid applies a per-cell metres->cells conversion; compare the
            # sign of each axis rather than the scaled magnitude.
            assert np.sign(d_lon) == np.sign(ref["wind_u"].iloc[0]), bearing
            assert np.sign(d_lat) == np.sign(ref["wind_v"].iloc[0]), bearing


def shifted_lat_is_unchanged(lat, bearing, speed):
    """True when a purely zonal wind leaves the latitude axis untouched."""
    lon = np.zeros_like(lat)
    _, shifted_lat = advective_shift(lon, lat, bearing, speed)
    return shifted_lat.mean() == pytest.approx(lat.mean(), abs=1e-9)


class TestComputeGrid:
    def test_empty_when_no_forecasts(self):
        stations = [_station("A", 28.6, 77.2)]
        result = compute_ncr_grid(stations, {}, horizon_hours=24)
        assert result["cells"] == []

    def test_cells_have_valid_categories(self):
        stations = [
            _station("A", 28.61, 77.20),
            _station("B", 28.50, 77.05),
            _station("C", 28.70, 77.30),
        ]
        forecasts = {
            "A": [{"horizon_hours": 24, "aqi_pred": 45}],
            "B": [{"horizon_hours": 24, "aqi_pred": 150}],
            "C": [{"horizon_hours": 24, "aqi_pred": 330}],
        }
        result = compute_ncr_grid(stations, forecasts, horizon_hours=24,
                                  wind_dir=315.0, wind_speed=3.0)
        assert result["horizon_hours"] == 24
        assert len(result["cells"]) > 0
        assert result["cells"][0]["aqi_category"] in {
            "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
        }

    def test_shift_blends_field_into_downwind_cells(self):
        stations = [_station("A", 28.6, 77.2)]
        forecasts = {"A": [{"horizon_hours": 24, "aqi_pred": 200}]}
        windy = compute_ncr_grid(stations, forecasts, 24, wind_dir=90.0, wind_speed=6.0)
        assert windy["cells"] != []

    def test_advection_does_not_punch_nan_holes_at_the_border(self):
        """Cells pushed past the domain edge must fall back, not vanish.

        The shifted field is NaN outside the displaced footprint. Blending that
        NaN straight into the result removed those cells from the map, so strong
        winds silently deleted part of the domain.
        """
        stations = [_station("A", 28.6, 77.2), _station("B", 28.5, 77.0)]
        forecasts = {
            "A": [{"horizon_hours": 24, "aqi_pred": 200}],
            "B": [{"horizon_hours": 24, "aqi_pred": 180}],
        }
        calm = compute_ncr_grid(stations, forecasts, 24)
        windy = compute_ncr_grid(
            stations, forecasts, 24, wind_dir=90.0, wind_speed=12.0
        )
        # A high wind pushing the field off-domain must not reduce coverage.
        assert len(windy["cells"]) == len(calm["cells"])
        assert all(c["aqi"] is not None for c in windy["cells"])
