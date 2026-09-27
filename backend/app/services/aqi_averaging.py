"""CPCB averaging-window aggregation for observed AQI inputs.

The CPCB National AQI does not score instantaneous sensor values. It scores
*window means*:

======================  ===================
Pollutant               Averaging period
======================  ===================
PM2.5, PM10             24 hours
NO2, SO2, NH3, Pb       24 hours
O3                      8 hours
CO                      8 hours
======================  ===================

This module turns a station's raw ``PollutionReading`` time series into those
window means so :mod:`app.services.aqi_calculator` receives correctly averaged
inputs.

Design rules
------------
Trailing windows only
    The window is ``[as_of - window, as_of]``, both ends inclusive. A reading
    stamped after ``as_of`` is **excluded**, so evaluating history for time *t*
    can never see *t+1*. This is what prevents look-ahead leakage when the same
    helper is used to build features or backtests.

Real timestamps only
    Windows are positioned by each reading's own ``timestamp`` field, never by
    row order or array index, so gaps and irregular sampling are handled
    correctly.

No fabrication
    A pollutant with too few valid observations in its window yields ``None``
    ("unavailable") rather than a mean over whatever happens to be present. A
    single reading in a 24-hour window is not a 24-hour average, and emitting it
    as one would mislabel the station.

No double averaging
    Callers pass raw observations here exactly once. Results are already
    window means, so they go straight to :func:`~app.services.aqi_calculator.
    calculate_aqi` and must not be averaged again downstream.

The O3 1-hour fallback
    O3 is the one pollutant whose scoring period is not fixed. CPCB's
    *About National AQI* (footnote 5) states that when the 8-hourly O3 average
    exceeds 208 ug/m3, the 1-hourly O3 value is used **instead** for the
    sub-index. :func:`select_o3_concentration` implements that choice, so this
    module - not the calculator - decides *which* O3 concentration is scored.
    :func:`~app.services.aqi_calculator.calculate_iaqi` stays a pure
    concentration-to-IAQI mapping against the unchanged 8-hour O3 table; no
    separate 1-hour breakpoint table exists, because CPCB publishes none.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

# `_truncate_to_precision` is private to `aqi_calculator` by name only. It is
# imported deliberately: the 208 comparison *must* use the same truncation the
# sub-index uses, and a second copy of that rule is a second thing to drift.
from .aqi_calculator import (
    AVERAGING_WINDOW_HOURS,
    IAQI_CONCENTRATION_PRECISION,
    POLLUTANT_UNITS,
    _truncate_to_precision,
)

__all__ = [
    "MIN_WINDOW_COVERAGE",
    "DEFAULT_MIN_OBSERVATIONS",
    "O3_FALLBACK_THRESHOLD_UG_M3",
    "O3_BASIS_8H",
    "O3_BASIS_1H_FALLBACK",
    "O3_BASIS_8H_FALLBACK_UNAVAILABLE",
    "O3_BASIS_UNAVAILABLE",
    "normalise_timestamp",
    "min_observations_for",
    "truncate_concentration",
    "rolling_means",
    "rolling_means_by_station",
    "latest_valid_observation",
    "select_o3_concentration",
]

#: A window must be at least this fraction covered by valid observations before
#: a mean is trusted. 24 h -> 6 readings, 8 h -> 2 readings.
MIN_WINDOW_COVERAGE = 0.25

#: Floor so that even a short window needs more than a single sample.
DEFAULT_MIN_OBSERVATIONS = 2

#: CPCB *About National AQI*, footnote 5: when the 8-hourly O3 average exceeds
#: 208 ug/m3 the 1-hourly O3 value is used instead for computing the sub-index.
#: The trigger is strict - an 8-hour mean of exactly 208 still uses 8 hours.
O3_FALLBACK_THRESHOLD_UG_M3 = 208

#: Provenance labels for the O3 concentration that was actually handed to
#: :func:`~app.services.aqi_calculator.calculate_iaqi`.
O3_BASIS_8H = "8h"
O3_BASIS_1H_FALLBACK = "1h_fallback"
O3_BASIS_8H_FALLBACK_UNAVAILABLE = "8h_fallback_unavailable"
O3_BASIS_UNAVAILABLE = "unavailable"


def normalise_timestamp(value: Any) -> datetime | None:
    """Return a naive-UTC ``datetime`` for ``value``, or ``None`` if unusable.

    The application stores naive UTC wall-clock (``cpcb_service`` strips the
    tzinfo before insert), so tz-aware inputs are converted to UTC and made
    naive to keep comparisons consistent.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).replace(
                tzinfo=None
            )
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def min_observations_for(window_hours: int) -> int:
    """Minimum valid observations required to publish a window mean."""
    return max(
        DEFAULT_MIN_OBSERVATIONS,
        int(math.ceil(window_hours * MIN_WINDOW_COVERAGE)),
    )


def truncate_concentration(pollutant: str, value: float) -> float:
    """Truncate ``value`` to the precision CPCB's table for ``pollutant`` uses.

    CPCB defines the sub-index on the *truncated* concentration, so any
    threshold test on a pollutant has to truncate the same way. For O3 that is
    whole ug/m3, which is what makes an 8-hour mean of 208.9 compare as 208
    against :data:`O3_FALLBACK_THRESHOLD_UG_M3`.
    """
    return _truncate_to_precision(value, IAQI_CONCENTRATION_PRECISION.get(pollutant, 0))


def _read_field(reading: Any, name: str) -> Any:
    """Read ``name`` from an ORM row or a mapping."""
    if isinstance(reading, Mapping):
        return reading.get(name)
    return getattr(reading, name, None)


def _usable(value: Any) -> float | None:
    """Return a finite, non-negative float, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except (TypeError, ValueError):
            return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number) or number < 0.0:
        return None
    return number


def rolling_means(
    readings: Iterable[Any],
    as_of: Any,
    *,
    pollutants: Sequence[str] | None = None,
    windows: Mapping[str, int] | None = None,
    min_observations: Mapping[str, int] | None = None,
) -> dict[str, float | None]:
    """Trailing window means for one station, keyed by pollutant.

    Parameters
    ----------
    readings:
        Rows for a **single** station. Each may be an ORM
        ``PollutionReading`` or a mapping with ``timestamp`` plus pollutant
        keys. Order is irrelevant; rows outside the window are ignored.
    as_of:
        Trailing edge of the window (inclusive). Usually the latest
        observation's timestamp, or ``datetime.utcnow()`` for a live view.
    pollutants:
        Subset to average. Defaults to every pollutant in
        :data:`~app.services.aqi_calculator.AVERAGING_WINDOW_HOURS`.
    windows / min_observations:
        Optional overrides for the window length / coverage requirement.

    Returns
    -------
    dict
        ``pollutant -> mean`` for every requested pollutant. ``None`` means the
        window lacked enough valid observations. Pollutants absent from the
        input are still present in the result with ``None``, so callers can
        distinguish "not measured" from "not requested".
    """
    edge = normalise_timestamp(as_of)
    window_map = dict(windows or AVERAGING_WINDOW_HOURS)
    targets = tuple(pollutants) if pollutants is not None else tuple(window_map)

    sums: dict[str, float] = {p: 0.0 for p in targets}
    counts: dict[str, int] = {p: 0 for p in targets}
    seen_any = False

    if edge is not None:
        for reading in readings:
            stamp = normalise_timestamp(_read_field(reading, "timestamp"))
            if stamp is None:
                continue
            for pollutant in targets:
                window_hours = window_map.get(pollutant)
                if window_hours is None:
                    continue
                # Trailing window, both bounds inclusive, future rows excluded.
                if stamp > edge or stamp < edge - timedelta(hours=window_hours):
                    continue
                seen_any = True
                value = _usable(_read_field(reading, pollutant))
                if value is None:
                    continue
                sums[pollutant] += value
                counts[pollutant] += 1

    result: dict[str, float | None] = {}
    for pollutant in targets:
        required = (min_observations or {}).get(
            pollutant, min_observations_for(window_map.get(pollutant, 24))
        )
        if edge is None or not seen_any or counts[pollutant] < required:
            result[pollutant] = None
        else:
            result[pollutant] = sums[pollutant] / counts[pollutant]
    return result


def latest_valid_observation(
    readings: Iterable[Any],
    as_of: Any,
    *,
    pollutant: str,
) -> float | None:
    """Most recent valid ``pollutant`` value stamped at or before ``as_of``.

    This is *not* a window mean: no second averaging pass is applied. It
    returns the newest finite, non-negative reading that is not in the future,
    which is how this project represents CPCB's "1-hourly O3 value" - the CPCB
    feed reports hourly (and, at a few stations, 15-minute) observations, and
    the audit confirmed the feed is not already an 8-hour average, so the latest
    stored observation is the 1-hour value.

    Future rows are never candidates, so a row stamped after ``as_of`` cannot be
    returned even if it is the highest value in the table. Validation follows
    :func:`_usable`, so ``None``, ``NaN``, infinities, negatives, booleans and
    non-numeric strings are skipped and the search continues backwards rather
    than being reported as a value. ``0.0`` is valid and is returned as ``0.0``.
    """
    edge = normalise_timestamp(as_of)
    if edge is None:
        return None

    best_stamp: datetime | None = None
    best_value: float | None = None
    for reading in readings:
        stamp = normalise_timestamp(_read_field(reading, "timestamp"))
        if stamp is None or stamp > edge:
            continue
        value = _usable(_read_field(reading, pollutant))
        if value is None:
            continue
        if best_stamp is None or stamp > best_stamp:
            best_stamp, best_value = stamp, value
    return best_value


def select_o3_concentration(
    readings: Iterable[Any],
    as_of: Any,
) -> tuple[float | None, str]:
    """Choose the O3 concentration to score, per the CPCB 8h -> 1h rule.

    Implements *About National AQI* footnote 5:

        8-hourly O3 average <= 208  ->  score the 8-hour value
        8-hourly O3 average  > 208  ->  score the 1-hour value instead

    The 1-hour value **substitutes** for the 8-hour one; this is not
    ``max(8h, 1h)``. The point of the rule is that a high 8-hour mean is often
    already stale - ozone peaks in the afternoon and has dissipated by the
    evening - and scoring the mean alone would overstate the current AQI. The
    same published O3 breakpoint table is used either way; CPCB defines no
    separate 1-hour table.

    The comparison is made on the **truncated** 8-hour mean, matching how
    :func:`~app.services.aqi_calculator.calculate_iaqi` selects a band, so the
    trigger and the sub-index can never disagree about the same concentration.

    Returns
    -------
    tuple
        ``(concentration, basis)`` where ``basis`` is one of the
        ``O3_BASIS_*`` labels. The 8-hour branch returns the *untruncated*
        window mean, so a station that does not trigger is scored bit-identically
        to before this rule existed; the calculator performs its own truncation.

    Missing-data policy
    -------------------
    CPCB does not state what to do when the 1-hour value is unavailable. This
    project retains the valid 8-hour concentration and reports
    :data:`O3_BASIS_8H_FALLBACK_UNAVAILABLE`, rather than dropping O3 out of the
    AQI or inventing a value. A 1-hour value is deliberately **not** used as an
    independent default when the 8-hour window is unavailable: CPCB's rule
    substitutes the 1-hour value *for a triggered 8-hour mean*, and does not
    authorise 1-hour-only O3. That yields
    :data:`O3_BASIS_UNAVAILABLE` / ``None``, and O3 is simply not scored.

    Note that with the 1-hour value taken as the latest valid observation, the
    ``8h_fallback_unavailable`` branch cannot in practice be reached from the
    same rows that produced the 8-hour mean (a mean needs at least two valid
    readings, any of which qualifies as "the latest valid observation"). It is
    kept so the policy stays explicit and survives a future change of the 1-hour
    source, and is covered by a test.
    """
    # Materialised because both the window mean and the 1-hour value are read
    # from the same rows; a one-shot iterator would otherwise be exhausted by
    # the first pass and silently degrade to "1-hour value unavailable".
    rows = tuple(readings)

    o3_8h = rolling_means(rows, as_of, pollutants=("o3",)).get("o3")
    if o3_8h is None:
        return None, O3_BASIS_UNAVAILABLE

    if truncate_concentration("o3", o3_8h) <= O3_FALLBACK_THRESHOLD_UG_M3:
        return o3_8h, O3_BASIS_8H

    o3_1h = latest_valid_observation(rows, as_of, pollutant="o3")
    if o3_1h is None:
        return o3_8h, O3_BASIS_8H_FALLBACK_UNAVAILABLE
    return o3_1h, O3_BASIS_1H_FALLBACK


def rolling_means_by_station(
    readings: Iterable[Any],
    as_of: Any,
    *,
    station_key: str = "station_id",
    **kwargs: Any,
) -> dict[Any, dict[str, float | None]]:
    """Group ``readings`` by station, then apply :func:`rolling_means` per group.

    This is the entry point for multi-station callers: pass every station's rows
    once and receive one averaged-concentration mapping per station. The
    trailing-edge timestamp is shared, so every station is evaluated as of the
    same instant - which matters when the results are averaged into an NCR-wide
    figure.
    """
    grouped: dict[Any, list[Any]] = {}
    for reading in readings:
        key = _read_field(reading, station_key)
        grouped.setdefault(key, []).append(reading)
    return {
        key: rolling_means(rows, as_of, **kwargs)
        for key, rows in grouped.items()
    }


def window_metadata() -> dict[str, dict[str, object]]:
    """Describe each pollutant's unit and averaging window, for API responses."""
    return {
        pollutant: {"unit": POLLUTANT_UNITS.get(pollutant, "unknown"),
                    "window_hours": window}
        for pollutant, window in AVERAGING_WINDOW_HOURS.items()
    }
