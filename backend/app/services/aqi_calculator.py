"""Indian National Air Quality Index (CPCB / MoEFCC) calculation.

This module is the **single source of truth** for AQI in AeroCast-NCR. No API
route, service, or frontend component re-implements breakpoint or category
logic; they all call in here.

Breakpoint tables
-----------------
``IAQI_BREAKPOINTS`` holds the CPCB National AQI sub-index bands exactly as
published by the Central Pollution Control Board in *About National Air Quality
Index* (``https://airquality.cpcb.gov.in/ccr_docs/About_AQI.pdf``). Each entry
is ``(C_low, C_high, IAQI_low, IAQI_high)`` and the bands are **transcribed, not
derived**::

    PM2.5  0-30, 31-60, 61-90, 91-120, 121-250, 251-380
    PM10   0-50, 51-100, 101-250, 251-350, 351-430, 431-510
    NO2    0-40, 41-80, 81-180, 181-280, 281-400, 401-520
    O3     0-50, 51-100, 101-168, 169-208, 209-748, 749-1000  (8-hour)
    SO2    0-40, 41-80, 81-380, 381-800, 801-1600, 1601-2620
    CO     0-1.0, 1.1-2.0, 2.1-10, 10.1-17, 17.1-34, 34.1-50  (8-hour, mg/m3)

Interpolation is CPCB's own formula, in which ``Cp`` is the **truncated**
concentration::

    Ip  = [(IHi - ILo) / (BPHi - BPLo)] * (Cp - BPLo) + ILo
    BPLo / BPHi  the published breakpoints less than / equal to, and greater
                 than, the truncated concentration

Truncation is what makes the published ranges work: the printed bands are
offset by one (``0-30`` then ``31-60``), so a raw concentration of ``30.5``
belongs to no band. Truncating to the table's own precision - whole units for
every pollutant except CO, which is published to one decimal - lands it on ``30``
and the gaps disappear. See :data:`IAQI_CONCENTRATION_PRECISION`.

Do **not** "close" the gaps by rewriting the lower bounds to the previous band's
upper bound. That produces a self-consistent table, but it is not the CPCB one:
it changes the interpolation endpoints and therefore the sub-index. The CPCB
document's own worked example pins this down - *"the sub-index is 51 at
31 ug/m3, 75 at 45 ug/m3, and 100 at 60 ug/m3"* - which only the transcribed
bands reproduce::

    PM2.5 45 -> 51 + (100-51)/(60-31) * (45-31) + 0 = 74.655 -> 75
    PM2.5 31 -> 51                                        (published 51)
    PM2.5 60 -> 100                                       (published 100)

A continuous-band rewrite returns 76, 53 and 100 for those three inputs, i.e. it
fails two of the three values CPCB states outright. An earlier revision of this
module did exactly that; the boundary-preserving version it replaced scored
ordinary in-band values correctly and only mis-handled the published holes,
which sent concentrations in those holes to AQI 500 ("Severe"). Truncation
fixes the holes *without* disturbing the in-band values.

Units
-----
Concentrations are stored and passed in the units below, which are the same
units the CPCB breakpoint tables are expressed in. **No unit conversion is
performed inside this module**, so there is exactly one conversion boundary in
the whole system (currently none, because the ingest already matches).

===================  ==========  =========================
Pollutant            Unit        CPCB averaging period
===================  ==========  =========================
PM2.5, PM10          ug/m3       24-hour
NO2, SO2             ug/m3       24-hour
O3                   ug/m3        8-hour
CO                   **mg/m3**    8-hour
NH3, Pb              ug/m3       24-hour
===================  ==========  =========================

.. note::
   **O3 1-hour fallback.** CPCB's O3 rule is an 8-hour average *plus* a
   substitution: *About National AQI* footnote 5 states that when the 8-hourly
   O3 average exceeds 208 ug/m3, the 1-hourly O3 value is used instead for
   computing the sub-index. That selection is implemented in
   :func:`app.services.aqi_averaging.select_o3_concentration`, which returns the
   concentration to score and a basis label; it is deliberately *not* here,
   because choosing an averaging period is not part of mapping a concentration
   onto a band. The 8-hour O3 table above is used for both periods - CPCB
   publishes no separate 1-hour table - and the 1-hour value replaces the 8-hour
   value rather than being combined with it.

   In this project the "1-hourly value" is the latest valid stored O3
   observation at the evaluation instant, not a second rolling mean: the CPCB
   feed reports hourly readings (15-minute at a few stations) and those readings
   are not themselves 8-hour averages, so the newest one already represents an
   hour. If the 8-hour mean triggers the substitution but no such observation
   exists, the 8-hour value is retained and reported as
   ``8h_fallback_unavailable``; that is a project reporting policy chosen
   because CPCB does not specify the case, not a CPCB requirement. An O3
   sub-index is never formed from a 1-hour value alone.

.. warning::
   **CO is mg/m3 in this project, not ug/m3.** This is asserted by the ingest
   code (``refresh_service.PMAP`` -> ``"CO (mg/m3)"``), by
   ``docs/SCIENTIFIC_METHODOLOGY.md`` and by the data itself (stored CO has a
   mean of ~1.5 and a max of ~80; values >= 100 do not occur). Because the CPCB
   CO breakpoints are also mg/m3, **no conversion is required or performed**.
   Dividing CO by 1000 here would drive a typical Delhi reading to ~0 and
   silently drop CO out of the AQI entirely.

Missing / invalid data
----------------------
A pollutant that is absent, ``None``, ``NaN``, infinite, or negative is
**unavailable**, not zero. AQI 0 means "pristine, Good air", so scoring a
missing pollutant as 0 would quietly drag the overall AQI down and mislabel a
polluted station as Good. Unavailable pollutants are *excluded* from the
maximum and reported through :attr:`AQIResult.data_availability`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "IAQI_BREAKPOINTS",
    "IAQI_CONCENTRATION_PRECISION",
    "AQI_CATEGORIES",
    "POLLUTANT_UNITS",
    "AVERAGING_WINDOW_HOURS",
    "AQI_MAX",
    "UNSCORED_POLLUTANTS",
    "AQIResult",
    "calculate_iaqi",
    "get_aqi_category",
    "get_dominant_pollutant",
    "calculate_aqi",
    "evaluate_aqi",
    "is_scorable",
]

# Upper bound of the AQI scale. CPCB publishes six bands from 0 to 500.
AQI_MAX = 500

# --------------------------------------------------------------------------
# Pollutant registry
# --------------------------------------------------------------------------

#: Storage/ingest unit per pollutant. CO is mg/m3 (see module docstring).
POLLUTANT_UNITS: dict[str, str] = {
    "pm25": "ug/m3",
    "pm10": "ug/m3",
    "no2": "ug/m3",
    "so2": "ug/m3",
    "o3": "ug/m3",
    "co": "mg/m3",
    "nh3": "ug/m3",
    "pb": "ug/m3",
}

#: CPCB sub-index averaging period in hours. PM/O3/CO are the ones the CPCB
#: rulebook pins to a specific window; NO2/SO2/NH3/Pb use the 24-hour period.
AVERAGING_WINDOW_HOURS: dict[str, int] = {
    "pm25": 24,
    "pm10": 24,
    "no2": 24,
    "so2": 24,
    "o3": 8,
    "co": 8,
    "nh3": 24,
    "pb": 24,
}

#: Pollutants the project tracks but cannot score yet, because no verified CPCB
#: breakpoint table for them is available. See ``NH3_PB_BREAKPOINT_STATUS``.
UNSCORED_POLLUTANTS: frozenset[str] = frozenset({"nh3", "pb"})

# --------------------------------------------------------------------------
# CPCB sub-index breakpoints  (transcribed from the published table)
# --------------------------------------------------------------------------

#: Decimal precision of each pollutant's published concentration breakpoints.
#:
#: CPCB's formula is defined on the **truncated** concentration, so ``Cp`` is
#: truncated to this many decimals *before* a band is selected. That is what
#: resolves the one-unit holes the printed table leaves between bands
#: (``0-30`` / ``31-60`` -> ``30.5`` truncates to ``30``). Every table is
#: published in whole units except CO, which is published to one decimal
#: (``1.1-2.0``, ``10.1-17``, ``17.1-34``), so CO truncates to one decimal and
#: ``co=10.05`` lands on ``10.0`` rather than falling into a hole.
#:
#: :func:`test_breakpoint_endpoints_match_declared_precision` asserts that every
#: concentration breakpoint in :data:`IAQI_BREAKPOINTS` is exactly
#: representable at its declared precision, so this map cannot silently drift
#: away from the table it describes.
IAQI_CONCENTRATION_PRECISION: dict[str, int] = {
    "pm25": 0,
    "pm10": 0,
    "o3": 0,
    "no2": 0,
    "so2": 0,
    "co": 1,
}

#: ``pollutant -> [(C_low, C_high, IAQI_low, IAQI_high), ...]``
#:
#: Transcribed from the CPCB *About National Air Quality Index* table; see the
#: module docstring for the full printed grid. Concentration endpoints are
#: written as the document prints them - whole numbers as ``int``, CO's decimal
#: endpoints as ``float`` - because the representation is what the
#: :data:`IAQI_CONCENTRATION_PRECISION` truncation is defined against.
#:
#: The ``C_high`` of the top band is the concentration at which the sub-index
#: reaches 500. *About National Air Quality Index* prints the Severe band as
#: open-ended (``250+``, ``430+``, ``748+``, ``400+``, ``1600+``, ``34+``) and
#: gives no upper endpoint, so the Severe band's upper bound is taken from
#: CPCB's own Control Room AQI calculator
#: (``https://cpcbccr.com/ccr_docs/AQI-Calculator.xls``), which publishes the
#: complete grid including IAQI 401-500:
#:
#: ==========  ==============  ==============
#: Pollutant   Severe band     Reaches 500 at
#: ==========  ==============  ==============
#: PM2.5       251-380         380
#: PM10        431-510         510
#: O3          749-1000        1000
#: NO2         401-520         520
#: SO2         1601-2620       2620
#: CO          34.1-50         50
#: ==========  ==============  ==============
#:
#: An earlier revision used project-chosen uppers (500/600/1200/800/2100/50)
#: because the PDF alone does not determine them. Those were arbitrary: with an
#: open-ended ``+`` band, any value is a guess, and the guess was wrong by up to
#: 69 AQI points (NO2 520 scored 431). Concentrations at or above these
#: published upper bounds clamp to 500.
#:
#: NH3 and Pb are deliberately absent. CPCB does publish sub-index bands for
#: them (the Control Room calculator renders them under the same IAQI scale), so
#: the gap is a sourcing decision, not an oversight: those tables are **not
#: vendored anywhere in this repository** and are not reproduced here, because
#: an unverified breakpoint table is worse than an acknowledged gap - it would
#: emit a confidently wrong regulatory number. They are registered in
#: :data:`POLLUTANT_UNITS` / :data:`AVERAGING_WINDOW_HOURS` so real
#: measurements are still stored and surfaced, and they are reported as
#: unavailable by :func:`evaluate_aqi`.
IAQI_BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    # PM2.5, ug/m3, 24-hour average
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 380, 401, 500),
    ],
    # PM10, ug/m3, 24-hour average
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 510, 401, 500),
    ],
    # O3, ug/m3, 8-hour average
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    # NO2, ug/m3, 24-hour average
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 520, 401, 500),
    ],
    # SO2, ug/m3, 24-hour average
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2620, 401, 500),
    ],
    # CO, mg/m3, 8-hour average  (see module docstring: already mg/m3)
    "co": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500),
    ],
}

#: Human-readable reason a tracked pollutant carries no sub-index, surfaced in
#: :attr:`AQIResult.data_availability` so the gap is visible rather than silent.
#:
#: CPCB does publish NH3 and Pb breakpoints; the reason recorded here is that
#: they are not *vendored and independently checked* in this repository, which
#: is what the scoring code is allowed to rely on.
NH3_PB_BREAKPOINT_STATUS: dict[str, str] = {
    "nh3": "official_cpcb_table_not_vendored_in_repository",
    "pb": "official_cpcb_table_not_vendored_in_repository",
}

#: AQI category bands as ``(IAQI_low_inclusive, IAQI_high_inclusive, label, level)``.
#: These are **contiguous** (``0-50``, ``51-100``, ``101-200``, ...). A previous
#: version shared the same integer gaps as the concentration bands, so
#: ``get_aqi_category(50.5)`` matched no band and fell through to
#: ``("Severe", 6)`` - reachable in production because ``grap.py`` passes a
#: float station average into this function.
AQI_CATEGORIES: list[tuple[float, float, str, int]] = [
    (0, 50, "Good", 1),
    (51, 100, "Satisfactory", 2),
    (101, 200, "Moderate", 3),
    (201, 300, "Poor", 4),
    (301, 400, "Very Poor", 5),
    (401, 500, "Severe", 6),
]


@dataclass(frozen=True)
class AQIResult:
    """Full outcome of an AQI evaluation.

    Attributes
    ----------
    aqi:
        Overall AQI = max of the valid sub-indices, rounded to the nearest int.
        ``0`` when no pollutant could be scored.
    category, category_level:
        AQI band label and its 1-6 severity level. ``("Unknown", 0)`` when
        nothing was scorable.
    dominant_pollutant:
        Pollutant that produced :attr:`aqi`. ``None`` when nothing was
        scorable.
    sub_indices:
        ``pollutant -> IAQI`` for every pollutant that could be scored.
    averaged_concentrations:
        The concentrations actually used, i.e. the *averaged* values when
        averaging was applied (or the raw values when it was not).
    data_availability:
        ``pollutant -> reason`` for each pollutant that was **excluded**,
        so callers can tell "no data" apart from "clean air".
    """

    aqi: int
    category: str
    category_level: int
    dominant_pollutant: str | None
    sub_indices: dict[str, float] = field(default_factory=dict)
    averaged_concentrations: dict[str, float] = field(default_factory=dict)
    data_availability: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable form used by API responses."""
        return {
            "aqi": self.aqi,
            "category": self.category,
            "category_level": self.category_level,
            "dominant_pollutant": self.dominant_pollutant,
            "sub_indices": {k: round(v, 1) for k, v in self.sub_indices.items()},
            "averaged_concentrations": {
                k: round(v, 3) for k, v in self.averaged_concentrations.items()
            },
            "pollutant_units": {
                k: POLLUTANT_UNITS.get(k, "unknown") for k in self.averaged_concentrations
            },
            "data_availability": dict(self.data_availability),
        }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _clean(concentration: object) -> float | None:
    """Return a usable, finite, non-negative float, else ``None``."""
    if concentration is None or isinstance(concentration, bool):
        return None
    if isinstance(concentration, str):
        # Tolerate numeric strings from CSV/API ingestion without inventing data.
        try:
            concentration = float(concentration.strip())
        except (TypeError, ValueError):
            return None
    if not isinstance(concentration, (int, float)):
        return None
    value = float(concentration)
    if math.isnan(value) or math.isinf(value):
        return None
    if value < 0.0:
        # Physically impossible; treat as unavailable rather than clamping to 0,
        # which would read as "pristine air".
        return None
    return value


def is_scorable(pollutant: str, concentration: object) -> bool:
    """True when ``concentration`` can produce a sub-index for ``pollutant``."""
    return pollutant in IAQI_BREAKPOINTS and _clean(concentration) is not None


#: Slack added before flooring so that a value which *is* exactly representable
#: at the target precision is not dropped a step by binary-float error
#: (``1.7 * 10`` evaluates to ``16.999999999999996``). Small enough that a
#: genuine sub-precision remainder still truncates down.
_TRUNCATION_GUARD = 1e-9


def _truncate_to_precision(value: float, decimals: int) -> float:
    """Truncate ``value`` toward zero at ``decimals`` decimal places.

    CPCB defines its sub-index formula on the *truncated* concentration, and
    this is the step that makes the published bands usable: they are printed
    with a one-unit offset between them, so a raw ``30.5`` matches no band
    until it is truncated to the whole unit the table is written in.
    """
    if decimals <= 0:
        return float(math.floor(value + _TRUNCATION_GUARD))
    factor = 10.0**decimals
    return math.floor(value * factor + _TRUNCATION_GUARD) / factor


# --------------------------------------------------------------------------
# Core calculation
# --------------------------------------------------------------------------


def calculate_iaqi(pollutant: str, concentration: object) -> float | None:
    """Interpolate the CPCB sub-index for one pollutant.

    Parameters
    ----------
    pollutant:
        Key into :data:`IAQI_BREAKPOINTS` (``pm25``, ``pm10``, ``no2``,
        ``so2``, ``o3``, ``co``).
    concentration:
        Concentration in the pollutant's :data:`POLLUTANT_UNITS` unit, already
        averaged over the CPCB window if the caller is computing an observed
        AQI.

    Returns
    -------
    float | None
        The sub-index, or ``None`` when the input cannot be scored: unknown or
        unscored pollutant, missing value, NaN, infinity, or negative
        concentration. ``None`` means *unavailable*; it must never be coerced to
        ``0``, which would mean "Good".

    Notes
    -----
    The concentration is truncated to the published precision of its table
    (:data:`IAQI_CONCENTRATION_PRECISION`) before a band is chosen, per the
    CPCB definition of ``Cp``. That is what closes the one-unit holes between
    the printed bands, so there is no concentration in range that fails to match
    a band and no path to a spurious AQI 500.

    Exact published breakpoints reproduce their published sub-index exactly,
    because a band is selected with closed bounds and first match wins. The
    published grid is contiguous - consecutive bands are offset by one
    (``121-250`` then ``251-380``) and truncation removes any in-between value -
    so no band overlaps itself and no value falls through.
    """
    value = _clean(concentration)
    if value is None:
        return None
    breakpoints = IAQI_BREAKPOINTS.get(pollutant)
    if not breakpoints:
        return None

    value = _truncate_to_precision(value, IAQI_CONCENTRATION_PRECISION.get(pollutant, 0))

    top = breakpoints[-1]
    if value >= top[1]:
        return float(AQI_MAX)

    for c_low, c_high, iaqi_low, iaqi_high in breakpoints:
        if c_low <= value <= c_high:
            if c_high == c_low:  # defensive: zero-width band cannot occur
                return float(iaqi_low)
            ratio = (iaqi_high - iaqi_low) / (c_high - c_low)
            return ratio * (value - c_low) + iaqi_low

    # Unreachable after truncation: at the table's own precision every value in
    # [0, top] lands inside a printed band. Kept so a future malformed table
    # reports "unavailable" rather than silently returning "Severe".
    return None


def get_aqi_category(aqi: object) -> tuple[str, int]:
    """Map an AQI value onto its CPCB category.

    Accepts floats (``grap.py`` passes a rounded station mean). Values between
    two integer band edges - e.g. ``50.5`` - resolve to the band whose numeric
    range contains them rather than falling through to "Severe".
    """
    value = _clean(aqi)
    if value is None:
        return "Unknown", 0
    if value > AQI_MAX:
        return "Severe", 6
    for lo, hi, label, level in AQI_CATEGORIES:
        if lo <= value <= hi:
            return label, level
    # Only reachable for a non-integer strictly between two bands, e.g. 50.5.
    # Resolve by numeric interval rather than dropping to "Severe".
    for (lo, _hi, label, level), (next_lo, _n_hi, _l, _lv) in zip(
        AQI_CATEGORIES, AQI_CATEGORIES[1:], strict=False
    ):
        if lo < value < next_lo:
            return label, level
    return "Severe", 6


# Canonical pollutant order, used for stable iteration and deterministic
# tie-breaking on the dominant pollutant (first match wins, as before).
_ORDER: tuple[str, ...] = ("pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb")


def evaluate_aqi(
    concentrations: Mapping[str, object],
    *,
    averaged: bool = False,
) -> AQIResult:
    """Evaluate a full AQI from a mapping of pollutant -> concentration.

    Parameters
    ----------
    concentrations:
        ``{"pm25": 45.0, "co": 1.5, ...}`` in :data:`POLLUTANT_UNITS` units.
        Unknown keys are ignored; missing/None/NaN/negative values are treated
        as unavailable.
    averaged:
        ``True`` when the supplied concentrations have already been averaged
        over their CPCB window. Purely descriptive - surfaced so API consumers
        know whether the numbers are window means or instantaneous.

    Returns
    -------
    AQIResult
    """
    sub_indices: dict[str, float] = {}
    used: dict[str, float] = {}
    availability: dict[str, str] = {}

    for pollutant in _ORDER:
        raw = concentrations.get(pollutant)
        if pollutant in UNSCORED_POLLUTANTS:
            availability[pollutant] = NH3_PB_BREAKPOINT_STATUS.get(
                pollutant, "no_cpcb_subindex_defined"
            )
            # Still record a usable concentration when one was measured, so the
            # value is not lost even though it cannot be scored.
            cleaned = _clean(raw)
            if cleaned is not None:
                used[pollutant] = cleaned
            continue

        iaqi = calculate_iaqi(pollutant, raw)
        if iaqi is None:
            availability[pollutant] = "no_valid_observation"
            continue
        sub_indices[pollutant] = iaqi
        used[pollutant] = _clean(raw)  # type: ignore[arg-type]

    if not sub_indices:
        return AQIResult(
            aqi=0,
            category="Unknown",
            category_level=0,
            dominant_pollutant=None,
            sub_indices={},
            averaged_concentrations=used,
            data_availability=availability,
        )

    dominant = max(sub_indices, key=lambda p: (sub_indices[p], -_ORDER.index(p)))
    aqi = int(round(max(sub_indices.values())))
    category, level = get_aqi_category(aqi)
    return AQIResult(
        aqi=aqi,
        category=category,
        category_level=level,
        dominant_pollutant=dominant,
        sub_indices=sub_indices,
        averaged_concentrations=used,
        data_availability=availability,
    )


def get_dominant_pollutant(
    pm25: float | None = None,
    pm10: float | None = None,
    o3: float | None = None,
    no2: float | None = None,
    so2: float | None = None,
    co: float | None = None,
    nh3: float | None = None,
    pb: float | None = None,
) -> str:
    """Return the pollutant with the highest sub-index.

    Kept for the existing positional callers. With nothing scorable it returns
    ``"pm25"`` to preserve the previous contract.
    """
    result = evaluate_aqi(
        {
            "pm25": pm25,
            "pm10": pm10,
            "o3": o3,
            "no2": no2,
            "so2": so2,
            "co": co,
            "nh3": nh3,
            "pb": pb,
        }
    )
    return result.dominant_pollutant or "pm25"


def calculate_aqi(
    pm25: float | None = None,
    pm10: float | None = None,
    o3: float | None = None,
    no2: float | None = None,
    so2: float | None = None,
    co: float | None = None,
    nh3: float | None = None,
    pb: float | None = None,
) -> tuple[int, str, str]:
    """Compute ``(aqi, category, dominant_pollutant)``.

    Signature-compatible with the previous implementation: the six original
    positional parameters keep their order and ``nh3``/``pb`` are appended as
    optional keywords, so every existing call site continues to work unchanged.

    Use :func:`evaluate_aqi` when the sub-indices, units and per-pollutant data
    availability are also needed.
    """
    result = evaluate_aqi(
        {
            "pm25": pm25,
            "pm10": pm10,
            "o3": o3,
            "no2": no2,
            "so2": so2,
            "co": co,
            "nh3": nh3,
            "pb": pb,
        }
    )
    return result.aqi, result.category, result.dominant_pollutant or "pm25"
