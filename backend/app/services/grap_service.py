"""Graded Response Action Plan (GRAP) for Delhi NCR.

Implements the standardised stage matrix notified by the Commission for Air
Quality Management (CAQM) for the National Capital Region.  Indexing is by the
current AQI (the same values used across the forecasting pipeline):

    Stage I    201-300   Poor        (pre-emptive, year-round actions)
    Stage II   301-400   Very Poor   (Severe-level abatement)
    Stage III  401-450   Severe      (construction/petrol-diesel curbs)
    Stage IV   451+      Severe+     (emergency curbs: odd-even, school closure)

Thresholds follow the October-2024 CAQM revision (Stage III spans 401-450 and
Stage IV begins above 450, splitting the old single >400 severe band).

This module is deliberately pure / dependency-free so it can be unit-tested in
isolation and reused by the API layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

GRAP_SOURCE = "CAQM graded response action plan (Delhi NCR) — revised October 2024"


class GrapStage(TypedDict):
    stage: int
    title: str
    aqi_range_low: int | None
    aqi_range_high: int | None
    categories: list[str]
    color: str
    summary: str
    measures: list[str]


GRAP_STAGES: list[GrapStage] = [
    {
        "stage": 1,
        "title": "Stage I — Poor",
        "aqi_range_low": 201,
        "aqi_range_high": 300,
        "categories": ["Poor"],
        "color": "#f97316",
        "summary": "Pre-emptive enforcement driven by the 6 AQI category colour tracking.",
        "measures": [
            "Strict enforcement of dust-control norms at construction and demolition sites",
            "Mechanised road sweeping and water sprinkling on major roads and pollution hotspots",
            "Ban on open burning of solid waste, leaves and biomass in all NCR jurisdictions",
            "Enforce PUC (pollution-under-control) compliance for vehicles",
            "Deploy anti-smog guns at identified high-traffic / construction hotspots",
        ],
    },
    {
        "stage": 2,
        "title": "Stage II — Very Poor",
        "aqi_range_low": 301,
        "aqi_range_high": 400,
        "categories": ["Very Poor"],
        "color": "#ef4444",
        "summary": "Severe-level abatement: intensive dust suppression plus source curbs.",
        "measures": [
            "All Stage I measures continue",
            "Ban on use of diesel generator sets in areas covered by piped natural gas",
            "Increase frequency of mechanised road sweeping and water sprinkling",
            "Periodic mechanised cleaning of roads with dust-suppression systems",
            "Enhance public transport frequency and augment bus services on NCR routes",
        ],
    },
    {
        "stage": 3,
        "title": "Stage III — Severe",
        "aqi_range_low": 401,
        "aqi_range_high": 450,
        "categories": ["Severe"],
        "color": "#dc2626",
        "summary": "Construction and vehicle-use curbs in addition to Stages I-II.",
        "measures": [
            "All Stage II measures continue",
            "Ban on non-essential construction and demolition activities in NCR",
            "Closure of stone crushers, hot-mix plants and concrete batching plants",
            "Restrict BS-III petrol and BS-IV diesel four-wheelers from plying in Delhi NCR",
            "Increase CNG / electric bus and Metro frequency",
        ],
    },
    {
        "stage": 4,
        "title": "Stage IV — Severe+ (emergency)",
        "aqi_range_low": 451,
        "aqi_range_high": None,
        "categories": ["Severe+", "Emergency"],
        "color": "#7f1d1d",
        "summary": "Emergency curbs to flatten a severe pollution episode over 450.",
        "measures": [
            "All Stage III measures continue",
            "Ban on entry of trucks into Delhi (except essential goods)",
            "Odd-even road-space rationing for private four-wheelers",
            "Work-from-home for 50% of government and private office staff",
            "Physical closure of schools, colleges and educational institutions (move online)",
            "Ban on commercial four-wheelers not operating on cleaner fuels",
        ],
    },
]

NOT_INVOKED: GrapStage = {
    "stage": 0,
    "title": "Not invoked",
    "aqi_range_low": None,
    "aqi_range_high": 200,
    "categories": ["Good", "Satisfactory", "Moderate"],
    "color": "#22c55e",
    "summary": "Current AQI is below the Stage I threshold of 201.",
    "measures": [
        "Continue routine monitoring and early-warning (daily AQI advisories)",
        "Maintain baseline dust-control and waste-burning enforcement",
        "Watch for a forecasted deterioration before the next non-severity season",
    ],
}


def get_grap_stages() -> list[GrapStage]:
    """Return the full GRAP stage matrix (including the not-invoked stage 0)."""
    return [NOT_INVOKED] + [stage.copy() for stage in GRAP_STAGES]


def stage_from_aqi(aqi: float | None) -> GrapStage:
    """Resolve the GRAP stage for a current AQI value.

    Stage 0 means the plan is not invoked. Values above 500 clamp to Stage IV.
    """
    if aqi is None or aqi <= 200:
        return NOT_INVOKED.copy()
    for stage in GRAP_STAGES:
        hi = stage["aqi_range_high"]
        if hi is None:
            return stage.copy()
        if aqi <= hi:
            return stage.copy()
    return GRAP_STAGES[-1].copy()


def reasonable_aqi(aqi: float | None) -> int | None:
    """Clamp an arbitrary AQI input to the reportable 0-500 range."""
    if aqi is None:
        return None
    if aqi < 0 or aqi != aqi:
        return None
    return int(min(500, max(0, round(aqi))))


def assess_grap(
    aqi: float | None,
    inversion_strength: float | None = None,
    fire_mean_frp_mw: float | None = None,
) -> dict:
    """Assess the operative GRAP stage and produce an actionable advisory.

    ``inversion_strength`` (0..1) and ``fire_mean_frp_mw`` are optional context
    used to explain *why* the current stage is active — e.g. a shallow boundary
    layer trapping emissions, or stubble burning advecting into the region.
    """
    aqi_val = reasonable_aqi(aqi)
    stage = stage_from_aqi(aqi_val)
    rationale: list[str] = []

    if aqi_val is None:
        rationale.append("No current AQI is available; stage is advisory only.")
    elif stage["stage"] == 0:
        rationale.append(f"Current AQI is {aqi_val} (≤ 200) — below the Stage I trigger of 201.")
    else:
        hi = stage["aqi_range_high"]
        stage_low = stage["aqi_range_low"]
        band = f"{stage_low}–{hi}" if hi is not None else f"> {stage_low - 1}" if stage_low is not None else "unbounded"
        rationale.append(f"Current AQI is {aqi_val} (band {band}) → invoked under GRAP {stage['title']}.")

    inversion_note = None
    if inversion_strength is not None:
        if inversion_strength > 0.6:
            inversion_note = (
                "High inversion strength — emissions are being trapped near the ground; "
                "source curbs will take longer to register in observed AQI."
            )
            rationale.append("High inversion strength (lapse-rate / PBL proxy) is suppressing vertical mixing.")
        elif inversion_strength > 0.3:
            inversion_note = "Moderate inversion present — expect slower dispersion during the next 24 h."
            rationale.append("Moderate inversion is partially limiting vertical dispersion.")

    fire_note = None
    if fire_mean_frp_mw is not None:
        if fire_mean_frp_mw >= 80:
            fire_note = (
                "Elevated regional fire activity (stubble burning) — follow GRAP source "
                "measures early even if local AQI is still below the band."
            )
            rationale.append("Regional fire forcing (mean FRP ≥ 80 MW) is elevated.")
        elif fire_mean_frp_mw >= 40:
            fire_note = "Moderate regional fire activity detected; monitor plume transport risk."
            rationale.append("Regional fire activity (mean FRP 40–80 MW) is moderate.")

    measures = stage["measures"]
    if aqi_val is not None and stage["stage"] > 0 and inversion_strength is not None and inversion_strength > 0.6:
        measures = measures + ["Extra advisory: hold discretionary outdoor activities while the inversion persists."]

    return {
        "assessed_at": datetime.now(UTC).replace(tzinfo=None),
        "stage": stage["stage"],
        "status": "ACTIVE" if stage["stage"] > 0 else "NOT_INVOKED",
        "title": stage["title"],
        "color": stage["color"],
        "aqi": aqi_val,
        "aqi_category": None,
        "dominant_pollutant": None,
        "inversion_strength": (round(inversion_strength, 3) if inversion_strength is not None else None),
        "inversion_note": inversion_note,
        "fire_mean_frp_mw": (round(fire_mean_frp_mw, 1) if fire_mean_frp_mw is not None else None),
        "fire_note": fire_note,
        "advisory": (
            "Stage not invoked — routine monitoring only."
            if stage["stage"] == 0
            else f"GRAP {stage['title']} measures are operative across Delhi NCR."
        ),
        "rationale": rationale,
        "measures": measures,
        "source": GRAP_SOURCE,
    }
