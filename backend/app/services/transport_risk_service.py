"""Estimated Regional Pollution Transport Risk engine for AeroCast-NCR.

This is an ESTIMATION system, NOT a guaranteed chemical plume model.

It produces a transparent 0-100 *"Estimated Regional Pollution Transport
Risk"* for the Delhi NCR region from **only the fire, weather and pollution
observations already stored in the database**. It never asserts that a given
fire (or cluster of fires) definitely caused Delhi pollution; the score is a
heuristic combination of upwind fire transport potential, atmospheric
ventilation conditions and the observed regional pollution burden.

Inputs used, where available:
  upwind fire activity   ``wind_aligned_fire_count`` (fires within 500 km
                          whose bearing roughly matches the upwind sector)
  fire intensity / FRP   ``fire_impact_score`` (FRP-weighted, wind-aligned,
                          distance-weighted logistic score from
                          ``ml.features.fire_impact.compute_fire_impact``)
  distance from NCR      ``nearest_fire_distance``
  wind speed / direction regional circular-mean of stored per-station winds
  wind alignment         ``wind_alignment_pct`` (share of fires upwind)
  PBL condition          regional mean ``pbl`` normalized feature
  inversion / stability  regional mean inversion strength (lapse-rate or proxy)
  atmospheric ventilation regional mean ventilation-coefficient normalized
  current pollution      regional mean observed PM2.5 (CPCB)

Formula (documented here once; echoed in the API response):

  fire_component     = WF_INTENSITY*fire_impact_score
                     + WF_PROXIMITY*proximity
                     + WF_COUNT*min(upwind_count/UPWIND_COUNT_SAT, 1)
                     + WF_ALIGNMENT*(wind_alignment_pct/100)
    proximity        = 1 - min(nearest_fire_distance/MAX_DISTANCE_KM, 1)

  atmo_component     = WA_VENT*(1 - ventilation_norm)
                     + WA_INVERSION*inversion_norm
                     + WA_PBL*(1 - pbl_norm)
    (poor ventilation / shallow PBL / inversion all raise the score)

  pollution_component = clip((region_pm25 - PM25_REF_GOOD)
                             / (PM25_REF_SEVERE - PM25_REF_GOOD), 0, 1)

  risk_score (0-100) = 100 * (W_FIRE*fire_component
                            + W_ATMO*atmo_component
                            + W_POLLUTION*pollution_component)
                            / (sum of weights whose inputs are available)

  Risk bands:  0-20 LOW | 21-40 MODERATE | 41-60 ELEVATED
               61-80 HIGH | 81-100 VERY HIGH

All constants are declared below with their rationale. Nothing is invented
silently: the constants dict is returned by the API.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("aerocast.transport_risk")

METHODOLOGY_VERSION = "1.0.0"
REGION = "Delhi NCR (5-station CPCB network + upwind tract)"

# ---------------------------------------------------------------------------
# Documented constants
# ---------------------------------------------------------------------------

# Radius (km) within which fires are considered "regional". Rationale: matches
# the existing ml.features.fire_impact.DEFAULT_MAX_DISTANCE_KM so the FRP/distance
# weighting is identical to the rest of the codebase.
MAX_DISTANCE_KM = 500.0

# Number of upwind fires that fully saturates the upwind-count factor. Rationale:
# a plume region with tens of simultaneous aligned hotspots is a well-developed
# transport source; beyond ~50 the marginal information gain plateaus.
UPWIND_COUNT_SATURATION = 50.0

# PM2.5 normalization anchors (µg/m3) — reuse the atmosphere layer references
# (CPCB 24h "satisfactory" upper bound and emergency band).
PM25_REF_GOOD = 35.0
PM25_REF_SEVERE = 300.0

# Component-level weights (sum to 1.0). Rationale:
#   fire 0.45  — transport potential is fire-driven in this domain (stubble).
#   atmo 0.35  — how effectively the atmosphere can distribute/retain smoke.
#   poll 0.20  — observed loading is the current burden, not the transport path.
W_FIRE = 0.45
W_ATMO = 0.35
W_POLLUTION = 0.20

# Fire sub-weights (sum to 1.0). fire_impact_score already encodes FRP intensity,
# wind alignment and distance, so it carries the largest share.
WF_INTENSITY = 0.50
WF_PROXIMITY = 0.25
WF_COUNT = 0.15
WF_ALIGNMENT = 0.10

# Atmosphere sub-weights (sum to 1.0). Ventilation (wind x PBL) is the primary
# mixing/removal agent; inversion and shallow PBL are secondary contributors.
WA_VENT = 0.45
WA_INVERSION = 0.30
WA_PBL = 0.25

# Risk bands, exactly as requested: 0-20 / 21-40 / 41-60 / 61-80 / 81-100.
RISK_BANDS: list[tuple[float, str]] = [
    (20.0, "LOW"),
    (40.0, "MODERATE"),
    (60.0, "ELEVATED"),
    (80.0, "HIGH"),
    (float("inf"), "VERY HIGH"),
]

COMPASS_POINTS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

DISCLAIMER = (
    "Estimated Regional Pollution Transport Risk — a transparent heuristic "
    "built from stored fire, weather and pollution observations. It is an "
    "ESTIMATION, not a guaranteed chemical plume model, and it does not assert "
    "that any specific fire caused Delhi pollution."
)


def _clip01(v) -> float | None:
    if v is None:
        return None
    try:
        return round(float(np.clip(float(v), 0.0, 1.0)), 4)
    except (TypeError, ValueError):
        return None


def compass(direction: float | None) -> str | None:
    if direction is None:
        return None
    idx = int((float(direction) % 360.0 + 11.25) / 22.5) % 16
    return COMPASS_POINTS[idx]


def classify_risk(score: float | None) -> str:
    """Map a 0-100 score to its band (exact user-specified bands)."""
    if score is None:
        return "UNAVAILABLE"
    s = float(score)
    if s <= 20.0:
        return "LOW"
    if s <= 40.0:
        return "MODERATE"
    if s <= 60.0:
        return "ELEVATED"
    if s <= 80.0:
        return "HIGH"
    return "VERY HIGH"


def _mean_wind(stations: list[dict[str, Any]]) -> dict[str, Any]:
    """Regional circular-mean wind (FROM direction) across analysed stations.

    Direction averaging uses unit vectors in the direction the wind is blowing
    TOWARD (meteorological FROM direction + 180); the resultant is converted
    back to the mean FROM direction. Speed is the scalar mean.
    """
    sin_sum = 0.0
    cos_sum = 0.0
    speeds: list[float] = []
    n = 0
    for s in stations:
        spd = s["wind"].get("wind_speed_mps")
        deg = s["wind"].get("wind_direction_deg")
        if spd is None or deg is None:
            continue
        toward = math.radians(float(deg) + 180.0)
        sin_sum += math.sin(toward)
        cos_sum += math.cos(toward)
        speeds.append(float(spd))
        n += 1
    if n == 0 or (sin_sum == 0.0 and cos_sum == 0.0):
        return {
            "wind_speed_mps": None,
            "wind_direction_deg": None,
            "compass_from": None,
            "basis": "no stored wind data",
            "stations_used": n,
        }
    toward_deg = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0
    from_deg = (toward_deg + 180.0) % 360.0
    return {
        "wind_speed_mps": round(float(np.mean(speeds)), 2),
        "wind_direction_deg": round(from_deg, 1),
        "compass_from": compass(from_deg),
        "basis": f"circular mean of stored wind over {n} station(s), winds are reported FROM the given direction",
        "stations_used": n,
    }


def compute_regional_risk(
    fires_df: pd.DataFrame | None,
    wind_speed: float | None,
    wind_direction: float | None,
    ventilation_norm: float | None,
    pbl_norm: float | None,
    inversion_norm: float | None,
    region_pm25: float | None,
    *,
    ncr_lat: float = 28.6139,
    ncr_lon: float = 77.2090,
) -> dict[str, Any]:
    """Pure-function core: compute the risk from already-assembled inputs.

    Returns the full risk packet (score, level, components, factors). Pure and
    deterministic so it can be unit-tested without a database.
    """
    from ml.features.fire_impact import compute_fire_impact

    if fires_df is not None and not fires_df.empty:
        impact = compute_fire_impact(
            fires_df,
            ncr_lat,
            ncr_lon,
            wind_dir=wind_direction if wind_direction is not None else 0.0,
            wind_speed=wind_speed or 0.0,
            max_distance_km=MAX_DISTANCE_KM,
        )
    else:
        impact = {
            "fire_count": 0,
            "fire_impact_score": 0.0,
            "nearest_fire_distance": MAX_DISTANCE_KM + 1.0,
            "wind_aligned_fire_count": 0,
            "wind_alignment_pct": 0.0,
            "transport_time_hours": None,
            "transport_risk": 0.0,
            "transport_risk_level": "none",
            "stubble_impact_score": 0.0,
        }

    fire_count = int(impact["fire_count"])
    upwind_count = int(impact["wind_aligned_fire_count"])
    alignment_pct = float(impact["wind_alignment_pct"])
    impact_score = float(impact["fire_impact_score"])
    nearest_km = float(impact["nearest_fire_distance"])

    # ---- fire component ----
    proximity = _clip01(1.0 - min(1.0, nearest_km / MAX_DISTANCE_KM)) or 0.0
    count_factor = _clip01(upwind_count / UPWIND_COUNT_SATURATION) or 0.0
    fire_component = (
        WF_INTENSITY * impact_score
        + WF_PROXIMITY * proximity
        + WF_COUNT * count_factor
        + WF_ALIGNMENT * (alignment_pct / 100.0)
    )
    fire_component = round(float(np.clip(fire_component, 0.0, 1.0)), 4)

    # ---- atmosphere component ----
    vent_known = ventilation_norm is not None
    inv_known = inversion_norm is not None
    pbl_known = pbl_norm is not None
    if vent_known or pbl_known or inv_known:
        atmo_component = (
            WA_VENT * (1.0 - (ventilation_norm if ventilation_norm is not None else 0.5))
            + WA_INVERSION * (inversion_norm if inversion_norm is not None else 0.0)
            + WA_PBL * (1.0 - (pbl_norm if pbl_norm is not None else 0.5))
        )
        atmo_component = round(float(np.clip(atmo_component, 0.0, 1.0)), 4)
    else:
        atmo_component = None

    # ---- pollution component ----
    if region_pm25 is not None and region_pm25 > 0:
        pollution_component = _clip01((region_pm25 - PM25_REF_GOOD) / (PM25_REF_SEVERE - PM25_REF_GOOD))
    else:
        pollution_component = None

    # ---- weighted blend with availability normalization ----
    items = [(W_FIRE, fire_component), (W_ATMO, atmo_component), (W_POLLUTION, pollution_component)]
    weighting_notes: list[str] = []
    active = [(w, c) for w, c in items if c is not None]
    if not active:
        risk_score = None
        weighting_notes.append("No available input components; risk cannot be estimated.")
    else:
        denom = sum(w for w, _ in active) or 1.0
        raw = sum(w * c for w, c in active) / denom
        clipped = _clip01(raw)
        risk_score = int(round(100.0 * (0.0 if clipped is None else clipped)))
        excluded = [
            (name, w)
            for name, w, c in (
                ("fire", W_FIRE, fire_component),
                ("atmosphere", W_ATMO, atmo_component),
                ("current pollution", W_POLLUTION, pollution_component),
            )
            if c is None
        ]
        for name, _w in excluded:
            weighting_notes.append(
                f"{name} input unavailable; its weight was excluded and the remaining weights renormalized."
            )

    # ---- factors ----
    ranking: list[tuple[float, str]] = []
    fire_descriptors = [
        (
            W_FIRE * WF_INTENSITY * impact_score,
            f"Upwind fire activity: {upwind_count} of {fire_count} hotspots within {MAX_DISTANCE_KM:.0f} km are "
            f"upwind of NCR ({alignment_pct:.0f}% aligned); FRP-weighted impact {impact_score:.2f}",
        ),
        (
            W_FIRE * WF_INTENSITY * impact_score,
            f"Fire intensity: cumulative FRP-weighted transport potential {impact_score:.4f} (FRP x wind alignment / distance)",
        ),
        (
            W_FIRE * WF_PROXIMITY * proximity,
            f"Proximity: nearest fire {nearest_km:.0f} km from NCR centroid"
            if nearest_km < MAX_DISTANCE_KM
            else "Proximity: no fire within 500 km of NCR",
        ),
        (
            W_FIRE * WF_COUNT * count_factor,
            f"Upwind hotspot density: {upwind_count} upwind fires (saturates at {UPWIND_COUNT_SATURATION:.0f})",
        ),
        (
            W_FIRE * WF_ALIGNMENT * (alignment_pct / 100.0),
            f"Wind alignment: {alignment_pct:.0f}% of regional fires upwind of NCR",
        ),
    ]
    if impact["transport_time_hours"]:
        fire_descriptors.append(
            (
                W_FIRE * 0.5,
                f"Nearest-fire smoke advective arrival ~{impact['transport_time_hours']:.1f}h at the regional wind",
            )
        )
    if fire_count == 0:
        fire_descriptors = [
            (
                0.9,
                "No active fires within 500 km of NCR in the look-back window -> fire transport potential is negligible",
            )
        ]
    ranking.extend(fire_descriptors)

    if atmo_component is not None:
        ranking.extend(
            [
                (
                    # ``is not None`` guards, not ``or``: a genuine normalized
                    # 0.0 means fully capped ventilation / a fully inverted
                    # atmosphere, and ``or`` was replacing it with a mid-risk
                    # 0.5, reporting calm conditions as a neutral atmosphere.
                    W_ATMO * WA_VENT * (1.0 - (ventilation_norm if ventilation_norm is not None else 0.5)),
                    f"Atmospheric ventilation: coefficient normalized {ventilation_norm:.2f}, its inverse raises accumulation risk"
                    if vent_known
                    else "Atmospheric ventilation unavailable",
                ),
                (
                    W_ATMO * WA_INVERSION * (inversion_norm if inversion_norm is not None else 0.0),
                    f"Stability/inversion: strength normalized {inversion_norm:.2f}; inversions cap vertical mixing"
                    if inv_known
                    else "Stability/inversion unavailable",
                ),
                (
                    W_ATMO * WA_PBL * (1.0 - (pbl_norm if pbl_norm is not None else 0.5)),
                    f"Boundary layer: depth normalized {pbl_norm:.2f}; a shallower PBL concentrates surface pollution"
                    if pbl_known
                    else "Boundary layer unavailable",
                ),
            ]
        )
    if pollution_component is not None:
        ranking.append(
            (
                W_POLLUTION * pollution_component,
                f"Current observed pollution: regional mean PM2.5 "
                f"{region_pm25:.1f} ug/m3 (normalized {pollution_component:.2f}) raises the existing burden",
            )
        )
    else:
        ranking.append((W_POLLUTION, "No current NCR PM2.5 stored -> pollution component excluded"))

    ranking.sort(key=lambda t: t[0], reverse=True)

    main_contributing_factors = [pair[1] for pair in ranking[:4]] + weighting_notes[:2]

    return {
        "risk_score": risk_score,
        "risk_level": classify_risk(risk_score),
        "main_contributing_factors": main_contributing_factors,
        "upwind_fire_count": upwind_count,
        "fire_count": fire_count,
        "dominant_wind_direction": {
            "from_degrees": wind_direction,
            "compass_from": compass(wind_direction),
            "wind_speed_mps": wind_speed,
        },
        "atmospheric_condition": {
            "ventilation_norm": ventilation_norm,
            "pbl_norm": pbl_norm,
            "inversion_norm": inversion_norm,
        },
        "inputs": {
            "fire_lookback_hours_requested": None,
            "nearest_fire_distance_km": round(nearest_km, 1) if nearest_km < MAX_DISTANCE_KM + 1 else None,
            "wind_alignment_pct": round(alignment_pct, 1),
            "transport_time_hours": impact["transport_time_hours"],
            "fire_transport_risk_01": impact["transport_risk"],
            "fire_transport_risk_level": impact.get("transport_risk_level", "none"),
            "stubble_impact_01": impact["stubble_impact_score"],
            "region_pm25_ugm3": region_pm25,
        },
        "components": {
            "fire_component": fire_component,
            "atmo_component": atmo_component,
            "pollution_component": pollution_component,
            "weights": {"fire": W_FIRE, "atmosphere": W_ATMO, "pollution": W_POLLUTION},
            "renormalization_notes": weighting_notes,
        },
    }


def get_current_transport_risk(db, fire_window_hours: int = 72) -> dict[str, Any]:
    """Assemble real stored inputs and compute the current regional risk."""
    from ..config import get_settings
    from ..models.db_models import FireReading, Station
    from .atmosphere_service import analyze_station

    now = datetime.now(UTC).replace(tzinfo=None)

    stations = db.query(Station).order_by(Station.name).all()
    atmo_rows = [analyze_station(db, s, now) for s in stations]

    # Regional wind (circular mean) and mechanical means of normalized features.
    wind = _mean_wind(atmo_rows)

    def _mean(key: str) -> float | None:
        vals = [r["features"].get(key) for r in atmo_rows if r["features"].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    vent_norm = _mean("ventilation")
    pbl_norm = _mean("pbl")
    inv_norm = _mean("inversion")

    pm25_vals = [r["inputs"].get("pm25_ugm3") for r in atmo_rows if r.get("inputs", {}).get("pm25_ugm3") is not None]
    region_pm25 = round(float(np.mean(pm25_vals)), 1) if pm25_vals else None

    # Fires in the look-back window (real stored FIRMS rows).
    since = now - timedelta(hours=fire_window_hours)
    if get_settings().database_url.startswith("sqlite"):
        since = since.replace(tzinfo=None)
    fire_rows = db.query(FireReading).filter(FireReading.acq_date >= since).order_by(FireReading.acq_date.desc()).all()
    fires_df = None
    if fire_rows:
        fires_df = pd.DataFrame(
            [
                {
                    "lat": f.latitude,
                    "lon": f.longitude,
                    "frp": f.frp or 1.0,
                }
                for f in fire_rows
            ]
        )

    mean_frp = round(float(np.mean([f.frp for f in fire_rows if f.frp])), 1) if fires_df is not None else 0.0
    peak_frp = (
        round(float(max(f.frp for f in fire_rows if f.frp)), 1)
        if fires_df is not None and any(f.frp for f in fire_rows)
        else 0.0
    )

    centroid_lat = float(np.mean([s.latitude for s in stations]))
    centroid_lon = float(np.mean([s.longitude for s in stations]))

    risk = compute_regional_risk(
        fires_df,
        wind_speed=wind["wind_speed_mps"],
        wind_direction=wind["wind_direction_deg"],
        ventilation_norm=vent_norm,
        pbl_norm=pbl_norm,
        inversion_norm=inv_norm,
        region_pm25=region_pm25,
        ncr_lat=centroid_lat,
        ncr_lon=centroid_lon,
    )
    risk["inputs"]["fire_lookback_hours_requested"] = fire_window_hours
    risk["inputs"]["fire_rows_in_window"] = len(fire_rows)
    risk["inputs"]["mean_frp_mw"] = mean_frp
    risk["inputs"]["peak_frp_mw"] = peak_frp
    risk["inputs"]["region_pm25_ugm3"] = region_pm25

    risk["generated_at"] = now
    risk["region"] = REGION
    risk["disclaimer"] = DISCLAIMER
    risk["dominant_wind_direction"].update({"basis": wind["basis"], "basis_count_stations": wind["stations_used"]})
    risk["atmospheric_condition"].update(
        {
            "ventilation_condition": "poor"
            if vent_norm is not None and vent_norm < 0.5
            else "good"
            if vent_norm is not None and vent_norm >= 1.0
            else "moderate"
            if vent_norm is not None
            else None,
            "pbl_condition": "shallow"
            if pbl_norm is not None and pbl_norm < 0.2
            else "deep"
            if pbl_norm is not None and pbl_norm >= 0.33
            else "moderate"
            if pbl_norm is not None
            else None,
            "inversion_source_seen": sorted(
                {(r.get("inversion") or {}).get("source", "unavailable") for r in atmo_rows}
            ),
        }
    )
    risk["methodology"] = {
        "version": METHODOLOGY_VERSION,
        "disclaimer": DISCLAIMER,
        "formulas": {
            "fire_component": (
                f"0.50*fire_impact_score + 0.25*proximity + 0.15*min(upwind_count/{UPWIND_COUNT_SATURATION:.0f},1) "
                f"+ 0.10*wind_alignment_pct/100"
            ),
            "proximity": "1 - min(nearest_fire_distance_km/500, 1)",
            "atmo_component": (
                f"{WA_VENT:.2f}*(1 - ventilation_norm) + {WA_INVERSION:.2f}*inversion_norm "
                f"+ {WA_PBL:.2f}*(1 - pbl_norm)"
            ),
            "pollution_component": "clip((regional_mean_pm25 - 35)/265, 0, 1)",
            "risk_score_0_100": (
                "100 * (0.45*fire_component + 0.35*atmo_component + 0.20*pollution_component) "
                "/ (sum of weights for components with available inputs)"
            ),
            "fire_impact_score": (
                "from ml.features.fire_impact: sum over fires <=500km of FRP*max(cos(wind alignment),0)/(dist_km+1), "
                "mapped by 1/(1+exp(-(x-1))) [k=1, mid=1]"
            ),
        },
        "constants": {
            "max_fire_distance_km": MAX_DISTANCE_KM,
            "upwind_count_saturation": UPWIND_COUNT_SATURATION,
            "pm25_ref_good_ugm3": PM25_REF_GOOD,
            "pm25_ref_severe_ugm3": PM25_REF_SEVERE,
            "component_weights": {"fire": W_FIRE, "atmosphere": W_ATMO, "pollution": W_POLLUTION},
            "fire_sub_weights": {
                "intensity": WF_INTENSITY,
                "proximity": WF_PROXIMITY,
                "count": WF_COUNT,
                "alignment": WF_ALIGNMENT,
            },
            "atmo_sub_weights": {"ventilation": WA_VENT, "inversion": WA_INVERSION, "pbl": WA_PBL},
            "risk_bands": {
                "0-20": "LOW",
                "21-40": "MODERATE",
                "41-60": "ELEVATED",
                "61-80": "HIGH",
                "81-100": "VERY HIGH",
            },
        },
        "caveats": [
            "ESTIMATION only — not a chemical transport/plume model.",
            "The score does not assert that any fire caused Delhi pollution.",
            "Weather quantities (incl. PBL, ventilation, inversion) are Open-Meteo model fields, not in-situ measurements.",
            "PM2.5 is CPCB-observed; fires are NASA FIRMS observations.",
            "Wind alignment and upwind counts describe WHERE fires sit relative to regional wind; they do not place a plume on a map.",
        ],
    }
    risk["station_detail"] = [
        {
            "station": r["station"],
            "pm25_ugm3": r["inputs"].get("pm25_ugm3"),
            "vent_norm": r["features"].get("ventilation"),
            "pbl_norm": r["features"].get("pbl"),
            "inv_norm": r["features"].get("inversion"),
            "inversion_source": (r.get("inversion") or {}).get("source"),
        }
        for r in atmo_rows
    ]
    return risk
