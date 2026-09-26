"""Vertical atmospheric analysis — lapse-rate inversion and PBL classification.

Implements the scientifically defensible inversion detection required by
SIH26082: instead of (or in addition to) the PBL-height *proxy*, inversion is
detected from the **vertical temperature profile** by computing the atmospheric
temperature gradient with height.

Physical basis
--------------
In a "normal" tropospheric lapse the temperature *decreases* with height
(environmental lapse rate ~ -6.5 K/km). A **temperature inversion** occurs when
temperature *increases* with height over some layer, i.e. a *positive* vertical
temperature gradient. This traps pollutants close to the ground (stable layer),
which is exactly the phenomenon the problem statement asks us to track.

Conventions
-----------
* Temperatures are supplied as a mapping pressure_hPa -> temperature(degC or K).
* We require at least two levels to form a gradient. Levels are sorted by
  pressure (descending), i.e. from surface to top.
* gradient = dT / d(log p) approximated as (T2 - T1)/(ln p1 - ln p2) over the
  layer. Because pressure *decreases* upward, a positive temperature increase
  upward corresponds to a positive dT/d(ln p) with our sign convention
  (see ``_coarse_gradient``).

Units are transparent: temperatures in degC (callers are expected to pass degC
as returned by Open-Meteo). All derived strengths are normalised to [0, 1].
"""

from __future__ import annotations

import numpy as np

#: Pressure levels used for inversion analysis, surface -> top.
DEFAULT_LEVELS_HPA: list[float] = [1000, 925, 850, 700]

#: Inversion categories by gradient strength (K / 100 hPa), applied to the
#: strongest (most positive) layer gradient.
STRONG_INVERSION_THRESHOLD_K = 1.5  # > 1.5 K/100hPa -> strong
MODERATE_INVERSION_THRESHOLD_K = 0.6  # > 0.6  K/100hPa -> moderate
WEAK_INVERSION_THRESHOLD_K = 0.0  # > 0.0  K/100hPa -> weak inversion (T increases w/ height)

#: PBL height (m) classification used for the PBL proxy fallback.
PBL_STRONG_M = 150.0
PBL_MODERATE_M = 300.0
PBL_WEAK_M = 500.0
PBL_CLIMATOLOGICAL_DEFAULT_M = 600.0

# --- Height geometry -------------------------------------------------------
#: Specific gas constant for dry air, J/(kg K) (ICAO standard atmosphere).
R_DRY_AIR_J_KG_K = 287.05
#: Standard gravity, m/s^2.
GRAVITY_M_S2 = 9.80665
#: Reference pressure the height integration starts from when the profile does
#: not carry real geopotential heights: the 1000 hPa level.
HEIGHT_REFERENCE_HPA = 1000.0


def pressure_to_height_m(p_hPa: float, t_layer_c: float) -> float | None:
    """Height (m) of a pressure level above a reference, via the hypsometric equation.

    The layer-mean temperature is treated as constant over the layer, which is
    the standard first-order hypsometric approximation::

        dz = (R_d * T_mean / g) * ln(p_ref / p)

    Returns ``None`` for non-physical input rather than guessing, so a bad
    profile is visible as missing geometry instead of a plausible number.
    """
    if p_hPa is None or t_layer_c is None:
        return None
    try:
        p = float(p_hPa)
        t_mean_k = float(t_layer_c) + 273.15
    except (TypeError, ValueError):
        return None
    if not (p > 0) or t_mean_k <= 0:
        return None
    if p >= HEIGHT_REFERENCE_HPA:
        # At or below the reference level the height is zero by definition.
        return 0.0 if p == HEIGHT_REFERENCE_HPA else None
    return float((R_DRY_AIR_J_KG_K * t_mean_k / GRAVITY_M_S2) * np.log(HEIGHT_REFERENCE_HPA / p))


def layer_heights_m(
    temp_by_level: dict[float, float],
    height_by_level: dict[float, float] | None = None,
) -> dict[float, float]:
    """Geometric height (m) for each pressure level in a vertical profile.

    Real geopotential heights from the reanalysis/forecast feed win when
    present. Otherwise heights are integrated upward from
    :data:`HEIGHT_REFERENCE_HPA` with the hypsometric equation, so with the full
    1000/925/850/700 hPa profile the numbers are heights above the 1000 hPa
    level. If 1000 hPa itself is missing, the lowest available level becomes the
    reference and the returned heights are relative to it -- the caller sees a
    depth, not an altitude, which is what the layer analysis needs anyway.
    """
    heights: dict[float, float] = {}
    supplied = {float(p): float(h) for p, h in (height_by_level or {}).items()
                if h is not None and not (isinstance(h, float) and np.isnan(h))}
    temps = {float(p): float(t) for p, t in (temp_by_level or {}).items() if t is not None}
    if len(temps) < 2:
        # A single level carries no layer information, so there is no geometry
        # to report. Any real heights the feed supplied are still returned.
        return dict(supplied)

    # Walk from the lowest level upward (decreasing pressure). A supplied
    # geopotential height anchors that level; the rest are integrated from it,
    # so a partially-populated feed still yields a full profile.
    levels = sorted(temps, reverse=True)
    for i, p in enumerate(levels):
        if p in supplied:
            heights[p] = supplied[p]
        elif i == 0:
            # Reference level: zero by definition when it is the 1000 hPa
            # surface. If the profile starts higher up, heights become depths
            # relative to the lowest observed level.
            heights[p] = 0.0
        else:
            p_base = levels[i - 1]
            t_mean = (temps[p_base] + temps[p]) / 2.0
            dz = (R_DRY_AIR_J_KG_K * (t_mean + 273.15) / GRAVITY_M_S2) * np.log(p_base / p)
            heights[p] = heights.get(p_base, 0.0) + float(dz)
    # Levels that carry a real height but no temperature keep it.
    for p, h in supplied.items():
        heights.setdefault(p, h)
    return heights


def _level_pairs(levels_hpa: list[float]) -> list[tuple[float, float]]:
    """Return adjacent (base_pressure, top_pressure) pairs.

    Levels are sorted descending by pressure (base -> top). Each pair is
    ``(base_level, top_level)`` where base has *higher* pressure.
    """
    ordered = sorted(levels_hpa, reverse=True)  # base (highest p) first
    return [(base, top) for base, top in zip(ordered, ordered[1:], strict=False)]


def _coarse_gradient(t_base: float, t_top: float, p_base: float, p_top: float) -> float | None:
    """Approximate vertical temperature gradient over a layer (K per 100 hPa).

    A linear-in-pressure gradient (transparent, standard for layer reporting)::

        dT/dp = (T_top - T_base) / (p_base - p_top)     [K / hPa]
        gradient = dT/dp * 100                          [K / 100 hPa]

    With p_base > p_top (base is higher pressure / lower altitude), a warmer
    layer aloft (T_top > T_base) gives a **positive** gradient == temperature
    inversion (temperature increases with height).
    """
    if p_base is None or p_top is None or t_base is None or t_top is None:
        return None
    if not (p_base > 0 and p_top > 0) or p_base <= p_top:
        return None
    try:
        dp = float(p_base) - float(p_top)
        if abs(dp) < 1e-9:
            return None
        grad = (float(t_top) - float(t_base)) / dp * 100.0  # per 100 hPa
        return float(grad)
    except (TypeError, ValueError):
        return None


def compute_lapse_rates(temp_by_level: dict[float, float]) -> dict[tuple[float, float], float]:
    """Compute layer gradient (K/100 hPa) for each adjacent level pair.

    Args:
        temp_by_level: mapping pressure_hPa -> temperature(degC).

    Returns:
        dict mapping (base_pressure, top_pressure) -> gradient.
    """
    present = {p: t for p, t in (temp_by_level or {}).items() if t is not None}
    if len(present) < 2:
        return {}
    grades: dict[tuple[float, float], float] = {}
    for p_base, p_top in _level_pairs(list(present.keys())):
        g = _coarse_gradient(present[p_base], present[p_top], p_base, p_top)
        if g is not None:
            grades[(p_base, p_top)] = g
    return grades


def classify_gradient(
    gradients: dict[tuple[float, float], float],
    temp_by_level: dict[float, float] | None = None,
    height_by_level: dict[float, float] | None = None,
) -> dict[str, object]:
    """Classify inversion from a set of layer gradients.

    Returns a dict with:
      - inversion_detected       : bool
      - inversion_strength       : float in [0, 1] (based on strongest layer)
      - inversion_category       : none / weak / moderate / strong
      - inversion_base_pressure  : hPa or None
      - inversion_top_pressure   : hPa or None
      - inversion_base_height_m  : m or None  (hypsometric / geopotential)
      - inversion_top_height_m   : m or None
      - inversion_thickness_m    : m or None   (top height - base height)
      - inversion_thickness_hpa  : hPa or None (base pressure - top pressure)
      - strongest_layer_gradient : K/100 hPa of the strongest (most negative
                                   stability) layer, or None
      - profile_available        : bool (>=2 levels)
    """
    if not gradients:
        return {
            "inversion_detected": False,
            "inversion_strength": 0.0,
            "inversion_category": "none",
            "inversion_base_pressure": None,
            "inversion_top_pressure": None,
            "inversion_base_height_m": None,
            "inversion_top_height_m": None,
            "inversion_thickness_m": None,
            "inversion_thickness_hpa": None,
            "strongest_layer_gradient": None,
            "profile_available": False,
        }

    # Strongest layer == the layer with the most *positive* gradient (temperature
    # increasing most sharply upward) => the most trapping layer.
    (base_p, top_p), gmax = max(gradients.items(), key=lambda kv: (kv[1], kv[0]))

    if gmax > STRONG_INVERSION_THRESHOLD_K:
        category = "strong"
    elif gmax > MODERATE_INVERSION_THRESHOLD_K:
        category = "moderate"
    elif gmax > WEAK_INVERSION_THRESHOLD_K:
        category = "weak"
    else:
        category = "none"

    # Normalise strength: linear ramp from 0 at WEAK threshold up to 1 at a
    # physically strong cap (e.g. 4 K/100 hPa).
    strength = np.clip((gmax - WEAK_INVERSION_THRESHOLD_K) / (4.0 - WEAK_INVERSION_THRESHOLD_K), 0.0, 1.0)

    # --- Geometry: how deep the trapping layer sits and how thick it is. ---
    # The audit listed base/top *height* and thickness as unimplemented because
    # only the pressures were reported. Both are derivable from the profile
    # itself, so they are computed here rather than deferred to a future archive.
    heights = layer_heights_m(temp_by_level or {}, height_by_level)
    base_h = heights.get(float(base_p))
    top_h = heights.get(float(top_p))
    thickness_m = None
    if base_h is not None and top_h is not None:
        thickness_m = round(float(top_h - base_h), 1)
    thickness_hpa = round(float(base_p - top_p), 1)

    return {
        "inversion_detected": category != "none",
        "inversion_strength": round(float(strength), 4),
        "inversion_category": category,
        "inversion_base_pressure": base_p,
        "inversion_top_pressure": top_p,
        "inversion_base_height_m": None if base_h is None else round(float(base_h), 1),
        "inversion_top_height_m": None if top_h is None else round(float(top_h), 1),
        "inversion_thickness_m": thickness_m,
        "inversion_thickness_hpa": thickness_hpa,
        "strongest_layer_gradient": round(gmax, 4),
        "profile_available": True,
    }


def classify_pbl(
    pbl_height: float | None,
    *,
    low_pbl_threshold_m: float = PBL_MODERATE_M,
    pbl_valid: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Classify the planetary boundary layer height.

    Returns:
      - low_pbl_flag       : bool (True when pbl <= low_pbl_threshold_m)
      - pbl_category       : strong_trapping / moderate_trapping / weak_trapping / good_dispersion / unknown
      - dispersion_condition: TRApped / LIMITED / MODERATE / GOOD / UNKNOWN
    """
    pbl_valid = pbl_valid or (0.0, 5000.0)
    if pbl_height is None or (isinstance(pbl_height, float) and np.isnan(pbl_height)):
        return {
            "low_pbl_flag": False,
            "pbl_category": "unknown",
            "dispersion_condition": "UNKNOWN",
        }
    lo, hi = pbl_valid
    if not (lo <= pbl_height <= hi):
        return {
            "low_pbl_flag": False,
            "pbl_category": "unknown",
            "dispersion_condition": "UNKNOWN",
        }

    low = pbl_height <= low_pbl_threshold_m
    if pbl_height < PBL_STRONG_M:
        cat, cond = "strong_trapping", "TRAPPED"
    elif pbl_height < PBL_MODERATE_M:
        cat, cond = "moderate_trapping", "LIMITED"
    elif pbl_height < PBL_WEAK_M:
        cat, cond = "weak_trapping", "MODERATE"
    else:
        cat, cond = "good_dispersion", "GOOD"

    return {
        "low_pbl_flag": bool(low),
        "pbl_category": cat,
        "dispersion_condition": cond,
    }


def combine_inversion(
    pbl_height: float | None,
    temp_by_level: dict[float, float] | None,
    *,
    height_by_level: dict[float, float] | None = None,
    pbl_valid: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Combine vertical lapse-rate and PBL-proxy inversion status.

    When vertical data is available the lapse-rate result is authoritative
    (``source == "lapse_rate"``). When it is not, the PBL-height proxy is used
    and marked ``source == "pbl_proxy"`` so consumers can disclose the basis.

    Returns a flat dict with keys from classify_gradient()/classify_pbl() plus
    ``inversion_source``.
    """
    res = classify_gradient(compute_lapse_rates(temp_by_level or {}), temp_by_level, height_by_level)
    pbl = classify_pbl(pbl_height, pbl_valid=pbl_valid)

    if res["profile_available"]:
        out = dict(res)
        out["inversion_source"] = "lapse_rate"
    else:
        # PBL-height proxy (documented fallback; see docs/SIH_GAP_AUDIT.md).
        is_inv = pbl_height is not None and pbl_height < PBL_WEAK_M
        if pbl_height is not None and not (isinstance(pbl_height, float) and np.isnan(pbl_height)):
            strength = np.clip((PBL_WEAK_M - pbl_height) / PBL_WEAK_M, 0.0, 1.0)
            if pbl_height < PBL_STRONG_M:
                cat = "strong"
            elif pbl_height < PBL_MODERATE_M:
                cat = "moderate"
            elif pbl_height < PBL_WEAK_M:
                cat = "weak"
            else:
                cat = "none"
        else:
            strength, cat = 0.0, "unknown"
        out = {
            "inversion_detected": bool(is_inv),
            "inversion_strength": round(float(strength), 4),
            "inversion_category": cat,
            "inversion_base_pressure": None,
            "inversion_top_pressure": None,
            # No vertical profile means no defensible geometry. Reporting a
            # thickness here would be inventing a number.
            "inversion_base_height_m": None,
            "inversion_top_height_m": None,
            "inversion_thickness_m": None,
            "inversion_thickness_hpa": None,
            "strongest_layer_gradient": None,
            "profile_available": False,
            "inversion_source": "pbl_proxy",
        }

    out.update(pbl)
    return out
