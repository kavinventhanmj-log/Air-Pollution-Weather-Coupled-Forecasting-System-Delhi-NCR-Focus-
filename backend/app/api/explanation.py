from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import Station
from ..schemas.schemas import ExplanationResponse
from ..services import explanation_service, pm25_explanation_service
from ..services.aqi_calculator import calculate_aqi

router = APIRouter()


@router.get("/explanation/{station_name}", response_model=ExplanationResponse)
def get_explanation(
    station_name: str,
    db: Session = Depends(get_db),
    horizon: int = Query(default=24, ge=1, le=72),
):
    """SHAP explanation of the PM2.5 forecast for a station.

    Uses the deployed per-horizon PM2.5 model in ``models/pm25`` together with
    its own training-shaped feature builder (``build_feature_row``). The
    previous version loaded the flat ``xgboost_pm25_24h`` artifact, which
    expects 114 columns including ``mean_frp``/``max_bright``/season dummies
    that no code path in this repository produces - those were silently
    zero-filled, so the returned attributions described values that were never
    measured. That model and feature set are not a matched pair.
    """
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")

    try:
        payload = pm25_explanation_service.explain_pm25_forecast(db, station_name, horizon)
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith("station_not_found"):
            raise HTTPException(status_code=404, detail=detail) from exc
        insufficient_markers = (
            "Not enough recent pollution history",
            "No aligned observations",
            "No pollution readings for station",
        )
        if any(marker in detail for marker in insufficient_markers):
            # Data availability, not a bad request: the caller cannot fix it by
            # changing parameters, so it is a 503 like the forecast routes.
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "insufficient_data",
                    "station": station.name,
                    "reason": detail,
                },
            ) from exc
        raise HTTPException(status_code=400, detail=detail) from exc
    except RuntimeError as exc:
        # No trained model: refuse, never fabricate.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    drivers = list(payload.get("top_positive_drivers", [])) + list(
        payload.get("top_negative_drivers", [])
    )
    drivers = sorted(
        drivers, key=lambda d: abs(d.get("shap_value", 0.0)), reverse=True
    )[:6]
    top_features = [
        {
            "feature": d.get("feature"),
            "importance": abs(d.get("shap_value", 0.0)),
            "direction": d.get("direction", "increases"),
            "description": d.get("description", ""),
        }
        for d in drivers
    ]

    feature_values = payload.get("feature_values_used", {}) or {}
    natural_language = explanation_service.generate_natural_language(
        feature_values, top_features, {"pm25_pred": payload.get("forecast_pm25")}
    )

    # AQI is derived from the predicted PM2.5 alone here, so it is a PM2.5-only
    # sub-index, not a full six-pollutant AQI. The other sub-indices are left
    # absent rather than assumed.
    pm25_pred = payload.get("forecast_pm25")
    aqi_pred = None
    if pm25_pred is not None:
        aqi_pred, _, _ = calculate_aqi(pm25=float(pm25_pred), pm10=None, o3=None, no2=None, so2=None, co=None)

    return ExplanationResponse(
        station=payload["station"],
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        prediction={
            "pm25_pred": pm25_pred,
            "aqi_pred": aqi_pred,
            "aqi_basis": "pm25_sub_index_only",
            "horizon_hours": payload.get("horizon_hours"),
            "forecast_timestamp": payload.get("forecast_timestamp"),
        },
        top_features=top_features,
        natural_language=natural_language,
    )
