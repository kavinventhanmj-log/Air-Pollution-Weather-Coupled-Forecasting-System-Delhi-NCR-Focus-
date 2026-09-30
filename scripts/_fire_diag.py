"""Diagnose add_fire_features timing with progress logging."""

import sys
import time
from datetime import UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.app.database import SessionLocal
from backend.app.models.db_models import FireReading, PollutionReading, Station, WeatherReading
from ml.preprocessing.training_dataset import _bearing_array, _haversine_arrays, align_observations, coerce_utc_naive

LOG = Path(__file__).resolve().parent.parent / "scripts" / "_fire_diag.log"


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def df_from_model(db, model, cols):
    rows = db.query(model).all()
    return pd.DataFrame({c: [getattr(r, c) for r in rows] for c in cols})


db = SessionLocal()
log("load start")
stations = df_from_model(db, Station, ["id", "name", "latitude", "longitude"])
stations["station"] = stations["name"]
poll = df_from_model(db, PollutionReading, ["station_id", "timestamp", "pm25"])
wx = df_from_model(db, WeatherReading, [
    "station_id", "timestamp", "temperature", "humidity", "pressure", "pressure_msl",
    "surface_pressure", "wind_speed", "wind_direction", "precipitation", "cloud_cover",
    "pbl_height", "temperature_1000hPa", "temperature_925hPa", "temperature_850hPa",
    "temperature_700hPa"])
fires = df_from_model(db, FireReading, ["latitude", "longitude", "acq_date", "frp"])
fires.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
db.close()
log(f"load shapes {poll.shape} {wx.shape} {fires.shape}")

id_to_name = dict(zip(stations["id"], stations["station"], strict=True))
poll["station"] = poll["station_id"].map(id_to_name)
wx["station"] = wx["station_id"].map(id_to_name)
poll = poll.dropna(subset=["station"])
wx = wx.dropna(subset=["station"])
stats = stations[["station", "latitude", "longitude"]]

df, _ = align_observations(poll, wx, stats)
log(f"align rows={len(df)} station_counts={df.groupby('station').size().to_dict()}")

# FIRMS acquisition times are UTC by definition, so they must not be read as
# naive IST like the station observations.
fires["acq_timestamp"] = coerce_utc_naive(fires["acq_date"], assume_tz=UTC)
fires = fires.dropna(subset=["acq_timestamp", "lat", "lon"]).reset_index(drop=True)
log(f"fires valid={len(fires)}")
f_lat = fires["lat"].to_numpy(dtype=float)
f_lon = fires["lon"].to_numpy(dtype=float)
f_frp = fires["frp"].to_numpy(dtype=float)
f_frp = np.where(np.isnan(f_frp) | (f_frp <= 0), 1.0, f_frp)

row_ns = df["hour"].astype("int64").to_numpy()
r_lat = df["latitude"].to_numpy(dtype=float)
r_lon = df["longitude"].to_numpy(dtype=float)

win_ns = int(24 * 3.6e12)
positions = df.groupby("station", sort=False).indices
for stn, pos in positions.items():
    t0 = time.time()
    lat = float(np.unique(r_lat[pos])[0])
    lon = float(np.unique(r_lon[pos])[0])
    log(f"station={stn} pos={len(pos)} lat={lat} lon={lon}")

    t1 = time.time()
    dist = _haversine_arrays(lat, lon, f_lat, f_lon)
    bearing = _bearing_array(lat, lon, f_lat, f_lon)
    log(f"  geometry over {len(f_lat)} fires: {time.time()-t1:.2f}s")

    keep = dist <= 500.0
    log(f"  within 500km: {int(keep.sum())} ({100.0*keep.mean():.1f}%)")
    f_ts = fires["acq_timestamp"].astype("int64").to_numpy()[keep]
    t1 = time.time()
    order = np.argsort(f_ts, kind="stable")
    log(f"  argsort: {time.time()-t1:.2f}s")
    rts = row_ns[pos]
    t1 = time.time()
    lo = np.searchsorted(f_ts[order], rts - win_ns, side="left")
    hi = np.searchsorted(f_ts[order], rts, side="right")
    log(f"  searchsorted: {time.time()-t1:.2f}s  window sizes min/mean/max: "
        f"{int((hi-lo).min())}/{float((hi-lo).mean()):.0f}/{int((hi-lo).max())}")
    t1 = time.time()
    for k in range(len(pos)):
        if (k % 20000) == 0:
            log(f"    row {k}/{len(pos)} elapsed={time.time()-t1:.1f}s")
        lo_k, hi_k = int(lo[k]), int(hi[k])
        if hi_k <= lo_k:
            continue
        w_d = dist[keep][order][lo_k:hi_k]
        w_b = bearing[keep][order][lo_k:hi_k]
        _ = w_d.min()
        _ = w_b.max()
    log(f"  row loop: {time.time()-t1:.2f}s")
log("DONE")
