"""Generate seasonally-realistic fire data for the 2023-2024 training period.

NASA FIRMS public CSVs only cover the recent ~24-48h, so historical fire
observations for 2023-2024 are not available from the free download.  This
script synthesizes statistically-realistic stubble-burning fire records based
on well-documented seasonal behaviour:
  - Kharif stubble burning peaks Oct-Nov (westerly transport toward Delhi NCR)
  - Rabi burning peaks Apr-May
  - Fires concentrated in Punjab/Haryana farmlands north & west of Delhi
  - Afternoon/daytime acquisition (NASA Aqua ~13:30, Terra ~10:30 LST)
  - Moderate FRP values typical of agricultural fires

The existing real recent FIRMS records (September 2026) are preserved and
appended.  The synthetic data is clearly flagged so downstream consumers know
it is simulated, not retrieved live.
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIRE_CSV = PROJECT_ROOT / "data" / "fire" / "firms_fires.csv"

# Farmland burning regions: (lat_range, lon_range, weight)
# Punjab (Amritsar-Ludhiana-Patiala belt), Haryana (Karnal-Hisar belt),
# and western/central UP. Weights approximate the historical FIRMS distribution.
REGIONS = [
    # Punjab
    ((30.0, 32.0), (74.3, 76.2), 5.0),
    ((30.5, 31.5), (75.5, 76.0), 4.0),
    # Haryana
    ((29.0, 30.5), (75.0, 77.0), 3.0),
    ((28.8, 29.8), (76.0, 77.2), 2.0),
    # Western UP
    ((28.0, 29.5), (77.2, 79.0), 1.5),
    # Rajasthan (sparser)
    ((27.5, 29.0), (74.0, 76.0), 0.8),
]

# Daily base fire-count envelope by month (scaled per region)
# Mid-Oct to mid-Nov is the dominant kharif burning season.
MONTH_SCALE = {
    1: 0.10, 2: 0.08, 3: 0.20, 4: 0.55, 5: 0.65,
    6: 0.25, 7: 0.05, 8: 0.05, 9: 0.10, 10: 0.90,
    11: 1.00, 12: 0.30,
}

RNG = np.random.default_rng(42)


def day_scale(month: int) -> float:
    """Compute seasonal intensity envelope for a given month."""
    base = MONTH_SCALE[month]
    return base


def generate_fires_for_date(day: datetime.date, total_budget: int) -> list[dict]:
    """Generate fire records for a single date."""
    records = []
    month = day.month

    if month == 10:
        # Ramping up through October, peak in the last week.
        peak = max(0.15, (day.day - 5) / 27.0)
    elif month == 11:
        # Strong through mid-November, tapering off.
        peak = max(0.10, (25 - day.day) / 20.0)
    elif month == 4:
        peak = 0.3 + 0.7 * (day.day / 30.0)
    elif month == 5:
        peak = max(0.15, (20 - day.day) / 12.0)
    else:
        peak = 1.0

    # Build region list weighted by region weight * peak intensity
    region_pool = []
    for lat_rng, lon_rng, weight in REGIONS:
        reps = max(1, int(weight * 10))
        for _ in range(reps):
            region_pool.append((lat_rng, lon_rng))

    n_total = int(total_budget * peak * MONTH_SCALE[month] * RNG.uniform(0.85, 1.15))
    n_total = max(0, n_total)

    # Acquisition times: Aqua ~13:30, Terra ~10:30 local; afternoon peak burning
    local_times = [13, 14, 13, 14, 15, 11, 12, 13, 14]
    # Delhi local = UTC+5:30 → convert to UTC
    for _ in range(n_total):
        lat_rng, lon_rng = region_pool[RNG.integers(0, len(region_pool))]
        lat = RNG.uniform(lat_rng[0], lat_rng[1])
        lon = RNG.uniform(lon_rng[0], lon_rng[1])
        local_hhmm = RNG.integers(0, len(local_times))
        local_hour = local_times[local_hhmm]
        # convert to UTC integer HHMM
        utc_min = (local_hour * 60 + RNG.integers(0, 45)) - 330
        utc_min = utc_min % (24 * 60)
        utc_hhmm = int(f"{utc_min // 60:02d}{utc_min % 60:02d}")
        frp = max(0, round(RNG.normal(12.0, 9.0), 2))
        conf = RNG.choice(["nominal", "high", "high", "high"], p=[0.3, 0.35, 0.2, 0.15])
        records.append({
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "acq_date": day.strftime("%Y-%m-%d"),
            "acq_time": int(utc_hhmm),
            "frp": frp,
            "confidence": conf,
            "bright_ti4": round(315.0 + RNG.normal(20, 12), 2),
            "daynight": "D",
            "satellite": RNG.choice(["Aqua", "Terra"]),
            "synthetic": True,
            "source": "synthetic_sim",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate historical fire data for 2023-2024")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--daily-budget", type=int, default=1800,
                        help="Peak-season daily fire budget (controls total volume)")
    parser.add_argument("--output", default=str(FIRE_CSV))
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    all_records = []
    day = start
    while day <= end:
        records = generate_fires_for_date(day, args.daily_budget)
        all_records.extend(records)
        day += timedelta(days=1)

    synthetic = pd.DataFrame(all_records)
    print(f"Generated {len(synthetic):,} synthetic fire records "
          f"({args.start_date} -> {args.end_date})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep only pre-existing REAL (non-synthetic) records so re-runs are
    # deterministic and the seasonal synthetic signal is not diluted.
    existing = None
    if out_path.exists():
        try:
            df_existing = pd.read_csv(out_path, low_memory=False)
            synth_mask = df_existing.get("synthetic", pd.Series(False, index=df_existing.index)).fillna(False)
            existing = df_existing[~synth_mask.astype(bool)] if not df_existing.empty else None
            if existing is not None and not existing.empty:
                print(f"Preserving {len(existing):,} real (non-synthetic) fire records")
        except Exception:
            existing = None

    if existing is not None and not existing.empty:
        combined = pd.concat([synthetic, existing], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["latitude", "longitude", "acq_date", "acq_time"], keep="last"
        )
    else:
        combined = synthetic

    os.makedirs(out_path.parent, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined):,} total records -> {out_path}")
    print(f"  Date range: {combined['acq_date'].min()} to {combined['acq_date'].max()}")
    print("  Synthetic records flagged with column 'synthetic'=True and source='synthetic_sim'")


if __name__ == "__main__":
    main()
