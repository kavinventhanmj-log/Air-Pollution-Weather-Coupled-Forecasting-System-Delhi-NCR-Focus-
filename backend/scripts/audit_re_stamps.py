"""List legacy pollution rows that *may* be bootstrap re-stamps.

Read-only. This script never writes: it exists because the ``re_stamped``
migration (b8d3f1a9c4e2) deliberately does not guess. A row cannot be
distinguished from a real measurement after the fact, so classification is a
human decision, not a heuristic one.

What the review should look for
-------------------------------
``bootstrap_recent.py`` copied one station's latest reading forward across 24
hourly slots. The signature of that is a *run of consecutive hourly slots on
the same station carrying byte-identical values*. A station whose PM2.5 happens
to repeat once in a day is normal; a station whose full (pm25, pm10, o3, no2,
so2, co, aqi) tuple repeats across many consecutive hours is not.

Usage
-----
    python backend/scripts/audit_re_stamps.py                     # top runs
    python backend/scripts/audit_re_stamps.py --min-run 6 --limit 40
    python backend/scripts/audit_re_stamps.py --station "Anand Vihar"

To accept a finding and mark those rows honestly for future provenance:

    UPDATE pollution_observations SET re_stamped = 1 WHERE id IN (...);
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal  # noqa: E402
from app.models.db_models import PollutionReading, Station  # noqa: E402

#: The full set of scored values. A repeat of the whole tuple is far stronger
#: evidence than any single pollutant, which repeats constantly in real data.
VALUE_COLUMNS = (
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "aqi",
)


def _value_signature() -> list:
    """Group by the station plus the full value tuple, null-safely.

    NULLs are coalesced to a sentinel so rows that are entirely NULL collapse
    into one group; those are missing measurements, not re-stamps, and the
    caller is told to skip them explicitly.
    """
    return [PollutionReading.station_id] + [
        func.coalesce(getattr(PollutionReading, name), SENTINEL).label(name)
        for name in VALUE_COLUMNS
    ]


#: Sentinel standing in for NULL during grouping. -999 is outside the plausible
#: range of every scored pollutant, so it cannot collide with a real value.
SENTINEL = -999.0


def find_repeat_runs(session, min_run: int, station_name: str | None, limit: int):
    """Return the largest exact-value repeat groups.

    Each result is ``(station, value_tuple, count, first_ts, last_ts)``. Groups
    whose values are all NULL are excluded: they represent missing data, and
    labelling them as synthetic re-stamps would be wrong.
    """
    station = None
    if station_name:
        station = session.execute(
            select(Station).where(Station.name == station_name)
        ).scalar_one_or_none()
        if station is None:
            raise SystemExit(f"No station named {station_name!r}")

    signature = _value_signature()
    grouped = (
        select(
            *signature,
            func.count(PollutionReading.id).label("n"),
            func.min(PollutionReading.timestamp).label("first_ts"),
            func.max(PollutionReading.timestamp).label("last_ts"),
        )
        .where(PollutionReading.station_id == station.id if station else True)
        .group_by(*signature)
        .having(func.count(PollutionReading.id) >= min_run)
    ).subquery()

    rows = session.execute(
        select(
            Station.name,
            *(grouped.c[name] for name in VALUE_COLUMNS),
            grouped.c.n,
            grouped.c.first_ts,
            grouped.c.last_ts,
        )
        .select_from(grouped.join(Station, Station.id == grouped.c.station_id))
        .order_by(grouped.c.n.desc())
        .limit(limit)
    ).all()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-run", type=int, default=4,
                    help="minimum repeats of an identical value tuple to report (default: 4)")
    ap.add_argument("--limit", type=int, default=25, help="max groups to print (default: 25)")
    ap.add_argument("--station", help="restrict to one station name")
    args = ap.parse_args()

    session = SessionLocal()
    try:
        rows = find_repeat_runs(session, args.min_run, args.station, args.limit)
    finally:
        session.close()

    if not rows:
        print(f"No value tuple repeats {args.min_run}+ times. Nothing looks like a bootstrap re-stamp.")
        return 0

    print(f"Groups repeating an identical value tuple {args.min_run}+ times (READ ONLY - nothing written):\n")
    for name, pm25, pm10, o3, no2, so2, co, aqi, n, first_ts, last_ts in rows:
        values = (pm25, pm10, o3, no2, so2, co, aqi)
        if all(v == SENTINEL for v in values):
            print(f"  {name:<22} n={n:<5} all pollutants NULL, aqi=0 -> missing data, NOT a re-stamp; skip")
            continue
        shown = ", ".join(
            f"{k}=NULL" if v == SENTINEL else f"{k}={v}"
            for k, v in zip(VALUE_COLUMNS, values)
        )
        print(f"  {name:<22} n={n:<5} {first_ts} .. {last_ts}  {shown}")

    print("\nTo mark a confirmed group, re-stamp it explicitly - do not guess:")
    print("  UPDATE pollution_observations SET re_stamped = 1 WHERE id IN (...);")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
