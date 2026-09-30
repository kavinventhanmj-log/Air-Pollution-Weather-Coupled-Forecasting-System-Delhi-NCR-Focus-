"""One-command demo database reset: wipe, re-seed, re-stamp and refresh.

Reproduces the demo from scratch against any DATABASE_URL, e.g. a local
PostgreSQL instance:

    $env:DATABASE_URL="postgresql://aerocast:password@localhost:5432/aerocast_ncr"
    python -m backend.scripts.demo_reset
    Remove-Item Env:DATABASE_URL

Without DATABASE_URL set it rebuilds the local SQLite dev database instead.
All three steps (reset + seed, re-stamp into the last 24h) are the same ones
used by the docs; this is simply the single-shot orchestration of them.
"""

import argparse
from pathlib import Path

import backend.app.models.db_models  # noqa: F401  (registers all tables)
from backend.app.database import Base, SessionLocal, engine
from backend.scripts import bootstrap_recent, load_data


def main():
    parser = argparse.ArgumentParser(description="Reset and fully rebuild the demo database.")
    parser.add_argument("--csv", default=str(load_data.DEFAULT_CSV))
    parser.add_argument("--fire-csv", default=str(load_data.MODELS_DIR.resolve().parent / "data" / "fire" / "firms_fires.csv"))
    args = parser.parse_args()

    print("Dropping and recreating all tables ...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        print("Seeding coupled dataset ...")
        counts = load_data.load_coupled_data(db, Path(args.csv))
        print(f"  Pollution readings : {counts['pollution']:,}")
        print(f"  Weather readings   : {counts['weather']:,}")

        print("Loading fire data ...")
        n_fire = load_data.load_fire_data(db, Path(args.fire_csv))
        print(f"  Fire records       : {n_fire:,}")

        print("Loading model metrics ...")
        n_metrics = load_data.load_metrics(db, load_data.MODELS_DIR / "metrics.json")
        print(f"  Metric entries     : {n_metrics:,}")

        print("Seeding alerts ...")
        n_alerts = load_data.load_alerts_seed(db)
        print(f"  Seed alerts        : {n_alerts:,}")
    finally:
        db.close()

    print("Re-stamping newest observations into the last 24h ...")
    bootstrap_recent.main()

    print("\nDemo database is live. Open the app at http://localhost:5173")
    print("or run `make run-api` + `cd frontend && npm run dev` for a bare-metal build.")


if __name__ == "__main__":
    main()
