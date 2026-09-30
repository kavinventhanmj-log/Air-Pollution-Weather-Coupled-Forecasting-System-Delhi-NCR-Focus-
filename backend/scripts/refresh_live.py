"""Run a one-shot live data refresh on the backend database.

Usage:
    python -m backend.scripts.refresh_live [--once]
"""

import argparse
import json

from backend.app.services.refresh_service import run_refresh_once  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Refresh live weather/fire/pollution data.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON.")
    args = parser.parse_args()

    summary = run_refresh_once()
    if args.json:
        print(json.dumps(summary))
    else:
        for source, count in summary.items():
            print(f"  {source:10s}: {count:,} new rows")


if __name__ == "__main__":
    main()
