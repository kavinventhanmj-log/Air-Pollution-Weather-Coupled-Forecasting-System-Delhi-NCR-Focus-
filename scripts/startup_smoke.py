import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path[:1]:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI
from app.main import lifespan

async def main():
    print("BEFORE")
    try:
        async with lifespan(FastAPI()):
            print("INSIDE LIFESPAN")
    except BaseException as e:
        print("CAUGHT:", type(e).__name__, repr(e))
        raise

asyncio.run(main())
print("FINISHED")
