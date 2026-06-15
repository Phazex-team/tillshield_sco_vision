"""Launch the FastAPI app locally.

Runs startup integrity checks, then exec's uvicorn against
``app.main:app``. In production mode (``FRAUD_OFFLINE_MODE=1``) the
startup checks fail loudly when any required model bundle is missing.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3902)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--skip-checks", action="store_true",
                    help="Skip startup integrity checks (dev only).")
    args = ap.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_app")

    if not args.skip_checks:
        from app.startup import StartupCheckError, run_startup_checks
        try:
            summary = run_startup_checks()
        except StartupCheckError as exc:
            print(f"STARTUP FAILED:\n{exc}", file=sys.stderr)
            return 2
        log.info("startup checks ok: %s", summary)

    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
