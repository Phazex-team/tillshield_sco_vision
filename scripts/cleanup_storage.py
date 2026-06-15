"""Reclaim disk by deleting expired unlinked CCTV segments.

Dry-run by default. Pass ``--execute`` to actually unlink files +
delete rows. Linked segments (referenced by a video_window, artifact,
or review action) are NEVER deleted.

    python scripts/cleanup_storage.py --dry-run
    python scripts/cleanup_storage.py --execute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="(default) list what would be deleted")
    group.add_argument("--execute", action="store_true",
                       help="actually delete the expired unlinked files")
    args = ap.parse_args()

    execute = bool(args.execute)
    from db.session import get_sessionmaker
    from app.storage_guard import disk_status, run_cleanup

    SM = get_sessionmaker()
    with SM() as session:
        status = disk_status(session)
        report = run_cleanup(session, execute=execute)
        if execute:
            session.commit()

    out = {"disk": status, "cleanup": report}
    print(json.dumps(out, indent=2, default=str))
    if execute and report["deleted_rows"] == 0 and not report["failed"]:
        print("(no expired unlinked segments to delete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
