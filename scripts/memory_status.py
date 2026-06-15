"""Print the current memory guard status as JSON.

Equivalent of ``GET /api/v1/memory`` but available without the server.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from app.memory_guard import get_policy
    status = get_policy().poll()
    print(json.dumps(asdict(status), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
