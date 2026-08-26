#!/usr/bin/env python3
"""Measure importing the complete Expert module without creating devices."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "reports" / "startup_trace"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"import_only_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    start = time.monotonic_ns()
    rows = [{"event": "PROCESS_MAIN_ENTER", "monotonic_ns": start}]
    sys.path.insert(0, str(root))
    import tools.test_three_nut_closed_loop_v2  # noqa: F401
    end = time.monotonic_ns()
    rows.append({"event": "IMPORT_COMPLETE", "monotonic_ns": end, "duration_ms": (end - start) / 1e6})
    rows.append({"event": "SLEEP_START", "monotonic_ns": time.monotonic_ns(), "duration_s": 15.0})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"[IMPORT_ONLY] duration_ms={(end - start) / 1e6:.3f}", flush=True)
    time.sleep(15.0)
    rows.append({"event": "SLEEP_END", "monotonic_ns": time.monotonic_ns()})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"[IMPORT_ONLY] trace={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
