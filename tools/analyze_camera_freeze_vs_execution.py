#!/usr/bin/env python3
"""Correlate camera receive gaps with the execution trace (offline only)."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def load_camera(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            stamp = row.get("receive_monotonic_ns")
            if stamp is not None:
                rows.append({**row, "receive_monotonic_ns": int(stamp)})
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(rows, key=lambda row: row["receive_monotonic_ns"])


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("monotonic_ns") is not None:
                rows.append({**row, "monotonic_ns": int(row["monotonic_ns"])})
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(rows, key=lambda row: row["monotonic_ns"])


def gap_rows(camera: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for previous, following in zip(camera, camera[1:]):
        gap_ns = following["receive_monotonic_ns"] - previous["receive_monotonic_ns"]
        if gap_ns > 500_000_000:
            result.append({"previous": previous, "next": following, "gap_ns": gap_ns})
    return result


def event_text(row: dict[str, Any]) -> str:
    fields = [row.get("event", "?")]
    for key in ("phase", "device", "operation", "duration_ms", "result", "exception", "previous_device"):
        if row.get(key) is not None:
            fields.append(f"{key}={row[key]}")
    return " ".join(fields)


def correlate(gap: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    start = gap["previous"]["receive_monotonic_ns"]
    end = gap["next"]["receive_monotonic_ns"]
    before = [row for row in trace if row["monotonic_ns"] <= start]
    during = [row for row in trace if start < row["monotonic_ns"] < end]
    active = None
    for row in before:
        if row.get("phase") or row.get("device") or row.get("operation"):
            active = row
    return {**gap, "last_event_before": before[-1] if before else None, "active_at_gap_start": active, "during": during}


def format_gap(item: dict[str, Any], index: int) -> list[str]:
    previous = item["previous"]["receive_monotonic_ns"]
    following = item["next"]["receive_monotonic_ns"]
    lines = [
        f"### Gap {index}",
        "",
        f"- previous_camera_time_ns: `{previous}`",
        f"- next_camera_time_ns: `{following}`",
        f"- gap_ms: `{item['gap_ns'] / 1e6:.3f}`",
        "",
        "**execution_at_gap_start**",
        "",
    ]
    active = item.get("active_at_gap_start")
    if active:
        lines.append(f"- phase: `{active.get('phase', 'UNKNOWN')}`")
        lines.append(f"- device: `{active.get('device', 'UNKNOWN')}`")
        lines.append(f"- operation: `{active.get('operation', 'UNKNOWN')}`")
    else:
        lines.append("- OPERATION_NOT_PROVEN")
    lines.extend(["", "**last_event_before_gap**", ""])
    lines.append(f"- `{event_text(item['last_event_before']) if item.get('last_event_before') else 'NONE'}`")
    lines.extend(["", "**events_during_gap**", ""])
    if item["during"]:
        lines.extend(f"- `{event_text(row)}`" for row in item["during"])
    else:
        lines.append("- NONE")
    lines.append("")
    return lines


def analyze(camera: list[dict[str, Any]], trace: list[dict[str, Any]]) -> str:
    if len(camera) < 2:
        raise ValueError("at least two camera timestamps are required")
    gaps = gap_rows(camera)
    first = gaps[0] if gaps else None
    correlated = [correlate(gap, trace) for gap in gaps]
    lines = ["# Camera / Execution Correlation", "", f"- camera_frames: `{len(camera)}`", f"- execution_events: `{len(trace)}`", ""]
    for threshold in (500, 1000, 2000):
        lines.append(f"- gaps_gt_{threshold}ms: `{sum(item['gap_ns'] > threshold * 1e6 for item in gaps)}`")
    lines.append("")
    if first:
        lines.extend(["## FIRST_CAMERA_FREEZE", "", f"- previous_camera_time_ns: `{first['previous']['receive_monotonic_ns']}`", f"- next_camera_time_ns: `{first['next']['receive_monotonic_ns']}`", f"- gap_ms: `{first['gap_ns'] / 1e6:.3f}`", ""])
        active = correlated[0].get("active_at_gap_start")
        if active:
            lines.extend([f"- FIRST_CAMERA_FREEZE_PHASE = `{active.get('phase', 'UNKNOWN')}`", f"- FIRST_CAMERA_FREEZE_DEVICE = `{active.get('device', 'UNKNOWN')}`", f"- FIRST_CAMERA_FREEZE_OPERATION = `{active.get('operation', 'UNKNOWN')}`"])
        else:
            lines.extend(["- FIRST_CAMERA_FREEZE_PHASE = `OPERATION_NOT_PROVEN`", "- FIRST_CAMERA_FREEZE_DEVICE = `OPERATION_NOT_PROVEN`", "- FIRST_CAMERA_FREEZE_OPERATION = `OPERATION_NOT_PROVEN`"])
        lines.append("")
    else:
        lines.extend(["## RESULT", "", "No camera gap greater than 500 ms was found.", ""])
    lines.extend(["## Correlated gaps", ""])
    for index, item in enumerate(correlated, 1):
        lines.extend(format_gap(item, index))
    return "\n".join(lines)


def self_test() -> None:
    camera = [{"receive_monotonic_ns": value} for value in (0, 100_000_000, 700_000_000, 2_000_000_000)]
    trace = [
        {"event": "sdk_call_end", "monotonic_ns": 100_000_000, "phase": "RIGHT_LIFT", "device": "RIGHT_ARM", "operation": "get_joint_angles"},
        {"event": "phase_start", "monotonic_ns": 800_000_000, "phase": "LEFT_PLACE", "device": "LEFT_ARM"},
    ]
    report = analyze(camera, trace)
    assert "FIRST_CAMERA_FREEZE_PHASE = `RIGHT_LIFT`" in report
    assert "gap_ms: `600.000`" in report
    assert "gap_ms: `1300.000`" in report
    print("synthetic camera/execution correlation self-test: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlate camera receive gaps with execution trace timestamps.")
    parser.add_argument("--camera-jsonl", type=Path)
    parser.add_argument("--execution-trace", type=Path)
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic timestamp correlation test.")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.camera_jsonl is None or args.execution_trace is None:
        parser.error("--camera-jsonl and --execution-trace are required unless --self-test is used")
    report = analyze(load_camera(args.camera_jsonl), load_trace(args.execution_trace))
    output = args.output or (Path("reports/execution_trace") / f"camera_execution_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
