#!/usr/bin/env python3
"""Run or dry-run the Rabo three-nut expert.

Default behavior is dry-run. Use --execute only in the Rabo cloud sim after
reviewing the generated contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "three_nut_expert"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "RABO_THREE_NUT_EXPERT_IMPLEMENTATION_REPORT.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute the Rabo three-nut expert.")
    parser.add_argument("--mode", choices=["contract", "single", "three"], default="contract")
    parser.add_argument("--nut", choices=["A", "B", "C"], default="B")
    parser.add_argument("--order", default="A,B,C", help="Comma-separated nut order for --mode three.")
    parser.add_argument("--trials", type=int, default=1, help="Repeat single-nut run N times.")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually call SetEntityPose and robot motion APIs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser


def markdown_report(payload: dict[str, Any]) -> str:
    contract = payload["contract"]
    results = payload.get("results", [])
    lines = [
        "# Rabo Three-Nut Expert Implementation Report",
        "",
        "## 1. Result",
        "",
        f"Overall: {payload['overall']}",
        "",
        "## 2. Safety Mode",
        "",
        f"- Execute: `{payload['execute']}`",
        "- Default runner mode is dry-run. Robot motion requires explicit `--execute`.",
        "",
        "## 3. Source Of Truth",
        "",
        f"- {contract['source_of_truth']}",
        "- The legacy `agents/arm_hand_demo/__init__.py` file was not rewritten.",
        "- Snapshot: `docs/legacy/arm_hand_demo_legacy_snapshot.py`",
        "",
        "## 4. Device IDs",
        "",
        *[f"- {k}: `{v}`" for k, v in contract["device_ids"].items()],
        "",
        "## 5. Single-Nut Contract",
        "",
        f"- Proven nut: `{contract['single_nut_contract']['proven_nut']}`",
        f"- Right arm base XY: `{contract['single_nut_contract']['right_arm_base_xy']}`",
        f"- Grasp transform: `{contract['single_nut_contract']['grasp_transform']}`",
        f"- Right grasp force: `{contract['single_nut_contract']['right_grasp_force']}`",
        f"- Left grasp force: `{contract['single_nut_contract']['left_grasp_force']}`",
        "",
        "## 6. Gates",
        "",
        *[f"- {gate}" for gate in contract["gates"]],
        "",
        "## 7. Current Run Results",
        "",
    ]
    if results:
        lines.extend(
            [
                "| Mode | Nuts | Execute | Success | Steps Planned | Steps Executed | Error |",
                "| --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in results:
            lines.append(
                "| {mode} | {nuts} | {execute} | {success} | {planned} | {executed} | {error} |".format(
                    mode=item["mode"],
                    nuts=",".join(item["nut_keys"]),
                    execute=item["execute"],
                    success=item["success"],
                    planned=item["steps_planned"],
                    executed=item["steps_executed"],
                    error=item.get("error") or "",
                )
            )
    else:
        lines.append("- Contract generated only; no run requested.")
    lines.extend(
        [
            "",
            "## 8. Unknowns",
            "",
            *[f"- {item}" for item in contract["unknowns"]],
            "",
            "## 9. Cloud Commands",
            "",
            "Dry-run contract:",
            "",
            "```bash",
            "python3 tools/run_three_nut_expert.py --mode contract",
            "```",
            "",
            "Single B execute gate:",
            "",
            "```bash",
            "python3 tools/run_three_nut_expert.py --mode single --nut B --trials 1 --execute",
            "```",
            "",
            "Three-nut execute only after single gates pass:",
            "",
            "```bash",
            "python3 tools/run_three_nut_expert.py --mode three --order A,B,C --execute",
            "```",
            "",
            "## 10. Final Decision",
            "",
            f"- READY_FOR_SINGLE_NUT_CLOUD_TEST: {'YES' if not payload['execute'] else 'CHECK_RESULTS'}",
            "- READY_FOR_THREE_NUT_BATCH: NO until single B and A/C gates pass.",
            "- READY_FOR_RECORDER: NO until camera gate is fixed.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir = resolve(args.output_dir)
    args.report_path = resolve(args.report_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    from agents.three_nut_expert.expert import (
        build_demo_contract,
        execute_single_nut,
        execute_three_nut,
        print_contract,
        write_json,
    )

    contract = build_demo_contract()
    results: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    overall = "PASS"

    if args.mode == "contract":
        print_contract(contract)
    elif args.mode == "single":
        for i in range(args.trials):
            result, plan = execute_single_nut(
                args.nut,
                seed=args.seed + i,
                execute=args.execute,
                enable_jitter=not args.no_jitter,
            )
            results.append(asdict(result))
            plans.append(plan)
            print(json.dumps(asdict(result), ensure_ascii=False))
            if not result.success:
                overall = "FAIL"
                break
    elif args.mode == "three":
        order = [item.strip().upper() for item in args.order.split(",") if item.strip()]
        result, plan = execute_three_nut(
            order=order,
            seed=args.seed,
            execute=args.execute,
            enable_jitter=not args.no_jitter,
        )
        results.append(asdict(result))
        plans.append(plan)
        print(json.dumps(asdict(result), ensure_ascii=False))
        if not result.success:
            overall = "FAIL"

    payload: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "mode": args.mode,
        "execute": args.execute,
        "overall": overall,
        "contract": contract,
        "results": results,
        "plans": plans,
    }
    write_json(args.output_dir / "three_nut_expert_result.json", payload)
    args.report_path.write_text(markdown_report(payload), encoding="utf-8")
    print(f"Report: {args.report_path}")
    print(f"JSON: {args.output_dir / 'three_nut_expert_result.json'}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
