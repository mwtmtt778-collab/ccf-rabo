"""Three-nut expert entrypoint.

The default package entrypoint is intentionally dry-run only. Real robot/sim
motion must be launched through tools/run_three_nut_expert.py with --execute.
"""

from __future__ import annotations

from .expert import build_demo_contract, print_contract


def run() -> None:
    print("three_nut_expert default entrypoint is dry-run only.")
    print_contract(build_demo_contract())
