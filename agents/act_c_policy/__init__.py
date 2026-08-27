"""Official Rabo entrypoint for the fixed-point Nut C ACT policy."""

from __future__ import annotations

from .runtime import main

__all__ = ["run"]


def run() -> None:
    """First deployment stage: automatic integrated dry-run, never robot actuation."""
    raise SystemExit(main(["--dry-run", "--duration", "10"]))

