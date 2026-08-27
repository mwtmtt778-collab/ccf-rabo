"""Official Rabo entrypoint for the fixed-point Nut C ACT policy."""

from __future__ import annotations

from .runtime import main

__all__ = ["run"]


def run() -> None:
    """Official fixed-point C Agent: serial receding-horizon execution."""
    raise SystemExit(main(["--execute-mvp"]))
