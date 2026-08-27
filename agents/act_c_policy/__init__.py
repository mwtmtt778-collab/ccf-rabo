"""Rabo Case entrypoint for the fixed-point Nut C ACT policy."""

from __future__ import annotations

from .runtime import run_official_agent

__all__ = ["run"]


def run() -> None:
    """Automatically execute the fixed-point C task when Rabo starts this Agent."""
    print("Agent loaded: act_c_policy", flush=True)
    status = run_official_agent()
    if status:
        raise SystemExit(status)
