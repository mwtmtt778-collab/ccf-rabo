#!/usr/bin/env python3
"""Validate a Rabo ACT ONNX deployment without reading or commanding the robot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_FORMAT = "rabo_act_onnx_bundle_v1"
INPUT_NAMES = ("state", "cam_top", "cam_left_wrist", "cam_right_wrist")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(deployment_dir: Path) -> dict[str, Any]:
    manifest_path = deployment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"unsupported manifest format: {manifest.get('format')!r}")
    if manifest.get("action_dimension") != 26:
        raise ValueError("manifest action_dimension must be 26")
    names = manifest.get("action_names")
    if not isinstance(names, list) or len(names) != 26 or len(set(names)) != 26:
        raise ValueError("manifest action_names must contain 26 unique names")
    for filename, expected in manifest.get("sha256", {}).items():
        actual = sha256_file(deployment_dir / filename)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {filename}: {actual} != {expected}")
    return manifest


def make_session(model_path: Path, threads: int) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def load_golden_inputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        inputs = {name: np.asarray(archive[name], dtype=np.float32) for name in INPUT_NAMES}
    expected_shapes = {
        "state": (1, 26),
        "cam_top": (1, 3, 224, 224),
        "cam_left_wrist": (1, 3, 224, 224),
        "cam_right_wrist": (1, 3, 224, 224),
    }
    for name, expected_shape in expected_shapes.items():
        value = inputs[name]
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"invalid golden input {name}: {value.shape}")
    return inputs


def benchmark(deployment_dir: Path, threads: int, runs: int) -> dict[str, Any]:
    if threads <= 0 or runs <= 0:
        raise ValueError("threads and runs must be positive")
    manifest = load_manifest(deployment_dir)
    inputs = load_golden_inputs(deployment_dir / "golden_inputs.npz")
    expected = np.load(deployment_dir / "golden_output.npy", allow_pickle=False)
    session = make_session(deployment_dir / "act_policy.onnx", threads)
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError(f"CPU provider was not selected: {session.get_providers()}")

    actual = session.run(["action_chunk"], inputs)[0]
    if actual.shape != (1, 20, 26) or not np.isfinite(actual).all():
        raise ValueError(f"invalid model output: {actual.shape}")
    if expected.shape != actual.shape or not np.isfinite(expected).all():
        raise ValueError(f"invalid golden output: {expected.shape}")
    difference = np.abs(actual - expected)

    for _ in range(3):
        session.run(["action_chunk"], inputs)
    durations_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        output = session.run(["action_chunk"], inputs)[0]
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        if not np.isfinite(output).all():
            raise ValueError("model produced NaN/Inf during benchmark")

    max_difference = float(difference.max())
    p95_ms = float(np.percentile(durations_ms, 95))
    max_allowed_difference = float(manifest["acceptance"]["max_abs_difference"])
    max_allowed_p95_ms = float(manifest["acceptance"]["max_p95_inference_ms"])
    max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {
        "format": "rabo_act_onnx_benchmark_v1",
        "provider": session.get_providers()[0],
        "threads": threads,
        "runs": runs,
        "output_shape": list(actual.shape),
        "finite": True,
        "golden_max_abs_difference": max_difference,
        "golden_mean_abs_difference": float(difference.mean()),
        "latency_ms": {
            "mean": float(np.mean(durations_ms)),
            "p50": float(np.percentile(durations_ms, 50)),
            "p95": p95_ms,
            "max": float(np.max(durations_ms)),
        },
        "process_max_rss_mib": max_rss_kib / 1024.0,
        "acceptance": {
            "max_abs_difference": max_allowed_difference,
            "max_p95_inference_ms": max_allowed_p95_ms,
        },
        "pass": bool(
            math.isfinite(max_difference)
            and max_difference <= max_allowed_difference
            and p95_ms <= max_allowed_p95_ms
        ),
        "robot_commanded": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    result = benchmark(args.deployment_dir.resolve(), args.threads, args.runs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit("ACT_ONNX_BENCHMARK_FAILED")
    print("ACT_ONNX_BENCHMARK_PASS: no robot interface was opened", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
