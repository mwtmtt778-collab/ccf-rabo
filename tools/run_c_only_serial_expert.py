#!/usr/bin/env python3
"""Strictly serial, fail-stop Nut C Expert for raw ACT collection."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.three_nut_expert.config import (  # noqa: E402
    DEVICE_IDS,
    HAND_OPEN,
    KNOWN_FIXED_NUT_WORLD_POSE,
    LEFT_FIXED_PLACE_JOINT_PATHS,
    LEFT_PRE_JOINTS,
    RIGHT_GRASP_FORCE,
)
from agents.three_nut_expert.expert import (  # noqa: E402
    build_right_safe_lift_pose,
    compute_right_approach_pose,
    compute_right_grasp_pose,
    pose_to_list,
)
from expert.left_nut_grasp_planner import LeftNutGraspPlanner, result_failed  # noqa: E402
from tools.test_dual_closed_loop_v2_1 import (  # noqa: E402
    RIGHT_OBSERVATION_JOINTS,
    RIGHT_RELEASE_OPEN_WAIT_S,
    safe_lift_pose_from_grasp,
)
from tools.test_dual_closed_loop_v2_2 import (  # noqa: E402
    RIGHT_RELEASE_POSE,
)
from tools.test_right_release_stability import (  # noqa: E402
    DEFAULT_HOLD_AFTER_GRASP_S,
    DEFAULT_SETTLE_AFTER_RELEASE_S,
    DEFAULT_VISION_TARGET_RADIUS_M,
    TARGET_NUT_B_WORLD_POSE,
)


# [VERIFIED] Unmodified SDK-order samples from the successful
# RIGHT_RELEASE_SAFE_HEIGHT trajectory in episode_20260826_144708_105746.
# Intermediate points are telemetry.npz states[199,201,203,205, 7:14]; the
# terminal point is expert_states RIGHT_SAFE_RETREAT.move.joint_after.
RIGHT_RELEASE_SAFE_JOINT_PATH = (
    (
        0.7453176379203796,
        0.04430386796593666,
        1.0963411331176758,
        -1.3637889623641968,
        -1.3782968521118164,
        0.42015182971954346,
        -1.317345142364502,
    ),
    (
        0.8841877579689026,
        0.016344869509339333,
        1.1822478771209717,
        -1.4212772846221924,
        -1.5596474409103394,
        0.3535405099391937,
        -1.3983210325241089,
    ),
    (
        0.9969002604484558,
        -0.006356131751090288,
        1.251999020576477,
        -1.467968463897705,
        -1.706965684890747,
        0.2994275987148285,
        -1.4640942811965942,
    ),
    (
        1.1160832643508911,
        -0.030346693471074104,
        1.3257124423980713,
        -1.5172969102859497,
        -1.862575650215149,
        0.24226896464824677,
        -1.533569574356079,
    ),
    (
        1.2227206244205195,
        -0.051811932772827037,
        1.391666589012047,
        -1.5614329713680626,
        -2.0018056990128485,
        0.19112702025328832,
        -1.595731698495557,
    ),
)


ROS_NAMESPACE = "/gs_1eebee6f37512bbc1d125b25511e912c"
TOP_RGB_TOPIC = f"{ROS_NAMESPACE}/r6ef2dc_tp_cam_303d2b1ce0"
ARM_TOPIC_SUFFIXES = (
    "08fa69bc43", "1pa9vnh1mr", "2m4fzrdssg", "4jjp9rlwus",
    "rsqed0qcrb", "y1vmospyub", "y8sm0inqbu",
)
HAND_TOPIC_SUFFIXES = (
    "1ff4n99g6p", "26hns2y9iz", "3id78faqov", "c344tguind",
    "f1re3fuf56", "gww3yfamdt", "k3y7rwcnkq", "kfl2g3asap",
    "l2kojhzci0", "o6aptfg62g", "zp3f47zr30",
)
RIGHT_HAND_TOPIC_SUFFIXES = (
    "3gpp28c61u", "4d9q0h9sn8", "8n8u1jtrj3", "cxjshiucra",
    "ilteetn3js", "msti8uutyd", "q2al5ws592", "up6h8dvvum",
    "v0b7dxyble", "wtszomzf2i", "zrbc9dcp12",
)
STATE_TOPICS = {
    "left_arm": [f"{ROS_NAMESPACE}/rbd03eb_tp_ps_{value}" for value in ARM_TOPIC_SUFFIXES],
    "right_arm": [f"{ROS_NAMESPACE}/r412d23_tp_ps_{value}" for value in ARM_TOPIC_SUFFIXES],
    "left_hand": [f"{ROS_NAMESPACE}/r136d7b_tp_ps_{value}" for value in HAND_TOPIC_SUFFIXES],
    "right_hand": [f"{ROS_NAMESPACE}/rcd72e2_tp_ps_{value}" for value in RIGHT_HAND_TOPIC_SUFFIXES],
}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return repr(value)


class EventWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def write(
        self,
        phase: str,
        device: str,
        command: str,
        target: Any,
        result: Any,
        *,
        event: str,
        **fields: Any,
    ) -> None:
        row = {
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "timestamp_wall_ns": time.time_ns(),
            "event": event,
            "phase": phase,
            "device": device,
            "command": command,
            "target": jsonable(target),
            "result": jsonable(result),
            **jsonable(fields),
        }
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


@dataclass(frozen=True)
class ArmObservation:
    sequence: int
    timestamp_monotonic_ns: int
    state: tuple[float, ...]


class PassiveArmState:
    """Background ROS cache; its only thread receives passive state."""

    def __init__(self, history_hz: float = 5.0) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import JointState
        except Exception as exc:
            raise RuntimeError(f"passive ROS JointState unavailable: {exc!r}") from exc
        self.rclpy = rclpy
        rclpy.init(args=None)
        self.node = rclpy.create_node("act_c_serial_passive_motion_state")
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.latest: dict[str, dict[int, float]] = {"left_arm": {}, "right_arm": {}}
        self.history: dict[str, deque[ArmObservation]] = {
            "left_arm": deque(maxlen=512),
            "right_arm": deque(maxlen=512),
        }
        self.sequences = {"left_arm": 0, "right_arm": 0}
        self.last_history_ns = {"left_arm": 0, "right_arm": 0}
        self.history_period_ns = int(round(1e9 / history_hz))
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscriptions = []
        for arm in ("left_arm", "right_arm"):
            for index, topic in enumerate(STATE_TOPICS[arm]):
                callback = lambda msg, a=arm, i=index: self._callback(a, i, msg)
                self.subscriptions.append(self.node.create_subscription(JointState, topic, callback, qos))
        self.thread = threading.Thread(target=self._run, name="act-c-passive-arm-state", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.executor.spin_once(timeout_sec=0.05)
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def _callback(self, arm: str, index: int, msg: Any) -> None:
        positions = list(getattr(msg, "position", []))
        if positions:
            now_ns = time.monotonic_ns()
            with self.condition:
                self.latest[arm][index] = float(positions[0])
                if len(self.latest[arm]) == 7 and now_ns - self.last_history_ns[arm] >= self.history_period_ns:
                    self.sequences[arm] += 1
                    observation = ArmObservation(
                        self.sequences[arm],
                        now_ns,
                        tuple(self.latest[arm][joint] for joint in range(7)),
                    )
                    self.history[arm].append(observation)
                    self.last_history_ns[arm] = now_ns
                    self.condition.notify_all()

    def spin(self, seconds: float) -> None:
        self.stop_event.wait(seconds)

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while time.monotonic() < deadline:
                if self.error is not None:
                    raise RuntimeError(f"passive ROS executor failed: {self.error!r}")
                if all(self.history[arm] for arm in ("left_arm", "right_arm")):
                    return
                self.condition.wait(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            missing = {arm: sorted(set(range(7)) - set(values)) for arm, values in self.latest.items()}
        raise RuntimeError(f"passive arm state timeout: {missing}")

    def observation(self, arm: str) -> ArmObservation:
        with self.lock:
            if not self.history[arm]:
                raise RuntimeError(f"passive {arm} state incomplete")
            return self.history[arm][-1]

    def history_since(self, arm: str, sequence: int) -> list[ArmObservation]:
        with self.lock:
            return [row for row in self.history[arm] if row.sequence > sequence]

    def wait_for_update(self, arm: str, sequence: int, timeout_s: float) -> None:
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"passive ROS executor failed: {self.error!r}")
            if not self.history[arm] or self.history[arm][-1].sequence <= sequence:
                self.condition.wait(timeout=timeout_s)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class SerialExpert:
    def __init__(self, events: EventWriter, passive: PassiveArmState, right: Any, left: Any, args: argparse.Namespace) -> None:
        self.events = events
        self.passive = passive
        self.right = right
        self.left = left
        self.args = args
        self.hand6 = {"LEFT_HAND": [0.0] * 6, "RIGHT_HAND": [0.0] * 6}

    def arm_command(self, phase: str, arm_name: str, command: str, target: Sequence[float], fn: Callable[[], Any]) -> None:
        device = "LEFT_ARM" if arm_name == "left_arm" else "RIGHT_ARM"
        target_values = [float(value) for value in target]
        baseline = self.passive.observation(arm_name)
        command_start_ns = time.monotonic_ns()
        self.events.write(
            phase, device, command, target_values, None, event="command_start",
            command_start_ns=command_start_ns,
            pre_command_state=list(baseline.state),
            baseline_sequence=baseline.sequence,
            baseline_timestamp_monotonic_ns=baseline.timestamp_monotonic_ns,
        )
        try:
            result = fn()
        except BaseException as exc:
            command_return_ns = time.monotonic_ns()
            self.events.write(
                phase, device, command, target_values, repr(exc), event="command_return",
                command_start_ns=command_start_ns,
                command_return_ns=command_return_ns,
                command_call_ms=(command_return_ns - command_start_ns) / 1e6,
            )
            raise
        command_return_ns = time.monotonic_ns()
        observations_during_call = [
            row for row in self.passive.history_since(arm_name, baseline.sequence)
            if command_start_ns <= row.timestamp_monotonic_ns <= command_return_ns
        ]
        movement_during_call = self._movement_observed(baseline.state, observations_during_call)
        self.events.write(
            phase, device, command, target_values, result, event="command_return",
            command_start_ns=command_start_ns,
            command_return_ns=command_return_ns,
            command_call_ms=(command_return_ns - command_start_ns) / 1e6,
            movement_observed_during_call=movement_during_call,
        )
        failed, reason = result_failed(result)
        if failed:
            raise RuntimeError(f"{phase}: SDK return failed: {reason}")
        settled = self.wait_motion_settled(
            arm_name,
            baseline,
            command_start_ns=command_start_ns,
            command_return_ns=command_return_ns,
        )
        self.events.write(phase, device, command, target_values, settled, event="motion_settled")
        print("[ARM_SERIAL]", flush=True)
        print(
            f"phase={phase} command_call_ms={settled['command_call_ms']:.3f} "
            f"movement_observed_during_call={settled['movement_observed_during_call']} "
            f"movement_started={settled['movement_started']} "
            f"already_settled={settled['already_settled']} "
            f"settle_elapsed_ms={settled['settle_elapsed_ms']:.3f} result=PASS",
            flush=True,
        )
        self.passive.spin(self.args.post_settle_s)

    def _movement_observed(self, baseline: Sequence[float], observations: Sequence[ArmObservation]) -> bool:
        previous = tuple(float(value) for value in baseline)
        for row in observations:
            adjacent_delta = max(abs(a - b) for a, b in zip(row.state, previous))
            baseline_delta = max(abs(a - b) for a, b in zip(row.state, baseline))
            if max(adjacent_delta, baseline_delta) > self.args.start_threshold_rad:
                return True
            previous = row.state
        return False

    def wait_motion_settled(
        self,
        arm_name: str,
        baseline: ArmObservation,
        *,
        command_start_ns: int,
        command_return_ns: int,
    ) -> dict[str, Any]:
        deadline = command_start_ns + int(self.args.motion_timeout_s * 1e9)
        previous = baseline.state
        last_sequence = baseline.sequence
        movement_started = False
        stable_count = 0
        no_motion_stable_count = 0
        max_delta = 0.0
        movement_during_call = False
        while time.monotonic_ns() < deadline:
            observations = self.passive.history_since(arm_name, last_sequence)
            for row in observations:
                if row.timestamp_monotonic_ns < command_start_ns:
                    last_sequence = row.sequence
                    previous = row.state
                    continue
                adjacent_delta = max(abs(a - b) for a, b in zip(row.state, previous))
                baseline_delta = max(abs(a - b) for a, b in zip(row.state, baseline.state))
                max_delta = max(max_delta, adjacent_delta, baseline_delta)
                significant = max(adjacent_delta, baseline_delta) > self.args.start_threshold_rad
                if not movement_started and significant:
                    movement_started = True
                    movement_during_call = movement_during_call or row.timestamp_monotonic_ns <= command_return_ns
                    stable_count = 0
                elif movement_started and adjacent_delta < self.args.settle_threshold_rad:
                    stable_count += 1
                elif movement_started:
                    stable_count = 0
                elif adjacent_delta < self.args.settle_threshold_rad:
                    no_motion_stable_count += 1
                else:
                    no_motion_stable_count = 0
                previous = row.state
                last_sequence = row.sequence
            now_ns = time.monotonic_ns()
            already_settled = (
                not movement_started
                and no_motion_stable_count >= self.args.settle_samples
                and now_ns - command_return_ns >= int(self.args.no_motion_grace_s * 1e9)
            )
            if (movement_started and stable_count >= self.args.settle_samples) or already_settled:
                return {
                    "command_call_ms": (command_return_ns - command_start_ns) / 1e6,
                    "movement_observed_during_call": movement_during_call,
                    "movement_started": movement_started,
                    "already_settled": already_settled,
                    "stable_samples": stable_count if movement_started else no_motion_stable_count,
                    "max_observed_delta_rad": max_delta,
                    "settle_elapsed_ms": (now_ns - command_start_ns) / 1e6,
                }
            self.passive.wait_for_update(arm_name, last_sequence, 1.0 / self.args.observe_hz)
        raise TimeoutError(f"{arm_name} motion did not settle within {self.args.motion_timeout_s:g}s")

    def move_joints(self, phase: str, arm_name: str, arm: Any, target: Sequence[float]) -> None:
        values = [float(value) for value in target]
        self.arm_command(phase, arm_name, "move_joints", values, lambda: arm.move_joints(values, blocking=False))

    def move_to(self, phase: str, arm_name: str, arm: Any, target: Sequence[float]) -> None:
        values = [float(value) for value in target]
        self.arm_command(
            phase, arm_name, "move_to", values,
            lambda: arm.move_to(*values[:3], roll=values[3], pitch=values[4], yaw=values[5], blocking=False),
        )

    def hand_command(
        self,
        phase: str,
        device: str,
        command: str,
        target: Any,
        fn: Callable[[], Any],
        *,
        clench6_after: Sequence[float] | None = None,
        grasp_mode_after: int | None = None,
        dwell_s: float | None = None,
    ) -> None:
        payload = dict(target) if isinstance(target, dict) else {"value": jsonable(target)}
        payload["clench6_after"] = list(clench6_after) if clench6_after is not None else None
        if grasp_mode_after is not None:
            payload["grasp_mode_after"] = int(grasp_mode_after)
        self.events.write(phase, device, command, payload, None, event="command_start")
        try:
            result = fn()
        except BaseException as exc:
            self.events.write(phase, device, command, payload, repr(exc), event="command_return")
            raise
        self.events.write(phase, device, command, payload, result, event="command_return")
        failed, reason = result_failed(result)
        if failed:
            raise RuntimeError(f"{phase}: SDK return failed: {reason}")
        if clench6_after is not None:
            self.hand6[device] = [float(value) for value in clench6_after]
        time.sleep(self.args.hand_dwell_s if dwell_s is None else dwell_s)

    def run(self) -> None:
        right_arm, right_hand = self.right.right_arm, self.right.right_hand
        left_arm, left_hand = self.left.left_arm, self.left.left_hand

        for index, target in enumerate(RIGHT_OBSERVATION_JOINTS, start=1):
            self.move_joints(f"RIGHT_READY_{index}", "right_arm", right_arm, target)
        for index, target in enumerate(LEFT_PRE_JOINTS, start=1):
            self.move_joints(f"LEFT_READY_{index}", "left_arm", left_arm, target)

        nut_xyz = pose_to_list(KNOWN_FIXED_NUT_WORLD_POSE["C"])[:3]
        right_grasp = compute_right_grasp_pose(nut_xyz)
        right_approach = compute_right_approach_pose(right_grasp)
        self.move_to("RIGHT_APPROACH", "right_arm", right_arm, pose_to_list(right_approach))
        right_thumb = list(self.hand6["RIGHT_HAND"])
        right_thumb[1] = 1.0
        self.hand_command(
            "RIGHT_THUMB_TUCK", "RIGHT_HAND", "clench", {"thumb_rotation": 1.0},
            lambda: right_hand.clench(thumb_rotation=1.0), clench6_after=right_thumb,
        )
        self.move_to("RIGHT_GRASP", "right_arm", right_arm, pose_to_list(right_grasp))
        self.hand_command(
            "RIGHT_GRASP_FORCE", "RIGHT_HAND", "grasp_force", dict(RIGHT_GRASP_FORCE),
            lambda: right_hand.grasp_force(**RIGHT_GRASP_FORCE), grasp_mode_after=1,
            dwell_s=max(self.args.hand_dwell_s, float(DEFAULT_HOLD_AFTER_GRASP_S)),
        )
        self.move_to("RIGHT_LIFT", "right_arm", right_arm, pose_to_list(build_right_safe_lift_pose(right_grasp)))
        self.move_to("RIGHT_TRANSFER", "right_arm", right_arm, pose_to_list(RIGHT_RELEASE_POSE))
        release_started = time.monotonic()
        self.hand_command(
            "RIGHT_RELEASE", "RIGHT_HAND", "clench", list(HAND_OPEN),
            lambda: right_hand.clench(*list(HAND_OPEN)), clench6_after=HAND_OPEN, grasp_mode_after=0,
            dwell_s=max(self.args.hand_dwell_s, float(RIGHT_RELEASE_OPEN_WAIT_S)),
        )
        safe_index = len(RIGHT_RELEASE_SAFE_JOINT_PATH)
        for index, target in enumerate(RIGHT_RELEASE_SAFE_JOINT_PATH, start=1):
            phase = "RIGHT_RETREAT_SAFE" if index == safe_index else f"RIGHT_RETREAT_WP{index}"
            self.move_joints(phase, "right_arm", right_arm, target)
        for index, target in enumerate(RIGHT_OBSERVATION_JOINTS, start=1):
            self.move_joints(f"RIGHT_RETURN_READY_{index}", "right_arm", right_arm, target)

        remaining_settle = max(0.0, self.args.settle_after_release_s - (time.monotonic() - release_started))
        if remaining_settle > 0:
            time.sleep(remaining_settle)
        release_target = tuple(float(value) for value in TARGET_NUT_B_WORLD_POSE[:3])
        from tools.test_right_release_stability import detect_released_nut
        released = detect_released_nut(release_target, self.args.vision_target_radius_m)
        if not released.get("detected") or not released.get("nut_world_xyz"):
            raise RuntimeError(f"LEFT_VISION: release point-cloud detection failed: {released.get('error')}")
        released_xyz = [float(value) for value in released["nut_world_xyz"][:3]]
        self.events.write(
            "RELEASED_NUT_DETECTION",
            "TOP_POINTCLOUD",
            "detect_released_nut",
            {"roi_center_world_xyz": list(release_target), "radius_m": self.args.vision_target_radius_m},
            {"detected": True},
            event="perception_result",
            nut="C",
            detected_world_xyz=released_xyz,
        )

        # Only the measured point-cloud result is passed to the left grasp planner.
        planner = LeftNutGraspPlanner(left_arm=left_arm, left_hand=left_hand)
        plan = planner.build_plan(released_xyz)
        self.hand_command(
            "LEFT_OPEN", "LEFT_HAND", "clench", list(HAND_OPEN),
            lambda: left_hand.clench(*list(HAND_OPEN)), clench6_after=HAND_OPEN, grasp_mode_after=0,
        )
        for item in plan["path"]:
            if item["stage"] in {"SAFE_HIGH", "PREGRASP"}:
                self.move_to(f"LEFT_{item['stage']}", "left_arm", left_arm, item["pose"])
        left_thumb = list(self.hand6["LEFT_HAND"])
        left_thumb[1] = 1.0
        self.hand_command(
            "LEFT_THUMB_TUCK", "LEFT_HAND", "clench", {"thumb_rotation": 1.0},
            lambda: left_hand.clench(thumb_rotation=1.0), clench6_after=left_thumb,
        )
        for item in plan["path"]:
            if item["stage"].startswith("DESCENT_"):
                self.move_to(f"LEFT_{item['stage']}", "left_arm", left_arm, item["pose"])
        grasp_item = next(item for item in plan["path"] if item["stage"] == "GRASP")
        self.move_to("LEFT_GRASP", "left_arm", left_arm, grasp_item["pose"])
        self.hand_command(
            "LEFT_GRASP_FORCE", "LEFT_HAND", "grasp_force", dict(planner.config.grasp_force),
            lambda: left_hand.grasp_force(**planner.config.grasp_force), grasp_mode_after=1,
        )
        time.sleep(0.5)
        for item in plan["lift_waypoints"]:
            self.move_to(f"LEFT_{item['stage']}", "left_arm", left_arm, item["pose"])
        extra_lift = safe_lift_pose_from_grasp(plan["grasp_pose"])
        self.move_to("LEFT_SAFE_LIFT_FINAL", "left_arm", left_arm, extra_lift)

        fixed = LEFT_FIXED_PLACE_JOINT_PATHS["C"]
        transport = [list(item.joints) for item in fixed["transport"]]
        self.move_joints("LEFT_FIXED_ENTRY_ALIGN_C", "left_arm", left_arm, transport[0])
        for index, target in enumerate(transport[1:], start=2):
            self.move_joints(f"LEFT_FIXED_TRANSPORT_C_{index}", "left_arm", left_arm, target)
        place = [list(item.joints) for item in fixed["place"]]
        for index, target in enumerate(place, start=1):
            self.move_joints(f"LEFT_FIXED_PLACE_C_{index}", "left_arm", left_arm, target)
        self.hand_command(
            "LEFT_RELEASE", "LEFT_HAND", "clench", list(HAND_OPEN),
            lambda: left_hand.clench(*list(HAND_OPEN)), clench6_after=HAND_OPEN, grasp_mode_after=0,
        )
        self.events.write("DONE", "EXPERT", "done", {"sequence": "C"}, "PASS", event="expert_status")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the strict serial Nut C Expert and write action_events.jsonl.")
    parser.add_argument("--action-events", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--observe-hz", type=float, default=5.0)
    parser.add_argument("--start-threshold-rad", type=float, default=0.003)
    parser.add_argument("--settle-threshold-rad", type=float, default=0.002)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument("--post-settle-s", type=float, default=0.2)
    parser.add_argument("--motion-timeout-s", type=float, default=20.0)
    parser.add_argument("--no-motion-grace-s", type=float, default=1.0)
    parser.add_argument("--hand-dwell-s", type=float, default=0.7)
    parser.add_argument("--passive-start-timeout-s", type=float, default=5.0)
    parser.add_argument("--settle-after-release-s", type=float, default=DEFAULT_SETTLE_AFTER_RELEASE_S)
    parser.add_argument("--vision-target-radius-m", type=float, default=DEFAULT_VISION_TARGET_RADIUS_M)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = EventWriter(args.action_events.resolve())
    passive = right = left = None
    try:
        passive = PassiveArmState(args.observe_hz)
        passive.wait_ready(args.passive_start_timeout_s)
        from tools.test_right_release_stability import make_right_bundle
        from tools.test_dual_closed_loop_v2 import make_left_bundle
        right = make_right_bundle()
        left = make_left_bundle()
        SerialExpert(events, passive, right, left, args).run()
        return 0
    except BaseException as exc:
        events.write("FAULT", "EXPERT", "fault", None, repr(exc), event="expert_status")
        print(f"[FAULT] {exc!r}", file=sys.stderr, flush=True)
        return 1
    finally:
        if right is not None:
            from tools.test_right_release_stability import shutdown_bundle
            with contextlib.suppress(Exception):
                shutdown_bundle(right)
        if left is not None:
            from tools.test_dual_closed_loop_v2 import shutdown_left_bundle
            with contextlib.suppress(Exception):
                shutdown_left_bundle(left)
        if passive is not None:
            with contextlib.suppress(Exception):
                passive.close()
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
