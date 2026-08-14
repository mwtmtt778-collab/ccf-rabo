# Rabo Three-Nut Expert Implementation

> 注意：
> 本文件为实现设计/阶段说明，不再由运行脚本自动覆盖。
>
> 最新运行报告：
> `outputs/three_nut_expert/latest_report.md`
>
> 原始日志：
> `logs/three_nut_expert.log`
>
> 最新结构化结果：
> `outputs/three_nut_expert/three_nut_expert_result.json`

## 1. Result

Overall: PASS

## 2. Safety Mode

- Execute: `False`
- Default runner mode is dry-run. Robot motion requires explicit `--execute`.
- `success` in this report means SDK/API completion only; physical task success must be verified visually until perception checks are added.
- Raw log: `/home/liyi/agent_system/logs/three_nut_expert.log`

## 3. Source Of Truth

- agents/arm_hand_demo/__init__.py and docs/legacy/arm_hand_demo_legacy_snapshot.py
- The legacy `agents/arm_hand_demo/__init__.py` file was not rewritten.
- Snapshot: `docs/legacy/arm_hand_demo_legacy_snapshot.py`

## 4. Device IDs

- RIGHT_ARM: `r412d237980e3167577d7aece10f7aedb`
- RIGHT_HAND: `rcd72e2daf71f064c29aa45d4eeceeca9`
- LEFT_ARM: `rbd03ebf4ebf83c6a6a64754454bc520a`
- LEFT_HAND: `r136d7b4b6e527ea3875679b4bf7eeb7d`

## 5. Single-Nut Contract

- Proven nut: `B`
- Right arm base XY: `[-0.6816, -0.004]`
- Grasp transform: `target_x = base_x - nut_x + 0.06; target_y = base_y - nut_y - 0.01; z=-0.33; rpy=(0,0.8,0)`
- Right grasp force: `{'strength': 1.0, 'fingers': [1, 3, 4]}`
- Left grasp force: `{'strength': 0.5, 'fingers': None}`

## 6. Gates

- dry-run plan renders without importing Rabo SDK
- cloud single B execute succeeds 5-10 times
- only then run A/C single validation
- only then run three-nut sequence
- camera gate must pass before Recorder

## 7. Current Run Results

| Mode | Nuts | Execute | API Success | Task Success | Steps Planned | Steps Executed | Error |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| single | B | False | True | UNVERIFIED_BY_SCRIPT | 19 | 0 |  |
| single | B | False | True | UNVERIFIED_BY_SCRIPT | 19 | 0 |  |

## 8. Unknowns

- A/C staged spawn/place poses are not proven
- RGB camera frames are unstable in latest runtime map
- no read-only object pose API has been confirmed

## 9. Cloud Commands

Dry-run contract:

```bash
python3 tools/run_three_nut_expert.py --mode contract
```

Single B execute gate:

```bash
python3 tools/run_three_nut_expert.py --mode single --nut B --trials 1 --no-jitter --execute
```

Diagnostic single B with extra settling:

```bash
python3 tools/run_three_nut_expert.py --mode single --nut B --trials 5 --no-jitter --settle-after-pose-s 1.0 --hold-after-grasp-s 1.0 --step-delay-s 0.2 --execute
```

Three-nut execute only after single gates pass:

```bash
python3 tools/run_three_nut_expert.py --mode three --order A,B,C --execute
```

## 10. Final Decision

- READY_FOR_SINGLE_NUT_CLOUD_TEST: YES
- READY_FOR_THREE_NUT_BATCH: NO until single B and A/C gates pass.
- READY_FOR_RECORDER: NO until camera gate is fixed.
