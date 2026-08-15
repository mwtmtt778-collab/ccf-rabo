# Rabo 操作手册

## 基本启动

- [DOCUMENTED] Rabo 平台控制器默认运行：

```bash
python3 -u main.py
```

- [VERIFIED] 当前 `main.py` 无参数启动会先运行机器人状态只读测试，而不是直接执行 `arm_hand_demo.run()`。

## 机器人状态只读检查

- [DOCUMENTED] 当前允许用以下命令确认 Rabo SDK / 机器人连接环境：

```bash
python3 -u main.py
```

- [DOCUMENTED] 该测试只应调用状态读取能力：`get_joint_angles()`、`get_pose()`、`get_clench()`。

## 相机只读检查

- [DOCUMENTED] 相机测试命令：

```bash
python3 tools/test_camera_stream.py --duration 10 --output-dir outputs/camera_test --verbose
```

- [DOCUMENTED] 该测试不得调用机械臂运动、手部动作、物体 pose、reset 或相机参数修改 API。

## Runtime 接口只读映射

- [DOCUMENTED] Runtime 探针：

```bash
python3 tools/inspect_rabo_runtime.py --duration 8 --output-dir outputs/runtime_probe --verbose
```

- [DOCUMENTED] Runtime interface map：

```bash
python3 tools/map_rabo_runtime_interfaces.py --duration 10 --verbose
```

## 工作空间 V2

- [DOCUMENTED] 坐标解析：

```bash
python3 tools/resolve_workspace_coordinates.py
```

- [DOCUMENTED] 可达性测试：

```bash
python3 tools/test_dual_arm_reachability.py
```

- [DOCUMENTED] 当前坐标门禁未通过前，不运行 hover motion、抓取、handoff、Recorder 或 ACT。

## 三螺母 Expert

- [DOCUMENTED] Dry-run 合约：

```bash
python3 tools/run_three_nut_expert.py --mode contract
```

- [DOCUMENTED] 单螺母 B 执行 gate：

```bash
python3 tools/run_three_nut_expert.py --mode single --nut B --trials 1 --no-jitter --execute
```

- [PENDING] 执行命令只能在用户确认安全边界和 Rabo 场景状态后运行。

## 现场待确认

- [PENDING] 当前 Rabo 终端路径。
- [PENDING] Rabo 端 Git remote 名称和指向。
- [PENDING] 当前 Rabo runtime namespace。
- [PENDING] 当前 `rabo-dev-kit`、`rabo_robocap` 是否在正式控制器和普通 shell 中都可 import。

