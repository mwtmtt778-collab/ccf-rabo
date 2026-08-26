# Official vs Current Expert Startup

源码对照范围：进入第一次机器人动作之前。官方参考为 `agents/arm_hand_demo/__init__.py`；当前 Expert 为 `tools/test_three_nut_closed_loop_v2.py` 及其 startup helpers。

| 项目 | OFFICIAL | CURRENT | DIFFERENCE | POSSIBLE_RUNTIME_SIDE_EFFECT |
|---|---|---|---|---|
| import | 模块顶层导入 SDK、`threading`、`random` | 模块顶层导入配置、Expert、planner、monitor、录制/设备 helper；SDK 多在 runtime import | current top-level 依赖更多 | 只能作为待观测差异，未证明资源问题 |
| world / SetEntityPose | 创建 `SetEntityPose(world=WORLD_ID)` 并调用 `set` | 默认不创建；`--reset-to-nominal` 才创建并 reset scene | current 默认不做 world reset | UNKNOWN |
| arm 创建顺序 | RIGHT_ARM → LEFT_ARM | RIGHT_ARM → LEFT_ARM | 顺序一致 | UNKNOWN |
| hand 创建顺序 | RIGHT_HAND → LEFT_HAND | RIGHT_HAND → LEFT_HAND | 顺序一致 | UNKNOWN |
| mode | 所有设备 `mode="sim"` | 所有设备 `mode="sim"` | 无 | UNKNOWN |
| bundle / wrapper | 无 bundle，局部变量直接控制 | `SimpleNamespace` bundles + `ExpertStateRunner` | current 多 wrapper | 额外 Python 对象本身未证明影响 |
| ROS/rclpy context | 源码未创建 | 源码未显式创建 | UNKNOWN | 不猜测隐藏 SDK 行为 |
| executor | 未创建 | 未显式创建 | UNKNOWN | 不猜测 |
| thread | 双臂预摆位后才创建两个 `threading.Thread` | startup 前不创建 Expert motion thread；collector 另有 sampler/watcher（standalone 无） | current standalone 更晚创建 | UNKNOWN |
| monitor | 无 | 创建 `MotionMonitor`；可由 `--no-motion-gate` 禁用 gate | current-only | monitor 初始化/配置可能增加工作，但无源码证据证明 camera 影响 |
| coordinator | 无 | `--coordinated-execution` 时创建 `ExecutionCoordinator` | current-only | 仅对象初始化阶段，需实测 |
| state sampler | 无 | standalone 无；collector 才创建 `StateSampler` | collector-only | 不适用于 standalone startup |
| recording probe | 无 | 仅显式 `--recording-probe` 且 READY 后启动 | current optional | 默认不在首次动作前启动 |
| report/logger | 无专门 report 初始化 | 创建本地 report path、运行报告字典、startup trace | current-only | 本地文件 I/O |
| initial getters | 官方 READY 前无 getter | standalone READY 前无 getter；`ExpertStateRunner` 创建不读设备 | 无已证差异 | UNKNOWN |
| shutdown/finally | `shutdown()` 在 run 末尾 | `finally` 中 shutdown bundles/pose setter/probe | current 有更复杂 cleanup | stop probe 使用无 cleanup 退出 |

## 当前源码顺序

`PROCESS_MAIN_ENTER → ARGS_PARSED → IMPORT_RUNTIME_READY → CONFIG_LOADED → REPORT_INIT → SDK_IMPORT_READY → WORLD_CONTEXT_INIT → RIGHT_ARM → RIGHT_HAND → LEFT_ARM → LEFT_HAND → DEVICE_BUNDLE_READY → COORDINATOR_INIT → MOTION_MONITOR_INIT → EXPERT_OBJECT_READY → RUN_ENTER → EPISODE_INIT`。

## 证据边界

官方代码本身在初始化后立即进入双臂预摆位；本 probe 刻意不执行动作。当前 startup trace 对每个 runtime 初始化边界写入 absolute `monotonic_ns` 和耗时。任何 SDK 内部创建的 ROS context/executor 若源码不可见，均标记 UNKNOWN。
