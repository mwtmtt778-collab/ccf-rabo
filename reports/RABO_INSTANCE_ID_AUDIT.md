# Rabo Instance ID Audit

## 审计范围与结论

- [VERIFIED] 扫描根目录：`/home/liyi/agent_system`。
- [VERIFIED] 扫描了仓库内 Python、JSON、Markdown、文本、YAML、TOML 和 shell 文件；二进制模型、NPZ、PT、ZIP 与 `.git/` 不作为源码绑定事实来源。
- [VERIFIED] 找到一组一致的旧实例标识：4 个完整 SDK device ID、1 个 world ID、1 个 ROS runtime namespace、5 个 ROS generated entity prefix、相机/关节/手部 sensor suffix，以及 4 个 scene object ID。
- [VERIFIED] 没有找到独立定义且可确认语义的 `scene_id`。
- [VERIFIED] 没有找到硬件 `serial_number` 值。源码中的 `serial` 主要表示“串行执行”，相机脚本中“camera serial suffix”是作者命名，不能证明它跨实例稳定。
- [VERIFIED] `agents/act_c_policy/` 没有旧 ID、旧 `/gs_...`、旧 camera topic、world/scene ID 的直接字面量；它通过 Expert bundle factory 间接消费旧 `DEVICE_IDS`。
- [VERIFIED] 旧完整实例值的直接字面量还出现在 `agents/arm_hand_demo/__init__.py`、`tools/test_nut_camera_calibration.py`、`tools/run_c_only_serial_expert.py`、`tools/map_rabo_runtime_interfaces.py`、`docs/legacy/arm_hand_demo_legacy_snapshot.py` 及若干历史报告中。历史报告只作为旧值佐证，不作为新实例配置来源。

## 必须迁移检查

### LEFT_ARM

- 旧完整 SDK device ID：`rbd03ebf4ebf83c6a6a64754454bc520a`
- 旧 ROS generated entity prefix：`rbd03eb`
- 定义：`agents/three_nut_expert/config.py:25` 的 `DEVICE_IDS["LEFT_ARM"]`
- 重复历史定义：`agents/arm_hand_demo/__init__.py:16`
- 其他硬编码副本：`tools/map_rabo_runtime_interfaces.py:39`、若干历史报告/legacy snapshot
- 用途：传入 `rabo_robocap.LinkerArmA7(robot_id=..., mode="sim")`
- 消费入口：`tools/test_dual_closed_loop_v2.py:66-72` 的 `make_left_bundle()`
- ACT 间接消费：`agents/act_c_policy/runtime.py:403-418`
- [VERIFIED] 旧 arm state topics 由 `tools/run_c_only_serial_expert.py:126-130` 使用 `/gs_.../rbd03eb_tp_ps_<suffix>` 构造。

### RIGHT_ARM

- 旧完整 SDK device ID：`r412d237980e3167577d7aece10f7aedb`
- 旧 ROS generated entity prefix：`r412d23`
- 定义：`agents/three_nut_expert/config.py:23` 的 `DEVICE_IDS["RIGHT_ARM"]`
- 重复历史定义：`agents/arm_hand_demo/__init__.py:14`
- 其他硬编码副本：`tools/map_rabo_runtime_interfaces.py:40`、若干历史报告/legacy snapshot
- 用途：传入 `rabo_robocap.LinkerArmA7(robot_id=..., mode="sim")`
- 消费入口：`tools/test_right_release_stability.py:113-119` 的 `make_right_bundle()`
- ACT 间接消费：`agents/act_c_policy/runtime.py:403-418`
- [VERIFIED] 旧 arm state topics 由 `tools/run_c_only_serial_expert.py:126-130` 使用 `/gs_.../r412d23_tp_ps_<suffix>` 构造。

### LEFT_HAND

- 旧完整 SDK device ID：`r136d7b4b6e527ea3875679b4bf7eeb7d`
- 旧 ROS generated entity prefix：`r136d7b`
- 定义：`agents/three_nut_expert/config.py:26` 的 `DEVICE_IDS["LEFT_HAND"]`
- 重复历史定义：`agents/arm_hand_demo/__init__.py:17`
- 其他硬编码副本：`tools/map_rabo_runtime_interfaces.py:41`、若干历史报告/legacy snapshot
- 用途：传入 `rabo_robocap.LinkerHandO6Left(robot_id=..., mode="sim")`
- 消费入口：`tools/test_dual_closed_loop_v2.py:66-72` 的 `make_left_bundle()`
- ACT 间接消费：`agents/act_c_policy/runtime.py:403-418`

### RIGHT_HAND

- 旧完整 SDK device ID：`rcd72e2daf71f064c29aa45d4eeceeca9`
- 旧 ROS generated entity prefix：`rcd72e2`
- 定义：`agents/three_nut_expert/config.py:24` 的 `DEVICE_IDS["RIGHT_HAND"]`
- 重复历史定义：`agents/arm_hand_demo/__init__.py:15`
- 其他硬编码副本：`tools/map_rabo_runtime_interfaces.py:42`、若干历史报告/legacy snapshot
- 用途：传入 `rabo_robocap.LinkerHandO6Right(robot_id=..., mode="sim")`
- 消费入口：`tools/test_right_release_stability.py:113-119` 的 `make_right_bundle()`
- ACT 间接消费：`agents/act_c_policy/runtime.py:403-418`

### TOP_CAMERA

- 旧 RGB topic：`/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_cam_303d2b1ce0`
- 旧 generated entity prefix：`r6ef2dc`
- 旧 RGB sensor suffix：`303d2b1ce0`
- 旧 depth entity：`r6ef2dc_tp_dcam_861ff01d8c`
- 旧 pointcloud topic：`/gs_1eebee6f37512bbc1d125b25511e912c/r6ef2dc_tp_dcam_861ff01d8c/points`
- 代码位置：`tools/run_c_only_serial_expert.py:106-107`、`tools/test_nut_camera_calibration.py:34`
- 其他硬编码 suffix consumers：`tools/probe_act_recording_sources.py:45-48`、`tools/test_camera_stream.py:23-26`、`tools/test_camera_throughput.py:67-70`、`tools/map_rabo_runtime_interfaces.py:45-49`
- ACT 正式路径：`config/act_c_policy.json:2` 为 `"auto"`，`agents/act_c_policy/runtime.py:1152-1155` 动态选择，不直接消费旧 topic。

### WRIST_CAMERAS

- 左腕 RGB：`/gs_1eebee6f37512bbc1d125b25511e912c/rbd03eb_tp_cam_069a6739f3`
- 左腕 depth：`rbd03eb_tp_dcam_3d5a0139cf`
- 右腕 RGB：`/gs_1eebee6f37512bbc1d125b25511e912c/r412d23_tp_cam_3c67aef2bc`
- 右腕 depth：`r412d23_tp_dcam_a7a6349c11`
- 主要代码位置：`tools/map_rabo_runtime_interfaces.py:45-49`、`tools/probe_act_recording_sources.py:45-48`、`tools/test_camera_stream.py:23-26`、`tools/test_camera_throughput.py:67-70`
- [VERIFIED] 这些 topic 同时绑定旧 `/gs_...` namespace、旧 arm generated prefix 与旧 camera suffix。

### RUNTIME_NAMESPACE

- 旧值：`/gs_1eebee6f37512bbc1d125b25511e912c`
- 代码定义：`tools/run_c_only_serial_expert.py:106`
- 由谁消费：该文件的 `TOP_RGB_TOPIC` 和 `STATE_TOPICS`；rosbag 收集/转换工具又导入这些常量。
- [DOCUMENTED] `docs/RABO_RUNTIME_INTERFACE_MAP_REPORT.md:17-30` 将它记录为一次 live runtime namespace。

### WORLD_ID

- 旧值：`ww4b1e6f8392584c89902a6783a9f53075`
- 主定义：`agents/three_nut_expert/config.py:14`
- 重复定义：`agents/arm_hand_demo/__init__.py:8`、`tools/test_nut_camera_calibration.py:31`、legacy snapshot
- 消费：`rabo_dev_kit.SetEntityPose(world=WORLD_ID)`，例如 `agents/three_nut_expert/expert.py:225`。
- [VERIFIED] ACT policy runtime 本身不使用 world ID 或 `SetEntityPose`。

### SCENE_ID

- 旧值：`UNRESOLVED`
- [VERIFIED] 全仓没有独立 `scene_id` 常量或已确认 scene metadata 值。
- [UNKNOWN] `/gs_1e...` 是否就是平台 scene ID；现有代码只把它当 ROS runtime namespace，不能升级为 scene ID。

### SCENE OBJECT IDS

- Nut A：`thing_3eb64241-4166-4c4e-9144-edf733c885f1`
- Nut B：`thing_e937f633-faed-4b2e-b609-a7b80825a64a`
- Nut C：`thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7`
- Storage box：`thing_8e768252-b4a8-47d7-82fc-320981b50c01`
- 定义：`agents/three_nut_expert/config.py:16-20`；storage box 位于 `tools/resolve_workspace_coordinates.py:58`，并在 `tools/test_nut_camera_calibration.py:41` 有副本。
- [VERIFIED] 这些 ID 被 reset/calibration/Expert scene manipulation 使用；ACT policy runtime 不使用它们。

### SCENE GEOMETRY / POSES

- Nut A/B/C fixed world poses：`agents/three_nut_expert/config.py:62-68`
- robot roots、base frames、storage box pose：`tools/resolve_workspace_coordinates.py:42-90`
- left grasp planner base world：`expert/left_nut_grasp_planner.py:13`
- [PENDING] 新 Rabo Case 是否复制了完全相同的 scene geometry。即使 ID 能找到，这些坐标仍需迁移核对。

## 旧 ROS generated topic inventory

### Arm JointState sensor suffixes

定义于 `tools/run_c_only_serial_expert.py:108-115`：

| SDK joint | passive slot | old suffix |
|---|---:|---|
| J1 | 0 | `08fa69bc43` |
| J2 | 5 | `y1vmospyub` |
| J3 | 4 | `rsqed0qcrb` |
| J4 | 2 | `2m4fzrdssg` |
| J5 | 1 | `1pa9vnh1mr` |
| J6 | 6 | `y8sm0inqbu` |
| J7 | 3 | `4jjp9rlwus` |

- 旧 passive order：`[J1,J5,J4,J7,J3,J2,J6]`
- 旧 conversion：`PASSIVE_TO_SDK_INDICES = (0,5,4,2,1,6,3)`
- [PENDING] 新实例的 generated suffix 与 passive topic order 是否相同；不能只凭旧 mapping 自动套用。

### Hand JointState sensor suffixes

- 左手：`1ff4n99g6p`, `26hns2y9iz`, `3id78faqov`, `c344tguind`, `f1re3fuf56`, `gww3yfamdt`, `k3y7rwcnkq`, `kfl2g3asap`, `l2kojhzci0`, `o6aptfg62g`, `zp3f47zr30`
- 右手：`3gpp28c61u`, `4d9q0h9sn8`, `8n8u1jtrj3`, `cxjshiucra`, `ilteetn3js`, `msti8uutyd`, `q2al5ws592`, `up6h8dvvum`, `v0b7dxyble`, `wtszomzf2i`, `zrbc9dcp12`
- 定义：`tools/run_c_only_serial_expert.py:116-130`
- [PENDING] suffix 是否跟设备模型还是实例生成；当前证据不足。

## DEVICE_IDS 完整来源与调用链

### 定义

`agents/three_nut_expert/config.py:22-27`：

```text
RIGHT_ARM  = r412d237980e3167577d7aece10f7aedb
RIGHT_HAND = rcd72e2daf71f064c29aa45d4eeceeca9
LEFT_ARM   = rbd03ebf4ebf83c6a6a64754454bc520a
LEFT_HAND  = r136d7b4b6e527ea3875679b4bf7eeb7d
```

### C Expert runtime chain

```text
agents.three_nut_expert.config.DEVICE_IDS
  ↓ imported by tools.test_right_release_stability
tools.test_right_release_stability.make_right_bundle()       [lines 113-119]
  ├─ DEVICE_IDS["RIGHT_ARM"]
  │    → rabo_robocap.LinkerArmA7(robot_id=..., mode="sim")
  └─ DEVICE_IDS["RIGHT_HAND"]
       → rabo_robocap.LinkerHandO6Right(robot_id=..., mode="sim")

agents.three_nut_expert.config.DEVICE_IDS
  ↓ imported by tools.test_dual_closed_loop_v2
tools.test_dual_closed_loop_v2.make_left_bundle()             [lines 66-72]
  ├─ DEVICE_IDS["LEFT_ARM"]
  │    → rabo_robocap.LinkerArmA7(robot_id=..., mode="sim")
  └─ DEVICE_IDS["LEFT_HAND"]
       → rabo_robocap.LinkerHandO6Left(robot_id=..., mode="sim")
```

`tools/run_c_only_serial_expert.py:669-683` 在 passive state ready 后依次调用 `make_right_bundle()`、`make_left_bundle()`，再把 bundle 交给 `SerialExpert.run()`。`lines 689-700` 使用同模块的 shutdown helpers 清理。

### Hybrid replay chain

`tools/test_hybrid_action_replay.py:42-55` 导入相同 factories；`lines 529-531` 依次创建 right/left bundle。因此 replay 与成功 C Expert 没有第二套 device ID 来源。

### ACT Agent chain

```text
agents.act_c_policy.runtime.SdkDeviceBundle.__init__()         [lines 399-418]
  → tools.test_right_release_stability.make_right_bundle()
  → tools.test_dual_closed_loop_v2.make_left_bundle()
  → 上述 DEVICE_IDS
  → LinkerArmA7 / LinkerHandO6Right / LinkerHandO6Left constructors
```

[VERIFIED] ACT Agent 没有复制四个 ID，但仍受旧 `DEVICE_IDS` 的传递性绑定影响。

## ACT Agent 旧实例残留检查

### 直接残留

- [VERIFIED] `agents/act_c_policy/__init__.py`：无旧 arm/hand ID、`gs_`、camera topic、scene/world ID、robot alias。
- [VERIFIED] `agents/act_c_policy/runtime.py`：无旧 arm/hand ID、`gs_`、camera topic、scene/world ID、`linker_left/linker_right`。
- [VERIFIED] `config/act_c_policy.json`：camera 为 `auto`，没有 arm/hand name 或 device ID。

### 间接残留

- [VERIFIED] `agents/act_c_policy/runtime.py:403-418` 通过 `make_right_bundle()` / `make_left_bundle()` 间接使用 `agents/three_nut_expert/config.py:22-27` 的旧四个 `DEVICE_IDS`。
- [VERIFIED] camera 路径使用 ROS graph auto discovery，不间接导入旧 `TOP_RGB_TOPIC`。

## 已确认不需要迁移

- [VERIFIED] state26 contract：left arm7 + right arm7 + left hand6 + right hand6。
- [VERIFIED] Hybrid28 contract：state26 target + left/right grasp mode。
- [VERIFIED] 模型 repo-relative 路径：`models/act_c_fixed_point_v1.onnx`。
- [VERIFIED] 模型输入输出维度和 chunk contract。
- [VERIFIED] SDK device class：`LinkerArmA7`、`LinkerHandO6Left`、`LinkerHandO6Right`。
- [VERIFIED] SDK arm order语义 `J1..J7`、A7/O6 设备类型、joint hard-limit policy、delta/margin safety policy。
- [VERIFIED] `SERIAL_RECEDING_HORIZON_MVP` 算法和 task mode order 不属于实例 ID。

## 尚不确定

- [UNKNOWN] 新 Rabo 是否提供可枚举所有 devices、display name、full device ID、side、serial 的官方只读 metadata API；本地未安装 SDK，仓库文档没有确认该 API。
- [DOCUMENTED] 本地保存的官方 API 文档确认 `LinkerArmA7` 构造参数包含 `robot_id`、`mode`、可选 `scene_id`；它没有记录一个可安全调用的全设备枚举接口。因此探针不会为了“发现设备”实例化 `LinkerArmA7` 或手部 client。
- [UNKNOWN] full SDK device ID 在“复制同一个 Case”与“创建新实例”时是否保持。
- [UNKNOWN] `/gs_...` 与平台 scene ID 的确切关系。
- [UNKNOWN] camera、joint、force/position sensor suffix 是模型固定还是实例生成。
- [UNKNOWN] 单凭 ROS generated entity prefix 无法可靠判定 left/right；探针只在官方 metadata 或明确 ROS graph name 提供 side 时解析角色。
- [UNKNOWN] JointState 消息内 generated joint name 是否包含 J1..J7 语义；若不包含，新的 J1..J7 topic mapping 仍需人工/官方 metadata 确认。
- [PENDING] world ID、thing IDs、robot roots、base frames、Nut/box world poses 是否在新 Case 保持。

## 新实例探针边界

`tools/probe_rabo_instance.py`：

- 读取 ROS topic/service/node graph。
- 只订阅 `sensor_msgs/msg/JointState`，记录 joint names 与 topic family。
- 列出所有 image topics，不丢弃多候选。
- 列出所有识别出的 arm/hand interface family、未分类 generated entity、hand-related topic/service，以及完整 ROS service/node 清单。
- 从 topic family 识别 A7/O6 **候选**，不把接口形状升级成官方设备类型事实。
- 只从安全环境变量或 `/world/<id>/...` service name 提取 scene/world 信息。
- 仅检查 `rabo_robocap` / `rabo_dev_kit` package 是否存在；不构造 SDK control client。
- 角色或 full device ID 没有明确 metadata 证据时写 `UNRESOLVED`。
- 保存 `reports/rabo_instance_probe.json`；不修改任何正式配置。
