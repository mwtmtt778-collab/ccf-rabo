# Rabo Runtime Probe Method Report

生成时间：2026-08-13

## 目标

本阶段目标不是控制机器人，也不是采集数据或训练 ACT，而是先用只读方式探明 Rabo 比赛/仿真运行环境接口：

- Camera
- Robot State
- Robot Control
- Nut Objects
- Target Areas
- Simulation Reset
- Control / State / Camera Frequency

探针脚本：

```bash
tools/inspect_rabo_runtime.py
```

输出文件：

```bash
docs/RABO_RUNTIME_REPORT.md
outputs/runtime_probe/runtime_probe.json
```

## 严格只读原则

探针第一版没有执行以下操作：

- 没有发送机械臂控制命令
- 没有发送灵巧手控制命令
- 没有移动机器人
- 没有修改螺母或任何物体 pose
- 没有 reset 场景
- 没有调用 destructive API
- 没有运行 `main.py`
- 没有运行 `agents/arm_hand_demo`

脚本只做：

- 环境变量读取
- ROS2 discovery
- ROS2 topic/service/action 列表读取
- 候选 topic 只读订阅采样
- 源码静态搜索
- Python SDK import 检查
- 报告生成

如果某个接口只能通过写操作确认，脚本会记录为 UNKNOWN / NOT FOUND，不会实际调用。

## 使用的方法

### 1. 环境检查

脚本启动时记录：

- hostname
- pwd
- Python version
- ROS_DISTRO
- ROS_DOMAIN_ID
- RMW_IMPLEMENTATION
- PYTHONPATH

目的是判断当前 shell 是否处在真正 Rabo runtime 上下文中。

### 2. ROS2 CLI 只读检查

脚本尝试执行等价命令：

```bash
ros2 node list --no-daemon
ros2 topic list -t --no-daemon
ros2 service list -t --no-daemon
ros2 action list -t --no-daemon
```

这些命令只读取 ROS graph，不发布控制消息。

### 3. ROS2 Python API discovery

脚本使用 `rclpy` 创建一个只读探针节点：

```text
rabo_runtime_readonly_probe
```

然后调用：

- `get_node_names()`
- `get_topic_names_and_types()`
- `get_service_names_and_types()`
- `get_publishers_info_by_topic()`
- `get_subscriptions_info_by_topic()`

对 topic 记录：

- topic name
- message type
- publisher count
- subscriber count
- QoS 信息，如果可获取

### 4. Topic 关键词筛选

脚本不会假设固定 topic 名称，而是扫描全部 topic，再按关键词分类。

Camera 关键词：

```text
camera, image, rgb, depth
```

Robot state 关键词：

```text
joint, state, qpos, position, arm, hand, o6, a7, p7
```

Control 关键词：

```text
command, cmd, control, action, trajectory, servo, move
```

Object / target / reset 关键词：

```text
nut, object, pose, thing, target, goal, placement, area, bin, tray, reset, restart
```

### 5. Camera 只读采样

如果发现 `sensor_msgs/msg/Image` 或 `sensor_msgs/msg/CompressedImage` 类型的 camera topic，脚本会只读订阅若干秒，统计：

- width
- height
- encoding / format
- frames
- avg FPS
- min FPS
- max FPS

如果传入 `--save-images`，仅保存第一帧到：

```bash
outputs/runtime_probe/
```

保存图片是本地文件写入，不修改仿真环境。

### 6. Robot State 只读采样

如果发现 `sensor_msgs/msg/JointState` 或疑似 state topic，脚本会只读订阅，记录：

- joint names
- joint count
- position shape
- velocity shape
- effort shape
- state FPS

不假设 `7 + 7 + 6 + 6`。

### 7. Control 接口只发现，不调用

脚本只记录疑似 control topic/service/action：

- command
- cmd
- control
- action
- trajectory
- servo
- move

不会 publish command。

### 8. 源码静态搜索

脚本还会扫描当前仓库源码：

```bash
rg -n -i "camera|image|rgb|joint|qpos|state|command|control|arm|hand|o6|a7|p7|nut|object|pose|target|reset|expert|demo|pick|place"
```

目的是找已有：

- camera subscriber
- state reader
- command publisher
- reset 方法
- demo / example
- scripted task
- Rabo SDK 调用

### 9. SDK import 检查

脚本尝试 import：

```python
rabo_robocap
rabo_dev_kit
```

如果可 import，则记录模块路径和部分类/方法名。

如果不可 import，不报错退出，只记录 SDK unavailable。

## 本机运行结果

本机路径：

```bash
/home/liyi/agent_system
```

本机运行结论：

- ROS2 存在，`ROS_DISTRO=humble`
- 但本机 ROS graph 只看到探针自身、`/rosout`、`/parameter_events`
- 没有 Rabo camera topic
- 没有 Rabo robot state topic
- 没有 Rabo control topic/service/action
- `rabo_robocap` 不可 import
- `rabo_dev_kit` 不可 import

这不是异常。

原因是：Rabo 仿真和比赛接口实际在云端 runtime 内，本机只是新增开发端和 Git 同步端。本机没有运行 Rabo 仿真，也没有接入云端 DDS/SDK runtime，所以本机找不到 camera/state/control 是预期结果。

本机探针的价值是：

1. 验证脚本不会执行控制或 reset。
2. 验证脚本可运行、可生成 JSON/Markdown。
3. 验证源码静态信息可以被提取。
4. 通过 GitHub 同步到云端，在真正 Rabo runtime 中执行。

## 云端运行结果

云端路径：

```bash
/workspace/agent_system
```

云端已执行：

```bash
python3 tools/inspect_rabo_runtime.py \
  --duration 10 \
  --save-images \
  --output-dir outputs/runtime_probe \
  --verbose
```

云端输出摘要：

```text
Cameras:
  NONE
Robot state:
  NONE
Robot control:
  NONE
Nut objects:
  NUT_A_ID: thing_3eb64241-4166-4c4e-9144-edf733c885f1
  NUT_B_ID: thing_e937f633-faed-4b2e-b609-a7b80825a64a
  NUT_C_ID: thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7
Target regions:
  NONE
Reset:
  NONE
Camera FPS:
  []
State FPS:
  []
Can read nut pose: NO
Can control arm: NOT FOUND
Can control hand: NOT FOUND
```

这说明：云端当前命令所在 shell 能运行项目，但该 shell 里仍未暴露 Rabo camera/state/control 的 ROS graph 接口。

当前还不能据此断定 Rabo 没有接口，只能说明：

- 当前云端 shell 下，通过 ROS2 topic/service/action discovery 没找到接口。
- 当前脚本没有通过 Python SDK 找到只读 pose/state API。
- Rabo 仿真接口可能不是 ROS graph 形式暴露。
- Rabo SDK 可能只在平台正式运行 agent 时注入，普通 shell 下不可见或不可用。
- 也可能需要特定环境变量、启动方式、场景 runtime、容器权限或平台入口。

## 已从源码确认的静态信息

文件：

```bash
agents/arm_hand_demo/__init__.py
```

已知 world ID：

```text
WORLD_ID = ww4b1e6f8392584c89902a6783a9f53075
```

已知螺母 ID：

```text
NUT_A_ID = thing_3eb64241-4166-4c4e-9144-edf733c885f1
NUT_B_ID = thing_e937f633-faed-4b2e-b609-a7b80825a64a
NUT_C_ID = thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7
```

已知机器人 ID：

```text
RIGHT_ARM_ID  = r412d237980e3167577d7aece10f7aedb
RIGHT_HAND_ID = rcd72e2daf71f064c29aa45d4eeceeca9
LEFT_ARM_ID   = rbd03ebf4ebf83c6a6a64754454bc520a
LEFT_HAND_ID  = r136d7b4b6e527ea3875679b4bf7eeb7d
```

已知源码中存在控制 SDK 用法：

```python
from rabo_robocap import LinkerArmA7, LinkerHandO6Right, LinkerHandO6Left
from rabo_dev_kit import SetEntityPose
```

并存在以下会改变仿真状态的调用：

```python
SetEntityPose(...).set(...)
LinkerArmA7(...).move_joints(...)
LinkerArmA7(...).move_to(...)
LinkerHandO6Right(...).clench(...)
LinkerHandO6Right(...).grasp_force(...)
LinkerHandO6Left(...).clench(...)
LinkerHandO6Left(...).grasp_force(...)
```

探针没有执行这些调用，只静态记录它们的位置。

## 当前接口结论

### Camera

真实 camera topic/API：

```text
UNKNOWN
```

当前本机和云端普通 shell 的 ROS graph 均未发现 camera topic。

### Robot State

机械臂 state topic/API：

```text
UNKNOWN
```

O6 state topic/API：

```text
UNKNOWN
```

当前未发现 JointState 或其它 state topic。

### Robot Control

机械臂 control topic/API：

```text
源码静态可见 LinkerArmA7.move_joints / move_to
运行时只读 discovery 未找到 ROS control topic/service/action
```

O6 control topic/API：

```text
源码静态可见 LinkerHandO6Right / LinkerHandO6Left 的 clench / grasp_force
运行时只读 discovery 未找到 ROS control topic/service/action
```

### Nut Object

已知 nut object IDs：

```text
thing_3eb64241-4166-4c4e-9144-edf733c885f1
thing_e937f633-faed-4b2e-b609-a7b80825a64a
thing_4a38ffae-a85b-43c9-8b20-abb0cf6dcac7
```

能否读取 nut ground-truth pose：

```text
NO / UNKNOWN
```

当前只知道 `SetEntityPose` 可写 pose，但没有发现只读 get-pose API。
不能为了确认而调用写接口。

### Target Areas

真实 target object / placement region：

```text
UNKNOWN
```

当前源码和 ROS discovery 都没有明确目标区域对象。

### Reset

reset 方法：

```text
UNKNOWN
```

当前没有发现可确认的 reset service/action/API。
即使发现 reset，也不能在 probe 中调用。

### Frequency

camera FPS：

```text
UNKNOWN
```

state FPS：

```text
UNKNOWN
```

control Hz：

```text
UNKNOWN
```

因为没有发现 camera/state topic，无法只读采样估计频率。

## 下一步建议

下一步仍然不是写抓取、不是训练、不是控制机器人。

应该继续在云端确认 Rabo SDK/API 的只读能力。

建议云端执行：

```bash
python3 - <<'PY'
mods = ["rabo_robocap", "rabo_dev_kit"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__file__", None))
        names = [
            x for x in dir(mod)
            if "Pose" in x or "Entity" in x or "Arm" in x or "Hand" in x
            or "Camera" in x or "State" in x or "Joint" in x
        ]
        print("attrs", names[:120])
    except Exception as e:
        print(m, "FAIL", repr(e))
PY
```

如果 SDK 可 import，再进一步只读 inspect 类方法签名：

```bash
python3 - <<'PY'
import inspect
from rabo_robocap import LinkerArmA7, LinkerHandO6Right, LinkerHandO6Left

for cls in [LinkerArmA7, LinkerHandO6Right, LinkerHandO6Left]:
    print("\\nCLASS", cls)
    for name, fn in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if callable(fn):
            try:
                print(name, inspect.signature(fn))
            except Exception:
                print(name)
PY
```

注意：

只 inspect 类和方法签名，不实例化、不调用 move、set、reset。

## 当前状态一句话

只读探针已经写好并在本机、云端普通 shell 中运行；本机找不到 Rabo 接口是预期，因为仿真在云端；云端普通 shell 也未通过 ROS graph 暴露 camera/state/control，因此下一步应在云端继续确认 Rabo SDK 是否提供只读 state / camera / object pose API，或确认平台是否需要通过正式 agent runtime 注入这些接口。

