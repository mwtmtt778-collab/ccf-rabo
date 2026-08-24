# ACT 采样、训练与回放 MVP 实施计划

## 1. 目标与范围

- [VERIFIED] 本计划对应项目根目录 `/home/liyi/agent_system`。
- [VERIFIED] 本地 ACT 环境实际位于 `/home/liyi/ccf_act_baseline`。
- [DOCUMENTED] LeRobot 源码版本为 0.6.2，使用 LeRobotDataset v3。
- [PENDING] 先用 5 Hz、2–3 个 Episode 打通“Rabo 采样 → 本地转换 → ACT 训练 → 离线评估 → Rabo 回放”。
- [PENDING] 第一轮不要求 Expert 完成任务，也不声明模型具有任务能力；产物统一标记为 `PIPELINE_SMOKE_TEST`。

本轮不处理严格三路 10 Hz、正式 commanded action、Expert 成功率、深度图、点云和正式在线 ACT 性能优化。

## 2. LeRobot 标准与本项目分层

本机 `/home/liyi/ccf_act_baseline/lerobot` 中的 LeRobotDataset v3 采用：

- 低维状态、动作和时间戳：Apache Parquet；
- 多相机视觉：MP4；
- schema、FPS、Episode、任务和文件索引：`meta/`；
- 创建数据集时逐帧 `add_frame()`，逐 Episode `save_episode()`；
- 全部写完后必须调用 `finalize()`，否则 Parquet footer 和 metadata 不完整。

Rabo 端只生成便于可靠打包下载的“原始 Episode 包”，不直接生成 LeRobotDataset。原因是 Rabo 控制器应优先保证 ROS/SDK 采集稳定，不应同时承担 PyTorch、LeRobot、Parquet 和最终 MP4 数据集构建。

```text
Rabo /home/.../agent_system/outputs/act_samples/
  原始 Episode（状态 NPZ + PPM/PGM 或原始压缩图 + 时间戳 + metadata）
                    ↓ 打包下载
本机 /home/liyi/ccf_act_baseline/data/rabo_raw/
  原始包解压与完整性检查
                    ↓ 本地转换
本机 /home/liyi/ccf_act_baseline/data/lerobot/rabo_act_mvp_5hz/
  标准 LeRobotDataset v3（Parquet + MP4 + meta）
                    ↓
ACT 训练、checkpoint、离线评估、部署包
```

## 3. 固定的 26 维数据契约

统一顺序：

| 索引 | 名称 | 读取接口 | 执行接口 | 单位/范围 |
| --- | --- | --- | --- | --- |
| `0:7` | 左臂 J1–J7 | `get_joint_angles()` | `move_joints(..., blocking=False)` | rad |
| `7:14` | 右臂 J1–J7 | `get_joint_angles()` | `move_joints(..., blocking=False)` | rad |
| `14:20` | 左手 clench 六参数 | `get_clench()` | `clench(..., blocking=False)` | `[0,1]` |
| `20:26` | 右手 clench 六参数 | `get_clench()` | `clench(..., blocking=False)` | `[0,1]` |

手部六参数顺序固定为：拇指旋转、拇指弯曲、食指、中指、无名指、小指。

第一版临时动作定义：

```text
observation.state[t] = state[t]
action[t] = state[t + 1]
```

这是 next-state proxy，只用于打通流程。正式行为克隆需要后续记录真实下发命令。

## 4. 阶段 A：Rabo 原始 Episode 采样

### A1. 输出位置

Rabo 项目内固定保存到：

```text
outputs/act_samples/<episode_id>/
outputs/act_samples/<episode_id>.tar.gz
```

`outputs/` 已被 Git 忽略，采样数据不得提交到仓库。

### A2. 原始包结构

```text
<episode_id>/
├── metadata.json
├── telemetry.npz
├── quality_report.json
└── cameras/
    ├── cam_top/
    │   ├── frame_000000.ppm
    │   ├── ros_timestamps.npy
    │   └── arrival_times_s.npy
    ├── cam_left_wrist/
    └── cam_right_wrist/
```

`telemetry.npz` 至少包含：

- `qpos`: `N×26`；
- `actions`: `N×26`，第一版由下一状态派生；
- `timestamps`: `N`，以 Episode 起点为零点的 monotonic 秒；
- `read_durations`: `N`；
- `source_states`: 派生 action 前的完整状态序列，用于审计。

相机以各自原始异步频率保存每个独立新帧。5 Hz 是状态/动作时间轴，不伪造三路 5 Hz 同步相机。

### A3. 运行方式

第一版 Recorder 为只读独立进程，不调用任何运动接口。Recorder 启动后，在另一个 Rabo 终端运行现有 Expert。即使 Expert 失败，Recorder 也保留数据并标记任务状态。

Rabo 终端 A（先启动 Recorder）：

```bash
cd /home/liyi/agent_system
python3 -u tools/record_act_episode.py \
  --duration 60 \
  --fps 5 \
  --start-delay 10 \
  --episode-id episode_000000 \
  --task rabo_single_b_pipeline_smoke
```

Rabo 终端 B（在倒计时内准备，在 Recorder 显示 `RECORDING` 后运行）：

```bash
cd /home/liyi/agent_system
python3 -u tools/run_three_nut_expert.py \
  --mode single \
  --nut B \
  --trials 1 \
  --no-jitter \
  --execute
```

`--execute` 会产生真实仿真运动，必须由用户在 Rabo 场景确认后执行。若 60 秒不能覆盖完整动作，下一次只增加 `--duration`；不要通过伪造或重复帧补长度。

结束后 Recorder 会打印唯一需要下载的文件，例如：

```text
/home/liyi/agent_system/outputs/act_samples/episode_000000.tar.gz
```

若 Rabo 不允许同一设备存在两组 SDK 客户端，表现为 Recorder 或 Expert 初始化失败，则停止使用双终端模式，改为把 Recorder 接入 Expert 已创建的同一组 device bundle；不通过反复重试规避节点冲突。

验收条件：

- 三路相机均至少有一个有效帧；
- 至少有 3 个有效 26 维状态；
- qpos/action 无 NaN/Inf；
- 时间戳单调；
- 每路保存帧数、丢队列帧数、重复时间戳数可审计；
- `.tar.gz` 可生成并下载。

## 5. 阶段 B：下载与本地验证

从 Rabo 下载：

```text
outputs/act_samples/<episode_id>.tar.gz
```

放到本机：

```text
/home/liyi/ccf_act_baseline/data/rabo_raw/archives/
```

解压到：

```text
/home/liyi/ccf_act_baseline/data/rabo_raw/<episode_id>/
```

本地验证必须先检查 archive、metadata、NPZ shape、数值有限性、三路图片解码和时间戳单调，再允许转换。Rabo 原始 `rgb8/bgr8` 图像使用无需额外依赖的 PPM 保存；若 ROS 话题本身提供 JPEG/PNG 压缩图则保留原格式。

## 6. 阶段 C：转换为 LeRobotDataset v3

转换目标：

```text
/home/liyi/ccf_act_baseline/data/lerobot/rabo_act_mvp_5hz/
```

转换规则：

- dataset FPS 固定为 5；
- 相机键固定为 `cam_top`、`cam_left_wrist`、`cam_right_wrist`；
- 每个控制时刻只选 `camera_timestamp <= control_timestamp` 的最新图像，禁止使用未来帧；
- 记录或报告每个控制步的相机帧龄和复用比例；
- feature names 使用本计划中的 26 维真实名称；
- 每个原始 Episode 对应一次 `save_episode()`；
- 全部 Episode 完成后必须调用 `finalize()`；
- 转换后必须通过现有 `validate_lerobot_dataset.py` 和 DataLoader smoke test。

## 7. 阶段 D：ACT 最小训练与离线结果

第一轮配置：

```text
fps=5
三路 RGB=224×224
state_dim=26
action_dim=26
batch_size=1
chunk_size=20
n_action_steps=1
steps=200–1000
```

先用一个 Episode 过拟合，输出 checkpoint、loss、预测动作、整体及四设备分组 MAE/MSE。结果只标记为 pipeline smoke test。

## 8. 阶段 E：回放

### E1. 最快闭环：预计算轨迹回放

本地对 Episode 生成 `predicted_actions.npy`，Rabo 回放器只加载 `N×26` 动作并按 5 Hz 下发。这不是真正视觉闭环，但能最快验证训练产物、26 维拆分和 SDK 执行链路。

### E2. 正式闭环：Rabo 在线推理

在 Rabo 环境验证 PyTorch/LeRobot 可安装、checkpoint 可加载、三相机单次推理低于周期预算后，再升级为在线 ACT。若 Rabo CPU 达不到 5 Hz，再评估降低相机/分辨率、增加 `n_action_steps` 或本地 GPU 推理服务。

### E3. 运动门禁

回放器必须默认 dry-run；只有显式 `--execute` 才运动。执行前必须完成：当前值回写、单维小扰动、5 Hz 非阻塞目标更新/抢占测试、绝对限位、步长限制、NaN/Inf、初始状态偏差和停止/超时保护。

## 9. 实施顺序与状态

1. [VERIFIED] 已实现 `tools/record_act_episode.py`，本地语法、CLI、图像转换、next-state action、NPZ 和 archive 纯数据 smoke test 通过；尚未在 Rabo 运行。
2. [PENDING] 在 Rabo 只运行一次短采样并下载 archive。
3. [PENDING] 在 `ccf_act_baseline` 增加 Rabo raw 校验与 LeRobot v3 转换。
4. [PENDING] 一个 Episode 过拟合并生成第一组离线指标。
5. [PENDING] 生成预计算动作并完成 Rabo dry-run 回放。
6. [PENDING] 通过 SDK 小扰动门禁后，决定是否执行仿真轨迹回放。
7. [PENDING] 最后升级到在线 ACT。
