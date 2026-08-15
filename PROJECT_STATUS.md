# PROJECT_STATUS.md

## 当前阶段

- [VERIFIED] 项目管理体系 V1 已进入收口清理阶段。
- [DOCUMENTED] 当前工程围绕 Rabo 双臂 / 三螺母任务，先确认 runtime、相机、工作空间和策略，再进入更高风险的抓取、Recorder、ACT。

## 当前目标

- [VERIFIED] 完成项目管理体系 V1 收口，让仓库回到可继续正常开发的状态。
- [VERIFIED] 不重构业务代码目录，不修改现有业务代码。

## 已完成

- [VERIFIED] 本地 GitHub remote 已配置为 `origin git@github-ccf-pc2:mwtmtt778-collab/ccf-rabo.git`。
- [DOCUMENTED] PC-to-GitHub-to-Rabo 同步曾用临时文件验证过。
- [DOCUMENTED] Rabo runtime 只读探针、相机测试、接口映射、三螺母 Expert dry-run、工作空间 V2 坐标门禁均已有文档或脚本。
- [VERIFIED] `.gitignore` 当前已忽略 `logs/` 和 `outputs/`。
- [VERIFIED] `.ai/` 已定义为本地 AI 临时工作区并加入 `.gitignore`。

## 正在进行

- [VERIFIED] `tools/test_dual_arm_reachability.py` 有未提交修改。
- [VERIFIED] `expert/` 与 `tools/resolve_workspace_coordinates.py` 是未跟踪源码，应保护。
- [VERIFIED] 当前正在整理项目管理文档，不处理业务逻辑。

## 当前阻塞

- [DOCUMENTED] 工作空间 V2 尚未完成，停在坐标门禁阶段。
- [PENDING] Box A/B/C 真实 world pose、左右臂 base_link world pose、右臂完整 base_link pose 或 TF frame 仍需 Rabo 现场确认。
- [PENDING] Rabo 端真实 remote 名称和指向需要在 Rabo 终端确认。
- [PENDING] README 与当前 `main.py` 启动行为冲突，需要后续决定修 README 还是修入口行为。

## 下一步

- [VERIFIED] `agent_system_backup.tar.gz` 应从 Git 索引移除但保留本地文件，并用 `.gitignore` 防止重新进入 Git。
- [PENDING] 用户确认是否进一步归档 Rabo HTML 快照、WPS 文件和仍有技术证据价值的 RABO 阶段报告。
- [PENDING] 在 Rabo 端运行最小只读命令，确认 remote、Python、ROS、SDK 和 runtime namespace。

## 最近一次有效验证

- [DOCUMENTED] `PROJECT_STATUS_SUMMARY.txt` 记录：用户报告 Rabo 端机器人状态只读测试四个设备连接成功，且无错误。
- [DOCUMENTED] `docs/RABO_CAMERA_TEST_REPORT.md` 记录：Rabo 端相机只读测试发现部分 RGB topic 可采样，结果为 CHECK。
- [DOCUMENTED] `docs/RABO_WORKSPACE_V2_CLOSURE_REPORT.md` 记录：本地 `py_compile` 和 `test_dual_arm_reachability.py --skip-runtime` 通过，但工作空间坐标门禁仍阻塞。
