# AGENTS.md

本文件是当前项目给 AI 编程助手和 Codex 使用的工作规则入口。

## 当前项目边界

- [VERIFIED] 项目根目录是 `/home/liyi/agent_system`。
- [DOCUMENTED] 本项目是 Rabo 平台控制器工程，平台按 `requirements.txt` 安装依赖，并以 `python3 -u main.py` 启动控制器。
- [VERIFIED] 当前代码目录保持为：`agents/`、`core/`、`drivers/`、`expert/`、`tools/`、`main.py`、`requirements.txt`。
- [DOCUMENTED] README 中的开发约定仍然有效，但 README 中的启动说明与当前 `main.py` 实际行为存在冲突，见 `PROJECT_STATUS.md`。

## 必须保护的现有工作

- [VERIFIED] 当前仓库存在未提交修改和未跟踪源码。
- [VERIFIED] `tools/test_dual_arm_reachability.py` 是已跟踪且已修改文件。
- [VERIFIED] `expert/` 是未跟踪源码目录。
- [VERIFIED] `tools/resolve_workspace_coordinates.py` 是未跟踪源码文件。

任何任务默认不得删除、覆盖、reset、clean、stash 或丢失这些现有修改。

## 代码修改规则

- 不要重构现有代码目录结构。
- 不要新建 `src/`、`tests/`、`config/`、`experiments/` 等目录，除非用户明确要求。
- 不要修改业务代码、实验脚本、`main.py`、`requirements.txt`，除非用户明确要求并说明目标。
- 不要把 Secret、API Key、Token、SSH 私钥或密码写入仓库。
- 不要把 `logs/`、`outputs/` 这类运行产物当作文档事实来源提交。

## 事实标记

项目管理文档统一使用：

- `[VERIFIED]`：实际运行或用户确认。
- `[DOCUMENTED]`：已有文档明确记录。
- `[PENDING]`：待确认。
- `[OBSOLETE]`：已经失效。
- `[UNKNOWN]`：当前无法判断。

不要把 Codex 历史报告中的推测直接升级为事实。

## AI 工作区

- `.ai/inbox/current_task.md` 只记录当前任务。
- `.ai/handoff/latest_result.md` 只记录最近一次 Codex 执行结果摘要。
- 不要把旧报告批量搬入 `.ai/`。
- [VERIFIED] `.ai/` 是本地 AI 临时工作区，已加入 `.gitignore`。

## Git 规则

- 默认只读检查可以运行 `git status`、`git diff`、`git ls-files`。
- 不要运行 `git add`、`commit`、`push`、`pull`、`checkout`、`switch`、`reset`、`clean`，除非用户明确要求。
- 不要用 Git 操作回滚用户或前序工作。
