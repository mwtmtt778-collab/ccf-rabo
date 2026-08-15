# GitHub 与 Rabo 同步

## 本地 GitHub 状态

- [VERIFIED] 本地仓库路径：`/home/liyi/agent_system`。
- [VERIFIED] 当前分支：`main`。
- [VERIFIED] 当前本地 remote：

```text
origin git@github-ccf-pc2:mwtmtt778-collab/ccf-rabo.git
```

- [DOCUMENTED] `PROJECT_STATUS_SUMMARY.txt` 记录：本地 PC 到 GitHub 私有仓库的 read/pull/push 曾验证过。
- [DOCUMENTED] `SYNC_TEST.txt` 记录：`local to rabo sync ok`、`rabo to local sync ok`。

## Rabo 端状态

- [PENDING] Rabo 端 remote 名称和指向需要现场终端证据。
- [DOCUMENTED] `docs/GIT_WORKFLOW_REPORT.md` 写过：云端 checkout 可能使用 `github` 指向同一 GitHub 仓库、`origin` 指向 Rabo 平台仓库。
- [PENDING] 上述 remote 命名只是历史记录，不能当作当前事实。

## 最小只读确认命令

在 Rabo 端运行：

```bash
pwd
git status --short --branch
git remote -v
git rev-parse HEAD
```

## 同步风险

- [VERIFIED] 当前本地有未提交修改和未跟踪源码。
- [VERIFIED] 不应在未确认前执行 `pull`、`push`、`reset`、`clean`。
- [PENDING] `agent_system_backup.tar.gz` 当前被 Git 跟踪，建议后续评估是否从 Git 移除。
- [PENDING] Rabo HTML、WPS、旧 Codex 文档是否纳入 GitHub 需单独判断。

