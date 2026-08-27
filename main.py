"""项目入口: 选择并启动一个 agent。

────────────────────────────────────────────────────────────────────────
在 Rabo 平台上，控制器无参数启动本入口，实际运行下面 DEFAULT_AGENT
指定并由 Case 交付的 Agent。命令行参数仅保留给本地开发调试。

(可选)本地手动调试时, 可以用命令行参数显式指定某个 agent —— 平台用不到这一项:
    python main.py                  # 启动 DEFAULT_AGENT(当前 act_c_policy)
    python main.py <your_agent>     # 启动 agents/<your_agent>/(子包须在 __init__.py 暴露 run())
────────────────────────────────────────────────────────────────────────

为什么用 importlib 动态加载 (而不是 if/elif 硬编码)?
  - 加新 agent 时不用改 main.py, 只要在 agents/ 下加个目录就行
  - main.py 永远只有这几行, 不会随 agent 数量膨胀

如果你有需要在所有 agent 启动前都做的全局副作用 (设环境变量、关闭 SSL 验证、
注入 monkey patch 等), 写在 import 之前, 例如:

    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 必须最先!
"""

import importlib
import sys


# 平台无参启动时运行的 agent —— 换 agent 就改这里(本地调试也可用命令行参数覆盖)。
DEFAULT_AGENT = "act_c_policy"
RUN_ROBOT_STATE_TEST_ON_DEFAULT_START = False


def main():
    if len(sys.argv) == 1 and RUN_ROBOT_STATE_TEST_ON_DEFAULT_START:
        from tools.test_robot_state import main as test_robot_state_main

        test_robot_state_main()
        return

    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AGENT

    try:
        module = importlib.import_module(f"agents.{name}")
    except ModuleNotFoundError as e:
        print(f"找不到 agent: {name} ({e})", file=sys.stderr)
        print("可用 agent 在 agents/ 目录下", file=sys.stderr)
        sys.exit(1)

    if not hasattr(module, "run"):
        print(
            f"agents/{name}/__init__.py 必须暴露 run() 函数",
            file=sys.stderr,
        )
        sys.exit(1)

    module.run()


if __name__ == "__main__":
    main()
