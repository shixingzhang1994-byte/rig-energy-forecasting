from __future__ import annotations

"""V15正式调度基准入口。

控制器实现保留在单变量诊断模块中，以保证V14失败复现与V15正式评估使用
完全相同的动作定义；本入口只提供不含“diagnostic”歧义的冻结复现命令。
"""

from run_v15_generator_first_diagnostic import main


if __name__ == "__main__":
    main()
