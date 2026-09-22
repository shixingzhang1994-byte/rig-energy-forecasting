# 论文工作区

当前投稿候选位于 [`submission/`](submission/)，使用 Elsevier CAS 单栏模板：

- 主文件：[`submission/main.tex`](submission/main.tex)
- 正文：[`submission/body.tex`](submission/body.tex)
- 参考文献：[`submission/refs.bib`](submission/refs.bib)
- 当前 PDF：[`submission/main.pdf`](submission/main.pdf)（20 个 PDF 页面，其中 1 页 Highlights、正文编号 19 页；2026-09-22 从当前源码干净编译）

## 当前状态

这份 PDF 已与 V24、V27 和 V28 的当前证据同步，可作为下一轮作者核验与投稿预检的候选稿：

1. V24 的 12 种子确认性控制器比较、成本权衡和组件干预已纳入摘要与结果；
2. V27 的异构仿真不完整结论已在局限中披露；
3. V28 的 8 个公开数据集家族、真实负荷预测、BESS 观测分析及钻井状态持续时间负面结果已纳入；
4. 论文明确区分“真实测量的组件/假设外部验证”和“目标钻机闭环现场控制器验证”。

当前剩余工作主要是作者身份与声明核验、更新 V24/V28 公共归档包，以及按目标期刊最新要求执行最终投稿预检。

## 目录说明

- `submission/`：当前 CAS 投稿候选，唯一应继续编辑的 LaTeX 稿。
- `latex/`：V21/早期 `elsarticle` 工作稿，仅用于追溯。
- `archive/latex-revision-20260911/`：已被后续稿件取代的修订副本。
- `experiments/`：论文侧重算、比较矩阵和结果。
- `figures/`：论文图件及图件说明。
- `review/`：返修计划、核查、模拟审稿与补丁记录。
- `checkpoints/`：冻结恢复点，不随普通清理删除。
- `STATUS.md`：论文层当前状态与投稿前清单。
- `progress.md`：历史开发日志，不作为当前状态入口。

项目总体状态统一见 [`../README.md`](../README.md)。
