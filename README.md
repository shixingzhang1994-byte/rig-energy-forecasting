# 钻机多源动力系统协同优化项目

> **项目唯一入口 · 最后更新：2026-09-22**  
> 当前结论：核心控制器已完成 V24 同源仿真确认性评估；V27 异构仿真仍因 1/12 个预注册单元不可评估而保持“验证不完整”。V28 公开真实数据及正式实验均已完成：8 个数据集家族已准备，真实负荷预测的 24 个数据集—时域组合和 2 个真实 BESS 观测分析均已保留。按六个论文关键证据块计，4/6（66.7%）主要由真实测量支持，但目标钻机闭环现场验证仍为 0%。项目不能宣称目标钻机现场功率验证、跨钻机泛化或工程部署完成。

## 一眼看进度

| 工作线 | 当前状态 | 可用结论 | 关键证据 | 下一步 |
|---|---|---|---|---|
| 核心控制方法 | **V24 已完成** | 12 个冻结种子中，RSS 相对三个匹配基线的 EENS 均值差为负；同时运行成本均值更高，应分别报告可靠性与成本 | [`analysis_manifest.json`](rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json) | 把 V24 主结果同步进论文 |
| 异构参数验证 | **V27 不完整** | 11 个单元通过技术门；`20261304` 在调度前因稳定钻进场景为 0 个合格窗口而不可评估。按预注册规则不得补种子、替换单元或计算完整疗效结论 | [`eligibility.json`](rig-energy-forecasting/artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json)、[`v27_heterogeneous_validation.yaml`](rig-energy-forecasting/configs/v27_heterogeneous_validation.yaml) | 若继续，另立 V28 协议；不要改写 V27 |
| 公开真实数据 | **V28 数据与实验均已完成** | 8 个数据集家族、10 个清洗后电气序列；预测实验 8 序列 × 3 时域全部完成，BESS 观测实验覆盖 M5BAT 与 Tsukuba。1 分钟结果为 5/8 方向改善、3/8 Holm 显著；15 和 60 分钟均为 8/8 优于持久性 | [`EXPERIMENT_RESULTS.md`](rig-energy-forecasting/artifacts/v28_public_real_validation/final_evidence/EXPERIMENT_RESULTS.md)、[`experiment_evidence_manifest.json`](rig-energy-forecasting/artifacts/v28_public_real_validation/final_evidence/experiment_evidence_manifest.json) | 将结果、负面发现和证据边界写入论文 |
| 论文 | **证据同步版已完成** | 当前 CAS 稿已纳入 V24、V27 和 V28，明确报告真实数据支持、负面外部有效性结果及现场验证边界；2026-09-22 编译为 20 个 PDF 页面 | [`paper/submission/main.pdf`](paper/submission/main.pdf)、[`paper/STATUS.md`](paper/STATUS.md) | 作者核验、更新归档包并执行投稿预检 |
| 可复现发布 | **V21 草稿已冻结** | GitHub 工作副本与校验通过的 Zenodo ZIP 均保留；它们不是 V24/V27 的最新发布包 | [`releases/README.md`](releases/README.md) | 论文定稿后重建新版本发布包 |

V24 的三个主要 EENS 差值（RSS 减基线）分别为：相对 Rule-Based `-5.239 kWh`（95% bootstrap CI `[-9.428, -1.294]`，Holm 校正 `p=0.0396`）；相对 Point-Forecast-MILP `-15.588 kWh`（`[-21.454, -9.976]`，`p=0.00293`）；相对 Risk-Adaptive-Residual-CVaR-MILP `-5.492 kWh`（`[-8.698, -2.798]`，`p=0.00586`）。对应运行成本均值差分别为 `+195.98`、`+280.17`、`+153.83` 元，因此这是可靠性—成本权衡，不是双重改善。

当前代码回归：**172 passed，5 warnings**（2026-09-21；warning 均为 PyTorch nested-tensor 提示）。

V28 数据准备结果：10 个清洗后电气序列共 **13,571,131** 个一分钟时刻，其中 **12,351,203** 个（**91.01%**）通过数据集各自的固定质量门；目标列均未插补。已完成的真实实验使用 6 个独立数据家族、共 12 个“数据集—实验角色”实例（2 个钻井过程记录、8 个预测序列、2 个 BESS 记录；Tsukuba 在预测和 BESS 两类分析中各承担一个角色）。

V28 最终结果不能压缩成“真实数据证明控制器有效”。可审计的口径是：六个论文关键证据块中，钻井状态分布、钻井时序外部有效性、跨域真实负荷预测、真实 BESS 行为四块由真实测量主要支持，V24 控制器疗效与组件消融两块由仿真支持，故真实测量证据块占 **4/6 = 66.7%**。直接目标钻机闭环现场验证仍为 **0%**。

## 当前证据边界

可以说：

- V24 在同一仿真器、冻结协议和 12 个种子下提供了 RSS 相对匹配基线的确认性可靠性证据。
- V27 保留了 11 个技术通过单元和 1 个不可评估单元，证明异构验证流程执行过，但整体按协议不完整。
- 公开真实数据支持钻井过程/状态/转移的外部审计，并明确指出当前仿真时序持续性不受这些数据支持。
- REFIT、UCI 和 Tsukuba 的冻结测试段已完成跨域负荷预测实验；15 和 60 分钟时域均为 8/8 优于持久性，1 分钟时域结果混合，不能作普遍改善主张。
- M5BAT 和 Tsukuba 已完成真实储能功率/SOC 观测分析，支持物理一致性与控制信号合理性，但不支持论文控制器的因果疗效。

不能说：

- 已完成目标钻机现场总有功功率验证、跨站点/跨钻机泛化、HIL 或部署验证。
- V27 已通过，或用 11 个通过单元替代预注册的 12 单元总体分析。
- Petrobras 3W 生产异常标签等同于供能风险标签。
- 可靠性改善同时意味着运行成本下降。

## 目录结构

```text
QZ/
├── README.md                       # 本文件：唯一当前状态入口
├── rig-energy-forecasting/         # 权威代码、配置、测试、数据和实验产物
├── paper/
│   ├── submission/                 # 当前 CAS 投稿候选源码与 PDF
│   ├── latex/                      # V21/早期 elsarticle 工作稿
│   ├── experiments/                # 论文侧重算与表格
│   ├── figures/                    # 论文图件
│   ├── review/                     # 审稿模拟、核查和返修记录
│   └── archive/                    # 已被当前投稿稿取代的修订副本
├── docs/
│   ├── 目标钻机真实数据需求与V15外部测试规范.md
│   ├── 项目任务书.pdf
│   └── history/                    # 历史研发状态和旧收敛方案
├── data/                           # 用户交付的 V15 数据包；保留原始证据身份
└── releases/
    ├── github-reproducibility-package/  # V21 GitHub 发布工作副本
    └── reserve_soc_...draft.zip          # 已校验的 V21 Zenodo 草稿包
```

`rig-energy-forecasting/artifacts/` 中大小写不统一的历史版本目录没有重命名，因为冻结清单、复现命令和论文证据路径依赖原路径。这里的“整洁”优先服从证据可追溯性。

## 从哪里开始

- 看当前结论：本页。
- 做代码或实验：[`rig-energy-forecasting/README.md`](rig-energy-forecasting/README.md)。
- 改论文：[`paper/README.md`](paper/README.md)。
- 准备现场数据：[`docs/目标钻机真实数据需求与V15外部测试规范.md`](docs/目标钻机真实数据需求与V15外部测试规范.md)。
- 查 V21 及以前的完整历史：[`docs/history/研发状态_V21_20260909.md`](docs/history/研发状态_V21_20260909.md)。

## 复核命令

```bash
cd /home/zsx/桌面/QZ/rig-energy-forecasting

# 代码回归
conda run -n qz-rig-energy-ml pytest -q

# 重建公开真实数据审计
conda run -n qz-rig-energy-ml python scripts/analyze_public_real_datasets.py

# 重建最终实验证据台账并执行哈希/产物完整性检查
conda run -n qz-rig-energy-ml python scripts/build_v28_experiment_evidence.py

# 检查 V27 完整性；当前应以退出码 2 报告缺少不可评估单元的 result.json
conda run -n qz-rig-energy-ml python scripts/analyze_v27_heterogeneous.py

# 校验冻结的 V21 发布 ZIP
cd /home/zsx/桌面/QZ/releases
sha256sum -c reserve_soc_reproducibility_package_v1.0.0-draft.zip.sha256
unzip -t reserve_soc_reproducibility_package_v1.0.0-draft.zip
```

## 维护规则

1. 本页是唯一“当前状态”；历史长日志只放 `docs/history/`。
2. 新实验必须使用新版本目录和冻结配置，不覆盖 V15/V19–V27 证据。
3. `data/` 与 `artifacts/` 不按“看起来重复”删除；只有经过清单/压缩包校验的副本才能清理。
4. 缓存、临时渲染、LaTeX 中间文件和误放安装包不进入项目目录。
5. 论文中的每个关键数字都应能链接到 JSON/CSV、复现命令或冻结清单。
