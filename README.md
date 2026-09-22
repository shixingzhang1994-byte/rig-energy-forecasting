# 钻机多源动力系统协同优化

面向钻机多源动力系统的高拟真仿真研究代码，覆盖：

- 公开证据约束的钻机负荷数据生成与物理合理性校验；
- 短时总负荷预测及现代基线对比；
- 四级供需风险识别与安全保护通道；
- 网电、柴油发电机、储能的风险自适应协同调度；
- 可复现的验收审计与证据清单。

> **当前状态（2026-09-21）**：V24 已完成 12 种子确认性 matched-controller 评估；V27 异构仿真因 12 个预注册单元中 1 个在控制器运行前不可评估而按协议保持不完整。V28 已完成公开真实数据准备、8 序列 × 3 时域预测实验以及 M5BAT/Tsukuba BESS 观测分析。六个论文关键证据块中 4 个主要由真实测量支持（66.7%），但目标钻机闭环现场验证仍为 0%。不得宣称目标钻机现场功率验证、跨钻机泛化或部署验证已经完成。项目级进度以 [`../README.md`](../README.md) 为准。

## 当前证据入口

- V24 确认性分析：`artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json`
- V27 冻结协议：`configs/v27_heterogeneous_validation.yaml`
- V27 不可评估单元：`artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json`
- 多公开数据集审计：`artifacts/public_real_multi_dataset_validation/manifest.json`
- 时序外部有效性评估：`artifacts/public_real_multi_dataset_validation/temporal_external_validity_assessment.json`
- V28 数据就绪清单：`artifacts/v28_public_real_validation/data_readiness/dataset_readiness_manifest.json`
- V28 预测确认性分析：`artifacts/v28_public_real_validation/forecast_confirmatory/forecast_confirmatory_analysis.json`
- V28 BESS 观测报告：`artifacts/v28_public_real_validation/bess_observational/bess_observational_report.json`
- V28 最终证据台账：`artifacts/v28_public_real_validation/final_evidence/experiment_evidence_manifest.json`

V15/V19–V27 的历史失败、冻结配置和产物仍按原路径保留；不要重命名或覆盖。

## 环境

```bash
conda env create -f environment.yml
conda activate qz-rig-energy-ml
```

RTX 50 系列可按 `scripts/setup_conda_env.sh` 安装 CUDA 版 PyTorch；CPU 环境可在实验脚本中使用 `--cpu`。

## 快速开始

```bash
# 生成一份可复现的合成数据
python scripts/generate_synthetic.py \
  --seed 20260825 \
  --output data/processed/rig_load_synthetic_v3.parquet \
  --artifact-dir artifacts/v3_data

# 运行快速 CPU 冒烟测试
python scripts/run_benchmark.py --quick --cpu
pytest -q
```

## V18收敛复核

```bash
# 审计官方公开真实WITSML，不解压、不升级其证据身份
python scripts/audit_energistics_witsml.py \
  data/public_external/energistics_well_b/NA-NA-EnergisticsWell2016-B.zip \
  artifacts/v18_public_real_drilling_anchor

# 全量回归并生成V18证据哈希清单
pytest -q
python scripts/build_v18_convergence_manifest.py
```

V18机器可读协议为 `configs/v18_evidence_triage_convergence.yaml`。预测工程候选冻结为V17目标域iTransformer；项目方法的创新边界是预测、不确定性、供能风险、CVaR/MILP启停、SOC监督和应急负荷保护组成的安全协同链，而不是把iTransformer本身作为自有创新。

## V3 全流程

已有预测结果可复用时：

```bash
python scripts/run_v3_pipeline.py --reuse-forecasts
```

完整重建三种独立随机种子时，去掉 `--reuse-forecasts`。运行结果默认写入 `artifacts/`，该目录以及本地数据、模型文件被 Git 忽略，以避免把大型生成物和潜在现场数据提交到仓库。

## 现场数据接口

最小输入字段为 `timestamp` 和 `total_active_power_kw`，可选加入 `operation_state` 及供能约束字段。字段映射示例见 `configs/scada_mapping.example.yaml`。

```bash
python scripts/prepare_real_data.py \
  --input /path/to/scada.csv \
  --mapping configs/scada_mapping.example.yaml \
  --output data/processed/rig_load_real.parquet
```

## 目录结构

```text
configs/       实验配置与验收门槛
scripts/       数据生成、训练、调度和审计入口
src/           rig_energy Python 包
tests/         单元测试
data/          本地数据目录（默认不提交）
artifacts/     实验产物目录（默认不提交）
```
