# 钻机多源动力系统协同优化

面向钻机多源动力系统的高拟真仿真研究代码，覆盖：

- 公开证据约束的钻机负荷数据生成与物理合理性校验；
- 短时总负荷预测及现代基线对比；
- 四级供需风险识别与安全保护通道；
- 网电、柴油发电机、储能的风险自适应协同调度；
- 可复现的验收审计与证据清单。

> 当前处于V18技术收敛：V15已完成冻结仿真留出，V17已完成用户指定72小时数据回放、目标域强基线、HiGHS 1.12.0锁版本复核和应急负荷保护；V18新增Energistics官方真实WITSML钻井过程锚点。用户指定72小时文件仍保留其原始合成来源，公开WITSML不含总有功功率，因此不得宣称目标钻机现场功率验证已完成。

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
