# 钻机多源动力系统协同优化

面向钻机多源动力系统的高拟真仿真研究代码，覆盖：

- 公开证据约束的钻机负荷数据生成与物理合理性校验；
- 短时总负荷预测及现代基线对比；
- 四级供需风险识别与安全保护通道；
- 网电、柴油发电机、储能的风险自适应协同调度；
- 可复现的验收审计与证据清单。

> 当前数据为公开资料约束的高拟真仿真数据，不是现场 SCADA 实测数据。现场验证、在线漂移监测和多机组离散启停约束仍需后续工程化。

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
