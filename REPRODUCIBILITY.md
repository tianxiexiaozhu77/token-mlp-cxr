# 实验链路与入口说明

这份文档定位当前论文的原始实现。代码按原工作区结构归档，未做训练逻辑重构，也未在整理过程中重新训练。

隔离环境安装、七模型权重下载、数据获取与固定清单要求，以及表 1–15 的具体实验/导出命令，见 [README.md](README.md)。新增 `tools/prepare_workspace.py` 与 `tools/setup_environment.py` 默认只打印计划，传入 `--execute` 才执行；它们不生成缺失的历史数据清单或审计记录。

## 按顺序理解代码

1. **数据与分组**：`RadZero/scripts/prepare_openi_adapter.py` 提供 OpenI 的分组准备；`experiments/direct_anchor_classification/prepare_data.py` 准备 CheXpert/NIH 主划分。标签、prompts、样本顺序和 split membership 都属于复现输入。公开数据集名称或随机种子本身不足以恢复本论文的本地子集。
2. **冻结编码器、缓存与原生输出一致性**：AlphaRAD/RadZero 使用 `experiments/direct_anchor_classification/cache.py` 和 `validate_native.py`；其余五个模型使用 `experiments/baseline_token_classification/cache.py` 及对应 `TokenBackend`。缓存阶段检查原生输出路径，不是额外的适配器训练。
3. **共享 Token MLP**：`RadZero/scripts/adapter_model.py:ResidualAdapter` 和 `experiments/baseline_token_classification/common.py:residual_mlp` 实现相同的残差结构：`d -> 64 -> d`、GELU、up projection 零初始化。`FrozenClassificationHead` 重建 AlphaRAD/RadZero 的固定聚合与打分。
4. **主结果训练与选择**：`experiments/direct_anchor_classification/train.py` 服务 AlphaRAD/RadZero；`experiments/baseline_token_classification/train.py` 服务其余五个模型。主研究使用固定 split、三种子、验证 AUROC 选点、epoch 0 可选。对应模型启动器见 README。
5. **Post、pooling、Both、gate/loss**：依次查看 `adapter_placement_classification`、`both_adapter_classification`、`gated_asl_classification`。部分对照复用 Pre 结果，不应把各目录所有记录数相加当作新增训练数。
6. **监督基线**：`supervised_baseline_classification/components.py` 包含 LinearProbe 和 ClipGlobalAdapter；AlphaRAD 与 RadZero 的特征导出和训练位于各自监督基线实验目录。全局适配器并不等同于只改变 MLP 的插入位置。
7. **更大 NIH 划分**：`large_patient_multilabel/prepare.py -> features.py -> freeze.py -> run.py`，然后 `report.py`/`uncertainty.py`。`freeze.py` 固定协议；所有适配器拟合和 NIH 验证选点先完成，再评估测试集。
8. **NIH→CheXpert**：`nih_chexpert_transfer/run.py` 调用 `prepare.py`、共享 `cache.py`、`evaluate.py`、`report.py`。`common.py:MAPPING` 固定七标签映射，`selected_module` 加载 NIH checkpoint；该目录没有适配器训练步骤。
9. **不确定性**：`classification_uncertainty/bootstrap.py` 及 `chexchonet_stability_20261024/cluster_bootstrap.py` 提供共享统计逻辑；NIH 与迁移实验按患者成对 bootstrap，使用 2,000 次抽样。
10. **结果进入论文**：`latex_ours/scripts/build_token_mlp_*.py` 核验并导出已有结果。`export_token_mlp_figure_assets.py -> draw_token_mlp_pipeline.py` 生成真实影像主图。不同 exporter 的核验范围见它们各自的 docstring；表格算术核验不等于重做模型推理。

## 原工作区中的典型入口

下面是资源恢复之后使用的入口示例，不是对空目录执行即可成功的安装流程。所有路径相对于本包根目录；GPU 编号需要与实际服务器一致。

```bash
# 查看 AlphaRAD/RadZero 共享训练器的参数；不会开始训练。
bash AlphaRAD/local_setup/python_local.sh AlphaRAD/local_setup/train_direct_anchor.py --help

# 查看五模型训练器的参数，以 KAD 环境为例。
bash baselines/KAD/local_setup/python_local.sh experiments/baseline_token_classification/train.py --help

# 查看消融训练器和 NIH 训练器的参数。
bash AlphaRAD/local_setup/python_local.sh experiments/adapter_placement_classification/train.py --help
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/train.py --help
```

`--help` 是否可用也取决于模块导入依赖已安装。不要对当前已完成历史 run 目录启动新训练。`nih_chexpert_transfer/run.py`、部分 `prepare.py`、controller 使用固定常量，应先完成新机器的路径/GPU 配置；直接运行会准备或执行实验，并非纯参数查看。

## 需要外部准备的资源

| 资源 | 本地实现预期的位置/来源 |
|---|---|
| AlphaRAD 模型及 MPNet/DINO 资源 | `AlphaRAD/local_assets/`；主权重 `model/xraydinov2_mpnet/eccv_mpnet_518.tar` |
| RadZero 模型及 remote-code 文件 | `RadZero/checkpoints/RadZero/`，还需其模型配置引用的依赖 |
| CARZero 权重、文本模型 | `baselines/CARZero/pretrain_model/`、`local_assets/BioClinicalMPBERT/` |
| KAD 权重、文本模型 | `baselines/KAD/local_assets/KAD_224_best_valid.pt`、`Med-KEBERT/` |
| BiomedCLIP 模型、配置、tokenizer | `baselines/BiomedCLIP/checkpoints/` |
| BioViL-T 图像和文本模型 | `baselines/BioViL-T/checkpoints/` |
| MaCo 权重和文本资源 | `baselines/MaCo/checkpoints/` 及原生 `ZeroShot` 实现所需文件 |
| 原始数据与冻结划分 | README 中的三个 primary manifests，以及新的 NIH split 和 CheXpert target manifest |
| 历史结果完整审计 | 对应 run 下的协议、checkpoint、训练历史、冻结 feature/head、逐样本预测和哈希记录 |

## 整理时完成的检查

- 复制清单中的文件与原文件 SHA-256 一致。
- `PUBLISHED_LOGS.json` 中的归档日志逐文件核对大小和 SHA-256；未登记的日志会使核验失败。
- 所有归档 Python 文件进行语法解析，Shell 文件进行 `bash -n` 检查。
- 静态检查归档 Python 中显式 `experiments.*` 依赖是否在包内。
- 归档范围排除图像、权重、数组缓存、完整 runs 和虚拟环境；原始文本日志另存于 `logs/`。

这些检查由 `tools/verify_snapshot.py` 执行，结果写入 `VALIDATION.json`。本次未执行模型推理、训练、完整单元测试、预测重算、全套 LaTeX 编译或全新环境安装。
