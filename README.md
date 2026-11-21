# Token MLP for Chest X-ray Vision–Language Models

论文：**Shared Token Adaptation before Disease-Conditioned Aggregation in Frozen Chest X-ray Vision–Language Models**。

本仓库整理了七个冻结胸片视觉语言模型的 Token MLP 实现、实验程序、论文表格、汇总结果和原始日志。论文源文件是 [`latex_ours/main_token_mlp.tex`](latex_ours/main_token_mlp.tex)。以下表号对应这份稿件：表 3 是七模型主结果，表 4 是 AlphaRAD/RadZero 监督基线，表 5 是更大 NIH 测试划分，表 6 是 NIH→CheXpert。

**复现状态：**源码和日志已核验；安装说明和命令已按现有代码核对，但尚未在全新机器上完成端到端复现。本仓库不包含图像、模型权重、固定样本清单、完整历史 `runs/`、特征缓存或逐样本预测。下载公开数据集之后，还需要恢复下文列出的固定清单和实验依赖，才能重算论文数字。

## 1. 准备独立工作目录

建议将“准备上传的源码仓库”和“实际下载数据、训练的工作目录”分开。以下命令假定已经下载或解压本仓库，在仓库根目录执行；`CXR_WORKSPACE` 请换成一个**尚不存在**的目录。

```bash
set -euo pipefail
export CXR_RELEASE="$PWD"
export CXR_WORKSPACE=/path/to/token_mlp_workspace

# 仅显示计划，不联网、不写入工作目录。
python3 tools/prepare_workspace.py --workspace "$CXR_WORKSPACE"

# 克隆七个上游仓库、checkout 固定 commit、应用 AlphaRAD 补丁，再覆盖本仓库接口。
python3 tools/prepare_workspace.py --workspace "$CXR_WORKSPACE" --execute
cd "$CXR_WORKSPACE"
```

[`tools/prepare_workspace.py`](tools/prepare_workspace.py) 读取 [`MODEL_SOURCES.json`](MODEL_SOURCES.json)，使用其中保存的上游地址和 commit；先核对源码/补丁哈希，拒绝已有的目标目录。完整上游源码没有重复打包。AlphaRAD 上游本地修改见 [`upstream_patches/AlphaRAD.patch`](upstream_patches/AlphaRAD.patch)。若某个上游版本已不可访问，需要取得该版本的源码存档；替换为最新版本不等于复现历史实现。

后续命令均在 `CXR_WORKSPACE` 根目录执行。**已有实验服务器只查看这些命令时，不要对已完成的历史运行目录重新执行 prepare/train。**

## 2. 搭建隔离环境

使用 Linux、NVIDIA GPU 和支持相应 PyTorch CUDA wheel 的驱动。七模型分别使用独立环境，避免 Transformers、NumPy、PyTorch 版本冲突；不使用 `--system-site-packages`。

| 模型 | Python | PyTorch / torchvision | CUDA wheel | 环境目录 |
|---|---|---|---|---|
| AlphaRAD | 3.10.22 | 2.9.0 / 0.24.0 | cu128 | `AlphaRAD/.venv` |
| RadZero | 3.12.7 | 2.2.1 / 0.17.1 | cu121 | `.envs/radzero` |
| CARZero | 3.10.12 | 2.2.2 / 0.17.2 | cu121 | `baselines/CARZero/.venv` |
| KAD | 3.10.22 | 2.2.2 / 0.17.2 | cu121 | `baselines/KAD/.venv` |
| BiomedCLIP | 3.10.22 | 2.2.2 / 0.17.2 | cu121 | `baselines/BiomedCLIP/.venv` |
| BioViL-T | 3.10.22 | 2.2.2 / 0.17.2 | cu121 | `baselines/BioViL-T/.venv` |
| MaCo | 3.10.12 | 2.2.2 / 0.17.2 | cu121 | `baselines/MaCo/.venv` |

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装 `uv`，系统还需 `git`、`curl`、`tar`、`unzip`。然后执行：

```bash
# 预览单个环境的安装命令。
python3 tools/setup_environment.py AlphaRAD --workspace "$PWD"

# 建立七个互相隔离的环境；已有同名环境会报错，不会被覆盖。
for model in AlphaRAD RadZero CARZero KAD BiomedCLIP BioViL-T MaCo; do
  python3 tools/setup_environment.py "$model" --workspace "$PWD" --execute
done

# 启动器使用的公共缓存和临时目录。
mkdir -p baselines/.cache/tmp baselines/.cache/huggingface baselines/.cache/torch
mkdir -p RadZero/runs/gpu_locks

# 单独建立下载工具环境，避免改变模型环境的依赖。
uv venv --seed --python 3.10 .envs/downloads
.envs/downloads/bin/python -m pip install 'huggingface-hub==0.36.0' 'gdown==5.2.0'
```

安装脚本依次安装指定 Python、创建 venv、从 PyTorch 官方 wheel 源安装 torch/torchvision、安装对应依赖、执行 `pip check` 并打印版本。环境配置见 [`environments/index.json`](environments/index.json)，各模型依赖和原始 lock/freeze 一并保留。MaCo 依赖清单另补入其接口使用的 OpenCV；安装配方不表示已经验证过所有新机器。参考：[uv 环境管理](https://docs.astral.sh/uv/pip/environments/)、[PyTorch 历史版本安装](https://pytorch.org/get-started/previous-versions/)。

运行模型时使用其启动器，不要混用 Python：

```bash
bash AlphaRAD/local_setup/python_local.sh -c 'import torch; print(torch.__version__)'
bash RadZero/scripts/python_local.sh -c 'import torch; print(torch.__version__)'
bash baselines/KAD/local_setup/python_local.sh -c 'import torch; print(torch.__version__)'
```

多数历史训练程序只允许**物理 GPU 4–7**，下面单任务示例用 `--gpu 4`。控制器还会查询 `nvidia-smi` 并申请 GPU 文件锁。只有 GPU 0 的服务器需要先修改代码中的资源配置、重新冻结实验协议；仅设置 `CUDA_VISIBLE_DEVICES=0` 不能绕过物理编号校验。位置索引见 [`PORTABILITY_AUDIT.json`](PORTABILITY_AUDIT.json)。

## 3. 下载预训练模型

### 3.1 下载地址与预期位置

| 模型 | 权重来源 | 本地目标 |
|---|---|---|
| AlphaRAD | 历史 HF 仓库 `maxxyouu/AlphaRAD`，原项目记录见 `AlphaRAD/README.md` | `AlphaRAD/local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar` |
| RadZero | [Deepnoid/RadZero](https://huggingface.co/Deepnoid/RadZero) | `RadZero/checkpoints/RadZero/`，包含权重、配置、tokenizer、模型 Python 文件 |
| CARZero | [作者代码与下载说明](https://github.com/laihaoran/CARZero) | `baselines/CARZero/pretrain_model/CARZero_best_model.ckpt` |
| KAD | [作者代码与下载说明](https://github.com/xiaoman-zhang/KAD) | `baselines/KAD/local_assets/KAD_224_best_valid.pt` |
| BiomedCLIP | [Microsoft 模型页](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | `baselines/BiomedCLIP/checkpoints/` |
| BioViL-T | [Microsoft 模型页](https://huggingface.co/microsoft/BiomedVLP-BioViL-T) | `baselines/BioViL-T/checkpoints/` |
| MaCo | [作者代码与下载说明](https://github.com/SZUHvern/MaCo) | `baselines/MaCo/checkpoints/maco.pth` |

AlphaRAD 的 HF 地址来自实验时的本地记录，本次未能确认其当前在线可用性；若下载失败，需要使用原作者提供的同版资源或原服务器存档。其余下载也受原平台访问条件限制。本仓库不代发权重。

### 3.2 Hugging Face 权重、文本模型及配置

以下使用 [Hugging Face `snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download)。有固定 revision 的资源显式指定；没有记录 revision 的辅助资源仍需与历史哈希核对，不能把后续下载的 `main` 自动视作原实验版本。

```bash
# 下载时不使用强制 offline 的模型启动器。
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 .envs/downloads/bin/python - <<'PY'
from huggingface_hub import snapshot_download

# 忽略非本实验使用的其他框架导出，保留 PyTorch 权重/配置/tokenizer。
ignore = ['onnx/*', 'openvino/*', '*.onnx', '*.h5', '*.msgpack', '*.ot', '*.tflite']

def fetch(repo, destination, revision=None, allow=None, extra_ignore=None):
    snapshot_download(repo_id=repo, revision=revision, local_dir=destination,
                      allow_patterns=allow, ignore_patterns=ignore + (extra_ignore or []))

fetch('maxxyouu/AlphaRAD', 'AlphaRAD/local_assets',
      allow=['model/xraydinov2_mpnet/eccv_mpnet_518.tar'])
for repo in ['sentence-transformers/all-mpnet-base-v2', 'StanfordAIMI/dinov2-base-xray-224']:
    fetch(repo, 'AlphaRAD/local_assets/pretrained/' + repo)

# RadZero 使用 checkpoint 内的自定义模型代码，不能只下载 model.safetensors。
fetch('Deepnoid/RadZero', 'RadZero/checkpoints/RadZero', extra_ignore=['data/*'])
fetch('sentence-transformers/all-mpnet-base-v2', 'RadZero/dependencies/all-mpnet-base-v2')
fetch('StanfordAIMI/dinov2-base-xray-224', 'RadZero/dependencies/dinov2-base-xray-224')

fetch('Laihaoran/BioClinicalMPBERT', 'baselines/CARZero/local_assets/BioClinicalMPBERT')
fetch('xmcmic/Med-KEBERT', 'baselines/KAD/local_assets/Med-KEBERT')
fetch('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
      'baselines/BiomedCLIP/checkpoints', '9f341de24bfb00180f1b847274256e9b65a3a32e')
# OpenCLIP 接口从 checkpoints/ 本地读取 tokenizer。
fetch('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
      'baselines/BiomedCLIP/checkpoints',
      allow=['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'])
fetch('microsoft/BiomedVLP-BioViL-T', 'baselines/BioViL-T/checkpoints',
      '692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23')
PY

# BiomedCLIP 接口单独读取的 BERT 结构配置，与历史本地配置一致。
mkdir -p baselines/BiomedCLIP/checkpoints/text_config
cp resources/BiomedCLIP_text_config.json baselines/BiomedCLIP/checkpoints/text_config/config.json
```

MaCo 使用上游仓库根目录的 `mimic_wordpiece.json` 及 checkpoint 中的文本权重；无需改成其他 BERT。确认 `baselines/MaCo/mimic_wordpiece.json` 存在。

### 3.3 Google Drive 权重及已记录哈希

```bash
mkdir -p baselines/CARZero/pretrain_model baselines/KAD/local_assets baselines/MaCo/checkpoints
.envs/downloads/bin/gdown 'https://drive.google.com/uc?id=1kYF-k5otW5DHwz1En5d_ScV3zu2E27Ch' \
  -O baselines/CARZero/pretrain_model/CARZero_best_model.ckpt
.envs/downloads/bin/gdown 'https://drive.google.com/uc?id=1EOuWd6v4Us3lAYEwGcSPtH29616pELED' \
  -O baselines/KAD/local_assets/KAD_224_best_valid.pt
.envs/downloads/bin/gdown 'https://drive.google.com/uc?id=1B87JuLJnjh_CoQpjT7RWGTp6alfXV8_0' \
  -O baselines/MaCo/checkpoints/maco.pth

sha256sum -c <<'HASHES'
d9257c4865898becf3b2ff9ea4bdc4c5ec57da28bc99f4a714812cc589052b33  baselines/CARZero/pretrain_model/CARZero_best_model.ckpt
eae032570a2b041b6da629ad2e2bfc36283e4995d56d64779dc3d1a6a70021ea  baselines/KAD/local_assets/KAD_224_best_valid.pt
52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be  baselines/BiomedCLIP/checkpoints/open_clip_pytorch_model.bin
b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77  baselines/BioViL-T/checkpoints/biovil_t_image_model_proj_size_128.pt
6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c  baselines/BioViL-T/checkpoints/pytorch_model.bin
08bfd049637431cf1e475bfc1aefd322810551ce2efc0d9ceecea980aa0f6627  baselines/MaCo/checkpoints/maco.pth
HASHES
```

## 4. 下载数据集并恢复论文划分

### 4.1 论文实际使用的规模

| 用途 | 数据集 | 标签数 | Train / Val / Test 图像 | 分组单位 |
|---|---|---:|---:|---|
| 主实验 | OpenI | 18 | 5,966 / 755 / 749 | 报告 UID 与重复图像组；不是已验证的纵向患者 ID |
| 主实验 | CheXpert 本地队列 | 12 | 534 / 67 / 67 | 患者与完全重复文件组 |
| 主实验 | ChestXray14 本地队列 | 14 | 18,140 / 2,097 / 2,196 | 患者与完全重复文件组 |
| 表 5 | 同一 ChestXray14 队列重新划分 | 14 | 13,406 / 2,300 / 6,727 | 患者；划分 seed 20261121 |
| 表 6 | NIH 训练 → CheXpert 目标评估 | 7 个共同标签 | 目标不训练、不选点；评估 668 张 | 目标 500 位患者 |

CheXpert 是 **668 张图像/500 位患者**的本地队列；ChestXray14 是 **22,433 张图像/6,187 位患者**的本地队列。主实验将这些本地队列重新划分为训练/验证/测试，**不是官方大规模 train/test 协议**。表 5 仍来自 NIH，同机构重新划分；表 6 才是 NIH→Stanford 的跨机构评估，但目标是同一份已使用过的本地 CheXpert 队列，不是新增盲测数据。

### 4.2 数据下载入口与命令

- **OpenI / IU X-ray**：[NLM OpenI](https://openi.nlm.nih.gov/faq.php)。官方入口本次查询处于维护状态，下面为历史归档地址，当前可用性需实际下载确认。原准备程序还依赖 `indiana_reports.csv`、`indiana_projections.csv` 和作者的 OpenI 标签 CSV；官方图像/报告压缩包本身不自动生成这些文件。
- **CheXpert**：[Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)。本实验相关的 test 图像入口与标签见 [官方 test-set-labels 仓库](https://github.com/rajpurkarlab/cheXpert-test-set-labels)，按其链接申请/获取图像。不要用 v1.0-small 训练集直接替代这 668 张图像。
- **ChestXray14**：[NIH 官方下载目录](https://nihcc.app.box.com/v/ChestXray-NIHCC)。获取图像包、标签及列表文件后，按本地队列清单选取 22,433 张，保留患者 ID 和图像文件名。
- **作者预处理元数据**：[RadZero 官方说明](https://github.com/deepnoid-ai/RadZero)将评估元数据放在 HF 仓库 `data/`；其中标签/顺序还需与原实验固定清单比对。

```bash
mkdir -p data/downloads/OpenI data/downloads/CheXpert data/downloads/ChestXray14
curl --fail --location --continue-at - \
  https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz \
  --output data/downloads/OpenI/NLMCXR_png.tgz
curl --fail --location --continue-at - \
  https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz \
  --output data/downloads/OpenI/NLMCXR_reports.tgz
tar -xzf data/downloads/OpenI/NLMCXR_png.tgz -C data/downloads/OpenI
tar -xzf data/downloads/OpenI/NLMCXR_reports.tgz -C data/downloads/OpenI

curl --fail --location \
  https://raw.githubusercontent.com/rajpurkarlab/cheXpert-test-set-labels/main/groundtruth.csv \
  --output data/downloads/CheXpert/groundtruth.csv

# 只下载三个相关数据集的作者元数据；保留 HF 中的目录结构。
HF_HUB_OFFLINE=0 .envs/downloads/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Deepnoid/RadZero', local_dir='data/downloads/RadZero_metadata',
                  allow_patterns=['data/OpenI/*', 'data/Chexpert/*',
                                  'data/CheXpert/*', 'data/ChestXray14/*'])
PY
```

CheXpert/NIH 平台生成的实际下载链接可能随授权或时间变化，不能把网页地址当作图像包 URL。获取链接后，可用以下模板；填写真实链接再执行，文件名及解压工具按平台提供的格式调整：

```bash
export CXR_CHEXPERT_ARCHIVE_URL='填写官方平台提供的图像压缩包下载链接'
export CXR_NIH_ARCHIVE_URL='填写官方平台提供的某个 images_XXX.tar.gz 下载链接'
curl --fail --location --continue-at - "$CXR_CHEXPERT_ARCHIVE_URL" \
  --output data/downloads/CheXpert/images.zip
curl --fail --location --continue-at - "$CXR_NIH_ARCHIVE_URL" \
  --output data/downloads/ChestXray14/images_001.tar.gz
# NIH 如有多个图像包，需要下载覆盖本地队列中全部样本的包。
```

### 4.3 数据目录和固定清单

原代码读取的主要路径如下。这里列出的是**需要恢复的输入**，不是当前源码包里已经提供的文件：

```text
data/radzero/metadata/OpenI/                     custom.csv、图像标签 CSV、文本 prompts
data/radzero/raw/OpenI/iu_xray_kaggle/            indiana_reports.csv、indiana_projections.csv 等
data/radzero/prepared/OpenI/                     原实现预期的 OpenI 图像
data/radzero/prepared/ChestXray14/                图像及 chestxray14_test_text.json
data/alpharad/CheXpert/                          CheXpert 图像
RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json
AlphaRAD/local_setup/chexpert_query_data_20261022_shot5_seed42/full_catalog.json
AlphaRAD/local_setup/data_20261021_complete/catalog.json
data/direct_anchor_classification_20261118/chexpert/dataset.json
data/direct_anchor_classification_20261118/chestxray14/dataset.json
```

`dataset.json` 引用 `samples_file`，并记录 `samples_sha256`、`class_names`、`prompts`；样本清单包含图像路径、二元标签、分组 ID 和 split。CheXpert/NIH 的源 catalog 还记录 CSV 路径、哈希及标签顺序。完整复现必须保留样本成员、顺序、图像内容、标签映射和选点规则，不能只依赖“seed=42”。

数据准备入口：

```bash
# 前提：恢复 OpenI 原始 CSV、作者标签、图像及重复组审阅输入。
# 输出到 data/radzero/adapter_openi/，不是自动恢复历史 runs/.../inputs/。
bash RadZero/scripts/python_local.sh RadZero/scripts/prepare_openi_adapter.py \
  --seed 42 --workers 8

# 前提：历史 OpenI inputs/dataset.json 及其 samples_file、两个源 catalog/CSV 已恢复。
bash AlphaRAD/local_setup/python_local.sh experiments/direct_anchor_classification/prepare_data.py \
  --output data/direct_anchor_classification_20261118
```

OpenI 准备脚本还会核验已有的其他 prepared 数据和下载清单；其完整输入清单以脚本为准。目前没有从三个公开压缩包直接生成全部历史 catalog/inputs 的一键程序。**若没有原始固定清单，以下训练可在新协议下重做，但不能声称逐数值复现论文。** 新机器路径变化后应生成新的来源记录，不要改写历史哈希冒充原记录。

## 5. 论文表格对应哪些实验

表内数值通常为 **macro AUROC / macro AP（百分数）**；AP 在代码中用 `macro_auprc` 命名，采用 average precision 实现。适配器结果按种子 42/43/44 汇总，`±` 表示种子间标准差，差值为百分点；具体以各表 caption 为准。原模型没有适配器训练。公共训练设置为 AdamW、lr=1e-4、weight decay=1e-4、有效 batch=32、最多 20 epoch、最少 8 epoch、patience=5，按验证集 macro AUROC 选点，epoch 0 可被选中。

下表是当前稿件的完整映射；更细的文件索引见 [`TABLE_INDEX.csv`](TABLE_INDEX.csv)。

| 论文表 | 内容 | 生成数值的实验 | 导出程序（`latex_ours/scripts/` 下） |
|---|---|---|---|
| 1 | 模型结构与维度 | 架构统计，人工维护，无训练命令 | `tables/token_mlp_backbones.tex`，相对 `latex_ours/` |
| 2 | 数据集和划分 | 数据清单统计 | `build_token_mlp_dataset_splits.py` |
| 3 | 七模型 Original / Pre MLP | direct_anchor + baseline_token | `build_token_mlp_tables.py` |
| 4 | AlphaRAD/RadZero：原模型、Linear、Global、Post、Pre | supervised + radzero_supervised + placement，复用主结果 | `build_token_mlp_supervised_table.py` |
| 5 | 更大 NIH 测试集 | large_patient_multilabel | `build_token_mlp_large_patient_table.py` |
| 6 | NIH→CheXpert | nih_chexpert_transfer，复用表 5 的 NIH 选中模型 | `build_token_mlp_transfer_table.py` |
| 7 | Aggregation-only 对照 | baseline_token：BiomedCLIP/BioViL-T/MaCo | `build_token_mlp_tables.py` |
| 8 | MaCo pooling × placement | adapter_placement | `build_token_mlp_ablation_tables.py` |
| 9 | 验证集选中 epoch | 表 3 的训练记录 | `build_token_mlp_tables.py` |
| 10–12 | OpenI/CheXpert/NIH 的 Both MLP | both_adapter | `build_token_mlp_ablation_tables.py` |
| 13–15 | OpenI/CheXpert/NIH 的 Gating × Loss | gated_asl | `build_token_mlp_ablation_tables.py` |

下面是按现有 CLI 整理的实验命令，**不是完整历史 shell 会话的逐字记录**。需要先完成各段说明的依赖；日期目录保留论文来源映射。只更改命令中的 `--run-dir` 并不能迁移所有路径，因为部分公共模块、report 和 exporter 还使用固定常量。

### 5.1 表 3/9：AlphaRAD、RadZero 主实验

首先生成冻结编码器缓存。以下为逐分片执行示例；缓存位置会被后续对照引用。原缓存的分片、批次配置以 `cache_*.json` 为准，精确审计时需保持一致。

```bash
CXR_DIRECT=experiments/direct_anchor_classification/runs/20261118
CXR_ALPHA=experiments/direct_anchor_classification/runs/20261118_alpha_token_mlp

for dataset in openi chexpert chestxray14; do
  if [ "$dataset" = openi ]; then
    manifest=RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json
  else
    manifest="data/direct_anchor_classification_20261118/$dataset/dataset.json"
  fi
  for model in alpharad radzero; do
    if [ "$model" = alpharad ]; then
      launcher=AlphaRAD/local_setup/python_local.sh
    else
      launcher=RadZero/scripts/python_local.sh
    fi
    for shard in 0 1 2 3; do
      bash "$launcher" experiments/direct_anchor_classification/cache.py \
        --model "$model" --dataset "$manifest" \
        --output "$CXR_DIRECT/cache/$model/$dataset" \
        --gpu 4 --shards 4 --shard "$shard" --batch-size 12
    done
  done
done
```

核心训练命令：AlphaRAD 的 Pre 叫 `token_mlp`，RadZero 的 Pre 在历史代码中叫 `existing`；不要把 AlphaRAD 的 `existing` 当作 Token MLP。

```bash
for dataset in openi chexpert chestxray14; do
  if [ "$dataset" = openi ]; then
    manifest=RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json
  else
    manifest="data/direct_anchor_classification_20261118/$dataset/dataset.json"
  fi
  # AlphaRAD 原模型，只评估一次。
  bash AlphaRAD/local_setup/python_local.sh AlphaRAD/local_setup/train_direct_anchor.py \
    --dataset "$manifest" --cache "$CXR_DIRECT/cache/alpharad/$dataset" \
    --output "$CXR_DIRECT/results/alpharad/$dataset/seed42" \
    --gpu 4 --seed 42 --methods baseline
  for seed in 42 43 44; do
    bash AlphaRAD/local_setup/python_local.sh AlphaRAD/local_setup/train_direct_anchor.py \
      --dataset "$manifest" --cache "$CXR_DIRECT/cache/alpharad/$dataset" \
      --output "$CXR_ALPHA/results/alpharad/$dataset/seed$seed" \
      --gpu 4 --seed "$seed" --methods token_mlp
    methods=existing
    if [ "$seed" = 42 ]; then methods=baseline,existing; fi
    bash RadZero/scripts/python_local.sh RadZero/scripts/train_direct_anchor.py \
      --dataset "$manifest" --cache "$CXR_DIRECT/cache/radzero/$dataset" \
      --output "$CXR_DIRECT/results/radzero/$dataset/seed$seed" \
      --gpu 4 --seed "$seed" --methods "$methods"
  done
done
```

结果保存为 `results/<model>/<dataset>/seed<seed>/<method>/result.json`、`training.json`、`adapter.pt`、`test_predictions.npz`。表 9 的 epoch 从这里读取。

历史 AlphaRAD 调度器的调用如下，可替代上面的 AlphaRAD Pre 循环。**其最终 report 还要求历史 bias/direct/query-refinement 对照齐全**，仅重跑论文 Pre 不满足整套历史 report：

```bash
bash AlphaRAD/local_setup/python_local.sh experiments/direct_anchor_classification/warm_alpha_cache.py \
  --reference-run "$CXR_DIRECT" --output "$CXR_ALPHA/cache_checksums"
bash AlphaRAD/local_setup/python_local.sh experiments/direct_anchor_classification/run_alpha_token.py \
  --run-dir "$CXR_ALPHA" --reference-run "$CXR_DIRECT" --gpus 4,5,6,7
```

### 5.2 表 3/7/9：其余五模型

前提是三套固定数据清单及模型资源已恢复。`cache.py` 为指定模型缓存全部三个数据集，`train.py` 每次只训练一个模型/数据集/种子。

```bash
CXR_BASE=experiments/baseline_token_classification/runs/20261119
bash AlphaRAD/local_setup/python_local.sh experiments/baseline_token_classification/prepare.py \
  --run-dir "$CXR_BASE"
for model in CARZero KAD BiomedCLIP BioViL-T MaCo; do
  launcher="baselines/$model/local_setup/python_local.sh"
  bash "$launcher" experiments/baseline_token_classification/cache.py \
    --model "$model" --run-dir "$CXR_BASE" --gpu 4
  for dataset in openi chexpert chestxray14; do
    for seed in 42 43 44; do
      bash "$launcher" experiments/baseline_token_classification/train.py \
        --model "$model" --dataset "$dataset" --run-dir "$CXR_BASE" --seed "$seed" --gpu 4
    done
  done
done
bash AlphaRAD/local_setup/python_local.sh experiments/baseline_token_classification/report.py \
  --run-dir "$CXR_BASE" --allow-partial
```

历史默认队列还包含 MedKLIP/MAVL/CheXzero，当前论文只使用上面五个模型，所以这里不调用默认 `run.py`。历史 `report.py` 默认检查全部八模型；五模型子集需 `--allow-partial`，产生的报告会明确标为 partial，不能当作八模型完成证明。论文的七模型表由独立 exporter 选择并核验。

### 5.3 表 4/8：Post MLP 与 MaCo 池化位置对照

前提是 AlphaRAD/KAD/MaCo 的主结果和缓存已完成。每个 `train.py` 任务内部运行三个种子；整个 placement campaign 包括 45 次新增拟合。

```bash
CXR_PLACE=experiments/adapter_placement_classification/runs/20261119
bash AlphaRAD/local_setup/python_local.sh experiments/adapter_placement_classification/prepare.py \
  --run-dir "$CXR_PLACE"
bash AlphaRAD/local_setup/python_local.sh experiments/adapter_placement_classification/run.py \
  --run-dir "$CXR_PLACE" --gpus 4,5,6,7
bash AlphaRAD/local_setup/python_local.sh experiments/adapter_placement_classification/report.py \
  --run-dir "$CXR_PLACE"
```

该实验提供 AlphaRAD/KAD Post，以及 MaCo 的文本引导/原生平均池化 × Pre/Post。RadZero Post 由下一节生成。

### 5.4 表 4：Linear probe、全局适配器和 RadZero Post

历史 AlphaRAD 监督基线 campaign 同时跑 AlphaRAD/KAD/MaCo（54 次新增训练），最终表 4 只取 AlphaRAD。RadZero 扩展增加 27 次训练。各 `train.py` 自动构建所需 global/context 特征并运行三个种子，无需把 `features.py` 当成独立 CLI。

**额外输入要求：**历史 prepare/finalizer 还绑定 RSNA 辅助实验，需要 `experiments/rsna_fixed_patient_cleanup_20261110/corrected_test_manifest.json` 及图像、患者映射/审计资源。RadZero prepare 还读取父实验的 `COMPLETION.json`、`external/PROTOCOL.json`、`cohort_audit.json` 和 `image_hashes.json`。这些资源没有打包，不能仅凭三套主数据直接完成该历史 campaign。

```bash
CXR_SUP=experiments/supervised_baseline_classification/runs/20261120
bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/prepare.py \
  --run-dir "$CXR_SUP"
bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/external_audit.py \
  --run-dir "$CXR_SUP"
bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/run.py \
  --run-dir "$CXR_SUP"
# 包括历史 RSNA 评估与 bootstrap；这些输出不等于当前表 4 的列。
bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/run_additional.py
bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/finalize.py

CXR_RAD=experiments/radzero_supervised_classification/runs/20261120
bash RadZero/scripts/python_local.sh experiments/radzero_supervised_classification/prepare.py \
  --run-dir "$CXR_RAD"
bash RadZero/scripts/python_local.sh experiments/radzero_supervised_classification/run.py \
  --run-dir "$CXR_RAD"
bash RadZero/scripts/python_local.sh experiments/radzero_supervised_classification/finish.py
```

只看单个训练任务如何调用：在相应 `PROTOCOL.json` 已冻结、输入检查已通过时，以下命令各跑一个数据集的全部相关方法与三个种子。它们是总队列中对应任务的替代调用，不需在总队列完成后再次运行。

```bash
bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/train.py \
  --run-dir "$CXR_SUP" --model AlphaRAD --dataset chestxray14 --gpu 4
bash RadZero/scripts/python_local.sh experiments/radzero_supervised_classification/train.py \
  --run-dir "$CXR_RAD" --model RadZero --dataset chestxray14 --gpu 4
```

### 5.5 表 10–12：Both MLP

前提：七模型主结果、placement 对照及冻结缓存齐全。Both 在聚合前后各有独立 MLP，参数量是单个 MLP 的两倍；63 次新增训练。

```bash
CXR_BOTH=experiments/both_adapter_classification/runs/20261119
bash AlphaRAD/local_setup/python_local.sh experiments/both_adapter_classification/prepare.py \
  --run-dir "$CXR_BOTH"
bash AlphaRAD/local_setup/python_local.sh experiments/both_adapter_classification/snapshot.py \
  --run-dir "$CXR_BOTH"
bash AlphaRAD/local_setup/python_local.sh experiments/both_adapter_classification/run.py \
  --run-dir "$CXR_BOTH" --gpus 4,5,6,7
bash AlphaRAD/local_setup/python_local.sh experiments/both_adapter_classification/report.py \
  --run-dir "$CXR_BOTH"
```

### 5.6 表 13–15：Gating × Loss

前提：Both 的 `PROTOCOL.json`、完全一致的缓存和历史比较结果。七模型 × 三数据集 × 三种子 × 四种条件，共 252 次新拟合。

```bash
CXR_GATE=experiments/gated_asl_classification/runs/20261120_recovered
bash AlphaRAD/local_setup/python_local.sh experiments/gated_asl_classification/prepare.py \
  --run-dir "$CXR_GATE"
bash AlphaRAD/local_setup/python_local.sh experiments/gated_asl_classification/snapshot.py \
  --run-dir "$CXR_GATE"
bash AlphaRAD/local_setup/python_local.sh experiments/gated_asl_classification/run.py \
  --run-dir "$CXR_GATE" --gpus 4,5,6,7
bash AlphaRAD/local_setup/python_local.sh experiments/gated_asl_classification/report.py \
  --run-dir "$CXR_GATE"
```

`20261120_recovered` 是论文最终读取的历史目录名；全新运行和历史故障恢复不是同一过程。恢复已有任务需要 `migrate.py` 对源目录及恢复材料做校验，不能对已有 recovered 目录重复 prepare。

### 5.7 表 5：更大 NIH 测试划分

使用同一 NIH 队列，患者级 60/10/30 划分；AlphaRAD/RadZero × 四方法 × 三种子，共 24 次训练。先完成全部拟合和 NIH 验证选点，再锁定模型并评估测试集。

该实验复用早期冻结 token 缓存，需要原缓存哈希记录。`freeze.py` 还要求 `test_checks.json`、`source_integrity.json`、`ENCODER_AUDIT.json`；原 encoder 审计引用的 `encoder_audit/numerical_diagnostics.json` 属于历史辅助记录，也需恢复。下面先准备数据与特征，再在这些审计输入齐全时冻结并运行：

```bash
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/prepare.py --workers 12
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/features.py
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/check.py

# 审计原图重新编码与缓存一致：先构造固定 TRAIN/VAL 抽查批次。
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/prepare_encoder_audit.py
CXR_AUDIT=experiments/large_patient_multilabel/runs/20261121/encoder_audit/matched_batch
bash AlphaRAD/local_setup/python_local.sh experiments/direct_anchor_classification/cache.py \
  --model alpharad --dataset "$CXR_AUDIT/dataset.json" --output "$CXR_AUDIT/alpharad" \
  --gpu 4 --batch-size 12
bash RadZero/scripts/python_local.sh experiments/direct_anchor_classification/cache.py \
  --model radzero --dataset "$CXR_AUDIT/dataset.json" --output "$CXR_AUDIT/radzero" \
  --gpu 4 --batch-size 12
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/audit_encoder.py

bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/freeze.py
bash AlphaRAD/local_setup/python_local.sh experiments/large_patient_multilabel/run.py \
  --run-dir experiments/large_patient_multilabel/runs/20261121
```

`run.py` 负责训练、生成选点锁、测试、患者成对 bootstrap 和 report。输出包括 `summary.csv`、`per_seed_results.csv`、`uncertainty/`、`COMPLETION.json`。

### 5.8 表 6：仅 NIH 训练与验证 → CheXpert 测试

前提：表 5 的训练、NIH 验证选点锁和 checkpoint 全部完成，CheXpert 668 张目标图像与标签清单已恢复。

```bash
bash AlphaRAD/local_setup/python_local.sh experiments/nih_chexpert_transfer/run.py
```

该入口自动准备目标清单、运行检查、缓存目标图像、加载 NIH 选中 checkpoint、评估七个共同标签并汇总 bootstrap。原控制器使用 GPU 4/5，无 `--run-dir` 参数。两模型原模型各一次加 24 个适配器，共 26 次评估，**不在 CheXpert 上训练、调参或选 checkpoint**。

## 6. 将结果导出为论文表格

这些命令是**读取已完成实验、核验并生成表格**，不会训练模型。前提是上文的完整运行资源已恢复到 exporter 预期路径。原实现还检查协议、完成记录、预测和来源哈希；其中 NIH/transfer exporter 要求额外的历史 `FINAL_VERIFICATION.json`。只有随包 CSV 时，不满足全套 exporter 的输入条件。

```bash
# 表 2：数据集和划分。
bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_dataset_splits.py
# 表 3、7、9：核验 93 个保存的评估记录及逐样本预测。
bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_tables.py
# 表 4。
bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_supervised_table.py
# 表 5、6。
bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_large_patient_table.py
bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_transfer_table.py
# 表 8、10–15。
bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_ablation_tables.py
```

主表 exporter 还使用 `experiments/classification_results_summary_20261119/all_methods.csv` 及其固定来源记录。换数据、换路径或重新跑出不同数字时，需要为新实验更新来源映射并重新核验，而不是绕过历史哈希检查。`FINAL_VERIFICATION.json`、人工审计记录和历史 catalog 的完整生成流程尚未全部封装为可移植 CLI，这是当前代码快照的复现边界。

已生成文件保存在 [`latex_ours/tables/`](latex_ours/tables/)，三种子结果、汇总 CSV、置信区间和 provenance 保存在 [`latex_ours/supplement_data/`](latex_ours/supplement_data/)，可以直接查看。

主图的入口是 `latex_ours/scripts/export_token_mlp_figure_assets.py` 和 `draw_token_mlp_pipeline.py`；需要真实 NIH 图像、冻结 tokens、选中 checkpoint 和相应绘图环境。图像/NPZ/渲染资产未随包提供，不能仅凭本包完整编译图 1。

## 7. 核心代码和日志在哪里

| 内容 | 文件 |
|---|---|
| 共享残差 MLP、指标、冻结工具 | `experiments/baseline_token_classification/common.py` |
| AlphaRAD 接入 | `AlphaRAD/local_setup/token_adapter_model.py` |
| RadZero 残差适配器 | `RadZero/scripts/adapter_model.py` |
| AlphaRAD/RadZero 原生固定聚合与打分 | `experiments/direct_anchor_classification/model.py` |
| 其余五模型接入 | `baselines/<model>/local_setup/token_mlp_adapter.py` 及 `adapter.py` |
| Linear probe / 全局适配器 | `experiments/supervised_baseline_classification/components.py` |
| Gating / ASL | `experiments/gated_asl_classification/components.py` |
| 患者成对 bootstrap | `experiments/classification_uncertainty/bootstrap.py` |

AlphaRAD 的 Token MLP 复用 RadZero 残差模块，因此不能任意拆开两个目录。更细的代码说明见 [`CODE_INDEX.csv`](CODE_INDEX.csv) 与 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

随包保存 **399 份原始 `.log`，共 1,824,306 字节**，位于 `logs/<实验目录>/<运行日期>/<原目录内相对路径>`。例如：

```text
logs/direct_anchor_classification/20261118/
logs/direct_anchor_classification/20261118_alpha_token_mlp/
logs/baseline_token_classification/20261119/
logs/adapter_placement_classification/20261119/
logs/supervised_baseline_classification/20261120/
logs/radzero_supervised_classification/20261120/
logs/both_adapter_classification/20261119/
logs/gated_asl_classification/20261120_recovered/
logs/large_patient_multilabel/20261121/
logs/nih_chexpert_transfer/20261121/
```

逐文件来源、大小、SHA-256 见 [`LOG_INDEX.csv`](LOG_INDEX.csv) 和 [`PUBLISHED_LOGS.json`](PUBLISHED_LOGS.json)。日志包含缓存、调度、失败/恢复及历史辅助任务；日志数量不是训练次数，也不代表每份日志都是最终成功结果。完整 run、逐 epoch JSON、预测和 checkpoint 不包含在日志包中。

## 8. 核验与上传 GitHub

在**未下载模型/数据的源码发布目录**运行：

```bash
python3 tools/verify_snapshot.py
sha256sum -c FILE_CHECKSUMS.sha256
```

核验工具检查原源码复制哈希、399 份日志、Python AST、Shell 语法及显式 `experiments.*` 依赖；结果写入 `VALIDATION.json`。它不会启动训练、安装模型环境或重算论文指标。新增环境/工作目录工具还做了无副作用的命令预览检查，但没有执行全套下载和安装。

上传以本仓库目录为根，不要上传整个 `medical/` 工作区。`.gitignore` 已排除数据、权重、训练目录、环境和缓存，并允许 `logs/` 下已登记日志。现有各组件 `LICENSE` 属于其上游；本快照尚未替作者选择原创代码的根目录许可证。

整理工具：`tools/build_source_snapshot.py` 从原工作区复制明确列出的源码和汇总文件；`tools/package_logs.py` 归档原始日志；它们不会自动重建本 README、新增环境配方及全部发布文档，更新发布包时应一并保留。
