# Image Manipulation Localization Project

本项目是一个面向图像局部篡改检测 / image manipulation localization 的 PyTorch 工程骨架。目标是输入一张图像，输出像素级篡改概率图和二值 mask，并逐步扩展为支持主实验、消融实验、跨数据集测试、鲁棒性测试和可视化分析的完整科研项目。

当前阶段只搭建基础目录、配置、轻量可导入模块和运行入口，不一次性实现复杂训练逻辑或完整论文模型。

## 研究目标

- 像素级定位图像局部篡改区域。
- 兼顾 RGB 语义线索、噪声/残差/高频线索、局部纹理边界线索和全局依赖建模。
- 支持 CASIA、Columbia、COVERAGE、NIST16、IMD2020 等常见图像篡改检测数据集。
- 支持跨数据集泛化评估、鲁棒性测试和可解释可视化。

## 当前模型骨架

当前模型位于 `models/`，采用轻量版 RFB-TraceFormer 思路：

- `RGB main branch`：提取基础局部纹理特征。
- `frequency branch`：使用固定高通滤波生成残差/高频输入，再提取辅助特征。
- `global block`：预留 Mamba/Transformer 全局建模接口，当前为轻量可运行占位模块。
- `fusion`：融合 RGB 与高频特征。
- `decoder`：多尺度解码并输出 mask logits 与 boundary logits。

后续应逐步替换为更强的 Transformer/Mamba backbone、SRM 残差分支、多尺度频域特征和边界细化模块。

## 目录结构

```text
configs/
  train_casia.yaml
  test_cross_dataset.yaml
  ablation.yaml
datasets/
  tamper_dataset.py
  transforms.py
models/
  tamper_net.py
  backbone.py
  frequency_branch.py
  mamba_block.py
  fusion.py
  decoder.py
losses/
  losses.py
metrics/
  metrics.py
utils/
  seed.py
  logger.py
  checkpoint.py
  visualization.py
  config.py
tools/
  train.py
  test.py
  infer.py
  evaluate_robustness.py
  run_ablation.py
scripts/
  train_local.sh
  train_server.sh
  test_server.sh
  ablation_server.sh
```

## 安装依赖

建议先创建独立 Python 环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

依赖说明：

- `torch` / `torchvision`：模型、训练和图像张量处理。
- `PyYAML`：读取 YAML 配置。
- `Pillow` / `opencv-python`：图像读取、mask 处理和后续鲁棒性退化。
- `numpy` / `scikit-learn`：指标计算。
- `tqdm`：训练和评估进度条。
- `matplotlib` / `tensorboard`：曲线、日志和可视化。

## 基础运行检查

当前训练脚本是骨架入口，用于检查配置、随机种子、模型构建和 import 是否正常：

```bash
python -m tools.train --config configs/train_casia.yaml
```

Full training:

```bash
python -m tools.train --config configs/train_casia.yaml
```

Resume training:

```bash
python -m tools.train --config configs/train_casia.yaml --resume outputs/train_casia/checkpoints/last.pth
```

Two-GPU server training:

```bash
torchrun --nproc_per_node=2 -m tools.train --config configs/train_casia_manifest.yaml
```

Training outputs are saved under `experiment.output_dir`:

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `logs/train.log`
- `logs/tensorboard/`
- `curves/metrics.csv`
- `curves/*.png`
- `visualizations/epoch_xxxx/*_prob.png`
- `visualizations/epoch_xxxx/*_overlay.png`

模型 forward、输出 shape 和 backward 的最小检查：

```bash
python -m models.quick_test_model
```

数据读取检查入口会按 YAML 中的 `image_dir` 和 `mask_dir` 构建 `TamperDataset`，读取一个 batch 并打印 shape：

```bash
python -m tools.check_dataloader --config configs/train_casia.yaml --split train --batch-size 2
```

Manifest dataset check:

```bash
python -m tools.check_dataloader --config configs/train_casia_manifest.yaml --split train --batch-size 2
python -m tools.check_dataloader --config configs/test_cross_dataset_manifest.yaml --split test --dataset NIST16 --batch-size 1
```

跨数据集测试入口：

```bash
python -m tools.test --config configs/test_cross_dataset.yaml
```

消融实验入口：

```bash
python -m tools.run_ablation --config configs/ablation.yaml
```

鲁棒性测试入口：

```bash
python -m tools.evaluate_robustness --config configs/test_cross_dataset.yaml
```

推理入口：

```bash
python -m tools.infer --config configs/train_casia.yaml --image path/to/image.png
```

## 数据集配置

所有路径都必须写在 YAML 中，不允许写死在代码里。当前数据集读取优先使用 `image_dir` 和 `mask_dir`，目录建议整理为：

```text
data/CASIA/
  train/
    images/
      sample_0001.jpg
      sample_0002.png
    masks/
      sample_0001.png
      sample_0002.png
  val/
    images/
      sample_1001.jpg
    masks/
      sample_1001.png
```

图像支持扩展名：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`。mask 会按文件名 stem 自动匹配，例如 `sample_0001.jpg` 可以匹配 `sample_0001.png`、`sample_0001_mask.png`、`sample_0001_gt.png` 或 `sample_0001_label.png`。

配置示例：

```yaml
data:
  train:
    image_dir: data/CASIA/train/images
    mask_dir: data/CASIA/train/masks
    recursive: false
    strict_pairs: true
    mask_suffixes: ["", "_mask", "_gt", "_label"]
```

真实训练前，需要把这些路径替换为本地或服务器上的数据集路径。

Manifest format is also supported. Each line should use:

```text
image_path # mask_path # class_id
```

For authentic images, use `class_id=0` or `mask_path=null`. If `include_authentic: true`, the dataset will generate an all-zero mask for those samples. If `include_authentic: false`, authentic samples are skipped.

```yaml
data:
  train:
    manifest_files:
      - /data1/data/local/CASIA-D/authentic_img_mask_cls.txt
      - /data1/data/local/CASIA-D/splice_img_mask_cls.txt
    manifest_separator: " # "
    include_authentic: true
    skip_first: 200
```

## 下一步开发顺序

建议下一步优先实现 `datasets/tamper_dataset.py` 的真实数据读取和配对校验，并配套一个最小数据样例检查命令。原因是图像篡改定位任务中，image/mask 对齐错误会直接导致后续模型、loss 和指标全部失真。
