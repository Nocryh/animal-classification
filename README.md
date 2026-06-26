<div align="center">

# 🐾 Animal Classification

**基于 ResNet50 的 90 类动物图像分类模型**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset14-005ced.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-ResNet50-orange.svg)](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

</div>

## 📋 项目简介

使用 **ResNet50** 在 ImageNet 预训练权重上进行迁移学习，实现对 **90 种动物** 的高精度图像分类。模型导出为 **ONNX** 格式，支持跨平台高性能推理部署。

- **验证准确率**: 91.67%
- **输入尺寸**: 224×224 RGB
- **模型大小**: ~95 MB (ONNX / PyTorch)
- **推理框架**: ONNX Runtime

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/<your-username>/animal-classification.git
cd animal-classification

# 安装依赖
pip install -r requirements.txt
```

### 一键推理

```bash
python inference.py <图片路径>

# 示例
python inference.py examples/cat.jpg
```

输出示例：

```
Image: examples/cat.jpg
----------------------------------------
   1. cat                  99.93%  ████████████████████████████████████████
   2. tiger                0.02%
   3. hamster              0.01%
   4. possum               0.00%
   5. rat                  0.00%
```

### 在 Python 代码中使用

```python
from inference import AnimalClassifier

classifier = AnimalClassifier()
results = classifier.predict("path/to/image.jpg")

for rank, (name, prob) in enumerate(results, 1):
    print(f"{rank}. {name}: {prob:.2%}")
```

## 🏋️ 训练自己的模型

### 准备数据

按以下结构组织数据集：

```
data/
├── classes.txt          # 类别名称，每行一个
└── animals/             # ImageFolder 格式
    ├── cat/
    │   ├── cat_001.jpg
    │   └── cat_002.jpg
    ├── dog/
    │   └── ...
    └── ...
```

### 开始训练

```bash
# 基本训练
python train.py --data_dir ./data/animals --classes_file ./data/classes.txt

# 自定义参数
python train.py \
    --data_dir ./data/animals \
    --classes_file ./data/classes.txt \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --num_classes 90
```

### 导出 ONNX（独立使用）

```bash
python export_onnx.py --model_path animal_classifier.pth --classes_file animal_classifier_classes.txt
```

## 📊 模型信息

| 属性 | 值 |
|------|-----|
| 基础架构 | ResNet50 (ImageNet 预训练) |
| 分类头 | 单层 Linear (2048→90) |
| 输入 | 224×224×3 (RGB, 归一化) |
| 输出 | 90 类 logits |
| ONNX Opset | 14 |
| 动态 Batch | ✅ 支持 |
| 数据增强 | RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter |
| 优化器 | AdamW + CosineAnnealingLR |
| 验证准确率 | **91.67%** |

## 🐕 支持的 90 种动物

<details>
<summary>点击展开完整列表</summary>

antelope, badger, bat, bear, bee, beetle, bison, boar, butterfly, cat, caterpillar, chimpanzee, cockroach, cow, coyote, crab, crow, deer, dog, dolphin, donkey, dragonfly, duck, eagle, elephant, flamingo, fly, fox, goat, goldfish, goose, gorilla, grasshopper, hamster, hare, hedgehog, hippopotamus, hornbill, horse, hummingbird, hyena, jellyfish, kangaroo, koala, ladybugs, leopard, lion, lizard, lobster, mosquito, moth, mouse, octopus, okapi, orangutan, otter, owl, ox, oyster, panda, parrot, pelecaniformes, penguin, pig, pigeon, porcupine, possum, raccoon, rat, reindeer, rhinoceros, sandpiper, seahorse, seal, shark, sheep, snake, sparrow, squid, squirrel, starfish, swan, tiger, turkey, turtle, whale, wolf, wombat, woodpecker, zebra

</details>

## 📁 项目结构

```
animal-classification/
├── train.py                        # 训练脚本（支持命令行参数）
├── inference.py                    # ONNX 推理脚本 + AnimalClassifier 类
├── export_onnx.py                  # 独立 ONNX 导出脚本
├── animal_classifier.onnx          # 训练好的 ONNX 模型 (Git LFS)
├── animal_classifier.pth           # PyTorch 模型权重 (Git LFS)
├── animal_classifier_classes.txt   # 90 类动物名称
├── requirements.txt                # Python 依赖
├── .gitignore
├── .gitattributes                  # Git LFS 配置
├── LICENSE                         # MIT License
└── README.md
```

## 🔧 命令行参数

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data/animals` | 数据集目录（ImageFolder 格式） |
| `--classes_file` | `./data/classes.txt` | 类别名称文件 |
| `--output_dir` | 脚本所在目录 | 模型输出目录 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--num_classes` | 90 | 类别数量 |
| `--val_split` | 0.2 | 验证集比例 |
| `--seed` | 42 | 随机种子 |

### inference.py

```bash
python inference.py <image_path>
```

### export_onnx.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `animal_classifier.pth` | PyTorch 权重路径 |
| `--classes_file` | `animal_classifier_classes.txt` | 类别名称文件 |
| `--output_dir` | 脚本所在目录 | 输出目录 |
| `--num_classes` | 90 | 类别数量 |
| `--img_size` | 224 | 输入图像尺寸 |
| `--opset` | 14 | ONNX opset 版本 |

## 🧪 推理测试结果

| 测试图片 | 预测结果 | 置信度 |
|---------|---------|--------|
| 🐱 cat | cat | 99.93% |
| 🦁 lion | lion | 99.98% |
| 🐼 panda | panda | 99.52% |
| 🦅 eagle | eagle | 99.37% |

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) — 深度学习框架
- [ONNX Runtime](https://onnxruntime.ai/) — 高性能推理引擎
- [torchvision](https://pytorch.org/vision/) — 预训练模型
