"""
数据增强模块
============
- RandAugment: 随机选择增强操作，比固定 pipeline 更鲁棒
- MixUp: 样本线性混合
- CutMix: 区域级混合
- 用于训练和验证的双 pipeline
"""
import numpy as np
import random

import torch
import torch.nn.functional as F
from torchvision import transforms


# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ==============================================================================
# RandAugment 操作池
# ==============================================================================
# 参考: RandAugment: Practical automated data augmentation
# 针对动物分类特点：保留纹理增强，减少过度几何变换

def _shear_x(img, magnitude):
    """水平剪切"""
    return transforms.functional.affine(
        img, angle=0, translate=[0, 0], scale=1.0, shear=[magnitude, 0]
    )


def _shear_y(img, magnitude):
    """垂直剪切"""
    return transforms.functional.affine(
        img, angle=0, translate=[0, 0], scale=1.0, shear=[0, magnitude]
    )


def _translate_x(img, magnitude):
    """水平平移"""
    return transforms.functional.affine(
        img, angle=0, translate=[int(magnitude), 0], scale=1.0, shear=[0, 0]
    )


def _translate_y(img, magnitude):
    """垂直平移"""
    return transforms.functional.affine(
        img, angle=0, translate=[0, int(magnitude)], scale=1.0, shear=[0, 0]
    )


def _rotate(img, magnitude):
    """旋转"""
    return transforms.functional.rotate(img, magnitude)


def _brightness(img, magnitude):
    """亮度调整"""
    return transforms.functional.adjust_brightness(img, 1.0 + magnitude / 100)


def _contrast(img, magnitude):
    """对比度调整"""
    return transforms.functional.adjust_contrast(img, 1.0 + magnitude / 100)


def _sharpness(img, magnitude):
    """锐度调整"""
    return transforms.functional.adjust_sharpness(img, 1.0 + magnitude / 100)


def _posterize(img, magnitude):
    """色调分离"""
    bits = max(1, 8 - int(magnitude / 2))
    return transforms.functional.posterize(img, bits)


def _solarize(img, magnitude):
    """过度曝光"""
    threshold = 255 - magnitude
    return transforms.functional.solarize(img, threshold)


def _autocontrast(img, _):
    """自动对比度"""
    return transforms.functional.autocontrast(img)


def _equalize(img, _):
    """直方图均衡化"""
    return transforms.functional.equalize(img)


def _identity(img, _):
    """无操作"""
    return img


# RandAugment 操作池: (操作函数, 幅度范围)
# 针对动物分类优化：侧重纹理和光照变化，减少过度几何变形
RANDAUGMENT_OPS = [
    (_identity,         (0, 0)),
    (_autocontrast,     (0, 0)),
    (_equalize,         (0, 0)),
    (_rotate,           (-30, 30)),
    (_shear_x,          (-15, 15)),
    (_shear_y,          (-15, 15)),
    (_translate_x,      (-20, 20)),
    (_translate_y,      (-20, 20)),
    (_brightness,       (-30, 30)),
    (_contrast,         (-30, 30)),
    (_sharpness,        (-10, 10)),
    (_posterize,        (0, 8)),
    (_solarize,         (0, 200)),
]


class RandAugment:
    """
    RandAugment 变换（可 pickle，兼容 Windows multiprocessing）。

    参考: RandAugment: Practical automated data augmentation (Cubuk et al., 2020)
    """

    def __init__(self, num_ops: int = 2, magnitude: int = 9):
        self.num_ops = num_ops
        self.magnitude = magnitude

    def __call__(self, img):
        return _apply_randaugment(img, self.num_ops, self.magnitude)


def _apply_randaugment(img, num_ops: int = 2, magnitude: int = 9):
    """
    应用 RandAugment。

    Args:
        img: PIL Image
        num_ops: 每次随机选择的操作数
        magnitude: 全局增强幅度 (1-15)

    Returns:
        PIL Image
    """
    if num_ops <= 0:
        return img

    ops = random.sample(RANDAUGMENT_OPS, min(num_ops, len(RANDAUGMENT_OPS)))

    for op_fn, (min_mag, max_mag) in ops:
        # 将全局 magnitude 映射到操作的幅度范围
        if max_mag == 0:
            mag = 0
        else:
            mag = (magnitude / 15.0) * (max_mag - min_mag) + min_mag
        img = op_fn(img, mag)

    return img


# ==============================================================================
# MixUp 实现
# ==============================================================================

def mixup_data(x, y, alpha: float = 0.2):
    """
    MixUp 数据增强。

    Args:
        x: (B, C, H, W) 输入
        y: (B,) 或 (B, num_classes) 标签
        alpha: Beta 分布参数

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # 处理 one-hot 和 index 两种标签
    if y.dim() == 1:
        y_a, y_b = y, y[index]
    else:
        y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# 数据增强 Pipeline
# ==============================================================================

def get_train_transforms(img_size: int = 224, config: dict = None):
    """
    构建训练数据增强 pipeline。

    增强策略（针对动物分类优化）：
    1. RandomResizedCrop — 模拟不同尺度和视角
    2. RandAugment — 随机选择增强操作，避免过拟合固定模式
    3. RandomHorizontalFlip — 动物左右对称性
    4. 归一化（ImageNet 统计量）
    """
    if config is None:
        config = {}

    pipeline = []

    # 基础裁剪 + 翻转（始终启用）
    pipeline.append(transforms.RandomResizedCrop(
        img_size, scale=config.get("random_crop_scale", [0.6, 1.0])
    ))
    pipeline.append(transforms.RandomHorizontalFlip(
        p=config.get("random_horizontal_flip", 0.5)
    ))

    # RandAugment
    if config.get("randaugment", True):
        pipeline.append(RandAugment(
            num_ops=config.get("ra_num_ops", 2),
            magnitude=config.get("ra_magnitude", 9),
        ))
    else:
        # 回退到传统增强
        pipeline.append(transforms.RandomRotation(
            config.get("random_rotation", 20)
        ))
        cj = config.get("color_jitter", [0.2, 0.2, 0.2, 0.1])
        pipeline.append(transforms.ColorJitter(
            brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]
        ))

    # 固定结尾
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(pipeline)


def get_val_transforms(img_size: int = 224):
    """构建验证/测试数据变换 pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
