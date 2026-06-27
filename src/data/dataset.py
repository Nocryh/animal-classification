"""
数据加载与划分
==============
- 从 ImageFolder 格式目录加载数据
- 分层划分 train/val/test（保证每类样本比例一致）
- 数据完整性验证（训练前扫描）
- 类别分布统计
"""
import os
from pathlib import Path
import numpy as np
from collections import Counter
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from PIL import Image


def load_class_names(path: str) -> list:
    """从文本文件加载类别名称"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stratified_split(dataset: Dataset, val_split: float = 0.2,
                     test_split: float = 0.0, seed: int = 42) -> tuple:
    """
    分层划分数据集，保证每类在 train/val/test 中的比例一致。

    Args:
        dataset: ImageFolder dataset
        val_split: 验证集比例
        test_split: 测试集比例（0 表示不划分独立测试集）
        seed: 随机种子

    Returns:
        (train_indices, val_indices, test_indices)
    """
    labels = [label for _, label in dataset.samples]
    num_classes = len(dataset.classes)
    rng = np.random.RandomState(seed)

    train_idx, val_idx, test_idx = [], [], []

    for cls in range(num_classes):
        cls_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == cls]
        n = len(cls_indices)
        perm = rng.permutation(n)

        n_val = max(1, int(n * val_split))
        n_test = max(0, int(n * test_split))
        n_train = n - n_val - n_test

        val_idx.extend([cls_indices[i] for i in perm[:n_val]])
        if n_test > 0:
            test_idx.extend([cls_indices[i] for i in perm[n_val:n_val + n_test]])
        train_idx.extend([cls_indices[i] for i in perm[n_val + n_test:]])

    return train_idx, val_idx, test_idx


def compute_class_distribution(dataset: Dataset, class_names: list = None):
    """统计各类别样本数量"""
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    dist = {}
    for cls_idx, count in sorted(counter.items()):
        name = class_names[cls_idx] if class_names else str(cls_idx)
        dist[name] = count
    return dist


def create_dataloaders(data_dir: str, train_transform, val_transform,
                       batch_size: int = 32, num_workers: int = 2,
                       val_split: float = 0.2, test_split: float = 0.0,
                       seed: int = 42, pin_memory: bool = True) -> dict:
    """
    一站式创建 DataLoader。

    Returns:
        dict with keys: train_loader, val_loader, test_loader (or None),
                        class_names, train_size, val_size, test_size, class_distribution
    """
    dataset = datasets.ImageFolder(data_dir)

    train_idx, val_idx, test_idx = stratified_split(
        dataset, val_split=val_split, test_split=test_split, seed=seed
    )

    # 创建带有对应 transform 的子集
    train_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=train_transform), train_idx
    )
    val_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=val_transform), val_idx
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = None
    if test_idx:
        test_dataset = Subset(
            datasets.ImageFolder(data_dir, transform=val_transform), test_idx
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

    class_dist = compute_class_distribution(dataset, dataset.classes)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": dataset.classes,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx) if test_idx else 0,
        "class_distribution": class_dist,
    }


def validate_dataset(data_dir: str, img_size: int = 224) -> dict:
    """
    训练前扫描数据集，检测损坏/异常文件。

    检查项:
    - 文件是否能被 PIL 正常打开
    - 图像 mode 是否为 RGB（灰度/ RGBA 会转换）
    - 图像尺寸是否过小（< img_size）
    - 文件扩展名是否为常见图像格式

    Returns:
        {total, valid, corrupt, warnings, issues: [(path, reason), ...]}
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"total": 0, "valid": 0, "corrupt": 0, "warnings": [],
                "issues": [(str(data_path), "Directory not found")]}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    total = 0
    valid = 0
    corrupt = []
    warnings = []

    for ext in valid_extensions:
        for img_path in data_path.rglob(f"*{ext}"):
            # 跳过非文件（符号链接等）
            if not img_path.is_file():
                continue
            total += 1

            try:
                with Image.open(img_path) as img:
                    # 检查是否能正确加载
                    img.verify()
            except Exception:
                # verify 后文件指针状态异常，需要重新打开
                corrupt.append((str(img_path), "Corrupt file (cannot verify)"))
                continue

            try:
                with Image.open(img_path) as img:
                    mode = img.mode
                    w, h = img.size

                    if mode not in ("RGB", "RGBA", "L", "P", "CMYK"):
                        warnings.append((str(img_path), f"Unusual mode: {mode}"))

                    if w < img_size or h < img_size:
                        warnings.append(
                            (str(img_path), f"Small image: {w}x{h} (min: {img_size})")
                        )

                    if w < 10 or h < 10:
                        corrupt.append((str(img_path), f"Tiny image: {w}x{h}"))
                        continue

                valid += 1

            except Exception as e:
                corrupt.append((str(img_path), f"Open error: {e}"))

    # 汇总报告
    report = {
        "total": total,
        "valid": valid,
        "corrupt": len(corrupt),
        "warnings": len(warnings),
        "corrupt_files": corrupt[:50],    # 最多展示前 50 个
        "warn_files": warnings[:50],
    }

    # 打印摘要
    if total == 0:
        print(f"[DATA] No images found in {data_dir}. Check the path.")
    else:
        status = "PASS" if len(corrupt) == 0 else f"WARN ({len(corrupt)} corrupt)"
        print(f"[DATA] Scanned {total} images: {valid} valid, "
              f"{len(corrupt)} corrupt, {len(warnings)} warnings — {status}")

        if corrupt:
            print(f"[DATA] Corrupt files (first 10):")
            for path, reason in corrupt[:10]:
                print(f"  - {path}\n    Reason: {reason}")
            if len(corrupt) > 10:
                print(f"  ... and {len(corrupt) - 10} more")

        if warnings and not corrupt:
            print(f"[DATA] Warnings (first 5):")
            for path, reason in warnings[:5]:
                print(f"  - {path}\n    Reason: {reason}")

    return report
