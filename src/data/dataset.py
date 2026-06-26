"""
数据加载与划分
==============
- 从 ImageFolder 格式目录加载数据
- 分层划分 train/val/test（保证每类样本比例一致）
- 支持类别分布统计
"""
import os
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets


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
