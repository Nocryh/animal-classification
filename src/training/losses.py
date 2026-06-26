"""
损失函数模块
============
- CrossEntropy: 标准交叉熵
- Focal Loss: 聚焦困难样本，缓解类别不平衡
- Label Smoothing CrossEntropy: 标签平滑，防止过拟合
- Combined Loss: Focal + Label Smoothing 融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., 2017)

    核心思想: 对易分类样本降低权重，让模型聚焦于困难样本。
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: 聚焦参数，越大越关注困难样本 (default: 2.0)
        alpha: 类别权重，可为 float 或 tensor of shape (num_classes,)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none",
                                  weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing 交叉熵损失

    将 one-hot 标签替换为平滑分布:
        y_smooth = (1 - smoothing) * y_onehot + smoothing / num_classes

    效果: 防止模型对预测过于自信，提升泛化能力。

    Args:
        smoothing: 平滑因子 (default: 0.1)
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)

        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = torch.sum(-smooth_labels * log_probs, dim=-1)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Focal Loss + Label Smoothing 融合损失

    将 Focal Loss 的思想与 Label Smoothing 结合：
    1. 平滑标签（防止过拟合）
    2. 对平滑后的标签应用 focal 权重（聚焦困难样本）

    Args:
        gamma: focal 聚焦参数
        smoothing: 标签平滑因子
        alpha: 类别权重（可选）
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1,
                 alpha=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)

        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Focal weight: (1 - p_t)^gamma
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma

        loss = -focal_weight * smooth_labels * log_probs
        loss = loss.sum(dim=-1)

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                loss = loss * self.alpha[targets]

        return loss.mean()


def compute_class_weights(dataloader, num_classes: int, device: torch.device = None) -> torch.Tensor:
    """
    从训练集自动计算平衡的类别权重（逆频率）。

    Returns:
        weight tensor of shape (num_classes,)
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in dataloader:
        for label in labels:
            class_counts[label] += 1

    # 逆频率加权
    class_counts = class_counts.clamp(min=1)  # 防止除以 0
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes  # 归一化

    if device is not None:
        weights = weights.to(device)

    return weights


def get_loss_function(config: dict, dataloader=None, device=None) -> nn.Module:
    """
    根据配置创建损失函数。

    Args:
        config: 训练配置字典
        dataloader: (可选) 用于自动计算类别权重的 DataLoader
        device: (可选) 计算设备

    Returns:
        nn.Module 损失函数
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy")

    # 类别权重
    class_weights = loss_cfg.get("class_weights")
    if class_weights is None and dataloader is not None:
        num_classes = config.get("data", {}).get("num_classes", 90)
        class_weights = compute_class_weights(dataloader, num_classes, device)

    if loss_type == "cross_entropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        alpha = class_weights
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=alpha)

    elif loss_type == "label_smoothing":
        smoothing = loss_cfg.get("label_smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    elif loss_type == "combined":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        smoothing = loss_cfg.get("label_smoothing", 0.1)
        alpha = class_weights
        return CombinedLoss(gamma=gamma, smoothing=smoothing, alpha=alpha)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
