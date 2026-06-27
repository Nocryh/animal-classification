"""
模型工厂
========
统一接口创建 ResNet50 / EfficientNetV2 / ConvNeXt，输出各模型统计信息。
"""
import time
import torch
import torch.nn as nn
from torchvision import models


def gem_pooling(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Generalized Mean Pooling (GeM)
    比 AdaptiveAvgPool2d 更灵活，p=1 等价于平均池化，p→∞ 等价于最大池化
    """
    x = x.clamp(min=eps)
    h, w = x.shape[2], x.shape[3]
    return nn.functional.avg_pool2d(x.pow(p), (h, w)).pow(1.0 / p)


class GeMPool(nn.Module):
    """GeM Pooling 层"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_pooling(x, self.p, self.eps)


class ClassificationHead(nn.Module):
    """
    自定义分类头: GeM Pooling + Dropout + Linear
    比 ResNet 原生的 AdaptiveAvgPool2d + Linear 更现代化
    """
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3,
                 pool_type: str = "gem"):
        super().__init__()

        if pool_type == "gem":
            self.pool = GeMPool(p=3.0)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def create_model(architecture: str = "resnet50", num_classes: int = 90,
                 pretrained: bool = True, dropout: float = 0.3,
                 pool_type: str = "gem") -> tuple:
    """
    创建模型并返回 (model, model_info)。

    支持的架构:
        - resnet50:     ResNet50 (2016, 基线)
        - efficientnetv2_s: EfficientNetV2-S (2021, 效率优先)
        - convnext_tiny:    ConvNeXt-Tiny (2022, 现代化 ConvNet)

    Returns:
        (model, model_info_dict)
    """
    architecture = architecture.lower()

    if architecture == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        # avgpool + flatten 在 forward 里是硬编码的，分类头收到 2D 向量
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif architecture == "efficientnetv2_s":
        model = models.efficientnet_v2_s(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        in_features = model.classifier[1].in_features
        # 同样有 avgpool + flatten 在前，分类头收到 2D 向量
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        # ConvNeXt classifier: [LayerNorm, Flatten, Linear]
        # 只替换最后的 Linear 层，保持 LayerNorm 权重不变
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        # 在 Flatten 后插入 Dropout（通过修改 forward 更简单：直接加在最后）
        # 用 Sequential 重新包装以加入 Dropout
        model.classifier = nn.Sequential(
            model.classifier[0],   # LayerNorm (保持预训练权重)
            model.classifier[1],   # Flatten
            nn.Dropout(dropout),
            model.classifier[2],   # new Linear
        )

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Choose from: resnet50, efficientnetv2_s, convnext_tiny"
        )

    # 收集模型信息
    info = compute_model_info(model, architecture)

    return model, info


def compute_model_info(model: nn.Module, architecture: str) -> dict:
    """计算模型参数、FLOPs、推理速度等统计信息"""
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "architecture": architecture,
        "total_params": num_params,
        "trainable_params": trainable_params,
        "params_millions": round(num_params / 1e6, 2),
    }

    return info


def benchmark_inference_speed(model: nn.Module, device: torch.device,
                              input_size: tuple = (1, 3, 224, 224),
                              warmup: int = 10, repeats: int = 50) -> dict:
    """测量模型推理速度（FPS）"""
    model.eval()
    dummy_input = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    fps = repeats / elapsed
    latency_ms = (elapsed / repeats) * 1000

    return {
        "fps": round(fps, 2),
        "latency_ms": round(latency_ms, 2),
        "batch_size": input_size[0],
        "repeats": repeats,
    }
