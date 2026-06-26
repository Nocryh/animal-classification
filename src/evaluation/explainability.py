"""
模型可解释性模块
================
- Grad-CAM: 类别激活映射，可视化模型关注区域
- Grad-CAM++: 改进版，定位更精细
- 特征嵌入 t-SNE 可视化

参考:
    Grad-CAM: Selvaraju et al., 2017
    Grad-CAM++: Chattopadhyay et al., 2018
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


# ImageNet 反标准化参数
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """反标准化 ImageNet 归一化"""
    if tensor.dim() == 4:
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(tensor.device)
        std  = IMAGENET_STD.view(1, 3, 1, 1).to(tensor.device)
    else:
        mean = IMAGENET_MEAN.view(3, 1, 1).to(tensor.device)
        std  = IMAGENET_STD.view(3, 1, 1).to(tensor.device)

    return tensor * std + mean


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping

    通过目标类别对最后一层卷积特征图的梯度，生成热力图，
    可视化模型"看"图像的哪个部分来做出预测。

    Args:
        model: PyTorch 模型
        target_layer: 要可视化的卷积层
    """

    def __init__(self, model: torch.nn.Module, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册 hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        生成 Grad-CAM 热力图。

        Args:
            input_tensor: (1, C, H, W) 输入图像
            class_idx: 目标类别索引，None 则使用模型预测的最高分类别

        Returns:
            (H, W) 热力图，值域 [0, 1]
        """
        # Forward
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算 Grad-CAM
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        # 全局平均池化梯度得到权重
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # 加权组合激活图
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU + 归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 上采样到输入尺寸
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        return cam


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: 改进的 Grad-CAM

    使用梯度的高阶信息，对多目标场景定位更精确。
    """

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        # Grad-CAM++ 权重计算
        grad_power_2 = gradients ** 2
        grad_power_3 = grad_power_2 * gradients

        sum_activations = activations.sum(dim=[2, 3], keepdim=True)
        eps = 1e-7

        # alpha coefficients
        alpha_num = grad_power_2
        alpha_denom = 2 * grad_power_2 + sum_activations * grad_power_3 + eps
        alpha = alpha_num / alpha_denom

        # 正梯度权重
        positive_gradients = F.relu(gradients)
        weights = (alpha * positive_gradients).sum(dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    将热力图叠加到原始图像上。

    Args:
        image: (H, W, 3) 原始图像，值域 [0, 255] uint8
        heatmap: (H, W) 热力图，值域 [0, 1]
        alpha: 叠加透明度
        colormap: OpenCV colormap

    Returns:
        (H, W, 3) 叠加后的图像
    """
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 确保尺寸匹配
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = cv2.resize(
            heatmap_color, (image.shape[1], image.shape[0])
        )

    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed


def generate_gradcam_explanation(model: torch.nn.Module, image_tensor: torch.Tensor,
                                 original_image: np.ndarray, class_name: str,
                                 confidence: float, save_dir: str,
                                 filename: str = None, method: str = "gradcam",
                                 target_layer_name: str = None) -> dict:
    """
    为一个样本生成完整的 Grad-CAM 解释面板。

    输出一张 3 合 1 图：原始 | 热力图 | 叠加图

    Args:
        model: 模型
        image_tensor: (1, C, H, W) 预处理后的输入
        original_image: (H, W, 3) 原始图像 uint8
        class_name: 预测类别名
        confidence: 预测置信度
        save_dir: 保存目录
        filename: 文件名
        method: "gradcam" | "gradcamplus"

    Returns:
        {save_path, predicted_class, confidence}
    """
    os.makedirs(save_dir, exist_ok=True)

    # 查找目标卷积层
    target_layer = _find_conv_layer(model, target_layer_name)
    if target_layer is None:
        print(f"Warning: No conv layer found for Grad-CAM")
        return None

    # 生成热力图
    if method == "gradcamplus":
        gradcam = GradCAMPlusPlus(model, target_layer)
    else:
        gradcam = GradCAM(model, target_layer)

    heatmap = gradcam.generate(image_tensor)
    gradcam.remove_hooks()

    # 反标准化并转换原始图像用于显示
    denorm = denormalize(image_tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
    denorm = np.clip(denorm * 255, 0, 255).astype(np.uint8)

    # 生成叠加图
    overlay = overlay_heatmap(denorm, heatmap, alpha=0.5)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(denorm)
    axes[0].set_title(f"Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap\nTarget: {class_name}", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (Conf: {confidence:.2%})", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(f"Grad-CAM Explanation: {class_name} ({confidence:.2%})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if filename is None:
        filename = f"{class_name}_{confidence:.2f}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "save_path": save_path,
        "predicted_class": class_name,
        "confidence": confidence,
    }


def plot_tsne_embeddings(features: np.ndarray, labels: np.ndarray,
                         class_names: list, save_path: str,
                         perplexity: int = 30, max_samples: int = 2000):
    """
    t-SNE 可视化特征嵌入。

    Args:
        features: (N, D) 特征向量
        labels: (N,) 标签
        class_names: 类别名称
        save_path: 保存路径
        perplexity: t-SNE perplexity 参数
        max_samples: 最多采样数（类别太多时限制）
    """
    from sklearn.manifold import TSNE

    # 降采样
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000)
    embeddings = tsne.fit_transform(features)

    # 绘图
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(16, 12))

    cmap = plt.cm.get_cmap("tab20" if n_classes <= 20 else "gist_ncar", n_classes)

    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                label=name, alpha=0.6, s=8,
                color=cmap(i),
            )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize=6 if n_classes > 30 else 8)
    ax.set_title(f"t-SNE Feature Embedding Visualization ({n_classes} classes)",
                 fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _find_conv_layer(model: torch.nn.Module, target_name: str = None):
    """查找模型中的最后一个卷积层（用于 Grad-CAM）"""
    if target_name:
        for name, module in model.named_modules():
            if name == target_name:
                return module
        return None

    # 自动查找：找到最后一个 Conv2d 层
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    return last_conv
