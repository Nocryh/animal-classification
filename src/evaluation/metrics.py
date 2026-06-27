"""
评估指标模块
============
- 混淆矩阵生成与可视化
- Per-class Precision / Recall / F1 / Support
- Top-1 / Top-5 Accuracy
- 错误样本分析
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，支持服务器环境
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score, top_k_accuracy_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_dataset(model: nn.Module, dataloader: DataLoader,
                    device: torch.device, return_features: bool = False) -> dict:
    """
    对整个数据集进行预测。

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
        return_features: 是否返回倒数第二层特征（用于 t-SNE）

    Returns:
        dict with: predictions, labels, probabilities, (features), (filenames)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    all_paths = []

    for images, labels in dataloader:
        images = images.to(device)

        # 如果需要特征，使用 hook
        if return_features:
            features = {}
            def hook_fn(name):
                def hook(module, input, output):
                    features[name] = output.detach().cpu()
                return hook
            # 获取最后一层之前的特征
            if hasattr(model, "fc"):
                handle = model.fc.register_forward_hook(hook_fn("features"))
            elif hasattr(model, "classifier"):
                handle = model.classifier[-1].register_forward_hook(hook_fn("features"))
            else:
                handle = None

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

        if return_features and features:
            all_features.append(features.get("features", probs.cpu().numpy()))

        if handle:
            handle.remove()

    result = {
        "predictions": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "probabilities": np.concatenate(all_probs),
    }

    if all_features:
        result["features"] = np.concatenate(all_features)

    return result


def compute_metrics(predictions: np.ndarray, labels: np.ndarray,
                    class_names: list = None) -> dict:
    """
    计算全面的评估指标。

    Returns:
        dict with:
            - top1_acc, top5_acc
            - confusion_matrix (np.ndarray)
            - per_class: list of {class, precision, recall, f1, support}
            - macro_avg, weighted_avg
            - classification_report (str)
    """
    num_classes = len(class_names) if class_names else len(np.unique(labels))

    top1 = accuracy_score(labels, predictions)

    # Top-5 accuracy (从 one-hot 或索引计算)
    # 这里用 prediction index，对于 Top-5 需要概率
    top5 = None  # 由调用者传入概率计算

    cm = confusion_matrix(labels, predictions, labels=range(num_classes))

    # Per-class 指标
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=range(num_classes), zero_division=0
    )

    per_class = []
    for i in range(num_classes):
        per_class.append({
            "class": class_names[i] if class_names else str(i),
            "index": i,
            "precision": round(precision[i], 4),
            "recall": round(recall[i], 4),
            "f1": round(f1[i], 4),
            "support": int(support[i]),
        })

    # 宏平均和加权平均
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    report_str = classification_report(
        labels, predictions,
        target_names=class_names if class_names else [str(i) for i in range(num_classes)],
        zero_division=0,
    )

    results = {
        "top1_accuracy": round(top1, 4),
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1": round(weighted_f1, 4),
        },
        "classification_report": report_str,
        "total_samples": len(labels),
    }

    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          save_path: str, title: str = "Confusion Matrix",
                          figsize: tuple = (24, 20), normalize: bool = True,
                          max_classes: int = 90) -> str:
    """
    绘制并保存混淆矩阵热力图。

    Args:
        cm: 混淆矩阵 (num_classes, num_classes)
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图表尺寸
        normalize: 是否按行归一化（显示召回率）
        max_classes: 最多显示类别数（类别多时自动缩小字体）

    Returns:
        保存的文件路径
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues,
                   vmin=0, vmax=1 if normalize else cm.max())
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 标签
    n_classes = len(class_names)
    tick_interval = max(1, n_classes // 30)  # 类太多时跳过一些标签
    tick_indices = list(range(0, n_classes, tick_interval))
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)

    font_size = max(6, 12 - n_classes // 10)
    ax.set_xticklabels([class_names[i] for i in tick_indices],
                       rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels([class_names[i] for i in tick_indices],
                       fontsize=font_size)

    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_per_class_metrics(per_class: list, save_path: str,
                           top_n: int = 20, metric: str = "f1") -> str:
    """
    绘制每类 F1/Precision/Recall 的排序条形图。

    Args:
        per_class: per_class 指标列表
        save_path: 保存路径
        top_n: 显示最好/最差的 N 类
        metric: f1 | precision | recall

    Returns:
        保存的文件路径
    """
    sorted_data = sorted(per_class, key=lambda x: x[metric])

    best = sorted_data[-top_n:]
    worst = sorted_data[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Best
    names_best = [d["class"] for d in best]
    vals_best = [d[metric] for d in best]
    colors_best = plt.cm.Greens(np.linspace(0.5, 1, len(best)))
    ax1.barh(range(len(best)), vals_best, color=colors_best)
    ax1.set_yticks(range(len(best)))
    ax1.set_yticklabels(names_best)
    ax1.set_xlabel(metric.upper(), fontsize=12)
    ax1.set_title(f"Top {top_n} Best Classes ({metric.upper()})", fontsize=14)
    ax1.invert_yaxis()

    # Worst
    names_worst = [d["class"] for d in worst]
    vals_worst = [d[metric] for d in worst]
    colors_worst = plt.cm.Reds(np.linspace(0.5, 1, len(worst)))
    ax2.barh(range(len(worst)), vals_worst, color=colors_worst)
    ax2.set_yticks(range(len(worst)))
    ax2.set_yticklabels(names_worst)
    ax2.set_xlabel(metric.upper(), fontsize=12)
    ax2.set_title(f"Top {top_n} Worst Classes ({metric.upper()})", fontsize=14)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def analyze_confusion_pairs(cm: np.ndarray, class_names: list,
                            top_k: int = 10) -> list:
    """
    分析最易混淆的类别对。

    Returns:
        list of {true_class, pred_class, count, true_idx, pred_idx}
    """
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:top_k]
    pairs = []

    for idx in flat_indices:
        true_idx, pred_idx = np.unravel_index(idx, cm.shape)
        pairs.append({
            "true_class": class_names[true_idx],
            "pred_class": class_names[pred_idx],
            "count": int(cm[true_idx, pred_idx]),
            "true_idx": int(true_idx),
            "pred_idx": int(pred_idx),
        })

    return pairs


def find_misclassified(predictions: np.ndarray, labels: np.ndarray,
                       probabilities: np.ndarray, class_names: list,
                       top_k: int = 20) -> list:
    """
    找出预测置信度最高的误分类样本。

    Returns:
        list of {index, true_class, pred_class, confidence}
    """
    errors = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            errors.append({
                "index": i,
                "true_class": class_names[label] if class_names else str(label),
                "pred_class": class_names[pred] if class_names else str(pred),
                "confidence": float(probabilities[i, pred]),
            })

    errors.sort(key=lambda x: x["confidence"], reverse=True)
    return errors[:top_k]
