#!/usr/bin/env python
"""
Animal Classification — 评估入口
==================================
加载训练好的模型，在验证/测试集上生成完整评估报告：
- 混淆矩阵 + 每类指标
- Grad-CAM 可解释性分析
- t-SNE 特征嵌入可视化
- 错误分析

Usage:
    python evaluate.py --checkpoint experiments/xxx/best_model.pth
    python evaluate.py --checkpoint path/to/model.pth --data_dir ./test_images --mode gradcam
"""
import os
import sys
import argparse
import yaml
import numpy as np
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.augmentations import get_val_transforms
from src.models.factory import create_model
from src.evaluation.metrics import (
    predict_dataset, compute_metrics, plot_confusion_matrix,
    plot_per_class_metrics, analyze_confusion_pairs, find_misclassified,
)
from src.evaluation.explainability import (
    generate_gradcam_explanation, plot_tsne_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained animal classification model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Evaluation data directory (default: from config)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "metrics", "gradcam", "tsne", "errors"],
                        help="Evaluation mode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num_gradcam", type=int, default=20,
                        help="Number of Grad-CAM samples to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda / cpu)")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    """加载 checkpoint（支持完整 checkpoint 和仅权重）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        score = checkpoint.get("score", "unknown")
        print(f"Loaded checkpoint: epoch={epoch}, score={score}")
    else:
        # 仅权重文件
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded weights-only checkpoint")

    return model


def main():
    args = parse_args()

    # 配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 输出目录
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint), "evaluation"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}\n")

    # 模型
    model_cfg = config["model"]
    model, model_info = create_model(
        architecture=model_cfg.get("architecture", "resnet50"),
        num_classes=config["data"]["num_classes"],
        pretrained=False,  # 加载 checkpoint，不需要 pretrained
        dropout=model_cfg.get("dropout", 0.3),
        pool_type=model_cfg.get("pool", "gem"),
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    num_params = model_info["params_millions"]
    print(f"Model: {model_info['architecture']} ({num_params}M params)")

    # 数据
    data_dir = args.data_dir or config["data"]["data_dir"]
    val_transform = get_val_transforms(config["data"]["img_size"])

    eval_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    eval_loader = DataLoader(
        eval_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )
    class_names = eval_dataset.classes

    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Classes: {len(class_names)}\n")

    # ==================== Metrics ====================
    if args.mode in ["full", "metrics"]:
        print(f"{'=' * 60}")
        print("METRICS EVALUATION")
        print(f"{'=' * 60}")

        results = predict_dataset(
            model, eval_loader, device, return_features=(args.mode in ["full", "tsne"])
        )

        metrics = compute_metrics(
            results["predictions"], results["labels"], class_names
        )

        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
        print(f"Macro Avg: P={metrics['macro_avg']['precision']:.4f} | "
              f"R={metrics['macro_avg']['recall']:.4f} | F1={metrics['macro_avg']['f1']:.4f}")
        print(f"Weighted Avg: P={metrics['weighted_avg']['precision']:.4f} | "
              f"R={metrics['weighted_avg']['recall']:.4f} | F1={metrics['weighted_avg']['f1']:.4f}")

        # 混淆矩阵
        print("\nPlotting confusion matrix...")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            metrics["confusion_matrix"], class_names, cm_path,
            title=f"Confusion Matrix — {model_info['architecture']} "
                  f"(Top-1: {metrics['top1_accuracy']:.2%})",
        )
        print(f"  Saved: {cm_path}")

        # Per-class F1
        f1_path = os.path.join(output_dir, "per_class_f1.png")
        plot_per_class_metrics(metrics["per_class"], f1_path, top_n=20, metric="f1")
        print(f"  Saved: {f1_path}")

        # 分类报告
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(metrics["classification_report"])
        print(f"  Saved: {report_path}")

        # 错误分析
        confusion_pairs = analyze_confusion_pairs(
            metrics["confusion_matrix"], class_names, top_k=10
        )
        print("\nTop Confusion Pairs:")
        for i, pair in enumerate(confusion_pairs, 1):
            print(f"  {i}. {pair['true_class']} → {pair['pred_class']} "
                  f"({pair['count']} times)")

        # 完整的每类指标
        print("\nPer-Class Metrics (worst 5 by F1):")
        worst = sorted(metrics["per_class"], key=lambda x: x["f1"])[:5]
        for d in worst:
            print(f"  {d['class']:<20s} F1={d['f1']:.4f} "
                  f"P={d['precision']:.4f} R={d['recall']:.4f} "
                  f"sup={d['support']}")

    # ==================== Grad-CAM ====================
    if args.mode in ["full", "gradcam"]:
        print(f"\n{'=' * 60}")
        print("GRAD-CAM EXPLANATIONS")
        print(f"{'=' * 60}")

        gradcam_dir = os.path.join(output_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        # 选择样本：各类别各取 1 个 + 误分类样本
        correct_samples = []
        wrong_samples = []

        for images, labels in eval_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(len(images)):
                is_correct = preds[i] == labels[i]
                if is_correct and len(correct_samples) < args.num_gradcam // 2:
                    correct_samples.append((images[i:i+1], labels[i].item(), preds[i].item(),
                                           outputs.softmax(1)[i, preds[i]].item()))
                elif not is_correct and len(wrong_samples) < args.num_gradcam // 2:
                    wrong_samples.append((images[i:i+1], labels[i].item(), preds[i].item(),
                                         outputs.softmax(1)[i, preds[i]].item()))

                if len(correct_samples) >= args.num_gradcam // 2 and \
                   len(wrong_samples) >= args.num_gradcam // 2:
                    break

        # 生成 Grad-CAM
        for sample_type, samples in [("correct", correct_samples), ("wrong", wrong_samples)]:
            sample_dir = os.path.join(gradcam_dir, sample_type)
            for j, (img_tensor, true_label, pred_label, conf) in enumerate(samples):
                # 反标准化用于显示
                from src.evaluation.explainability import denormalize
                denorm = denormalize(img_tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
                denorm = np.clip(denorm * 255, 0, 255).astype(np.uint8)

                pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
                true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)

                filename = f"{sample_type}_{j}_{true_name}_pred_{pred_name}.png"

                generate_gradcam_explanation(
                    model, img_tensor, denorm, pred_name, conf,
                    sample_dir, filename, method="gradcam",
                )

        print(f"  Generated {args.num_gradcam} Grad-CAM visualizations")
        print(f"  Saved: {gradcam_dir}/")

    # ==================== t-SNE ====================
    if args.mode in ["full", "tsne"]:
        print(f"\n{'=' * 60}")
        print("t-SNE FEATURE EMBEDDING")
        print(f"{'=' * 60}")

        # Re-run with features if not already
        if args.mode not in ["full", "metrics"]:
            results = predict_dataset(
                model, eval_loader, device, return_features=True
            )
        else:
            results = predict_dataset(
                model, eval_loader, device, return_features=True
            )

        if "features" in results and results["features"] is not None:
            tsne_path = os.path.join(output_dir, "tsne_embeddings.png")
            plot_tsne_embeddings(
                results["features"], results["labels"], class_names, tsne_path,
                max_samples=2000,
            )
            print(f"  Saved: {tsne_path}")
        else:
            print("  Skipped: No features available")

    # ==================== Error Analysis ====================
    if args.mode in ["full", "errors"]:
        print(f"\n{'=' * 60}")
        print("ERROR ANALYSIS")
        print(f"{'=' * 60}")

        if args.mode not in ["full", "metrics"]:
            results = predict_dataset(model, eval_loader, device)

        errors = find_misclassified(
            results["predictions"], results["labels"],
            results["probabilities"], class_names, top_k=30,
        )

        error_path = os.path.join(output_dir, "error_analysis.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("Top Confused Misclassifications\n")
            f.write("=" * 60 + "\n\n")
            for i, e in enumerate(errors, 1):
                f.write(f"{i:2d}. True: {e['true_class']:<20s} → "
                        f"Pred: {e['pred_class']:<20s} "
                        f"(confidence: {e['confidence']:.2%})\n")

        print(f"  Top 30 misclassified samples saved to: {error_path}")

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Results: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
