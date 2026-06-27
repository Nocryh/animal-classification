#!/usr/bin/env python
"""
Animal Classification — 训练入口
==================================
支持 ResNet50 / EfficientNetV2 / ConvNeXt 三种架构，
完整训练流水线：数据加载 → 训练 → TensorBoard 可视化 → ONNX 导出。

Usage:
    python train.py --config config/default.yaml
    python train.py --model convnext_tiny --epochs 60 --batch_size 64
    python train.py --model efficientnetv2_s --loss focal --lr 3e-4
    python train.py --resume experiments/baseline/checkpoints/checkpoint_epoch_030.pt
"""
import os
import sys
import argparse
import yaml
from datetime import datetime

import torch

# 确保项目根目录在 path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import create_dataloaders, validate_dataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.factory import create_model, benchmark_inference_speed
from src.training.trainer import Trainer
from src.training.losses import get_loss_function


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an animal classification model"
    )
    # 配置文件
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file")

    # 快速覆盖（不需要修改 YAML）
    parser.add_argument("--model", type=str, default=None,
                        choices=["resnet50", "efficientnetv2_s", "convnext_tiny"],
                        help="Override model architecture")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override dataset directory")
    parser.add_argument("--classes_file", type=str, default=None,
                        help="Override classes file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["cross_entropy", "focal", "label_smoothing", "combined"],
                        help="Override loss function")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU ID (-1 for CPU)")

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行覆盖
    if args.model:
        config["model"]["architecture"] = args.model
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.classes_file:
        config["data"]["classes_file"] = args.classes_file
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.loss:
        config["loss"]["type"] = args.loss
    if args.no_amp:
        config["training"]["mixed_precision"] = False
    if args.gpu is not None:
        config["device"]["gpu_id"] = args.gpu
    if args.resume:
        config["checkpoint"]["resume"] = True
        config["checkpoint"]["resume_path"] = args.resume

    # 实验命名
    exp_name = args.exp_name or config["experiment"].get("name", "baseline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch = config["model"]["architecture"]
    exp_dir = os.path.join(
        PROJECT_ROOT, "experiments",
        f"{exp_name}_{arch}_{timestamp}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Experiment: {exp_name}_{arch}")
    print(f"Output: {exp_dir}")
    print(f"{'=' * 70}")

    # 设备
    gpu_id = config.get("device", {}).get("gpu_id", 0)
    device = torch.device(
        f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # 数据完整性扫描（训练前检查，避免中途崩溃）
    print("Scanning dataset for corrupt files...")
    data_report = validate_dataset(config["data"]["data_dir"], img_size=config["data"]["img_size"])
    if data_report["corrupt"] > 0:
        print(f"Warning: {data_report['corrupt']} corrupt files detected. "
              f"Consider removing them before training.")
    print()

    # 数据
    img_size = config["data"]["img_size"]
    train_transform = get_train_transforms(
        img_size=img_size,
        config=config.get("augmentation", {})
    )
    val_transform = get_val_transforms(img_size)

    dataloaders = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 2),
        val_split=config["data"]["val_split"],
        seed=config["experiment"]["seed"],
        pin_memory=config["data"].get("pin_memory", True),
    )

    print(f"Train: {dataloaders['train_size']} | Val: {dataloaders['val_size']}")

    # 类别分布摘要
    dist = dataloaders["class_distribution"]
    counts = list(dist.values())
    print(f"Class distribution — min: {min(counts)}, max: {max(counts)}, "
          f"mean: {sum(counts)/len(counts):.0f}, "
          f"imbalance ratio: {max(counts)/max(min(counts),1):.1f}:1")

    # 模型
    model_cfg = config["model"]
    model, model_info = create_model(
        architecture=model_cfg["architecture"],
        num_classes=config["data"]["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        pool_type=model_cfg.get("pool", "gem"),
    )

    print(f"\nModel: {model_info['architecture']}")
    print(f"Parameters: {model_info['params_millions']}M "
          f"({model_info['trainable_params']:,} trainable)")

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        config=config,
        experiment_dir=exp_dir,
    )

    history = trainer.train()

    # 保存最终模型
    final_model_path = os.path.join(exp_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # 推理速度测试
    print("\nBenchmarking inference speed...")
    speed_info = benchmark_inference_speed(model, device)
    print(f"  FPS: {speed_info['fps']} | Latency: {speed_info['latency_ms']}ms/batch")

    # ONNX 导出
    print("\nExporting ONNX model...")
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    onnx_path = os.path.join(exp_dir, "model.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"ONNX exported to: {onnx_path}")

    # 验证 ONNX
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX verification: PASSED")
    except ImportError:
        print("ONNX verification: SKIP (onnx not installed)")

    print(f"\n{'=' * 70}")
    print(f"Experiment complete!")
    print(f"Results: {exp_dir}")
    print(f"TensorBoard: tensorboard --logdir {exp_dir}/tensorboard")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
