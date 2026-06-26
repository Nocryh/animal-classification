"""
超参数自动优化
==============
基于 Optuna 的贝叶斯超参数搜索。

Usage:
    python -m src.utils.optuna_tune --n_trials 50 --config config/default.yaml

搜索空间:
    - 学习率 (log scale)
    - 权重衰减 (log scale)
    - Dropout 比率
    - 批量大小
    - 优化器选择
    - 损失函数选择
    - 增强策略参数
"""
import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime

import torch
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import create_dataloaders
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.factory import create_model
from src.training.trainer import Trainer
from src.training.losses import get_loss_function


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def suggest_hyperparameters(trial: Trial) -> dict:
    """
    定义 Optuna 搜索空间。

    搜索的超参数及其范围基于深度学习最佳实践设定。
    """
    hp = {}

    # 学习率: log-scale 搜索 [1e-5, 1e-2]
    hp["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # 权重衰减: [1e-6, 1e-2]
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Dropout: [0.1, 0.6]
    hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.6)

    # 优化器选择
    hp["optimizer"] = trial.suggest_categorical("optimizer", ["adamw", "sgd"])

    # 批量大小
    hp["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 损失函数
    hp["loss_type"] = trial.suggest_categorical(
        "loss_type", ["cross_entropy", "focal", "label_smoothing", "combined"]
    )

    # Focal gamma (仅影响 focal/combined)
    hp["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 5.0)

    # Label smoothing (仅影响 label_smoothing/combined)
    hp["label_smoothing"] = trial.suggest_float("label_smoothing", 0.05, 0.3)

    # 增强策略
    hp["ra_magnitude"] = trial.suggest_int("ra_magnitude", 5, 15)
    hp["mixup"] = trial.suggest_categorical("mixup", [True, False])
    hp["mixup_alpha"] = trial.suggest_float("mixup_alpha", 0.1, 1.0)

    # Warmup epochs
    hp["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 10)

    return hp


def objective(trial: Trial, base_config: dict, data_dir: str,
              classes_file: str, tuning_dir: str) -> float:
    """
    Optuna 目标函数：训练模型并返回验证准确率。
    """
    hp = suggest_hyperparameters(trial)

    # 更新配置
    config = yaml.safe_load(yaml.dump(base_config))  # 深拷贝

    # Training params
    config["training"]["lr"] = hp["lr"]
    config["training"]["weight_decay"] = hp["weight_decay"]
    config["training"]["optimizer"] = hp["optimizer"]
    config["training"]["batch_size"] = hp["batch_size"]
    config["training"]["epochs"] = 30  # 搜索时减少 epochs

    # Model params
    config["model"]["dropout"] = hp["dropout"]

    # Loss params
    config["loss"]["type"] = hp["loss_type"]
    config["loss"]["focal_gamma"] = hp["focal_gamma"]
    config["loss"]["label_smoothing"] = hp["label_smoothing"]

    # Augmentation
    config["augmentation"]["ra_magnitude"] = hp["ra_magnitude"]
    config["augmentation"]["mixup"] = hp["mixup"]
    config["augmentation"]["mixup_alpha"] = hp["mixup_alpha"]

    # Scheduler
    config["scheduler"]["warmup_epochs"] = hp["warmup_epochs"]

    # Experiment
    trial_dir = os.path.join(tuning_dir, f"trial_{trial.number:03d}")
    config["experiment"]["name"] = f"optuna_trial_{trial.number}"

    # Early stopping (缩短 patience 以加速搜索)
    config["early_stopping"]["enabled"] = True
    config["early_stopping"]["patience"] = 10

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 数据
        train_transform = get_train_transforms(
            img_size=config["data"]["img_size"],
            config=config["augmentation"]
        )
        val_transform = get_val_transforms(config["data"]["img_size"])

        dataloaders = create_dataloaders(
            data_dir=data_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=config["training"]["batch_size"],
            num_workers=2,
            val_split=config["data"]["val_split"],
            seed=config["experiment"]["seed"],
        )

        # 模型
        model_cfg = config["model"]
        model, model_info = create_model(
            architecture=model_cfg["architecture"],
            num_classes=config["data"]["num_classes"],
            pretrained=True,
            dropout=model_cfg["dropout"],
            pool_type=model_cfg.get("pool", "gem"),
        )

        # 训练
        trainer = Trainer(
            model=model,
            train_loader=dataloaders["train_loader"],
            val_loader=dataloaders["val_loader"],
            config=config,
            experiment_dir=trial_dir,
        )

        history = trainer.train()

        # 提取最佳验证准确率
        best_val_acc = max(h["val_acc"] for h in history)

        # 报告中间值（用于 pruning）
        for epoch, h in enumerate(history):
            trial.report(h["val_acc"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_acc

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_optuna_tuning(config_path: str, n_trials: int = 50,
                      timeout: int = None, study_name: str = None):
    """
    运行 Optuna 超参数搜索。

    Args:
        config_path: 基础配置文件路径
        n_trials: 最大试验次数
        timeout: 最大搜索时间（秒），None 表示不限制
        study_name: 实验名称
    """
    base_config = load_config(config_path)
    data_dir = base_config["data"]["data_dir"]
    classes_file = base_config["data"]["classes_file"]

    # 创建调优目录
    tuning_dir = os.path.join(PROJECT_ROOT, "experiments", "optuna_tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    if study_name is None:
        study_name = f"animal_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 创建 Optuna Study
    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{os.path.join(tuning_dir, 'optuna.db')}",
        load_if_exists=True,
    )

    # 执行搜索
    print(f"\n{'=' * 70}")
    print(f"Optuna Hyperparameter Optimization")
    print(f"Study: {study_name} | Trials: {n_trials}")
    print(f"Database: {tuning_dir}/optuna.db")
    print(f"{'=' * 70}\n")

    study.optimize(
        lambda trial: objective(trial, base_config, data_dir, classes_file, tuning_dir),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # 输出结果
    print(f"\n{'=' * 70}")
    print(f"Optimization Complete!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Val Acc: {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存最优参数
    best_params_path = os.path.join(tuning_dir, "best_params.yaml")
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.dump(study.best_params, f, allow_unicode=True)
    print(f"\nBest params saved to: {best_params_path}")

    # 参数重要性
    print(f"\nParameter Importance:")
    importances = optuna.importance.get_param_importances(study)
    for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {param}: {importance:.4f}")

    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to base config")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name")
    args = parser.parse_args()

    # 确保路径正确
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    run_optuna_tuning(
        config_path=config_path,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
    )


if __name__ == "__main__":
    main()
