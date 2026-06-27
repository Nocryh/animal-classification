"""
训练引擎
========
完整的训练循环，包含：
- 混合精度训练 (AMP)
- EMA (指数移动平均)
- TensorBoard 日志
- 断点续训
- 训练摘要报告
"""
import os
import yaml
import time
import copy
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .losses import get_loss_function
from .callbacks import EarlyStopping, CheckpointManager
from ..data.augmentations import mixup_data, mixup_criterion


class EMA:
    """
    指数移动平均 (Exponential Moving Average)

    维护一份模型权重的滑动平均副本，不参与梯度计算。
    验证时切换到 EMA 权重，通常带来 +0.5~1% 的稳定提升。

    参考: "Averaging Weights Leads to Wider Optima and Better Generalization"
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 为当前权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """每步训练后调用：shadow = decay * shadow + (1 - decay) * current"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average

    def apply_shadow(self):
        """将 EMA 权重应用到模型（验证前调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始训练权重（验证后调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]


class Trainer:
    """
    完整的深度学习训练器。

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config, experiment_dir)
        history = trainer.train()
    """

    def __init__(self, model: nn.Module, train_loader, val_loader,
                 config: dict, experiment_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_dir = experiment_dir

        # 实验子目录
        self.tb_dir = os.path.join(experiment_dir, "tensorboard")
        self.ckpt_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 设备
        gpu_id = config.get("device", {}).get("gpu_id", 0)
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # 配置提取
        self.training_cfg = config.get("training", {})
        self.epochs = self.training_cfg.get("epochs", 80)
        self.mixed_precision = self.training_cfg.get("mixed_precision", True)

        # 优化器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # 损失函数
        self.criterion = get_loss_function(config, train_loader, self.device)

        # 回调
        self.early_stopping = None
        es_cfg = config.get("early_stopping", {})
        if es_cfg.get("enabled", True):
            monitor = es_cfg.get("monitor", "val_loss")
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 15),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode="min" if "loss" in monitor else "max",
            )

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.ckpt_dir,
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            save_every_n=ckpt_cfg.get("save_every_n_epochs", 10),
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # 混合精度
        self.scaler = GradScaler("cuda") if self.mixed_precision and self.device.type == "cuda" else None

        # 日志
        self.log_interval = config.get("experiment", {}).get("log_interval", 50)
        self.history = []
        self.start_epoch = 1
        self.best_score = None

        # 断点续训
        if ckpt_cfg.get("resume", False):
            resume_path = ckpt_cfg.get("resume_path")
            if resume_path:
                self._resume(resume_path)
            else:
                self.start_epoch = self.checkpoint_manager.load_latest(
                    self.model, self.optimizer, self.scheduler
                ) + 1
                if self.start_epoch > 1:
                    print(f"Resumed from epoch {self.start_epoch}")

        # MixUp 配置
        aug_cfg = config.get("augmentation", {})
        self.use_mixup = aug_cfg.get("mixup", False)
        self.mixup_alpha = aug_cfg.get("mixup_alpha", 0.2)

        # EMA (指数移动平均)
        ema_cfg = config.get("ema", {})
        self.use_ema = ema_cfg.get("enabled", True)
        self.ema = EMA(model, decay=ema_cfg.get("decay", 0.999)) if self.use_ema else None

        # 保存实验配置快照
        self._save_config_snapshot()

    def _build_optimizer(self):
        """创建优化器"""
        opt_name = self.training_cfg.get("optimizer", "adamw").lower()
        lr = self.training_cfg.get("lr", 1e-4)
        weight_decay = self.training_cfg.get("weight_decay", 1e-4)

        if opt_name == "adamw":
            return optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            momentum = self.training_cfg.get("momentum", 0.9)
            return optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum,
                weight_decay=weight_decay, nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self):
        """创建学习率调度器"""
        sched_cfg = self.config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine_warmup")

        if sched_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=sched_cfg.get("lr_min", 1e-6),
            )

        elif sched_type == "cosine_warmup":
            # CosineAnnealingLR with linear warmup
            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            warmup_start_lr = sched_cfg.get("warmup_start_lr", 1e-6)
            base_lr = self.training_cfg.get("lr", 1e-4)

            # 先设置一个线性 warmup，然后用 cosine
            # 使用 LambdaLR 实现
            def warmup_cosine(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return warmup_start_lr / base_lr + (1 - warmup_start_lr / base_lr) * (epoch / warmup_epochs)
                else:
                    # Cosine decay
                    progress = (epoch - warmup_epochs) / max(1, self.epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))

            return lr_scheduler.LambdaLR(self.optimizer, warmup_cosine)

        elif sched_type == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5,
                patience=sched_cfg.get("patience", 10),
            )

        elif sched_type == "onecycle":
            return lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.training_cfg.get("lr", 1e-4),
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader),
            )

        return None

    def _resume(self, checkpoint_path: str):
        """从指定 checkpoint 恢复训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from checkpoint: {checkpoint_path} (epoch {self.start_epoch})")

    def _save_config_snapshot(self):
        """保存当前实验配置到实验目录"""
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

    def train(self):
        """执行完整训练流程"""
        print(f"\n{'=' * 70}")
        print(f"Training on {self.device} | Model: {self.config.get('model', {}).get('architecture', 'unknown')}")
        print(f"Loss: {self.config.get('loss', {}).get('type', 'cross_entropy')} | "
              f"Optimizer: {self.training_cfg.get('optimizer', 'adamw')} | "
              f"LR: {self.training_cfg.get('lr', 1e-4)}")
        print(f"Epochs: {self.epochs} | Batch Size: {self.training_cfg.get('batch_size', 32)}")
        print(f"TensorBoard: {self.tb_dir}")
        print(f"{'=' * 70}\n")

        global_step = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()

            # ========== Training ==========
            train_loss, train_acc = self._train_one_epoch(epoch, global_step)
            global_step += len(self.train_loader)

            # ========== Validation ==========
            if self.ema:
                self.ema.apply_shadow()
            val_loss, val_acc = self._validate()
            if self.ema:
                self.ema.restore()

            # ========== Scheduling ==========
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ========== Logging ==========
            epoch_time = time.time() - epoch_start
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "time": epoch_time,
            })

            # TensorBoard
            self.writer.add_scalars("Loss", {
                "train": train_loss, "val": val_loss
            }, epoch)
            self.writer.add_scalars("Accuracy", {
                "train": train_acc, "val": val_acc
            }, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # 终端输出
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s",
                end=""
            )

            # ========== Checkpointing ==========
            monitor_score = val_loss  # default: lower is better
            is_best = False
            if self.best_score is None or monitor_score < self.best_score:
                self.best_score = monitor_score
                is_best = True
                print("  [Best!]", end="")

            # 保存最佳模型
            if is_best:
                self.checkpoint_manager.save_best(self.model, epoch, monitor_score)

            # 定期备份
            self.checkpoint_manager.save_periodic(
                self.model, self.optimizer, self.scheduler,
                epoch, monitor_score, self.config
            )

            print()

            # ========== Early Stopping ==========
            if self.early_stopping:
                should_stop = self.early_stopping(monitor_score, self.model)
                if should_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    self.early_stopping.recover_best(self.model)
                    break

        # ========== 训练结束 ==========
        self.writer.close()
        self._save_training_summary()

        return self.history

    def _train_one_epoch(self, epoch: int, global_step: int) -> tuple:
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        samples_seen = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # MixUp
            if self.use_mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.mixup_alpha
                )

            # 前向传播（可选混合精度）
            if self.scaler:
                with autocast("cuda"):
                    outputs = self.model(images)
                    if self.use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                grad_clip = self.training_cfg.get("gradient_clip_norm")
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_clip = self.training_cfg.get("gradient_clip_norm")
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            # EMA 更新
            if self.ema:
                self.ema.update()

            # 统计
            running_loss += loss.item() * images.size(0)

            # MixUp 模式下的训练准确率（近似，在混合图像上对原始标签做预测）
            # 参考 timm/pytorch-image-models 的做法
            if self.use_mixup:
                _, predicted = outputs.max(1)
                total += labels_a.size(0)
                correct += (lam * predicted.eq(labels_a).float() +
                            (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            else:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # TensorBoard 批次数日志
            step = global_step + batch_idx
            if batch_idx % self.log_interval == 0:
                self.writer.add_scalar("Loss/train_batch", loss.item(), step)

        train_loss = running_loss / len(self.train_loader.dataset)
        train_acc = correct / max(total, 1)

        return train_loss, train_acc

    @torch.no_grad()
    def _validate(self) -> tuple:
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct / max(total, 1)

        return val_loss, val_acc

    def _save_training_summary(self):
        """保存训练历史为 CSV"""
        csv_path = os.path.join(self.experiment_dir, "training_history.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,time\n")
            for row in self.history:
                f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['train_acc']:.6f},"
                        f"{row['val_loss']:.6f},{row['val_acc']:.6f},{row['lr']:.8f},"
                        f"{row['time']:.1f}\n")
        print(f"\nTraining history saved to: {csv_path}")

        # 打印最佳结果
        if self.history:
            best_val_acc = max(h["val_acc"] for h in self.history)
            best_epoch = max(
                (h for h in self.history if h["val_acc"] == best_val_acc),
                key=lambda h: -h["val_loss"]
            )
            print(f"Best Val Acc: {best_val_acc:.2%} at epoch {best_epoch['epoch']}")
