"""
训练回调模块
============
- Early Stopping: 监控指标不提升时自动停止
- Checkpoint Manager: 保存/恢复 checkpoints
- Resume Training: 从断点完整恢复训练状态
"""
import os
import copy
import torch
import torch.nn as nn


class EarlyStopping:
    """
    早停机制。

    当监控指标在 patience 轮内没有提升 min_delta 以上时，触发停止信号。
    停止后可通过 recover_best() 恢复最佳权重。

    Args:
        patience: 容忍轮数
        min_delta: 最小提升阈值
        mode: 'min' (监控 loss) 或 'max' (监控 accuracy)
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """
        更新状态并返回是否应该停止。

        Returns:
            True if should stop, False otherwise
        """
        if self.mode == "min":
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            # 保存最佳模型状态
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def recover_best(self, model: nn.Module):
        """恢复最佳模型权重"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            return True
        return False


class CheckpointManager:
    """
    Checkpoint 管理器。

    功能：
    - 保存 top-K 最佳模型（按监控指标排序）
    - 定期备份（每 N epoch）
    - 断点续训 checkpoint（含 optimizer/scheduler/epoch）

    Args:
        save_dir: 保存目录
        save_top_k: 保留的最佳模型数量
        save_every_n: 定期备份间隔（epoch）
        monitor_mode: 'min' 或 'max'
    """

    def __init__(self, save_dir: str, save_top_k: int = 3,
                 save_every_n: int = 10, monitor_mode: str = "min"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_top_k = save_top_k
        self.save_every_n = save_every_n
        self.monitor_mode = monitor_mode
        self.best_scores = []  # list of (score, filepath)

    def save_checkpoint(self, model: nn.Module, optimizer, scheduler,
                        epoch: int, score: float, config: dict,
                        is_best: bool = False, filename: str = None):
        """
        保存完整 checkpoint（用于断点续训）。
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        filepath = os.path.join(self.save_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "score": score,
            "config": config,
        }
        torch.save(checkpoint, filepath)
        return filepath

    def save_best(self, model: nn.Module, epoch: int, score: float):
        """
        保存最佳模型（只保存权重，体积小）。
        自动维护 top-K 列表。
        """
        filename = f"best_epoch_{epoch:03d}_score_{score:.4f}.pth"
        filepath = os.path.join(self.save_dir, filename)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "score": score,
        }, filepath)

        # 维护 top-K
        self.best_scores.append((score, filepath))
        self.best_scores.sort(
            key=lambda x: x[0],
            reverse=(self.monitor_mode == "max")
        )

        # 删除溢出和重复的最佳模型
        while len(self.best_scores) > self.save_top_k:
            _, old_path = self.best_scores.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        return filepath

    def save_periodic(self, model: nn.Module, optimizer, scheduler,
                      epoch: int, score: float, config: dict):
        """定期备份"""
        if epoch % self.save_every_n == 0:
            return self.save_checkpoint(
                model, optimizer, scheduler, epoch, score, config,
                filename=f"periodic_epoch_{epoch:03d}.pt"
            )
        return None

    def load_latest(self, model: nn.Module, optimizer, scheduler) -> int:
        """
        加载最新的 checkpoint（用于断点续训）。

        Returns:
            续训的起始 epoch，若没有 checkpoint 则返回 0
        """
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith(".pt")]
        if not checkpoints:
            return 0

        # 按文件名排序取最新
        checkpoints.sort()
        latest = os.path.join(self.save_dir, checkpoints[-1])

        checkpoint = torch.load(latest, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get("epoch", 0)
