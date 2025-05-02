# training/vae_trainer.py

from __future__ import annotations


"""
Тренировочный скрипт для GraphVAE с адаптивными гиперпараметрами:
 - выбор устройства (GPU/CPU)
 - AMP (mixed precision)
 - KL-annealing (beta-schedule)
 - ReduceLROnPlateau для lr
 - адаптивный weight_decay
 - адаптивный batch_size
 - TensorBoard, прогресс-бары, логи
"""

from pathlib import Path
from typing import Any, Dict
import math

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from mol_gnn_project.models.losses import recon_loss, kld_loss

__all__ = ["run_training_vae"]

# -----------------------------------------------------------------------------

class VAETrainer:
    def __init__(
        self,
        model: nn.Module,
        train_ds: Any,
        val_ds: Any,
        cfg: Dict[str, Any],
        log_dir: str | Path,
    ) -> None:
        # Выбор устройства (GPU/CPU)
        use_gpu = cfg.get("use_gpu", True)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # модель на устройство
        self.model = model.to(self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg

        # начальные гиперпараметры
        base_lr = float(cfg.get("lr", 1e-3))
        base_wd = float(cfg.get("weight_decay", 1e-5))
        self.epochs = int(cfg.get("epochs", 100))

        # инициализация оптимизатора
        self.optim = optim.AdamW(
            self.model.parameters(), lr=base_lr, weight_decay=base_wd
        )

        # опциональный lr scheduler
        self.adapt_lr = cfg.get("adapt_lr", False)
        if self.adapt_lr:
            self.lr_scheduler = ReduceLROnPlateau(
                self.optim,
                mode="min",
                factor=cfg.get("lr_factor", 0.5),
                patience=cfg.get("lr_patience", 5),
                verbose=True,
            )

        # адаптивный weight_decay
        self.adapt_wd = cfg.get("adapt_wd", False)
        self.current_wd = base_wd
        self.wd_factor = cfg.get("wd_factor", 0.5)
        self.wd_patience = cfg.get("wd_patience", 5)
        self.wd_counter = 0

        # адаптивный batch_size
        self.adapt_bs = cfg.get("adapt_bs", False)
        self.bs = int(cfg.get("batch_size", 64))
        self.bs_max = int(cfg.get("max_batch_size", self.bs))
        self.bs_every = int(cfg.get("bs_increase_every", 10))

        # адаптивный beta (KL-annealing)
        self.adapt_beta = cfg.get("adapt_beta", False)
        self.beta_max = float(cfg.get("beta_max", 0.1))
        self.max_beta = float(cfg.get("max_beta", self.beta_max))
        self.warmup = int(cfg.get("warmup_epochs", 20))
        self.beta_rec_thresh = float(cfg.get("beta_rec_thresh", 0.0))
        self.beta_factor = float(cfg.get("beta_factor", 1.0))


        # AMP Gradient Scaler без лишних аргументов
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(log_dir))

        print(f"[INFO] Training for {self.epochs} epochs, initial beta_max={self.beta_max}")
        print(f"[INFO] Batch size: start={self.bs}, max={self.bs_max}, increase every={self.bs_every} epochs")

    # ------------------------------------------------------------------
    def _beta(
            self,
            epoch: int
    ) -> float:
        """Линейный разогрев β с 0 до beta_max."""
        return min(1.0, epoch / self.warmup) * self.beta_max

    # ------------------------------------------------------------------
    def _make_loader(
            self,
            dataset: Any,
            batch_size: int,
            shuffle: bool
    ) -> DataLoader:
        # утилита для создания DataLoader PyG
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def _step(
            self,
            batch: Any,
            train: bool,
            beta: float
    ) -> tuple[float, float, float]:
        # перенос на устройство и step одного батча
        batch = batch.to(self.device)

        # автокастинг (включён только при CUDA)
        with autocast(enabled=(self.device.type == "cuda")):
            x_hat, _adj_hat, mu, logvar = self.model(batch)
            loss_recon = recon_loss(x_hat, batch.x, mode=self.cfg.get("recon_mode", "mse"))
            loss_kld = kld_loss(mu, logvar)
            loss = loss_recon + beta * loss_kld

        if train:
            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
        return loss.item(), loss_recon.item(), loss_kld.item()

    # ------------------------------------------------------------------
    def _run_epoch(
            self,
            loader: DataLoader,
            epoch: int,
            train: bool
    ) -> Dict[str, float]:
        mode = "train" if train else "val"
        self.model.train(train)
        beta = self._beta(epoch)
        total_loss = total_rec = total_kld = 0.0

        print(f"[{mode.upper()}] Epoch {epoch}/{self.epochs}, beta={beta:.4f}")

        for batch in tqdm(loader, desc=f"{mode} epoch {epoch}", leave=False):
            l, r, k = self._step(batch, train, beta)
            total_loss += l
            total_rec += r
            total_kld += k

        n = len(loader)
        avg = {"loss": total_loss / n, "recon": total_rec / n, "kld": total_kld / n}
        # лог в TensorBoard
        self.writer.add_scalar(f"ELBO/{mode}", avg["loss"], epoch)
        self.writer.add_scalar(f"Recon/{mode}", avg["recon"], epoch)
        self.writer.add_scalar(f"KLD/{mode}", avg["kld"], epoch)
        if train:
            self.writer.add_scalar("beta", beta, epoch)

        print(f"[{mode.upper()}] Epoch {epoch} done: loss={avg['loss']:.4f}, recon={avg['recon']:.4f}, kld={avg['kld']:.4f}")
        return avg

    # ------------------------------------------------------------------
    def fit(self) -> None:
        """
        Основной цикл обучения с поддержкой early-stopping по patience_max.
        """
        best_val = float('inf')
        patience = 0
        patience_max = int(self.cfg.get("patience_max", self.cfg.get("patience", 15)))
        for epoch in range(1, self.epochs + 1):
            # адаптивный batch_size
            if self.adapt_bs and epoch % self.bs_every == 0 and self.bs < self.bs_max:
                self.bs = min(self.bs * 2, self.bs_max)
                print(f"[INFO] Increased batch_size to {self.bs}")

            # создаём загрузчики
            train_loader = self._make_loader(self.train_ds, self.bs, True)
            val_loader = self._make_loader(self.val_ds, self.bs, False)
            # прогоны train и val
            train_metrics = self._run_epoch(train_loader, epoch, True)
            val_metrics = self._run_epoch(val_loader, epoch, False)

            # получение метрик
            val_loss = val_metrics["loss"]
            # лучший чекпойнт
            if val_loss < best_val:
                best_val = val_loss
                early_counter = 0
                # сброс счётчика weight_decay
                self.wd_counter = 0
                # сохраняем модель
                ckpt_path = Path(self.writer.log_dir) / "vae_best.ckpt"
                torch.save({"model_state": self.model.state_dict(), "val_loss": val_loss}, ckpt_path)
                print(f"[INFO] New best checkpoint (val_loss={val_loss:.4f})")
            else:
                early_counter += 1
                # адаптивный weight_decay
                if self.adapt_wd:
                    self.wd_counter += 1
                    if self.wd_counter >= self.wd_patience:
                        self.current_wd *= self.wd_factor
                        for pg in self.optim.param_groups:
                            pg['weight_decay'] = self.current_wd
                        print(f"[INFO] Reduced weight_decay to {self.current_wd:.2e}")
                        self.wd_counter = 0
                # early stopping
                if early_counter >= patience_max:
                    print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {patience_max} epochs)")
                    break

            # адаптивный lr
            if self.adapt_lr:
                self.lr_scheduler.step(val_loss)
            # адаптивный beta
            if self.adapt_beta and train_metrics["recon"] < self.beta_rec_thresh and self.beta_max < self.max_beta:
                old = self.beta_max
                self.beta_max = min(self.beta_max * self.beta_factor, self.max_beta)
                print(f"[INFO] Increased beta_max from {old:.3f} to {self.beta_max:.3f}")

# -----------------------------------------------------------------------------
# Convenience API
# -----------------------------------------------------------------------------

def run_training_vae(
        model: nn.Module,
        train_ds,
        val_ds,
        cfg: Dict,
        log_dir: str | Path):
    """
       Удобный API: принимает датасеты, создаёт тренер и запускает обучение.
    """
    trainer = VAETrainer(
        model,
        train_ds,
        val_ds,
        cfg,
        log_dir
    )
    trainer.fit()
