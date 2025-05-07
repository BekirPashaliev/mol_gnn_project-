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
from typing import Any, Dict, List
import math
import time

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm.auto import tqdm

import mlflow

from mol_gnn_project.models.losses import recon_loss, kld_loss
from mol_gnn_project.utils import metrics as CM # chemical metrics
from mol_gnn_project.utils.visualizations import plot_training_curves, plot_times

__all__ = ["run_training_vae"]

# -----------------------------------------------------------------------------

class VAETrainer:
    def __init__(
        self,
        model: nn.Module,
        train_ds: Any,
        val_ds: Any,
        train_cfg: Dict[str, Any],
        log_cfg: Dict[str, Any],
    ) -> None:

        # save configs
        self.train_cfg = train_cfg
        self.log_cfg = log_cfg

        # Выбор устройства (GPU/CPU)
        use_gpu = train_cfg.get("use_gpu", True)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"\n[INFO] Using device: {self.device}")

        # модель на устройство
        self.model = model.to(self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds

        # начальные гиперпараметры
        base_lr = float(train_cfg.get("lr", 1e-3))
        base_wd = float(train_cfg.get("weight_decay", 0.0))
        self.epochs = int(train_cfg.get("epochs", 100))

        # инициализация оптимизатора
        self.optim = optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=base_wd
        )
        print(f"[INIT] Optimizer: AdamW(lr={base_lr}, weight_decay={base_wd})")

        # опциональный lr scheduler
        self.adapt_lr = train_cfg.get("adapt_lr", False)
        if self.adapt_lr:
            self.scheduler_type = train_cfg.get("scheduler_type", "plateau")
            if self.scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optim,
                    mode="min",
                    factor=train_cfg.get("lr_factor", 0.5),
                    patience=train_cfg.get("lr_patience", 5),
                )
            else:
                self.scheduler = CosineAnnealingLR(
                    self.optim,
                    T_max=train_cfg.get("cosine_T_max", 50),
                    eta_min=train_cfg.get("cosine_eta_min", 1e-6),
                )
            print(f"[INIT] LR Scheduler enabled: {self.scheduler_type}")

        # адаптивный weight_decay
        self.adapt_wd = train_cfg.get("adapt_wd", False)
        self.current_wd = base_wd
        self.wd_factor = train_cfg.get("wd_factor", 0.5)
        self.wd_patience = train_cfg.get("wd_patience", 5)
        self.wd_counter = 0
        print(f"[INIT] Adaptive weight_decay enabled: factor={self.wd_factor}, patience={self.wd_patience}")

        # адаптивный batch_size
        self.adapt_bs = train_cfg.get("adapt_bs", False)
        self.bs = int(train_cfg.get("batch_size", 64))
        self.bs_max = int(train_cfg.get("max_batch_size", self.bs))
        self.bs_every = int(train_cfg.get("bs_increase_every", 10))

        # адаптивный beta (KL-annealing)
        self.adapt_beta = train_cfg.get("adapt_beta", False)
        self.beta_max = float(train_cfg.get("beta_max", 1.0))
        self.max_beta = float(train_cfg.get("max_beta", self.beta_max))
        self.warmup = int(train_cfg.get("warmup_epochs", 20))
        self.beta_rec_thresh = float(train_cfg.get("beta_rec_thresh", 0.0))
        self.beta_factor = float(train_cfg.get("beta_factor", 1.0))


        # AMP Gradient Scaler без лишних аргументов
        self.scaler = GradScaler()
        print("[INIT] AMP gradient scaling enabled (GradScaler initialized)")

        # logging config
        self.track_metrics = log_cfg.get("track_metrics", True)
        self.metrics_freq = int(log_cfg.get("metrics_freq", 1))
        self.use_mlflow = log_cfg.get("use_mlflow", False)
        self.log_dir = Path(log_cfg.get("log_dir", "runs/"))
        self.plot_dir = Path(log_cfg.get("plot_dir", self.log_dir / "plots"))
        self.ckpt_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints/"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INIT] Logging directories created at: {self.log_dir}")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # MLflow
        if self.use_mlflow:
            print("[INIT] MLflow tracking enabled.")
            mlflow.set_experiment("mol_gnn_project")
            mlflow.start_run()

        print(f"\n[INFO] Training for {self.epochs} epochs, batch_size={self.bs}, beta_max={self.beta_max}")

    # ------------------------------------------------------------------
    def _beta(
            self,
            epoch: int
    ) -> float:
        """Линейный разогрев β с 0 до beta_max."""
        return min(1.0, epoch / self.warmup) * self.beta_max

    # ------------------------------------------------------------------
    # @staticmethod
    def _make_loader(
            self,
            dataset: Any,
            batch_size: int,
            shuffle: bool
    ) -> DataLoader:
        print(f"\n[LOADER] Creating DataLoader | batch_size={batch_size}, shuffle={shuffle}")

        # утилита для создания DataLoader PyG
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.train_cfg.get("num_workers", 4),
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
        batch = batch.to(self.device, non_blocking=True)


        # автокастинг (включён только при CUDA)
        with autocast(self.device.type):
            x_hat, _adj_hat, mu, logvar = self.model(batch)
            loss_recon = recon_loss(x_hat, batch.x, mode=self.train_cfg.get("recon_mode", "mse"))
            loss_kld = kld_loss(mu, logvar)
            loss = loss_recon + beta * loss_kld

        if train:
            # print(
            #     f"\n[STEP] Backward pass done. Loss={loss.item():.4f}, "
            #     f"Recon={loss_recon.item():.4f}, KLD={loss_kld.item():.4f}")

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

        print(f"\n[{mode.upper()}] Epoch {epoch}/{self.epochs}, beta={beta:.4f}")

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

        print(f"\n[{mode.upper()}] Epoch {epoch} done: loss={avg['loss']:.4f}, recon={avg['recon']:.4f}, kld={avg['kld']:.4f}")
        return avg


    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_all_metrics(self, loader: DataLoader, beta: float) -> Dict[str, float]:
        """
        Считает все метрики из CM.ALL_METRICS, включая VAE-losses (ELBO, Node_MSE и т.д.),
        классические и химические.
        """
        print("\n[METRICS] Starting _compute_all_metrics...")
        self.model.eval()

        # Для VAE-losses соберём последнюю батч-статистику
        print("[METRICS] Fetching last batch for VAE loss computation...")
        last_batch = None
        for batch in loader:
            last_batch = batch.to(self.device, non_blocking=True)

        # прямой прогон последнего батча
        print("[METRICS] Running forward pass on last batch...")
        with autocast(self.device.type):
            x_hat, edge_logits, mu, logvar = self.model(last_batch)

        metrics: Dict[str, float] = {}

        # VAE-losses
        print("[METRICS] Computing VAE losses: Node_MSE, KLD, ELBO...")
        rec = CM.compute_node_mse(x_hat, last_batch.x)
        kld = CM.compute_kld(mu, logvar)
        elbo = CM.compute_elbo(rec, kld, beta)
        metrics.update({
            "ELBO": elbo,
            "Node_MSE": rec,
            "KLD": kld,
        })
        print(f"[METRICS] ELBO={elbo:.4f}, Node_MSE={rec:.4f}, KLD={kld:.4f}")

        # Edge BCE только если есть предсказанные логиты
        if edge_logits is not None:
            print("[METRICS] Computing Edge_BCE...")
            bce = CM.compute_edge_bce(edge_logits, last_batch.edge_index, last_batch.num_nodes)
            metrics["Edge_BCE"] = bce
            print(f"[METRICS] Edge_BCE={bce:.4f}")
        else:
            print("[METRICS] Skipping Edge_BCE: edge_logits is None")

        # химические метрики по SMILES
        if hasattr(self.model, "decode_smiles"):
            print("[METRICS] Model has decode_smiles. Computing SMILES-based chemical metrics...")
            orig: List[str] = []
            recon: List[str] = []
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)

                _, _, z, _ = self.model(batch)
                orig.extend([d.smiles for d in batch.to_data_list()])
                recon.extend(self.model.decode_smiles(z))

            print("[METRICS] Computing individual chemical metrics...")
            for name, fn in CM.ALL_METRICS.items():
                if name in metrics or name in ("AUC", "PR", "MAE"):  # пропускаем не применимые
                    continue
                try:
                    if name == "Novelty":
                        value = fn(recon, orig)
                    elif name in ("Avg_Tanimoto", "Avg_GED", "Atom_Count_Diff", "Bond_Count_Diff", "Degree_JS"):
                        value = fn(orig, recon)
                    else:
                        value = fn(recon)
                    metrics[name] = value
                    print(f"[METRIC] {name:>20s} = {value:.4f}")
                except Exception as e:
                    print(f"[WARNING] Failed to compute metric '{name}': {e}")

        else:
            print("[METRICS] Skipping chemical metrics: model does not implement decode_smiles")

        print("[METRICS] Finished _compute_all_metrics.")

        return metrics


    # ------------------------------------------------------------------
    def fit(self) -> None:
        """
        Основной цикл обучения с поддержкой early-stopping по patience_max.
        """
        print("\n[INIT] Starting fit()")

        best_val = float('inf')
        early_counter = 0
        patience_max = int(self.train_cfg.get("patience_max", self.train_cfg.get("patience", 15)))

        print(f"\n[INIT] patience_max = {patience_max}")

        history = {
            "loss": {"train": [], "val": []},
            "recon": {"train": [], "val": []},
            "kld": {"train": [], "val": []},

            "ELBO": [],
            "Node_MSE": [],
            "Edge_BCE": [],
            'KLD': [],

            "AUC": [],
            "PR": [],
            "MAE": [],

            'Validity': [],
            'Uniqueness': [],
            'Novelty': [],
            'Avg_Tanimoto': [],
            'Avg_GED': [],
            'Atom_Count_Diff': [],
            'Bond_Count_Diff': [],
            'Degree_JS': [],
            'Internal_Diversity': []
        }
        train_times = []
        val_times = []

        # заранее инициализируем загрузчики и запомним текущий bs
        current_bs = self.bs
        print(f"\n[INIT] Initial batch size: {current_bs} "
              f"\n[INIT] Creating train_loader...")
        train_loader = self._make_loader(self.train_ds, current_bs, True)
        print("\n[INIT] train_loader created! "
              "\n[INIT] Creating val_loader...")
        val_loader = self._make_loader(self.val_ds, current_bs, False)
        print("\n[INIT] val_loader created.")

        def format_time(seconds: float) -> str:
            minutes = int(seconds) // 60
            sec = seconds % 60
            return f"{minutes}m {sec:.1f}s"

        total_start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            print(f"\n[EPOCH {epoch:03d}] ----------------------------")
            # — адаптивный batch_size: если включён и настал момент увеличения —
            new_bs = self.bs
            if self.adapt_bs and epoch % self.bs_every == 0 and self.bs < self.bs_max:
                new_bs = min(self.bs * 2, self.bs_max)
                print(f"\n[INFO] Increased batch_size to {new_bs}")

            # создаём загрузчики
            # — если batch_size изменился, пересоздаём загрузчики —
            if new_bs != current_bs:
                current_bs = new_bs
                print(f"\n[BATCH] Rebuilding loaders with batch size = {current_bs}"
                      f"\n[INIT] Creating train_loader...")
                train_loader = self._make_loader(self.train_ds, current_bs, True)
                print("\n[INIT] train_loader created! "
                      "\n[INIT] Creating val_loader...")
                val_loader = self._make_loader(self.val_ds, current_bs, False)
                print("\n[INIT] val_loader created.")

            # === TRAIN =================================================
            t_train0 = time.perf_counter()
            print("\n[TRAIN] Running training epoch...")
            train_metrics = self._run_epoch(train_loader, epoch, True)
            t_train = time.perf_counter() - t_train0
            train_times.append(t_train)
            print(f"\n[TRAIN] Time: {format_time(t_train)} | Metrics: {train_metrics}")


            # === VAL ===================================================
            t_val0 = time.perf_counter()
            print("\n[VAL] Running validation epoch...")
            val_metrics = self._run_epoch(val_loader, epoch, False)
            t_val = time.perf_counter() - t_val0
            val_times.append(t_val)
            print(f"\n[VAL] Time: {format_time(t_val)} | Metrics: {val_metrics}")

            # === TIME LOGGING =========================================
            epoch_time = t_train + t_val
            self.writer.add_scalar("Time/train", t_train, epoch)
            self.writer.add_scalar("Time/val", t_val, epoch)
            self.writer.add_scalar("Time/epoch", epoch_time, epoch)
            print(f"\n[TIME] Epoch duration: {format_time(epoch_time)}")

            # запись истории
            for m in ('loss', 'recon', 'kld'):
                history[m]['train'].append(train_metrics[m])
                history[m]['val'].append(val_metrics[m])

            if self.track_metrics:
                print("\n[METRICS] Computing additional metrics...")
                beta = self._beta(epoch)
                all_metrics = self._compute_all_metrics(val_loader, beta)
                for name, value in all_metrics.items():
                    val_metrics[name] = value
                    history[name].append(value)
                    print(f"\n[METRIC] {name:>15s} = {value:.4f}")
                    self.writer.add_scalar(f"Metrics/{name}", value, epoch)

            print(
                f"\n[SUMMARY] Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}"
            )

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
                print(f"\n[CHECKPOINT] New best model saved at: {ckpt_path} (val_loss={val_loss:.4f})")
            else:
                early_counter += 1
                print(f"\n[EARLY STOPPING] No improvement. Early counter: {early_counter}/{patience_max}")
                # адаптивный weight_decay
                if self.adapt_wd:
                    self.wd_counter += 1
                    if self.wd_counter >= self.wd_patience:
                        self.current_wd *= self.wd_factor
                        for pg in self.optim.param_groups:
                            pg['weight_decay'] = self.current_wd
                        print(f"\n[WD] Reduced weight_decay to {self.current_wd:.2e}")
                        self.wd_counter = 0
                # early stopping
                if early_counter >= patience_max:
                    print(f"\n[EARLY STOPPING] Stopping at epoch {epoch} (no improvement for {patience_max} epochs)")
                    break

            # адаптивный lr
            if self.adapt_lr:
                if self.scheduler_type == "plateau":
                    # ReduceLROnPlateau ждёт float-метрику
                    self.scheduler.step(val_loss)
                    print(f"\n[LR] Plateau scheduler step with val_loss = {val_loss:.4f}")
                else:
                    # CosineAnnealingLR не ждёт метрику — просто шаг по эпохе
                    self.scheduler.step()
                    print("\n[LR] Scheduler step (non-plateau)")

            # адаптивный beta
            if self.adapt_beta and train_metrics["recon"] < self.beta_rec_thresh and self.beta_max < self.max_beta:
                old = self.beta_max
                self.beta_max = min(self.beta_max * self.beta_factor, self.max_beta)
                print(f"\n[BETA] Increased beta_max from {old:.3f} to {self.beta_max:.3f}")


        save_dir = Path(self.writer.log_dir)
        print("[SAVE] Saving training curves and plots...")

        # Основные кривые: loss, recon, kld
        plot_training_curves(
            history=history,
            metrics=["loss", "recon", "kld"],
            save_dir=save_dir
        )
        # VAE‑losses: ELBO, Node_MSE, Edge_BCE, KLD
        plot_training_curves(
            history=history,
            metrics=["ELBO", "Node_MSE", "Edge_BCE", "KLD"],
            save_dir=save_dir
        )
        # Химические метрики (доля, новизна)
        plot_training_curves(
            history=history,
            metrics=['Validity', 'Uniqueness', 'Novelty'],
            save_dir=save_dir
        )
        # Химические метрики (сходство и диффы)
        plot_training_curves(
            history=history,
            metrics=['Avg_Tanimoto', 'Avg_GED', 'Atom_Count_Diff', 'Bond_Count_Diff', 'Degree_JS', 'Internal_Diversity'],
            save_dir=save_dir
        )
        # Время тренировки и валидации
        plot_times(
            train_times,
            val_times,
            save_dir
        )

        total_time = time.perf_counter() - total_start
        self.writer.add_scalar("Time/total_training", total_time)
        if self.use_mlflow:
            mlflow.end_run()
        print(f"\n[INFO] Total training time: {format_time(total_time)}")
        self.writer.close()

# -----------------------------------------------------------------------------
# Convenience API
# -----------------------------------------------------------------------------

def run_training_vae(
        model: nn.Module,
        train_ds,
        val_ds,
        train_cfg: Dict,
        log_cfg: Dict):
    """
       Удобный API: принимает датасеты, создаёт тренер и запускает обучение.
    """
    print("\n[ENTRY] Launching VAE training...")
    trainer = VAETrainer(
        model,
        train_ds,
        val_ds,
        train_cfg,
        log_cfg,
    )
    trainer.fit()
