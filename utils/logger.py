# utils/logger.py
from __future__ import annotations

"""Simple wrapper around TensorBoard (and optional MLflow).

* If TensorBoard is available → logs to `log_dir/run_name/`.
* If mlflow is installed → logs metrics to current experiment; otherwise, it silently skips.
"""

import os
from pathlib import Path
from typing import Dict

from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore

__all__ = ["Logger"]


class Logger:
    def __init__(self, log_dir: str | Path, run_name: str = "run") -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_writer = None
        if SummaryWriter is not None:
            tb_path = Path(log_dir) / f"{run_name}_{timestamp}"
            tb_path.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_path))

        self.use_mlflow = mlflow is not None and os.getenv("MLFLOW_TRACKING_URI") is not None
        if self.use_mlflow:
            mlflow.set_experiment("mol_gnn_project")
            mlflow.start_run(run_name=f"{run_name}_{timestamp}")

    # --------------------------------------------------------------
    def log_scalars(self, metrics: Dict[str, float], step: int):
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

    # --------------------------------------------------------------
    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_mlflow:
            mlflow.end_run()
