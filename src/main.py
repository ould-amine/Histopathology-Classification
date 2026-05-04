import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger

import hydra

from omegaconf import DictConfig, OmegaConf
import os
import datetime

from data import get_preprocessing_and_augmentation
from light import DatasetModule, get_module
from utilities import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training entry point.

    This function:
    - Initializes experiment tracking (MLflow)
    - Sets reproducibility parameters
    - Builds data preprocessing and augmentation pipelines
    - Instantiates model and data module
    - Launches training with PyTorch Lightning

    Args:
        cfg (DictConfig): Hydra configuration object containing all experiment parameters
    """
    # === Experiment tracking (MLflow) ===
    experiment_name = "Project-MVA-DLMI-Rapport"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    # Unique run name: model + augmentation + timestamp
    run_name = f"{cfg.models.name}_{cfg.augmentation.name}"
    run_name += "_" + datetime.datetime.now().strftime("%m%d_%H%M")

    tags = {"Mode": "Rapport"}

    # === Reproducibility & numerical precision ===
    set_seed(cfg.training.seed)
    torch.set_float32_matmul_precision(cfg.training.matmul_precision)

    # === Data preprocessing & augmentation ===
    preprocessing, image_transform = get_preprocessing_and_augmentation(cfg)

    # === Callbacks ===

    # Save best model based on validation accuracy
    checkpoint_cb = ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=1,
        mode="max",
        filename="best_model" + run_name,
        dirpath="checkpoint",
        save_weights_only=True,
    )

    # Early stopping to prevent overfitting
    early_stop_cb = EarlyStopping(
        monitor="val_accuracy", patience=cfg.training.patience, mode="max"
    )

    # Log learning rate evolution
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags,
        log_model=True,
    )

    # Log all hyperparameters from Hydra config
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # === Trainer ===
    trainer = Trainer(
        # profiler="simple",
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        # devices=2 if torch.cuda.is_available() else 1,
        # strategy="ddp" if torch.cuda.is_available() else "auto",
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        logger=mlflow_logger,
    )

    # === Data module ===
    data_module = DatasetModule(
        cfg, preprocessing=preprocessing, image_transform=image_transform
    )

    # === Model (Lightning module) ===
    lit_model = get_module(cfg)

    # === Training ===
    trainer.fit(lit_model, datamodule=data_module)


if __name__ == "__main__":
    main()
