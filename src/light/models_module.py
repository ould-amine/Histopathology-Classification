import torch
import pytorch_lightning as pl
import torchmetrics
import math

from models import get_model


class LitModule(pl.LightningModule):
    """
    Lightning module for LoRA and full fine-tuning.
    """

    def __init__(self, cfg):
        super().__init__()

        # === Model ===
        self.model = get_model(cfg)
        self.cfg = cfg

        # === Loss & metric ===
        self.criterion = getattr(torch.nn, cfg.training.loss)()
        self.metric = torchmetrics.classification.BinaryAccuracy()

        # === EMA (Exponential Moving Average) ===
        self.use_ema = cfg.training.ema.enabled
        self.ema_decay = cfg.training.ema.decay
        self.ema_state = None
        self._raw_state_backup = None

    def forward(self, x):
        return self.model(x)

    # === EMA utils ===
    @torch.no_grad()
    def _init_ema(self):
        self.ema_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def _update_ema(self):
        if not self.use_ema:
            return

        if self.ema_state is None:
            self._init_ema()
            return

        current_state = self.model.state_dict()

        for name, value in current_state.items():
            if name in self.ema_state:
                self.ema_state[name].mul_(self.ema_decay).add_(
                    value.detach(), alpha=1.0 - self.ema_decay
                )

    @torch.no_grad()
    def _swap_in_ema_weights(self):
        if not self.use_ema or self.ema_state is None:
            return

        self._raw_state_backup = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if k in self.ema_state
        }

        state = self.model.state_dict()
        for name, value in self.ema_state.items():
            state[name].copy_(value)

    @torch.no_grad()
    def _restore_raw_weights(self):
        if self._raw_state_backup is None:
            return

        state = self.model.state_dict()
        for name, value in self._raw_state_backup.items():
            state[name].copy_(value)

        self._raw_state_backup = None

    # === Hooks ===
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema()

    def on_validation_epoch_start(self):
        self._swap_in_ema_weights()

    def on_validation_epoch_end(self):
        self._restore_raw_weights()

    # === Steps ===
    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = torch.squeeze(self(images))
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = torch.squeeze(self(images))
        loss = self.criterion(outputs, labels)

        # Accuracy
        accuracy = self.metric(outputs, labels)

        self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("epoch_idx", float(self.current_epoch), on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        outputs = torch.squeeze(self(images))
        loss = self.criterion(outputs, labels)

        accuracy = self.metric(outputs, labels)

        self.log("Test_accuracy", accuracy, prog_bar=True, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    # === Optimizer ===
    def configure_optimizers(self):
        optimizer_name = getattr(self.cfg.training, "optimizer", "AdamW")
        optimizer_cls = getattr(torch.optim, optimizer_name)

        # === Special case: layer-wise LR ===
        if self.cfg.models.name == "dino_full_finetune":
            optimizer = optimizer_cls(
                [
                    {
                        "params": self.model.backbone.model.encoder.layer[
                            :6
                        ].parameters(),
                        "lr": self.cfg.training.lr.backbone_a,
                    },
                    {
                        "params": self.model.backbone.model.encoder.layer[
                            6:
                        ].parameters(),
                        "lr": self.cfg.training.lr.backbone_b,
                    },
                    {
                        "params": self.model.classifier.parameters(),
                        "lr": self.cfg.training.lr.classifier,
                    },
                ],
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.epochs,
            eta_min=self.cfg.training.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class LitModuleDANN(pl.LightningModule):
    """
    Lightning module for DANN training.
    """

    def __init__(self, cfg):
        super().__init__()

        self.model = get_model(cfg)
        self.cfg = cfg

        # === Losses ===
        self.criterion_classifier = getattr(torch.nn, cfg.training.loss.classifier)()
        self.criterion_center = getattr(torch.nn, cfg.training.loss.center)()

        self.center_weight = cfg.training.center_weight
        self.metric = torchmetrics.classification.BinaryAccuracy()

    def forward(self, x, lambda_=1.0):
        return self.model(x, lambda_=lambda_)

    def training_step(self, batch, batch_idx):
        x, y_cls, y_center = batch

        # === Lambda schedule (DANN) ===
        p = self.current_epoch / self.trainer.max_epochs
        lambda_ = 2 / (1 + math.exp(-10 * p)) - 1

        pred_cls, pred_center = self(x, lambda_)

        # === Loss ===
        loss_cls = self.criterion_classifier(torch.squeeze(pred_cls), y_cls)
        loss_center = self.criterion_center(pred_center, y_center)

        loss = loss_cls + self.center_weight * loss_center

        # === Logs ===
        self.log("lambda", lambda_, on_epoch=True, on_step=False)
        self.log(
            "train_classifier_loss",
            loss_cls,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_center_loss",
            loss_center,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_cls, _ = batch

        pred_cls, _ = self(x)

        loss = self.criterion_classifier(torch.squeeze(pred_cls), y_cls)

        acc = self.metric(torch.squeeze(pred_cls), y_cls.int())

        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("epoch_idx", float(self.current_epoch), on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.backbone.model.encoder.layer[:6].parameters(),
                    "lr": self.cfg.training.lr.backbone_a,
                },
                {
                    "params": self.model.backbone.model.encoder.layer[6:].parameters(),
                    "lr": self.cfg.training.lr.backbone_b,
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.cfg.training.lr.classifier,
                },
                {
                    "params": self.model.center_head.parameters(),
                    "lr": self.cfg.training.lr.center,
                },
            ],
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.epochs,
            eta_min=self.cfg.training.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# === Factory ===
def get_module(cfg):
    if cfg.models.method == "DANN":
        return LitModuleDANN(cfg)
    return LitModule(cfg)
