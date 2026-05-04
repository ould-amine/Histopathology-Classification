from models.lora import lora_model, get_backbone_with_classifier
from models.dann import DANNModel


def get_model(cfg):
    """
    Model factory based on configuration.
    """

    # === Sanity checks ===
    if cfg.models.method not in ["LoRA", "full_finetune", "DANN"]:
        raise ValueError(
            f"Method {cfg.models.method} not implemented "
            "(use: LoRA, full_finetune, DANN)"
        )

    if cfg.models.backbone not in ["DINO", "Phikon"]:
        raise ValueError(
            f"Backbone {cfg.models.backbone} not implemented (use: DINO, Phikon)"
        )

    # === Model selection ===
    if cfg.models.method == "LoRA":
        model = lora_model(cfg)

    elif cfg.models.method == "DANN":
        model = DANNModel(
            dino=(cfg.models.backbone == "DINO"),
            head_dropout=cfg.models.head_dropout,
        )
        model.train()

    elif cfg.models.method == "full_finetune":
        model = get_backbone_with_classifier(
            dino=(cfg.models.backbone == "DINO"),
            dropout=cfg.models.head_dropout,
        )
        model.train()

    else:
        raise ValueError(f"Method {cfg.models.method} not implemented")

    return model
