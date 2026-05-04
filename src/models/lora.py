import torch
from peft import LoraConfig, get_peft_model
from models.backbone_and_heads import (
    get_feature_extractor,
    get_linear_probing,
    Backbone,
)
from collections import OrderedDict


def get_backbone_with_classifier(dino=True, dropout=0.3):
    """
    Build backbone + classification head.
    """

    backbone = get_feature_extractor(
        from_hugging_face=True,
        dino=dino,
    )

    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("backbone", Backbone(backbone)),
                (
                    "classifier",
                    get_linear_probing(
                        backbone.config.hidden_size,
                        dropout=dropout,
                    ),
                ),
            ]
        )
    )

    return model


def lora_model(cfg):
    """
    Build LoRA model from backbone + classifier.
    """

    backbone_with_heads = get_backbone_with_classifier(
        dino=(cfg.models.backbone == "DINO"),
        dropout=cfg.models.head_dropout,
    )

    config = LoraConfig(
        r=cfg.models.r,
        lora_alpha=cfg.models.lora_alpha,
        target_modules=cfg.models.target_modules,
        lora_dropout=cfg.models.lora_dropout,
        use_rslora=True,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(backbone_with_heads, config)
    return model
