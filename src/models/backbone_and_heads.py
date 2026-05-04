import torch
from transformers import AutoModel

DINO_MODEL_NAME = "facebook/dinov2-small"
PHIKON_MODEL_NAME = "owkin/phikon-v2"


def get_feature_extractor(device=None, from_hugging_face=False, dino=True):
    """
    Load feature extractor (DINOv2 or Phikon).
    """

    if from_hugging_face:
        model_name = DINO_MODEL_NAME if dino else PHIKON_MODEL_NAME
        model = AutoModel.from_pretrained(model_name)
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if device:
            model.to(device)

    model.eval()
    return model


def get_linear_probing(in_features, hidden_dim=256, dropout=0.3, device=None):
    """
    Simple classification head.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, 1),
    )
    if device:
        model.to(device)
    return model


class Backbone(torch.nn.Module):
    """
    Wrapper to extract CLS token from transformer backbone.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(pixel_values=x)
        return output.last_hidden_state[:, 0]  # CLS token
