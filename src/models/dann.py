import torch
import torch.nn as nn
from models.backbone_and_heads import (
    Backbone,
    get_feature_extractor,
    get_linear_probing,
)


# === Gradient Reversal Layer ===
class GradientReversalLayer(torch.autograd.Function):
    """
    Reverse gradients during backpropagation (used in DANN).
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


def grad_reverse(x, lambda_):
    return GradientReversalLayer.apply(x, lambda_)


# === DANN model ===
class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network.

    - Main classifier: predicts target label
    - Domain classifier: predicts data source (center)
    """

    def __init__(self, n_centers=3, head_dropout=0.3, dino=True):
        super().__init__()

        # === Backbone ===
        feature_extractor = get_feature_extractor(from_hugging_face=True, dino=dino)
        hidden_size = feature_extractor.config.hidden_size

        self.backbone = Backbone(feature_extractor)

        # === Task head ===
        self.classifier = get_linear_probing(hidden_size)

        # === Domain head ===
        self.center_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, n_centers),
        )

    def forward(self, x, lambda_=1.0):
        features = self.backbone(x)

        # === Task prediction ===
        class_logits = self.classifier(features)

        # === Domain prediction (with gradient reversal) ===
        reversed_features = grad_reverse(features, lambda_)
        center_logits = self.center_head(reversed_features)

        return class_logits, center_logits
