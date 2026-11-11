"""The three architectures, with the paper's transfer-learning head.

From the supplement:

    "Removing the top layer, we appended a fully connected layer with 256
    features with a Rectified Linear Unit (ReLU) activation function for each
    model. This layer includes dropout regularization to prevent overfitting.
    The final fully connected layer maps the 256 features to the number of
    output classes. [...] Then, all layers were unfrozen and initialized with
    pre-trained weights except newly added layers, and backpropagation was run
    to train the new parameters."

So: ImageNet backbone, head replaced by
``Linear(->256) -> ReLU -> Dropout -> Linear(->n_classes)``, nothing frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import DROPOUT

# Which ImageNet checkpoint each backbone starts from (ResNet152: V2); the
# supplement says only "pre-trained on the ImageNet dataset".
_WEIGHTS = {
    "densenet169": models.DenseNet169_Weights.IMAGENET1K_V1,
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
}


def get_device(preference: str = "auto") -> torch.device:
    """Resolve ``"auto" | "cuda" | "cpu"`` to a device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def classifier_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """The 256-unit bottleneck head shared by the three architectures."""
    return nn.Sequential(
        nn.Linear(in_features, 256, bias=True),
        nn.ReLU(inplace=False),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes, bias=True),
    )


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Backbone + fresh classifier head, every parameter trainable."""
    try:
        dropout = DROPOUT[architecture]
    except KeyError:
        raise ValueError(
            f"unknown architecture {architecture!r}; expected one of {sorted(DROPOUT)}"
        ) from None

    weights = _WEIGHTS[architecture] if pretrained else None

    if architecture == "densenet169":
        model = models.densenet169(weights=weights)
        model.classifier = classifier_head(
            model.classifier.in_features, num_classes, dropout
        )
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier = classifier_head(
            model.classifier[1].in_features, num_classes, dropout
        )
    else:  # resnet152
        model = models.resnet152(weights=weights)
        model.fc = classifier_head(model.fc.in_features, num_classes, dropout)

    # "all layers were unfrozen": fine-tuning runs on the whole network.
    for param in model.parameters():
        param.requires_grad = True
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
