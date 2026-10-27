"""Shared pieces: seeding, the U-Net baseline, and the cross-entropy + Dice loss."""
import random

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

IGNORE = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(n_classes, encoder="resnet34"):
    """U-Net with an ImageNet-pre-trained ResNet-34 encoder (fully trainable)."""
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=n_classes)


class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=IGNORE, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.dice = DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)
