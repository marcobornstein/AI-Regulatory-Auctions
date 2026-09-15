import torch
from torch import nn
from torchvision import models


class FaceClassifier(nn.Module):
    """VGG-16 convolutional features (randomly initialized) with a linear classification head."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = models.vgg16(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))
