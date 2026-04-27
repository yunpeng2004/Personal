import torch
import torch.nn as nn
from torchvision import models


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        except Exception:
            backbone = models.resnet34(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))   # 1/2
        x = self.maxpool(stem)                      # 1/4
        e1 = self.layer1(x)                         # 1/4
        e2 = self.layer2(e1)                        # 1/8
        e3 = self.layer3(e2)                        # 1/16
        e4 = self.layer4(e3)                        # 1/32
        return stem, e1, e2, e3, e4
