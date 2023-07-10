from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from metrics import symmetric_mean_absolute_percentage_error_100, symmetric_mean_absolute_percentage_error_200
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and name.startswith("resnet")
                     and callable(models.__dict__[name]))


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def feature(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError
    

def load_resnet_backbone(arch: str, in_channels: int, pretrained: bool) -> nn.Module:
    assert arch in model_names
    assert arch.startswith('resnet'), 'Only support resnet familiy'

    backbone = models.__dict__[arch](pretrained=pretrained)

    # for in_channels inconsistent
    if in_channels != 3:
        backbone.conv1 = nn.Conv2d(in_channels,
                                   backbone.conv1.out_channels,
                                   kernel_size=backbone.conv1.kernel_size,
                                   stride=backbone.conv1.stride,
                                   padding=backbone.conv1.padding,
                                   bias=backbone.conv1.bias)

    backbone = nn.Sequential(OrderedDict([
        ('conv1', backbone.conv1),
        ('bn1', backbone.bn1),
        ('relu', backbone.relu),
        ('maxpool', backbone.maxpool),

        ('layer1', backbone.layer1),
        ('layer2', backbone.layer2),
        ('layer3', backbone.layer3),
        ('layer4', backbone.layer4),
    ]))

    return backbone


class VanillaRegressor(BaseModel):
    def __init__(self, arch: str, in_channels: int, pretrained: bool, num_outputs: int):
        super(VanillaRegressor, self).__init__()

        self.backbone = load_resnet_backbone(arch, in_channels, pretrained)
        self.regressor = nn.Linear(in_features=512, out_features=num_outputs)

        self.arch = arch
        self.in_channels = in_channels
        self.num_outputs = num_outputs

    def forward(self, x):
        # if x is fundus, x.shape: [m, 3, 384, 384]
        x = self.backbone(x)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [m, 512, 1, 1]
        x = torch.flatten(x, 1)  # [m, 512]
        x = self.regressor(x)
        return x

    def feature(self, x):
        return self.backbone(x)

    def loss(self, pred, target):
        return F.mse_loss(pred, target)
    
