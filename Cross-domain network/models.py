import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from config import (
    FEATURE_DIM,
    CLASS_NUM,
    SAMPLE_NUM_PER_CLASS,
    FEATURE_H,
    FEATURE_W,
    BATCH_NUM_PER_CLASS,
)

from torchsnooper import snoop

ACTIVATE = nn.ReLU
from mbv3 import MobileNetV3, make_divisible, conv_1x1_bn, Hswish

class CNNEncoder(nn.Module):
    def __init__(self, factor=1.0):
        super(CNNEncoder, self).__init__()
        self.mb1 = MobileNetV3(factor)

    def forward(self, x):
        x = self.mb1(x)
        return x

class RelationNetwork(nn.Module):
    def __init__(self, factor=1.0):
        super(RelationNetwork, self).__init__()
        last_conv = make_divisible(576 * factor)
        self.features = []
        self.features.append(
            conv_1x1_bn(make_divisible(96 * factor * 2), last_conv, nlin_layer=Hswish)
        )
        # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(
            nn.Conv2d(last_conv, make_divisible(128 * factor), 1, 1, 0)
        )
        self.features.append(Hswish(inplace=True))
        self.features.extend(
            [
                nn.Flatten(),
                nn.Dropout(p=0.8),
                nn.Linear(make_divisible(128 * factor), 1),
                nn.Sigmoid(),
            ]
        )

        self.features = nn.Sequential(*self.features)


    def forward(self, x):
        x = self.features(x)

        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)

if __name__ == "__main__":
    x = torch.randn(CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 30, 200)
    x2 = torch.randn(CLASS_NUM * BATCH_NUM_PER_CLASS, 3, 30, 200)
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(factor=1/3)
    sample_features = feature_encoder(x)
    sample_features = sample_features.view(
        CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, FEATURE_H, FEATURE_W
    )
    sample_features = torch.sum(sample_features, 1).squeeze(1)

    batch_features = feature_encoder(x2)

    sample_features_ext = sample_features.unsqueeze(0).repeat(
        BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1
    )
    batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(
        -1, FEATURE_DIM * 2, FEATURE_H, FEATURE_W
    )
    relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
