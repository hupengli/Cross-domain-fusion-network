import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def conv_bn(
    inp,
    oup,
    stride,
    conv_layer=nn.Conv2d,
    norm_layer=nn.BatchNorm2d,
    nlin_layer=nn.ReLU,
):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup, momentum=1, affine=True),
        nlin_layer(inplace=True),
    )


def conv_1x1_bn(
    inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU
):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup, momentum=1, affine=True),
        nlin_layer(inplace=True),
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def make_divisible(x, divisible_by=8):
    import numpy as np

    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl="RE"):
        super(MobileBottleneck, self).__init__()
        # assert stride in [1, 2]
        # assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == "RE":
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == "HS":
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = nn.Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, width_mult=1):
        super(MobileNetV3, self).__init__()
        input_channel = 16

        mobile_setting = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, True, "RE", 1],
            [3, 72, 24, False, "RE", 2],
            [5, 96, 40, True, "HS",  1],
            [5, 240, 80, True, "HS", (2,1)],
        
        ]

        # building first layer

        self.features = [conv_bn(3, input_channel, (1, 2), nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(
                MobileBottleneck(
                    input_channel, output_channel, k, s, exp_channel, se, nl
                )
            )
            input_channel = output_channel

        # # building last several layers
        # if mode == "large":
        #     last_conv = make_divisible(960 * width_mult)
        #     self.features.append(
        #         conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish)
        #     )
        #     self.features.append(nn.AdaptiveAvgPool2d(1))
        #     self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        #     self.features.append(Hswish(inplace=True))
        # elif mode == "small":
        #     last_conv = make_divisible(576 * width_mult)
        #     self.features.append(
        #         conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish)
        #     )
        #     # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        #     self.features.append(nn.AdaptiveAvgPool2d(1))
        #     self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        #     self.features.append(Hswish(inplace=True))
        # else:
        #     raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),  # refer to paper section 6
        #     nn.Linear(last_channel, n_class),
        # )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.mean(3).mean(2)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def test():
    from thop import profile

    net = MobileNetV3(width_mult=0.75)
    x = torch.randn(2, 3, 30, 200)
    y = net(x)
    macs, params = profile(net, inputs=(x,))
    print(y.size(), macs / 1000000, params / 1000000)


if __name__ == "__main__":

    test()
