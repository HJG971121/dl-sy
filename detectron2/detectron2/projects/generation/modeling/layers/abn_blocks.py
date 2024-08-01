import inplace_abn
from torch import nn
import torch
import math

abn_blocks = {
    0: inplace_abn.ABN,
    1: inplace_abn.InPlaceABN,
    2: inplace_abn.InPlaceABNSync
}

class BottleneckABN(nn.Module):
    def __init__(self, input_channel, stride=1, downsample=None, abn=0):
        super().__init__()
        self.expansion = 4
        self.downsample = downsample
        self.stride = stride
        mid_channel = input_channel // self.expansion
        abn_block = abn_blocks[abn]

        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=1, bias=False)
        self.bn1 = abn_block(input_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = abn_block(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, input_channel, kernel_size=1, bias=False)
        self.bn3 = abn_block(mid_channel)

    def forward(self, x):
        out = x.clone()

        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class DenseABN(nn.Sequential):
    def __init__(self, input_channel, growth_rate, stride=1, bn_size = 4, drop_rate=0, abn=0):
        super().__init__()
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.stride = stride
        abn_block = abn_blocks[abn]
        mid_channel = self.bn_size * growth_rate

        self.bn1 = abn_block(input_channel)
        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=1, bias=False)
        self.bn2 = abn_block(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # new_features = super(DenseABN, self).forward(x)
        out = x.clone()

        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = torch.nn.functional.dropout(out, p=self.drop_rate)

        return torch.cat([x, out], 1)


class Conv2dABN(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        abn=0,
        abn_para = {
            'activation': 'leaky_relu',
            'activation_param': 0.01
        },
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding,
                              dilation, groups, bias)
        self.abn = abn_blocks[abn](out_channel,
                                   activation = abn_para['activation'],
                                   activation_param = abn_para['activation_param'])

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x

class Padding2Even(nn.Module):
    def __init__(self, padding):
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(x,self.padding,mode='replicate')


class _Transition(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=False,
        abn=0):
        super().__init__()
        self.abn = abn_blocks[abn](in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1, bias=bias)

    def forward(self, x):
        x = self.abn(x)
        x = self.conv(x)
        return x

# Convolutional Block Attention Module
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7, abn=0):
        super(CBAMLayer, self).__init__()
        self.mid_channel = max(channel//reduction, 1)
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, self.mid_channel, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(self.mid_channel, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

# Convolutional Block Attention Module Revision
class CBAMRLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMRLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        y_channel = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        y_spatial = spatial_out * x

        return (y_channel+y_spatial)/2


class AttentionGate(nn.Module):
    def __init__(self, channel_Y, channel_S):
        super(AttentionGate, self).__init__()
        self.W_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_Y, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_Y)
        )
        self.W_S = nn.Sequential(
            nn.Conv2d(channel_S, channel_Y, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_Y)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(channel_Y, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, Y, S):
        Y1 = self.W_Y(Y)
        S1 = self.W_S(S)
        psi = self.relu(Y1+S1)
        psi = self.psi(psi)
        return S*psi
