import numpy as np
import torch
from torch import nn
from typing import Dict

from detectron2.modeling import Backbone
from detectron2.utils.comm import get_rank
from detectron2.utils.logger import setup_logger
from detectron2.layers import ShapeSpec

from ..utils import build_conv_layer
from ..layers import Padding2Even



class UNetBackbone(Backbone):
    """
    UNet backbone:
    Input shape: [n, c, y, x]
    Output shape: [n, feats, y, x]
    """

    def __init__(
        self,
        input_channel,
        feats = [16, 32, 64, 128, 128],
        blocks = [2, 4, 5, 6, 6],
        scales = [2, 2, 2, 2, 2],
        slim=True,
        abn=0):

        super().__init__()
        self.logger = setup_logger(name=__name__, distributed_rank=get_rank())
        self.input_channel = input_channel
        self.feats = feats
        self.blocks = blocks
        self.scales = scales
        self._size_divisibility = int(np.prod(scales))
        num_stages = len(feats)
        assert len(blocks) == num_stages
        assert len(scales) == num_stages

        for i in range(num_stages):
            k = 3 if slim and i>0 else 5
            if i == 0:
                self.add_module(
                    'l'+str(i),
                    build_conv_layer(
                        input_channel,
                        feats[i],
                        blocks[i],
                        stride=scales[i],
                        kernel_size=k,
                        abn=abn
                    )
                )
                self.add_module(
                    'rt'+str(i),
                    build_conv_layer(
                        feats[i],
                        feats[i],
                        blocks[i],
                        kernel_size=k,
                        abn=abn
                    )
                )
                self.add_module(
                    'up'+str(i),
                    nn.Upsample(scale_factor=scales[i], mode='bilinear', align_corners=True)
                    # nn.ConvTranspose2d(feats[i], feats[i - 1], kernel_size=k, stride=2, output_padding=1)
                )

            else:
                self.add_module(
                    'l'+str(i),
                    build_conv_layer(
                        feats[i-1],
                        feats[i],
                        blocks[i],
                        stride=scales[i],
                        kernel_size=k,
                        abn=abn
                    )
                )
                self.add_module(
                    'rt'+str(i),
                    build_conv_layer(
                        feats[i]+feats[i-1],
                        feats[i-1],
                        blocks[i],
                        kernel_size=k,
                        abn=abn
                    )
                )
                self.add_module(
                    'up'+str(i),
                    nn.Upsample(scale_factor=scales[i], mode='bilinear', align_corners=True)
                    # nn.ConvTranspose2d(feats[i],feats[i-1],kernel_size=k,stride=2,padding=1)
                )

        self._out_features = [f'p{i}' for i in range(num_stages)]
        self._out_feature_strides = {
            'p{}'.format(i): int(np.prod(scales[:i]))
            for i in range(num_stages)
        }
        self._out_feature_channels = {
            f'p{i}': feats[0] if i==0 else feats[i-1]
            for i in range(num_stages)
        }

    def output_shape(self):
        ret = {
            feat: ShapeSpec(
                channels=self._out_feature_channels[feat],
                stride=self._out_feature_strides[feat],
            )
            for feat in self._out_features
        }
        return ret

    def size_divisibility(self) -> int:
        return self._size_divisibility

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.l0(x)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)

        y4 = self.rt4(torch.cat([self.up4(x4), x3], dim=1))
        y3 = self.rt3(torch.cat([self.up3(y4), x2], dim=1))
        y2 = self.rt2(torch.cat([self.up2(y3), x1], dim=1))
        y1 = self.rt1(torch.cat([self.up1(y2), x0], dim=1))
        y0 = self.rt0(self.up0(y1))

        return [x4, x3, x2, x1, x0, x], [y4, y3, y2, y1, y0]
