import numpy as np
import torch
from torch import nn
from typing import Dict

from detectron2.modeling import Backbone
from detectron2.utils.comm import get_rank
from detectron2.utils.logger import setup_logger
from detectron2.layers import ShapeSpec

from ..utils import build_conv_layer, build_dense_layer
from ..layers import CBAMLayer, _Transition



class DenseUNetBackbone(Backbone):
    """
    UNet backbone:
    Input shape: [n, c, y, x]
    Output shape: [n, feats, y, x]
    """

    def __init__(
        self,
        input_channel,
        feats = [16, 32, 64, 128, 256],
        scales = [2, 2, 2, 2, 2],
        num_layers = [2, 2, 2, 2, 2],
        growth_rate = [8, 16, 32, 64, 128],
        slim=True,
        abn=0):

        super().__init__()
        self.logger = setup_logger(name=__name__, distributed_rank=get_rank())
        self.input_channel = input_channel
        self.feats = feats
        self.scales = scales
        self.num_layers = num_layers
        self._size_divisibility = int(np.prod(scales))
        num_stages = len(feats)
        assert len(num_layers) == num_stages
        assert len(scales) == num_stages

        for i in range(num_stages):
            k = 3 if slim and i>0 else 5
            if i == 0:
                self.add_module(
                    'ConvIn',
                    build_conv_layer(
                        input_channel,
                        feats[0],
                        stride=1,
                        kernel_size=3 if slim else 5,
                        abn=abn,
                    )
                )
            else:
                self.add_module(
                    'Denseblock_down'+str(i),
                    build_dense_layer(
                        feats[i-1],
                        stride=scales[i],
                        kernel_size=k,
                        abn=abn,
                        num_layer=num_layers[i],
                        growth_rate=growth_rate[i-1]
                    )
                )

                self.add_module(
                    'Transition_down'+str(i),
                    _Transition(
                        feats[i-1] + num_layers[i] * growth_rate[i-1],
                        feats[i],
                        kernel_size=k,
                        stride=scales[i],
                        abn=abn)
                )

                self.add_module(
                    'Denseblock_up'+str(i),
                    build_dense_layer(
                        feats[i],
                        kernel_size=k,
                        abn=abn,
                        num_layer=num_layers[i],
                        growth_rate=growth_rate[i]
                    )
                )

                self.add_module(
                    'up'+str(i),
                    nn.Upsample(scale_factor=scales[i], mode='bilinear', align_corners=True)
                    # nn.ConvTranspose2d(feats[i], feats[i - 1], kernel_size=k, stride=2, output_padding=1)
                )

                self.add_module(
                    'Transition_up'+str(i),
                    _Transition(
                        feats[i] + num_layers[i] * growth_rate[i] + feats[i-1],
                        feats[i-1],
                        kernel_size=k,
                        stride=1,
                        abn=abn)
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
        x0 = self.ConvIn(x)
        x1 = self.Transition_down1(self.Denseblock_down1(x0))
        x2 = self.Transition_down2(self.Denseblock_down2(x1))
        x3 = self.Transition_down3(self.Denseblock_down3(x2))
        x4 = self.Transition_down4(self.Denseblock_down4(x3))

        y3 = self.Transition_up4(torch.cat([self.up4(self.Denseblock_up4(x4)),x3], dim=1))
        y2 = self.Transition_up3(torch.cat([self.up3(self.Denseblock_up3(y3)),x2], dim=1))
        y1 = self.Transition_up2(torch.cat([self.up2(self.Denseblock_up2(y2)),x1], dim=1))
        y0 = self.Transition_up1(torch.cat([self.up1(self.Denseblock_up1(y1)),x0], dim=1))

        return [x4, x3, x2, x1, x0, x], [y3, y2, y1, y0]