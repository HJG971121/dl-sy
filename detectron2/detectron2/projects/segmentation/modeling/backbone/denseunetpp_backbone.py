import numpy as np
import torch
from torch import nn
from typing import Dict, List

from detectron2.modeling import Backbone
from detectron2.utils.comm import get_rank
from detectron2.utils.logger import setup_logger
from detectron2.layers import ShapeSpec

from ..utils import build_conv_layer, build_dense_layer
from ..layers import Padding2Even


# TODO 改
class DenseUNetPPBackbone(Backbone):
    """
    UNet++ backbone:
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
        abn=0,
        deep_supervision = True,):

        super().__init__()
        self.logger = setup_logger(name=__name__, distributed_rank=get_rank())
        self.input_channel = input_channel
        self.feats = feats
        self.blocks = blocks
        self.scales = scales
        self._size_divisibility = int(np.prod(scales))
        self.deep_supervision = deep_supervision
        num_stages = len(feats)
        assert len(blocks) == num_stages
        assert len(scales) == num_stages

        levels = [i for i in range(num_stages)]
        self.add_module(
            'conv0_0',
            build_conv_layer(
                input_channel,
                feats[0],
                blocks[0],
                kernel_size=3 if slim else 5,
                abn=abn
            )
        )

        for j in levels:
            if j == 0:
                for i in range(len(feats)-j):
                    if i != 0:
                        self.add_module(
                            'conv'+str(i)+'_'+str(j),
                            build_dense_layer(
                                feats[i-1],
                                feats[i],
                                blocks[i],
                                stride=scales[i],
                                kernel_size=3,
                                abn=abn
                            )
                        )

            else:
                for i in range(len(feats)-j):
                    self.add_module(
                        'conv'+str(i)+'_'+str(j),
                        build_dense_layer(
                            feats[i]*j+feats[i+1],
                            feats[i],
                            blocks[i],
                            kernel_size=3,
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up0(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up0(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up0(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up2(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up0(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up0(x1_3)], 1))

        if self.deep_supervision:
            return [x0_1, x0_2, x0_3, x0_4]
        else:
            return [x0_4]
