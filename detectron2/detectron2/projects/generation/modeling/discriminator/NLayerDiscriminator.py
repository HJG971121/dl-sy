import torch
import torch.nn as nn
from typing import List, Tuple, Union

from detectron2.projects.generation.modeling.layers.abn_blocks import Conv2dABN

class NLayerDiscriminator(nn.Module):

    def __init__(self,
                 in_channel: int,
                 feats = [32, 64, 128],
                 num_layer: int = 3,
                 abn: int = 0):

        super(NLayerDiscriminator, self).__init__()
        assert len(feats) == num_layer, f'len(feats) should be the same as num_layer {num_layer}, but got {len(feats)}.'

        layers = []
        for i in range(num_layer):
            if i == 0:
                layers+=[
                    nn.Conv2d(in_channels=in_channel, out_channels=feats[i], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                layers.append(Conv2dABN(in_channel=feats[i-1], out_channel=feats[i],
                                        kernel_size=4, stride=2, padding=1,bias=False, abn=abn,
                                        abn_para={'activation': 'leaky_relu',
                                                  'activation_param': 0.2}))
        layers+=[nn.Conv2d(feats[-1], 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

