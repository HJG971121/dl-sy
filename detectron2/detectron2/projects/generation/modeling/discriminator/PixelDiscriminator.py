import torch
import torch.nn as nn
from typing import List, Tuple, Union

from detectron2.projects.generation.modeling.layers.abn_blocks import Conv2dABN

class PixelDiscriminator(nn.Module):
    def __init__(self,
                 in_channel: int,
                 feat0: int = 64,
                 abn: int = 0):
        super(PixelDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_channel, out_channels=feat0, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            Conv2dABN(in_channel=feat0, out_channel=feat0*2,
                      kernel_size=1, stride=1, padding=0, bias=False, abn=abn, abn_para=0.2),
            Conv2dABN(feat0*2, 1, kernel_size=1, stride=1, padding=0, bias=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)