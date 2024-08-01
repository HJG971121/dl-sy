import logging
import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Callable, List



logger = logging.getLogger(__name__)

class OutLayer(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int = 1,
                 deep_supervision: int = 0):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.in_channels = in_channels
        self.out_channel = out_channel


        if deep_supervision == 0:
            self.add_module(
                'out_layer',
                torch.nn.Conv2d(in_channels[0], out_channel, kernel_size=1, stride=1)
            )

        elif deep_supervision == 1:
            for i in range(len(in_channels)):
                self.add_module(
                    'out_layer' + str(i),
                    torch.nn.Conv2d(in_channels[i], out_channel, kernel_size=1, stride=1)
                )

        elif deep_supervision in [2, 3]:
            for i in range(1, len(in_channels)):
                layers = []
                for ii in range(i, 0, -1):
                    layers.append(
                        nn.ConvTranspose2d(in_channels[ii], in_channels[ii - 1],
                                           kernel_size=2, stride=2, padding=0)
                    )

                self.add_module(
                    'interloss_up' + str(i),
                    nn.Sequential(*layers)
                )
            if deep_supervision == 2:
                for i in range(len(in_channels)):
                    self.add_module(
                        'out_layer' + str(i),
                        torch.nn.Conv2d(in_channels[0], out_channel, kernel_size=1, stride=1)
                    )
            elif deep_supervision == 3:
                self.add_module(
                    'out_layer',
                    torch.nn.Conv2d(in_channels[0]*len(in_channels), out_channel, kernel_size=1, stride=1)
                )

    def forward(self, y):
        deep_supervision = self.deep_supervision
        [y3, y2, y1, y0] = y

        if deep_supervision == 0:
            return [self.out_layer(y0)]
        elif deep_supervision == 1:
            return [out_layer(feature) for (out_layer, feature) in
                  zip([self.out_layer3, self.out_layer2, self.out_layer1, self.out_layer0], y)]
        elif deep_supervision in [2, 3]:
            y3 = self.interloss_up3(y3)
            y2 = self.interloss_up2(y2)
            y1 = self.interloss_up1(y1)
            if deep_supervision == 2:
                return [out_layer(feature) for (out_layer, feature) in
                  zip([self.out_layer3, self.out_layer2, self.out_layer1, self.out_layer0],
                      [y3, y2, y1, y0])]
            elif deep_supervision ==3:
                return [self.out_layer(torch.concatenate([y3, y2, y1, y0], dim=1))]


class MTTGnrtHead(nn.Module, ABC):
    def __init__(
        self,
        loss_function,
        in_channels: List[int],
        out_channel: int = 1,
        is_drop: bool = False,
        deep_supervision: int = 0,
        interloss_weight: List[int] = [0.125, 0.125, 0.125, 0.5]
    ):
        super().__init__()

        assert deep_supervision <= 3, f'deep_supervision should be 0, 1, 2, 3'

        self.is_drop = is_drop
        self.loss_function = loss_function
        self.deep_supervision = deep_supervision
        self.interloss_weight = interloss_weight

        self.outlayer = OutLayer(in_channels, out_channel, deep_supervision)

        if self.is_drop:
            self.drop_out = nn.Dropout(p = 0.2)


    def resize(self, image: torch.Tensor, size: tuple, mode):
        if mode!='nearest':
            return nn.functional.interpolate(image,size=size,mode=mode, align_corners=True)
        else:
            return nn.functional.interpolate(image, size=size, mode=mode)

    def forward(self, features: List[torch.Tensor], targets: Optional[torch.Tensor] = None):

        # drop layer
        if self.training and self.is_drop:
            features[-1] = self.drop_out(features[-1])
            # features = [self.drop_out(feature) for feature in features]

        # out layer
        logits = self.outlayer(features)
        # logits = [torch.sigmoid(logit) for logit in logits]

        if self.training:
            vmask = targets[:, 1:2]
            targets = targets[:, 0:1]
            # vmask = torch.cat([vmask, vmask], dim=1)
            # targets = torch.cat([targets, 1-targets], dim=1)
            losses = self.losses(logits, targets, vmask)
            return logits, losses
        else:
            return logits

    def losses(self, logits: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor, vmask: torch.Tensor):
        """
        define your losses
        """
        losses = {}
        for idx, logit in enumerate(logits):
            is_output = True if idx == len(logits) - 1 else False

            if self.deep_supervision == 1:
                loss = self.loss_function(is_output, logit,
                                          self.resize(targets, logit.shape[-2:], mode='bicubic'),
                                          self.resize(vmask, logit.shape[-2:], mode='nearest'))
            else:
                loss = self.loss_function(is_output, logit,targets, vmask)


            for k, v in loss.items():
                if k in losses.keys():
                        losses[k].append(v)
                else:
                    losses[k] = [v]

        if self.deep_supervision in [1, 2]:
            w_loss = [w * l for (w, l) in zip(self.interloss_weight, losses['total_loss'])]
            losses['total_loss'] = torch.stack(w_loss, dim=0).sum(dim=0)
        else:
            losses['total_loss'] = losses['total_loss'][-1]

        return losses





