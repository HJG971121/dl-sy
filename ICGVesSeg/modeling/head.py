import logging
import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Callable, List



logger = logging.getLogger(__name__)

class VesSegHead(nn.Module, ABC):
    def __init__(
        self,
        loss_function,
        in_channel: int = 64,
        out_channel: int = 1,
        is_drop: bool = False,

    ):
        super().__init__()

        self.is_drop = is_drop
        self.loss_function = loss_function

        self.out_layer = torch.nn.Conv2d(
            in_channel, out_channel,kernel_size=1,stride=1)

        if self.is_drop:
            self.drop_out = nn.Dropout(p = 0.5)

    def resize(self, image: torch.Tensor, size: tuple):
        return nn.functional.interpolate(image,size=size,mode='bicubic', align_corners=True)

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None):

        # drop layer
        if self.is_drop:
            features = self.drop_out(features)

        # out layer
        logits = self.out_layer(features)
        logits = torch.sigmoid(logits)

        if self.training:
            losses = self.losses(logits, targets)
            return logits, losses
        else:
            return logits, {}

    def losses(self, logits: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor):
        """
        define your losses
        """
        losses = {}
        loss = self.loss_function(True, logits, targets)
        for k, v in loss.items():
            losses[k]=v

        return losses







