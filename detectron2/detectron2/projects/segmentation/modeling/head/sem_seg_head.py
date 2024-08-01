import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
from typing import Dict, Optional, Tuple, Union, Callable
from detectron2.layers import ShapeSpec, Conv2d, get_norm

logger = logging.getLogger(__name__)

class BaseSegHead(nn.Module, ABC):
    def __init__(
        self,
        in_channel: int = 64,
        out_channel: int = 1,
        is_drop: bool = False
    ):
        super.__init__()

        self.out_channel = out_channel
        self.is_drop = is_drop

        # define your out layer
        self.out_layer = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1
        )

        if self.is_drop:
            self.drop_out = nn.Dropout(p = 0.5)

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if self.is_drop:
            features = self.drop_out(features)
        logits = self.out_layer(features)

        if self.training:
            return logits, self.losses(logits, targets)
        else:
            return logits, {}

    def losses(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        define your losses
        """

