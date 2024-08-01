# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional, Tuple, List
import torch
from torch import nn

from detectron2.modeling.backbone import Backbone
from detectron2.projects.segmentation.data import SegImageSample


class SemanticSegmentor(nn.Module):
    def __init__(self,
                 *,
                 backbone: Backbone,
                 sem_seg_head: nn.Module,
                 pixel_mean: float,
                 pixel_std: float,
                 ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, samples: List[SegImageSample]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = [x.image.to(self.device) for x in samples]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        targets = [x.label.to(self.device) for x in samples]

        return torch.Tensor(images), torch.Tensor(targets)

    def forward(self, samples: List[SegImageSample]) -> List[SegImageSample]:
        images, targets = self.preprocess_image(samples)

        if self.training:
            x = self.backbone(images)
            _, losses = self.sem_seg_head(x, targets)

            self.prelosses = losses

            return losses
        else:
            x = self.backbone(input)
            results, _ = self.sem_seg_head(x, targets)

        for result, sample in zip(results, samples):
            h, w = sample.img_size
            sample.pred_sem_seg = result[:, :h, :w]

        return samples

