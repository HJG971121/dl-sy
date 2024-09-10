
from typing import Optional, Tuple, List, Union
import torch
from torch import nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os

from detectron2.modeling.backbone import Backbone
from detectron2.projects.segmentation.data import ImageSample
from detectron2.utils.events import get_event_storage

class VesselSeg(nn.Module):
    def __init__(self,
                 *,
                 backbone: Backbone,
                 head: nn.Module,
                 pixel_mean: float,
                 pixel_std: float,
                 resize_size: Union[None, int],
                 output_dir: ''
                 ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.resize_size = resize_size
        self.output_dir = output_dir

    @property
    def device(self):
        return self.pixel_mean.device

    def resize(self, image: torch.Tensor, size: tuple):
        return nn.functional.interpolate(image,size=size,mode='bicubic', align_corners=True)

    def preprocess_image(self, samples: List[ImageSample]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = [x.image.to('cuda')[None] for x in samples]
        images = F.normalize(torch.cat(images, dim=0), self.pixel_mean, self.pixel_std)
        resize_size = self.resize_size
        if resize_size is not None:
            images = self.resize(images, (resize_size,resize_size))

        if self.training and samples[0].label is not None:
            targets = [x.label.to('cuda')[None] for x in samples]
            targets = torch.cat(targets, dim=0)
            if resize_size is not None:
                targets = self.resize(targets, (resize_size,resize_size))
        else:
            targets = None

        return images, targets

    def inference(self, images: torch.Tensor, targets: torch.Tensor):
        x = self.backbone(images)
        return self.head(x, targets)

    def forward(self, samples: List[ImageSample]) -> List[ImageSample]:
        images, targets = self.preprocess_image(samples)

        if self.training:
            storage = get_event_storage()
            self.iter = storage.iter

            logits, losses = self.inference(images, targets)
            self.prelosses = losses
            if self.iter % 250 == 0:
                self.save_img(samples[0].img_name, logits[0], images[0], targets[0], losses, is_save=True)
            del logits
            return losses
        else:
            results, _ = self.inference(images, targets)

        # print('results.shape: {}'.format(results.shape))
        for result, sample in zip(results, samples):
            if self.resize_size is not None:
                sample.pred = self.resize(result[None], (self.resize_size, self.resize_size))[0]
            sample.pred = result

        return samples


    def save_img(self, img_name, logit, image, target, losses, is_save=False):
        if not is_save:
            return

        assert len(logit.shape) == 3, f'logit.shape should be 3 dimensions, but got {len(logit.shape)}.'
        assert len(target.shape) == 3, f'target.shape should be 3 dimensions, but got {len(target.shape)}.'
        assert len(image.shape) == 3, f'image.shape should be 3 dimensions, but got {len(image.shape)}.'

        logit = torch.permute(logit, (1, 2, 0)).detach().cpu().numpy()
        image = torch.permute(image, (1, 2, 0)).detach().cpu().numpy()
        target = torch.permute(target, (1, 2, 0)).detach().cpu().numpy()

        loss = ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])

        fig, axes = plt.subplots(1,3,figsize = (10, 3), layout = 'tight')


        for i, (img, subtitle) in enumerate(zip([image, target, logit],
                                              ['input', 'target', 'pred'])):
            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(subtitle)
        fig.suptitle(f'iter {self.iter}, {img_name}, {loss}')

        os.makedirs(os.path.join(self.output_dir,'train'), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir,'train',f'iter{self.iter}_{img_name}.jpg'))

