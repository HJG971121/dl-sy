
from typing import Optional, Tuple, List, Union
import torch
from torch import nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from ..CAM.grad_cam import GradCAM
from ..CAM.utils import SemeanticSegmentationTarget
from detectron2.modeling.backbone import Backbone
from detectron2.projects.segmentation.data import ImageSample
from detectron2.utils.events import get_event_storage

class MTTGenerator(nn.Module):
    def __init__(self,
                 *,
                 backbone: Backbone,
                 head: nn.Module,
                 pixel_mean: float,
                 pixel_std: float,
                 resize_size: Union[None, int],
                 output_dir: '',
                 save_features: False,
                 save_cams: False
                 ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.resize_size = resize_size
        self.output_dir = output_dir
        self.save_features = save_features
        self.save_cams = save_cams
        if save_cams:
            self.normalized_masks = None
            self.target_layers = [self.backbone.Transition_up2,
                                  self.backbone.Attention_up2]

    @property
    def device(self):
        return self.pixel_mean.device

    def resize(self, image: torch.Tensor, size: tuple):
        return nn.functional.interpolate(image,size=size,mode='bicubic', align_corners=True)

    def preprocess_image(self, samples: List[ImageSample]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = [x.image.to('cuda')[None] for x in samples]
        resize_size = self.resize_size
        if self.training:
            if resize_size is not None:
                images = [self.resize(image, (resize_size,resize_size)) for image in images]
        images = torch.cat(images, dim=0)
        images = F.normalize(images, self.pixel_mean, self.pixel_std)
        # images = F.normalize(images, torch.mean(images, dim=[0,2,3]), torch.std(images, dim=[0,2,3]))

        # if self.training and samples[0].label is not None:
        targets = [x.label.to('cuda')[None] for x in samples]
        if resize_size is not None:
            targets = [self.resize(target, (resize_size,resize_size)) for target in targets]
        targets = torch.cat(targets, dim=0)

        # else:
            # targets = None

        return images, targets

    def inference(self, images: torch.Tensor, targets: torch.Tensor):
        x, y = self.backbone(images)
        if self.training:
            return self.head(y, targets)
        else:
            return x, y, self.head(y, targets)

    def forward(self, samples: List[ImageSample]) -> List[ImageSample]:
        images, targets = self.preprocess_image(samples)

        if self.training:
            storage = get_event_storage()
            self.iter = storage.iter
            logits, losses = self.inference(images, targets)
            self.prelosses = losses
            targets = targets[:,0:1]
            if self.iter % 500 == 0:
                self.save_img(samples[0].img_name, [x[0] for x in logits], images[0], targets[0], losses, is_save=True)
            del logits
            return losses
        else:
            # ts = time.time()
            x, y, results = self.inference(images, targets)
            # torch.cuda.synchronize()
            # te = time.time()
            # print(f'images.shape: {images.shape}, inference time: {(te-ts)}s')


        for idx, (result, sample) in enumerate(zip(results[-1], samples)):
            if self.resize_size is not None:
                sample.pred = self.resize(result[None], sample.img_size)[0]
            else:
                sample.pred = result

        if self.save_cams:
            if self.normalized_masks == None:
                self.normalized_masks = torch.nn.functional.softmax(results[-1], dim=1)
                self.targets = [SemeanticSegmentationTarget(self.normalized_masks, None)]

            sample.cam = self.get_cam(images)

        if self.save_features:
            sample.x = [xx[idx] for xx in x]
            sample.y = [yy[idx] for yy in y]
            sample.logits = [logit[idx] for logit in results]


        return samples


    def save_img(self, img_name, logits, image, target, losses, is_save=False):
        if not is_save:
            return

        num_logit = len(logits)
        assert len(logits[0].shape) == 3, f'logit.shape should be 3 dimensions, but got {len(logits[0].shape)}.'
        assert len(target.shape) == 3, f'target.shape should be 3 dimensions, but got {len(target.shape)}.'
        assert len(image.shape) == 3, f'image.shape should be 3 dimensions, but got {len(image.shape)}.'

        logit = [torch.permute(logit, (1, 2, 0)).detach().cpu().numpy().astype(np.float32) for logit in logits]
        image = torch.permute(image, (1, 2, 0)).detach().cpu().numpy().astype(np.float32)
        target = torch.permute(target, (1, 2, 0)).detach().cpu().numpy().astype(np.float32)
        # if logit[0].shape[-1] > 1:
        #     logit = [(x[:,:,0:1] + 1 - x[:,:,1:2])/2 for x in logit]

        loss_disp = {}
        for k, v in losses.items():
            if isinstance(v, list):
                loss_disp[k] = ', '.join([f'{vv:.4f}' for vv in v])
            else:
                loss_disp[k] = f'{v:.4f}'

        loss_disp = ', '.join([f'{k}: {v}' for k, v in loss_disp.items()])


        channel = np.shape(image)[-1]
        titles_input = ['LSI, input', 'RGB, input']
        if channel == 1:
            titles_input = ['LSI, input']
            input = [image]
        elif channel == 2:
            titles_input = ['LSI, input', 'R, input']
            input = [image[:,:,0:1], image[:,:,1:2]]
        elif channel == 3:
            titles_input = ['RGB, input']
            input = [image]
        elif channel == 4:
            titles_input = ['LSI, input', 'RGB, input']
            input = [image[:,:,0:1], image[:,:,1:4]]

        fig, axes = plt.subplots(1, num_logit + 1+len(titles_input), figsize=((num_logit + 3) * 2.5, 3), layout='tight')
        for i, (img, subtitle) in enumerate(zip(input + [target] + [x for x in logit],
                        titles_input+['MMT, target'] + [f'MTT, pred {num_logit-i-1}' for i in range(num_logit)])):
        # fig, axes = plt.subplots(1, 4 + len(titles_input),
        #                          figsize=((6) * 2.5, 3), layout='tight')
        # for i, (img, subtitle) in enumerate(zip(input + [target] + [logit[-1][:,:,i] for i in range(3)],
        #                 titles_input + ['MMT, target'] + ['MTT', 'pred_art', 'pred_vein'])):

            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(subtitle)
        fig.suptitle(f'iter {self.iter}, {img_name}, {loss_disp}')

        os.makedirs(os.path.join(self.output_dir,'train'), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir,'train',f'iter{self.iter}_{img_name}.jpg'))

    def get_cam(self, input):
        grayscale_cams = []
        for target_layer in self.target_layers:
            cam = GradCAM(model=self,target_layers=[target_layer], compute_input_gradient=False)
            grayscale_cam = cam(input_tensor=input, targets=self.targets)
            torch.cuda.empty_cache()
            grayscale_cams.append(torch.from_numpy(grayscale_cam).to(input.device))
        return grayscale_cams







