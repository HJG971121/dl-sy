
from typing import Optional, Tuple, List, Union
import torch
from torch import nn
import random
from torch.autograd import Variable
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
from abc import ABC
from ..data.data_sample import ImageSample
from detectron2.utils.events import get_event_storage
from detectron2.projects.generation.modeling.meta_arch import cycleGAN
from detectron2.projects.generation.utils.image_pool import ImagePool
from .MTTgnrtCycleGAN import MTTGeneratorCycleGAN


class MTTGeneratorGAN(nn.Module):
    def __init__(self,
                 netG: nn.Module,
                 netD: nn.Module,
                 pool_size,
                 loss_function,
                 weight,
                 optim_para,
                 pixel_mean,
                 pixel_std,
                 resize_size,
                 output_dir = ''):
        super(MTTGeneratorGAN, self).__init__()

        self.netG = netG
        if self.training:
            self.netD = netD

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=optim_para['lr'], weight_decay=optim_para['weight_decay'],
                                            betas=(optim_para['beta1'], 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=optim_para['lr'], weight_decay=optim_para['weight_decay'],
                                            betas=(optim_para['beta1'], 0.999))

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.fake_B_pool = ImagePool(pool_size)

        self.criterionGAN = loss_function['criterionGAN']
        self.weight = weight
        self.resize_size = resize_size
        self.output_dir = output_dir

    @property
    def device(self):
        return self.pixel_mean.device

    def resize(self, image: torch.Tensor, size: tuple, mode):
        if mode != 'nearest':
            return nn.functional.interpolate(image, size=size, mode=mode, align_corners=True)
        else:
            return nn.functional.interpolate(image, size=size, mode=mode)

    def preprocess_image(self, samples: List[ImageSample]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images_A = [x.image_A.to('cuda')[None] for x in samples]

        if self.training:
            assert samples[0].image_B is not None, f'samples.image_B should not be None.'
            images_B = [x.image_B.to('cuda')[None] for x in samples]
            v_masks = [x.v_mask.to('cuda')[None] for x in samples]

            resize_size = self.resize_size
            if resize_size is not None:
                images_A = [self.resize(image, (resize_size, resize_size), 'bicubic') for image in images_A]
                images_B = [self.resize(image, (resize_size, resize_size), 'bicubic') for image in images_B]
                v_masks = [self.resize(image, (resize_size, resize_size), 'nearest') for image in v_masks]

            images_A = torch.cat(images_A, dim=0)
            images_A = F.normalize(images_A, self.pixel_mean, self.pixel_std)
            images_B = torch.cat(images_B, dim=0)
            v_masks = torch.cat(v_masks, dim=0)
        else:
            images_B = None
            v_masks = None

        return images_A, images_B, v_masks

    def forward(self, samples: List[ImageSample]) -> List[ImageSample]:
        self.real_A, self.real_B, v_masks= self.preprocess_image(samples)

        if self.training:
            storage = get_event_storage()
            self.iter = storage.iter

            self.fake_B = self.netG(self.real_A)

            losses = self.losses()

            if self.iter % 500 == 0:
                self.save_img(samples[0].img_name, losses, is_save=True)

            return losses
        else:
            self.fake_B = self.netG(self.real_A)

        # print('results.shape: {}'.format(results.shape))
        for pred_A, sample in zip(self.fake_B, samples):
            if self.resize_size is not None:
                sample.pred_A = self.resize(pred_A[None], sample.img_size)[0]
            else:
                sample.pred_A = pred_A

        return samples


    def save_img(self, img_name, losses, is_save=False):
        if not is_save:
            return

        real_A = torch.permute(self.real_A[0], (1, 2, 0)).detach().cpu().numpy()
        fake_B = torch.permute(self.fake_B[0], (1, 2, 0)).detach().cpu().numpy()
        real_B = torch.permute(self.real_B[0], (1, 2, 0)).detach().cpu().numpy()


        # original images
        imshow = [
            {
                'LSI': real_A[:,:,0:1],
                'RGB': real_A[:,:,1:4],
                'MTT': real_B,
                'MTT_g': fake_B
            }]

        # loss display
        loss_disp = {}
        for k, v in losses.items():
            if isinstance(v, list):
                loss_disp[k] = ', '.join([f'{vv:.4f}' for vv in v])
            else:
                loss_disp[k] = f'{v:.4f}'
        loss_disp = ', '.join([f'{k}: {v}' for k, v in loss_disp.items()])


        # imshow
        r, c = len(imshow), np.max([len(x) for x in imshow])
        fig, axes = plt.subplots(r, c, figsize = (c*2.5, r*2.5), layout = 'tight')

        for i in range(r):
            for j, (subtitle, img) in enumerate(imshow[i].items()):
                ax = axes[j] if r==1 else axes[i,j]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(subtitle)
        fig.suptitle(f'iter {self.iter}, {img_name}, {loss_disp}')

        # save img
        os.makedirs(os.path.join(self.output_dir,'train'), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir,'train',f'iter{self.iter}_{img_name}.jpg'))


    def losses(self):

        # discriminator loss
        ridx = random.randint(0,3)
        # print(ridx)
        self.loss_D = self.D_loss(self.netD, self.real_B,
                    self.fake_B_pool.query(self.real_A[:,ridx:ridx+1] * torch.rand(self.real_B.shape).to(self.device)))

        # generator loss
        self.loss_G = self.G_loss()

        return {
            'loss_D': self.loss_D,
            'loss_G': self.loss_G
        }

    def G_loss(self):

        # GAN loss
        self.loss_G = self.criterionGAN(self.netD(self.fake_B), True)

        return self.loss_G

    def D_loss(self,
             netD: nn.Module,
             real: torch.Tensor,
             fake: torch.Tensor):

        pred_real = netD(real)
        pred_fake = netD(fake)

        # loss
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_fake+loss_D_real)*0.5

        return loss_D
