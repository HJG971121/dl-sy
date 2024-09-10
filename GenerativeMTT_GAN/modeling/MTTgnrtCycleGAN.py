
from typing import Optional, Tuple, List, Union
import torch
from torch import nn
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


class MTTGeneratorCycleGAN(nn.Module):
    def __init__(self,
                 netG_A: nn.Module,
                 netG_B: nn.Module,
                 netD_A: nn.Module,
                 netD_B: nn.Module,
                 pool_size,
                 loss_function,
                 weight,
                 optim_para,
                 pixel_mean,
                 pixel_std,
                 resize_size,
                 output_dir = ''):
        super(MTTGeneratorCycleGAN, self).__init__()

        self.netG_A, self.netG_B = netG_A, netG_B
        if self.training:
            self.netD_A, self.netD_B = netD_A, netD_B

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=optim_para['lr'], weight_decay=optim_para['weight_decay'],
                                            betas=(optim_para['beta1'], 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=optim_para['lr'], weight_decay=optim_para['weight_decay'],
                                            betas=(optim_para['beta1'], 0.999))

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        self.criterionGAN = loss_function['criterionGAN']
        self.criterionCycle = loss_function['criterionCycle']
        self.criterionIdt = loss_function['criterionIdt']
        self.weight = weight
        self.resize_size = resize_size
        self.output_dir = output_dir

    @property
    def device(self):
        return self.pixel_mean.device

    def resize(self, image: torch.Tensor, size: tuple, mode):
        if mode!='nearest':
            return nn.functional.interpolate(image,size=size,mode=mode, align_corners=True)
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
                images_A = [self.resize(image, (resize_size,resize_size), 'bicubic') for image in images_A]
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

            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)

            losses = self.losses()

            if self.iter % 500 == 0:
                self.save_img(samples[0].img_name, losses, is_save=True)

            return losses
        else:
            self.fake_B = self.netG_A(self.real_A)
            self.fake_A = self.netG_B(self.real_B)

        # print('results.shape: {}'.format(results.shape))
        for pred_A, pred_B, sample in zip(self.fake_B, self.fake_A, samples):
            if self.resize_size is not None:
                sample.pred_A = self.resize(pred_A[None], sample.imgA_size)[0]
                sample.pred_B = self.resize(pred_B[None], sample.imgB_size)[0]
            else:
                sample.pred_A = pred_A
                sample.pred_B = pred_B

        return samples


    def save_img(self, img_name, losses, is_save=False):
        if not is_save:
            return

        real_A = torch.permute(self.real_A[0], (1, 2, 0)).detach().cpu().numpy()
        fake_B = torch.permute(self.fake_B[0], (1, 2, 0)).detach().cpu().numpy()
        rec_A = torch.permute(self.rec_A[0], (1, 2, 0)).detach().cpu().numpy()

        real_B = torch.permute(self.real_B[0], (1, 2, 0)).detach().cpu().numpy()
        fake_A = torch.permute(self.fake_A[0], (1, 2, 0)).detach().cpu().numpy()
        rec_B = torch.permute(self.rec_B[0], (1, 2, 0)).detach().cpu().numpy()

        # original images
        imshow = [
            {
                'LSI': real_A[:,:,0:1],
                # 'RGB': real_A[:,:,1:4],
                'MTT': real_B
            },
            {
                'LSI_g': fake_A[:,:,0:1],
                # 'RGB_g': fake_A[:,:,1:4],
                'MTT_g': fake_B
            },
            {
                'LSI_r': rec_A[:, :, 0:1],
                # 'RGB_r': rec_A[:, :, 1:4],
                'MTT_r': rec_B
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
        fig, axes = plt.subplots(r, c, figsize = (r*2.5, c*2.5), layout = 'tight')

        for i in range(r):
            for j, (subtitle, img) in enumerate(imshow[i].items()):
                axes[i, j].imshow(img)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_title(subtitle)
        fig.suptitle(f'iter {self.iter}, {img_name}, {loss_disp}')

        # save img
        os.makedirs(os.path.join(self.output_dir,'train'), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir,'train',f'iter{self.iter}_{img_name}.jpg'))


    def losses(self):

        # discriminator loss
        self.loss_D_A = self.D_loss(self.netD_A, self.real_A, self.fake_A_pool.query(self.fake_A))
        self.loss_D_B = self.D_loss(self.netD_B, self.real_B, self.fake_B_pool.query(self.fake_B))

        # generator loss
        self.loss_G = self.G_loss()

        return {
            'loss_D_A': self.loss_D_A,
            'loss_D_B': self.loss_D_B,
            'loss_G': self.loss_G
        }

    def G_loss(self):
        labmda_idt = self.weight['labmda_idt']
        lambda_A = self.weight['lambda_A']
        lambda_B = self.weight['lambda_B']

        # Identity loss
        if labmda_idt>0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * labmda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * labmda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combine loss
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                 self.loss_cycle_A + self.loss_cycle_B + \
                 self.loss_idt_A + self.loss_idt_B
        return self.loss_G

    def D_loss(self,
             netD: nn.Module,
             real: torch.Tensor,
             fake: torch.Tensor):

        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        # loss
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_fake+loss_D_real)*0.5

        return loss_D
