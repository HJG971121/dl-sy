# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional, Tuple, List
import torch
from torch import nn
import copy
import itertools
from abc import ABC

from detectron2.modeling.backbone import Backbone
from detectron2.projects.segmentation.data import SegImageSample
from detectron2.projects.generation.utils.image_pool import ImagePool


class cycleGAN(ABC):
    def __init__(self,
                 netG_A,
                 netG_B,
                 netD_A,
                 netD_B,
                 pool_size,
                 loss_function,
                 weight,
                 optim_para,
                 pixel_mean,
                 pixel_std,
                 ):

        super().__init__()
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

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, samples: List[SegImageSample]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images_A = [x.image_A.to(self.device) for x in samples]
        images_A = [(x - self.pixel_mean[0]) / self.pixel_std[0] for x in images_A]
        images_B = [x.images_B.to(self.device) for x in samples]
        images_B = [(x - self.pixel_mean[1]) / self.pixel_std[1] for x in images_B]

        return torch.Tensor(images_A), torch.Tensor(images_B)

    def forward(self, samples: List[SegImageSample]) -> List[SegImageSample]:
        self.real_A, self.real_B = self.preprocess_image(samples)

        if self.training:
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)

            return self.losses()
        else:
            self.fake_B = self.netG_A(self.real_A)
            self.fake_A = self.netG_B(self.real_B)

        for pred_A, pred_B, sample in zip(self.fake_B, self.fake_A, samples):
            sample.pred_A = pred_A
            sample.pred_B = pred_B

        return samples

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


