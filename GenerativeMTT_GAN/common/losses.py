"""A VGG-based perceptual loss function for PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from pytorch_msssim import ssim, ms_ssim
from torchmetrics.image import PeakSignalNoiseRatio
from focal_frequency_loss import FocalFrequencyLoss as FFL


class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, 'forward', func)

    def extra_repr(self):
        return getattr(self.forward, '__name__', type(self.forward).__name__) + '()'


class LossList(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is not None:
            assert len(weights)==len(losses), f'The lengths of losses and weights should be the same, \
            but got len(weights)={len(weights)}, len(losses)={len(losses)}.'
            self.weights = weights
        else:
            self.weights=[1]*len(losses)

        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))

    def forward(self, is_output, *args, **kwargs):
        logits = args[0]
        targets = args[1]
        vmask = args[2]
        losses = {}
        loss_tot = 0
        for loss, weight in zip(self, self.weights):
            if loss.__class__.__name__ == 'VGGloss' and not is_output:
                continue
            if loss.__class__.__name__ == 'vMSEloss':
                loss_value = loss(logits, targets, vmask)
            else:
                loss_value = loss(logits, targets)
            losses[loss.__class__.__name__]=loss_value
            loss_tot+=loss_value*weight
        losses['total_loss'] = loss_tot
        return losses


class MSEloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.MSELoss()

    def forward(self, input, target=None):
        return self.criteria(input, target)

class vMSEloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target=None, vmask = None):
        return torch.sum(((input-target)**2)*vmask)/torch.sum(vmask)

class FMSEloss(nn.Module):
    def __init__(self, alpha = 1):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target=None):
        w = torch.abs(input - target)
        w = (w/torch.max(w))**self.alpha

        return torch.mean(w*(input-target)**2)

class L1loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.L1Loss()

    def forward(self, input, target=None):
        return self.criteria(input, target)

class SSIMloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target=None):
        return 1-ssim(input, target, data_range=1, size_average=True)


class FFloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = FFL(loss_weight=1, alpha=1)

    def forward(self, input, target=None):
        return self.criteria(input, target)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self,target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss



