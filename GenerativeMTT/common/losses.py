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
            loss_class = loss.__class__.__name__
            if loss_class == 'VGGloss' and not is_output:
                continue
            if loss_class in ['vMSEloss', 'vL1loss', 'vDICEloss']:
                loss_value = loss(logits, targets, vmask)
            else:
                loss_value = loss(logits, targets)

            if loss_class in ['vMSEloss', 'vL1loss', 'vDICEloss']:
                if loss.mode == 'lt':
                    loss_class+='_vein'
                elif loss.mode == 'st':
                    loss_class+='_artery'

            losses[loss_class]=loss_value
            loss_tot+=loss_value*weight
        losses['total_loss'] = loss_tot
        return losses


class vessel_loss(nn.Module):
    def __init__(self, weight=[1,1,1,1]):
        super().__init__()
        self.weight = weight
        self.CE = nn.CrossEntropyLoss()

    def forward(self, is_output, logit, target=None, vmask = None):

        art_mask = (target <= 0.25) * vmask
        vein_mask = (target >= 0.5) * vmask

        pred = torch.sigmoid(logit)
        # art_pred = input[1:2]
        # vein_pred = input[2:3]
        art_pred = (1-pred)*art_mask
        vein_pred = pred*vein_mask

        mse = torch.mean((pred-target)**2)
        vmse = torch.sum(vmask * (pred-target)**2)/torch.sum(vmask)

        ce_art = self.CE(art_pred,art_mask)
        ce_vein = self.CE(vein_pred,vein_mask)

        dice_art = 1 - 2*torch.sum(art_pred*art_mask)/(torch.sum(art_mask)+torch.sum(art_pred))
        dice_vein = 1 - 2 * torch.sum(vein_pred * vein_mask) / (torch.sum(vein_mask) + torch.sum(vein_pred))

        loss_total = 0
        for i, loss in enumerate([mse, vmse, dice_art, dice_art]):
            loss_total += self.weight[i]*loss

        return {
            'total_loss': loss_total,
            'MSEloss': mse,
            'vMSEloss': vmse,
            'CEloss_artery': dice_art,
            'CEloss_vein': dice_art,
        }


class TVloss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).

    The input must be 4D. If a target (second parameter) is passed in, it is
    ignored.

    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.

    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    """

    def __init__(self, p, reduction='mean', eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError('p must be 1 or 2')
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == 'mean':
            return diff.mean()
        if self.reduction == 'sum':
            return diff.sum()
        return diff


class VGGloss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            if batch.shape[1]==1:
                feats = self.model(batch.repeat(1,3,1,1))
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)



class MSEloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.MSELoss()

    def forward(self, input, target=None):
        return self.criteria(input, target)


class vDICEloss(nn.Module):
    def __init__(self, thr, mode = 'lt'):
        super().__init__()
        assert 0<=thr<=1, f'thr should be in [0, 1].'
        assert mode in ['lt', 'st'], f'mode should be either \'lt\' or \'st\'.'
        self.thr = thr
        self.mode = mode

    def forward(self, input, target=None, vmask = None):
        if self.mode == 'lt':
            mask = (target >= 0.5) * vmask
            pred = (input >= self.thr) * vmask
        elif self.mode == 'st':
            mask = (target <= self.thr) * vmask
            pred = (input <= self.thr) * vmask
        return 1-2*torch.sum(mask*pred)/(torch.sum(mask) + torch.sum(pred)+1E-5)

class vMSEloss(nn.Module):
    def __init__(self, thr, mode = 'lt'):
        super().__init__()
        assert 0<=thr<=1, f'thr should be in [0, 1].'
        assert mode in ['lt', 'st'], f'mode should be either \'lt\' or \'st\'.'
        self.thr = thr
        self.mode = mode
    def forward(self, input, target=None, vmask = None):
        if self.mode == 'lt':
            mask = (target >= self.thr) * vmask
        elif self.mode == 'st':
            mask = (target <= self.thr) * vmask
        return torch.sum(((input-target)**2)*mask)/torch.sum(mask)

class vL1loss(nn.Module):
    def __init__(self, thr, mode = 'lt'):
        super().__init__()
        assert 0<thr<1, f'thr should be in [0, 1].'
        assert mode in ['lt', 'st'], f'mode should be either \'lt\' or \'st\'.'
        self.thr = thr
        self.mode = mode
    def forward(self, input, target=None, vmask = None):
        if self.mode == 'lt':
            mask = (target >= self.thr) * vmask
        elif self.mode == 'st':
            mask = (target <= self.thr) * vmask
        return torch.sum(torch.abs(input-target)*mask)/torch.sum(mask)

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


class MS_SSIMloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target=None):
        return 1-ms_ssim(input, target, data_range=1, size_average=True, win_size=5)

class FFloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = FFL(loss_weight=1, alpha=1)

    def forward(self, input, target=None):
        return self.criteria(input, target)



