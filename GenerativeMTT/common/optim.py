import torch
from functools import partial
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

grad_clippers = dict(
    grad_norm_clipper = partial(clip_grad_norm_, max_norm = 2),
    grad_value_clipper = partial(clip_grad_value_, clip_value = 10)
)


SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)


AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
