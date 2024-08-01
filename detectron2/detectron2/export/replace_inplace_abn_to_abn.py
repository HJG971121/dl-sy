import warnings
import torch
from torch.nn import Module
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync

def replace_inplace_abn_to_abn(module: Module) -> None:

    if not isinstance(module, Module):
        return
    if isinstance(module, (InPlaceABN, InPlaceABNSync)):
        warnings.warn(f'Input module itself is not supposed to be an instance of {type(module)}.')

    for name, submodule in module.named_children():
        if isinstance(submodule, (InPlaceABN, InPlaceABNSync)):
            _abn = ABN(
                num_features=submodule.num_features,
                eps=submodule.eps,
                momentum=submodule.momentum,
                track_running_stats=submodule.track_running_stats,
                affine=submodule.affine,
                activation=submodule.activation,
                activation_param=submodule.activation_param,
            ).to(submodule.weight.device)
            _abn.register_parameter('weight', torch.nn.Parameter(torch.abs(submodule.weight) + submodule.eps))
            _abn.register_parameter('bias', submodule.bias)
            # _abn.register_parameter('running_mean', submodule.running_mean)
            # _abn.register_parameter('running_var', submodule.running_var)
            _abn.train(submodule.training)
            module.add_module(name, _abn)
        else:
            replace_inplace_abn_to_abn(submodule)