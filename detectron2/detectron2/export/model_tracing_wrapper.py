import torch.jit
from torch import nn, Tensor
from typing import Callable, Union

class ModelTracingWrapper(nn.Module):
    '''
    D2 - light里的model默认的输出输出为data sample或者List[data sample].
    在使用Tracing的方式将模型转换为TorchScript格式时，根据torch.jit.trace的要求，
    我们需要传入一个输入输出均为tensor或者Tuple[tensor]的模型对象。
    为此我们使用一个wrapper对模型进行wrap, 使得模型输入输出均为tensor或者tuple of tensor.
    具体而言，我们使用两个callable对象，分别在输入模型之前把tensor转换为data sample、
    在模型输出之后把data sample flatten成tensor或者tuple of tensor.
    这两个Callable对象的具体实现取决于具体的模型，在构造这个wrapper的时候由用户传入。

    Example1:
        class ToySegModel(nn.Module): -> SegImageSample
            def forward(x: SegImageSample):
                image = x.image
                x.pred_sem_seg = do_semantic_seg(image)
                return x

        seg_model = ToySegModel()  # its input and output are both SegImageSample.
        traceable_model = ModelTracingWrapper (  # its input and output are both torch.Tensor.
            seg_model,
            lambda image: SegImageSample(image=image),
            lambda sample: sample.pred_sem_seg,
        example_input = torch.randn(1, 64, 64, 64)
        traced_model = torch.jit.trace(traceable_model, example_input)
        pred_sem_seg = traced_model(example_input)
    '''

    def __init__(self, model: nn.Module, input_to_sample: Callable, sample_to_output: Callable):
        super().__init__()
        self.model = model
        self.input_to_sample = input_to_sample
        self.sample_to_output = sample_to_output

    def forward(self, *args, **kwargs) -> Union[tuple, Tensor]:
        # print('*args.len={} args[0].type={}'.format(
        #     len(args),
        #     type(args[0]),
        # ))
        sample_input = self.input_to_sample(*args, **kwargs)
        sample_output = self.model(sample_input)
        tuple_output = self.sample_to_output(sample_output)
        return tuple_output