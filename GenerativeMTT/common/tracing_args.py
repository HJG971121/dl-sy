import warnings
from torch import Tensor
from typing import List
from detectron2.projects.segmentation.data import ImageSample

def split_tensor_without_trace_warning(tensor: Tensor) -> List[Tensor]:
    assert isinstance(tensor, Tensor), 'invalid type {}.'.format(type(tensor))
    warnings.filterwarnings(
        action='ignore',
        category=RuntimeWarning,
        message='Iterating over a tensor might cause the trace to be incorrect.'
    )
    ret = [i for i in tensor]
    print('split tensor.shape={} ret.shape={}'.format(
        tensor.shape,
        [x.shape for x in ret],
    ))
    return ret

model_tracing_args = dict(
    sample_to_input = lambda samples: [x.image for x in samples],
    input_to_sample = lambda image: [ImageSample(image=image)],
    sample_to_output = lambda samples: tuple(x.pred for x in samples),
    output_to_sample = lambda outputs: [ImageSample(pred = x) for x in outputs],
)