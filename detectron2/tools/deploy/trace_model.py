'''
一个脚本，通过torch.jit.trace将D2-light里的模型转换为TorchScript格式，能够覆盖支持绝大部分简单场景（模型输入仅有图片一个字段，batch size=1).
作为一个脚本，用户可以随意进行修改和定制化。
主要流程有：
（1) 构造example input:
    -- 我们支持用户指定一张图片（但batch size必须为1),也支持实例化一个dataloader,读取前几张构造一个batch(batch1 size>1).这里的逻辑不够灵活，用户可以修改。
（2) 构造model,load参数，做一些处理：
    -- 把 inplace abn 换成 abn: 因为 inplace-abn 不支持转 trace,所以这里会把网络中的（sync)inplace abn 替换成 abn,并为了运算逻辑一致调整参数值。
    -- 处理模型接口：torch.jit.trace要求模型的输入输出为tensor或者tuple of tensor,而D2-light里的模型输入输出均为data sample.
       因此我们使用D2-light里的ModelTracingWrapper对model进行wrap,使其接口满足trace要求。
       注意：因为不同模型的参数不一样，转换为tuple的逻辑也不一样，我们要求模型自己实现两个转换逻辑，并写在cfg的tracing_args参数里。
       具体见ModelTracingWrapper的文档。
（3) 调torch.jit.trace,简单评估一下在example input上的tensor误差，保存转换以后的模型：
     这里仅在example input上进行简单的误差评估。我们建议另起一个脚本，对转换后的TorchScript模型在测试集上做完成的误差评估。
注意：
（1)转换后的模型支持动态image shape和动态batch size与否，完全取决于模型自己。
（2)转换后的模型可直接通过torch.jit.load读取并使用，输入输出均为tensor或者tuple of tensor,不依赖D2-light的接口和数据结构。
（若输入输出有多个tensor,其物理含义需要用户自己搞清楚）。

用户需要指定的参数：
（1) --config-file:config文件，其格式与训练时的config一样。主要使用其中的test dataloader、model、tracing_args、 train.device.
（2) --model-path:待转换的PyTorch模型的checkpoint路径（保存了参数）。
（3) --batch-size:模型的batch size,默认为1.
（4) --example-image:如果指定了，则读取指定的图片用作tracing的example input,此时要求batch size为1,
    如果不指定则从config里指定的test dataloader读取数据作用tracing的example input.
(5) --output-path: 转换后的TorchScript模型保存的路径。
'''

import argparse
import torch
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.export.model_tracing_wrapper import ModelTracingWrapper
from detectron2.export.replace_inplace_abn_to_abn import replace_inplace_abn_to_abn

logger = logging.getLogger('d2.trace_model')

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description= 'Export a model for deployment.')
    parser.add_argument('--config-file', metavar='FILE', help='path to config file')
    parser.add_argument('--model-path', default=None, type=str, help='pathto model weight')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--output-path', default='traced_model.pt', type=str)
    parser.add_argument('opts', help='Modify config options using the command-line.',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # cfg
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    # model
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    logger = setup_logger(name='trace_model')

    if args.model_path is not None:
        logger.info('Load model weight from path: {}'.format(args.model_path))
    else:
        logger.warning('Model weight is not specified.')
    model = model.eval()

    replace_inplace_abn_to_abn(model)

    traceable_model = ModelTracingWrapper(
        model,
        input_to_sample=cfg.tracing_args.input_to_sample,
        sample_to_output=cfg.tracing_args.sample_to_output
    )
    traceable_model = traceable_model.eval()

    # evaluate
    cfg.dataloader.test.batch_size = args.batch_size
    test_loader = instantiate(cfg.dataloader.test)
    loader_iter = iter(test_loader)
    data_sample = next(loader_iter)
    # print('before sample_to_input image.shape={}'.format(data_sample[0].image.shape))
    example_input = cfg.tracing_args['sample_to_input'](data_sample)
    # print('after sample_to_input image_tensor.shape={}'.format(example_input.shape))

    if not isinstance(example_input, (tuple, list)):
        example_input = (example_input,)
    # trace model
    with torch.no_grad():
        output = traceable_model(*example_input)
        traced_model = torch.jit.trace(traceable_model, example_input)

    # save traced model
    traced_model.save(args.output_path)
    print('Dump TorchScript model to: {}.'.format(args.output_path))








