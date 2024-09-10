#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or 1.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
from sklearn.model_selection import KFold

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    AMPTrainerKfold,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import print_csv_format
from detectron2.projects.segmentation.evaluation import inference_on_dataset
from detectron2.utils import comm
from detectron2.projects.segmentation.data import (BaseDataset,
                                                   build_train_loader,
                                                   build_test_loader)

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        if cfg.train.split_combine.enabled:
            from detectron2.projects.segmentation.transforms.split_combine import SplitCombineModelWrapper
            model = SplitCombineModelWrapper(model,
                    instantiate(cfg.train.split_combine.split_combiner),
                    batch_size= cfg.dataloader.test.batch_size)

        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_valid(cfg, model, dataloader):
    if "evaluator" in cfg.dataloader:
        if cfg.train.split_combine.enabled:
            from detectron2.projects.segmentation.transforms.split_combine import SplitCombineModelWrapper
            model = SplitCombineModelWrapper(model,
                    instantiate(cfg.train.split_combine.split_combiner),
                    batch_size= cfg.dataloader.test.batch_size)

        ret = inference_on_dataset(
            model, dataloader, instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    model = create_ddp_model(model, **cfg.train.ddp)

    train_loader = instantiate(cfg.dataloader.train)

    kf = KFold(n_splits=cfg.train.k_fold, shuffle=True, random_state=1121)
    dataset = instantiate(cfg.dataloader.train.dataset)
    data_list = dataset.data_list
    train_k_list = []
    val_k_list = []
    for train_index, val_index in kf.split(data_list):
        train_k_list.append([data_list[i] for i in train_index])
        val_k_list.append([data_list[i] for i in val_index])

    trainer = (
        AMPTrainerKfold(model, train_loader, optim, grad_clipper=cfg.train.amp.grad_clipper,
                   dataset=dataset, cfg=cfg, train_k_list=train_k_list, val_k_list=val_k_list)
        if cfg.train.amp.enabled
        else SimpleTrainer(model, train_loader, optim)
    )
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_scheduler)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        do_test(cfg, model)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.config_file = '..\GenerativeMTT\configs\\baseline_MADenseUNet_kfold.py'
    # args.resume = True
    # args.eval_only = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
