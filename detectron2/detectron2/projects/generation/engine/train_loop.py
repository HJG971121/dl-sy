# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import concurrent.futures
import logging
import numpy as np
import time
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
from detectron2.engine.train_loop import HookBase, TrainerBase

__all__ = ["HookBase", "TrainerBase", "SimpleGANTrainer", "AMPGANTrainer"]


class SimpleGANTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, 1, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            gather_metric_period: an int. Every gather_metric_period iterations
                the metrics are gathered from all the ranks to rank 0 and logged.
            zero_grad_before_forward: whether to zero the gradients before the forward.
            async_write_metrics: bool. If True, then write metrics asynchronously to improve
                training speed
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like 1 during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_obj = None
        self.optimizer_G = self.model.optimizer_G
        self.optimizer_D = self.model.optimizer_D
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        self.losses = self.model(data)

        self.optimize_parameters()

        self.after_backward()
        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, self.losses, data_time, iter=self.iter
            )
        else:
            self._write_metrics(self.losses, data_time)

    def optimize_parameters(self):
        # G_A and G_B
        self.set_requires_grad([self.model.netD_A, self.model.netD_B], False)
        self.optimizer_G.zero_grad()
        self.losses['loss_G'].backward()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.model.netD_A, self.model.netD_B], True)
        self.optimizer_D.zero_grad()
        self.losses['loss_D_A'].backward()
        self.losses['loss_D_B'].backward()
        self.optimizer_D.step()


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for para in net.parameters():
                    para.requires_grad = requires_grad

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleGANTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        # metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, list):
                for idx, loss in enumerate(v):
                    metrics_dict[f'{k}_{len(v)-idx-1}'] = loss.detach().cpu().item()
            else:
                metrics_dict[k] = v.detach().cpu().item()
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            # total_losses_reduced = metrics_dict['total_loss']
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            # storage.put_scalar(
            #     "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            # )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer_G"] = self.optimizer_G.state_dict()
        ret["optimizer_D"] = self.optimizer_D.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer_G.load_state_dict(state_dict["optimizer_G"])
        self.optimizer_D.load_state_dict(state_dict["optimizer_D"])

    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)


class AMPGANTrainer(SimpleGANTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
        grad_clipper = None,
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(
            model, data_loader, gather_metric_period, zero_grad_before_forward
        )

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            grad_scaler_G = GradScaler()
            grad_scaler_D = GradScaler()

        self.grad_scaler_G = grad_scaler_G
        self.grad_scaler_D = grad_scaler_D
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler
        self.optimizer_G = self.model.optimizer_G
        self.optimizer_D = self.model.optimizer_D
        self.grad_clipper = grad_clipper

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast(dtype=self.precision):
            self.losses = self.model(data)

        if 'CycleGAN' in self.model.__class__.__name__:
            self.optimize_parameters_cycleGAN()
        else:
            self.optimize_parameters_GAN()


        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler_G.get_scale())
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler_D.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, self.losses, data_time, iter=self.iter
            )
        else:
            self._write_metrics(self.losses, data_time)

        # self.grad_scaler_G.update()
        # self.grad_scaler_D.update()

    def optimize_parameters_cycleGAN(self):
        # G_A and G_B
        self.set_requires_grad([self.model.netD_A, self.model.netD_B], False)
        self.optimizer_G.zero_grad()
        self.losses['loss_G'].backward()
        if self.grad_clipper is not None:
            self.grad_clipper(self.model.netG_A.parameters())
            self.grad_clipper(self.model.netG_B.parameters())
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.model.netD_A, self.model.netD_B], True)
        self.optimizer_D.zero_grad()
        self.losses['loss_D_A'].backward()
        self.losses['loss_D_B'].backward()
        if self.grad_clipper is not None:
            self.grad_clipper(self.model.netD_A.parameters())
            self.grad_clipper(self.model.netD_B.parameters())
        self.optimizer_D.step()

    def optimize_parameters_GAN(self):
        # G
        self.set_requires_grad(self.model.netD, False)
        self.optimizer_G.zero_grad()
        self.losses['loss_G'].backward()
        if self.grad_clipper is not None:
            self.grad_clipper(self.model.netG.parameters())
        self.optimizer_G.step()

        # D
        self.set_requires_grad(self.model.netD, True)
        self.optimizer_D.zero_grad()
        self.losses['loss_D'].backward()
        if self.grad_clipper is not None:
            self.grad_clipper(self.model.netD.parameters())
        self.optimizer_D.step()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler_G"] = self.grad_scaler_G.state_dict()
        ret["grad_scaler_D"] = self.grad_scaler_D.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler_G.load_state_dict(state_dict["grad_scaler_G"])
        self.grad_scaler_D.load_state_dict(state_dict["grad_scaler_D"])
