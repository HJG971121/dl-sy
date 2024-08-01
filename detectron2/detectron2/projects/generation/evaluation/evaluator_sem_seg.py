import logging
from .evaluator_base import BaseEvaluator
from ..data.data_sample import SegImageSample
from typing import List, Union, Optional, Any
from .dice_metric import DiceMetric, calc_mask_dice_from_metrics, calc_mask_dice_img_metric

class SemSegEvaluator(BaseEvaluator):
    def __init__(
        self,
        metric_dump_path = None,
        metric_dump_formatter = None,
        prefix = None,
        output_per_cls_metric = False):
        super().__init__(__name__)
        self._logger = logging.getLogger(__name__)
        self._metric_dump_path = metric_dump_path
        self._output_per_cls_metric = output_per_cls_metric
        if self._metric_dump_path is not None:
            self._dice_per_img = []
            if metric_dump_formatter is None:
                metric_dump_formatter = default_dice_metric_dump_formatter
            self._metric_dump_formatter = metric_dump_formatter

    def process(self, data_samples: List[SegImageSample]):
        for sample in data_samples:
            label = sample.label.cuda()
            pred = sample.pred.cuda()
            ignore_class = self.dataset_meta.ignore_class
            res = calc_mask_dice_img_metric(pred, label, ignore_class)
            self.results.append(res)

            if self._metric_dump_path is not None:
                pred_res = self._metric_dump_formatter(sample, res)
                self._dice_per_img.append(pred_res)


def default_dice_metric_dump_formatter(sample: SegImageSample, metric: DiceMetric):
    f = {
        'img_name': sample.img_name,
        'mean_dice': float(metric.mean_dice),
    }
    if metric.dice_per_class is not None:
        f['dice_per_class'] = [float(x.cpu().numpy())
                               for x in metric.dice_per_class]
    return f

