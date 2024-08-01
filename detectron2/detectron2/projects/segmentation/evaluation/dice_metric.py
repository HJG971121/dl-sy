from typing import List, Dict, Optional
import torch
from dataclasses import dataclass

@dataclass
class DiceMetric:
    mean_dice: float
    class_names: Optional[List[str]] = None
    dice_per_class: Optional[torch.Tensor] = None

def _dice_to_float_dict(metric: DiceMetric) -> Dict[str, float]:
    ret = dict(
        mean_dice=metric.mean_dice
    )
    if metric.dice_per_class is not None:
        for cls_name, cls_dice in zip(metric.class_names, metric.dice_per_class):
            ret[f'class-{cls_name}-dice'] = float(cls_dice)
        return ret

def calc_mask_dice_img_metric(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        ignore_class: int = -1
) -> DiceMetric:
    _, dice_coefficient = dice_loss(pred_mask[None], gt_mask, smooth=1e-4, exponent=1, ignore_index=ignore_class)
    dice_mean = sum([x for x in dice_coefficient]) / max(1, len(dice_coefficient))

    result = DiceMetric(mean_dice=dice_mean.item(),
                        dice_per_class=torch.as_tensor(dice_coefficient))
    return result

def calc_mask_dice_from_metrics(
        img_metrics: List[DiceMetric],
        output_per_cls_metric: bool = True,
        class_names: List[str] = None
) -> DiceMetric:
    dice_total_mean = sum([x.mean_dice for x in img_metrics]) / max(1, len(img_metrics))
    res = DiceMetric(mean_dice=dice_total_mean)
    if output_per_cls_metric:
        assert class_names is not None
        res.class_names = class_names
        res.dice_per_class = sum([x.dice_per_class for x in img_metrics]) / max(1, len(img_metrics))

    return res

def calc_mask_dice(
        pred_masks: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        ignore_class: int,
        out_per_class_metric: bool = False,
        class_names: Optional[List[str]] = None
):
    assert len(pred_masks) == len(gt_masks)

    img_stats = []
    for pred_masks, gt_mask in zip(pred_masks, gt_masks):
        img_stats.append(calc_mask_dice_img_metric(pred_masks, gt_mask, ignore_class))

    return calc_mask_dice_from_metrics(img_stats, out_per_class_metric, class_names)