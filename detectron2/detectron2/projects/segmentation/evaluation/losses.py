import torch
import torch.nn

def binary_dice_loss(
        pred, target, valid_mask = None,
        smooth = 1, exponent = 2, ignore_index = 255
):
    assert pred.shape == target.shape

    if valid_mask is None and ignore_index is not None:
        valid_mask = (target != ignore_index).float()

    dice = (torch.sum(pred*target)*2+smooth)/(torch.sum(pred*valid_mask)+torch.sum(target*valid_mask)+smooth)
    return 1-dice

def dice_loss(
        pred, target, valid_mask = None,
        smooth = 1, exponent = 2, class_weight = None, ignore_index = 255
):
    assert pred.shape[0] == target.shape[0]

    if valid_mask is None and ignore_index is not None:
        valid_mask = (target != ignore_index).float()

    total_loss = 0
    dice_item = []
    num_classes = pred.shape[1]
    for i in range(num_classes):
        dice_loss = binary_dice_loss(
            pred[:,i].float(),
            (target == i).float(),
            valid_mask,
            smooth,
            exponent
        )
        dice_item.append(dice_loss)
        if class_weight is not None:
            dice_loss *=class_weight[i]
        total_loss += dice_loss
    return total_loss / num_classes, dice_item