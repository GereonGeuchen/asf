import torch


def wmse(input, target, weights):
    return torch.mean(
        weights * torch.nn.functional.mse_loss(input, target, reduction="none")
    )


def bpr_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl):
    return (
        -torch.log(torch.sigmoid(y_pred - y_pred_s))
        - torch.log(torch.sigmoid(y_pred_l - y_pred))
        - torch.log(torch.sigmoid(y_pred_l - y_pred_s))
    )


def tml_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl, margin=1.0, p=2):
    return torch.nn.functional.triplet_margin_loss(
        y_pred, y_pred_s, y_pred_l, margin=margin, p=p
    )
