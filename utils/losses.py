import torch
from torch import nn


def Focal_loss(pred, target,  alpha=0.5, gamma=2):
    logpt = -nn.CrossEntropyLoss(reduction='none')(pred, target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

