#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class Rank(nn.Module):
    """
    Pairwise rank loss between predictions
    """

    def __init__(self):
        super(Rank, self).__init__()

    def forward(self, p_is, cls_idx):

        # Assume p_is: [s x b x cls], cls_idx: [b]

        # get class probabilities
        probs = rearrange(p_is[:,:,cls_idx].squeeze(-1), 's b -> b s')

        # shift 


        return




class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        p = F.log_softmax(x, dim=-1)
        loss = torch.sum(-y.unsqueeze(-1) * p, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
