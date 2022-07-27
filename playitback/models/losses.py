#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
import sys

class Rank(nn.Module):
    """
    Pairwise rank loss between predictions
    """

    def __init__(self, reduction="mean"):
        super(Rank, self).__init__()
        self.reduction = reduction

    def forward(self, x, y, margin=5e-2, multilabel=False):

        # Assume
        # x: [p x b x cls] => x[0] are the predictons of the decoder x[1:] are predictions based on slot attention
        # y: [b] (single class) [b,c] (multi-class)

        # get class probabilities
        c_probs = rearrange(x, 's b c -> b c s')

        if multilabel:
            y = y.to(c_probs.device)
            c_probs = c_probs * y.unsqueeze(-1) # mask-out classes from multi-hot
            c_probs = rearrange(c_probs, 'b c s -> b s c')
            # calc difference across all S
            dif = abs(c_probs.unsqueeze(-2) - c_probs.unsqueeze(-3)) + margin # B x S x S x C
            dif = reduce(dif, 'b s1 s2 c -> b s1 s2','sum')# B x S x S
            dif = dif.squeeze(-1)
            mask = torch.tensor([[abs(i-j) for i in range(0,c_probs.shape[-2])] for j in range(0,c_probs.shape[-2])], device=c_probs.device)
        else:
            c_probs = torch.stack([c_probs[i,y[i]] for i in range(c_probs.shape[0])]) # B x S
            # calc difference across all S
            dif = abs(c_probs.unsqueeze(-1) - c_probs.unsqueeze(-2)) + margin # B x S x S
            mask = torch.tensor([[abs(i-j) for i in range(0,c_probs.shape[-1])] for j in range(0,c_probs.shape[-1])], device=c_probs.device)

        mask = 1./mask
        dif = dif * mask

        dif = torch.clamp(dif, min=0.)
        # loss-mask based on distance

        # Upper part of the distance matrix
        losses = torch.triu(dif, diagonal=1) # B x S x S
        losses = rearrange(losses, 'b s1 s2 -> b (s1 s2)')
        loss =reduce(losses, 'b s -> b', 'sum')

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

        return loss




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
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "rank":Rank
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
