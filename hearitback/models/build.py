#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch

from .tempr import TemPr
from .mvit import MViT


class HearItBack(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        name = cfg.MODEL.MODEL_NAME
        self.encoder = MViT(cfg=cfg)
        if not cfg.MODEL.IGNORE_DECODER:
            self.decoder = TemPr(cfg=cfg)
        else:
            self.decoder = torch.nn.Identity()

    def forward_encoder(self,x):
        return self.encoder(x)

    def forward_decoder(self,x,iter):
        return self.decoder(x,iter)

    def get_salient_region_idx(self,emb,temporal_dim=13):

        # Change the flattened dimension to time x frequency
        emb = rearrange(emb, 'b (h w) d -> b d w h',w=temporal_dim)
        # Reduce the frequency dimension
        emb = reduce(emb, 'b d w h -> b d w', 'mean')

        # Calculate the channel-wise min and max
        min = torch.min(emb, dim=-1, keepdim=True)[0]
        max = torch.max(emb, dim=-1, keepdim=True)[0]
        temporal_saliency = (emb - min) / (max - min)
        temporal_saliency = torch.sum(temporal_saliency, dim=1, keepdim=True)
        temporal_saliency = torch.nn.functional.interpolate(temporal_saliency, size=(400), mode='linear', align_corners=False)
        id = torch.argmax(temporal_saliency, dim=-1)
        return(temporal_saliency, id)




def build_model(cfg, gpu_id=None):
    """
    Builds the audio model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    encoder = HearItBack(cfg=cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        if cfg.NUM_GPUS > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(device=cur_device)

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True
        )
    return model
