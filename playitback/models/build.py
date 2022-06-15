#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch

from .tempr import TemPr
from .mvit import MViT

from einops import rearrange


class PlayItBack(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        name = cfg.MODEL.MODEL_NAME
        self.cfg = cfg
        self.encoder = MViT(cfg=cfg)
        if not cfg.MODEL.IGNORE_DECODER:
            self.decoder = TemPr(cfg=cfg)

    def scaled_region_estimator(id,length=400,ratio=2):
        # Calculate frames after the salient region in the scaled volume.
        right_max_margin = (length-id)*ratio
        # Calculate frames before the salient region in the scaled volume.
        left_max_margin = id*ratio
        # If both margins are more than length/2 then id*ratio will be the middle of the sampling region.
        if right_max_margin > length/2 and left_max_margin > length/2:
            return [int(left_max_margin-length/2),int(left_max_margin+length/2)]
        # If the left side of the salient region has less than length/2 frames, sample more frames from the right side.
        if left_max_margin < length/2:
            return [0, int(length)]
        # If the right side of the salient region has less than length/2 frames, sample more frames for the left side.
        if right_max_margin < length/2:
            return [int((length*ratio)-length), int(length*ratio)]

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
        return(temporal_saliency, id.item())

    def forward(self, x):

        en_feats = []
        id = 0

        for x_i,i in range(x.shape(0)):

            if i>0: # Only the first loop will not be required to be segmented
                start, end = self.scaled_region_estimator(id,length=data_length,ratio=2)
                x_i = x_i[:, start:end,:]
            else: # get the temporal length of the scpectrogram
                data_length = x_i.shape[-2]

            # Transfer the data to the current GPU device.
            x_i = x.cuda(non_blocking=True)
            if self.cfg.MODEL.FREEZE_ENCODER :
                with torch.no_grad():
                    out = self.encoder(x)
            else:
                out = self.encoder(x)

            # get labels and features
            if self.ENCODER.RETURN_EMBD:
                feats = out[0]
                preds = out[1]
            elif self.cfg.ENCODER.RETURN_EMBD_ONLY:
                feats = out
                preds = None
            else:
                preds = out[1]
                feats = None

            assert not (cfg.MODEL.IGNORE_DECODER and preds==None), "Cannot get predictions if Encoder only returns features and no Decoder is used. Either use a Decoder or set `ENCODER.RETURN_EMBD_ONLY` to False."

            # Should only be used with no playback so first predictions that are made are also returned
            if cfg.MODEL.IGNORE_DECODER:
                return preds

            assert feats, "Cannot use the Decoder without extracted features."

            # Get saliency
            _, id = get_salient_region_idx(feats)

            en_feats.append(feats)

        # get decoder preds
        de_preds = self.decoder(en_feats)

        return de_preds




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
    encoder = PlayItBack(cfg=cfg)

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
