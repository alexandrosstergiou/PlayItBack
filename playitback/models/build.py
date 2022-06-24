#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""
import sys
import torch

from .tempr import TemPr
from .mvit import MViT

from einops import rearrange, reduce


def approx_divisor(divisor,target):
    prev_div = 1
    for x in range(1,target+1):
        # store integer divisors
        if target%x == 0:
            curr_div = x
            # If later than the current divisor check previous divisor
            if curr_div > divisor:
                # divisor is closer to curr_div
                if curr_div-divisor < divisor-prev_div:
                    return curr_div
                # else return the previous divisor
                else:
                    return prev_div
            prev_div = x


class Mlp(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x

class PlayItBack(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        name = cfg.MODEL.MODEL_NAME
        self.cfg = cfg
        self.encoder = MViT(cfg=cfg)
        if not cfg.MODEL.IGNORE_DECODER:
            self.decoder = TemPr(cfg=cfg)

        self.mlps = torch.nn.ModuleList([])
        for i in range(cfg.DECODER.DEPTH):
            self.mlps.append(Mlp(
                in_features=self.cfg.DECODER.INPUT_CHANNELS,
                hidden_features=int(self.cfg.DECODER.INPUT_CHANNELS * 2),
                out_features=self.cfg.DECODER.INPUT_CHANNELS,
            ))

    def scaled_region_estimator(self,id,length=400,ratio=2):
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

    def get_salient_region_idx(self,emb,temporal_dim=13,idx=0):

        # Change the flattened dimension to time x frequency
        emb = rearrange(emb, 'b (h w) d -> b d w h',w=temporal_dim)
        # Reduce the frequency dimension
        emb = reduce(emb, 'b d w h -> b d w', 'mean')
        emb_mlp = self.mlps[idx](rearrange(emb, 'b d w -> b w d'))

        emb += rearrange(emb_mlp, 'b w d -> b d w')

        # Calculate the channel-wise min and max
        min = torch.min(emb, dim=-1, keepdim=True)[0]
        max = torch.max(emb, dim=-1, keepdim=True)[0]
        temporal_saliency = (emb - min) / (max - min)
        temporal_saliency = torch.sum(temporal_saliency, dim=1, keepdim=True)
        temporal_saliency = torch.nn.functional.interpolate(temporal_saliency, size=(400), mode='linear', align_corners=False)
        id = torch.argmax(temporal_saliency, dim=-1)
        if torch.is_tensor(id):
            id = id[0]
        return(temporal_saliency, id.item())


    def rearrange_batched_data(self,x):
        # assume x size: [ b x (playbacks x c=1) x t x f]
        playbacks = x.shape[1]
        base_t = x.shape[-2]/playbacks
        x = x.permute(1, 0, 2, 3)
        x_s = []
        for i,x_i in enumerate(x):
            x_s.append(x_i[:,-int(base_t*(i+1)):,:].unsqueeze(-3))
        return x_s

    def rearrange_data(self,x):
        # assume x size: [(playbacks x c=1) x t x f]
        playbacks = x.shape[0]
        base_t = x.shape[-2]/playbacks
        x_s = []
        for i,x_i in enumerate(x):
            x_s.append(x_i[-int(base_t*(i+1)):,:].unsqueeze(0))
        return x_s


    def forward(self, x):

        # Check if the data has been padded with zeros based on DATA_LOADER.TRAIN_CROP_SIZE
        # if they have (and are being fed from a DataLoader), their channel dimension will correspond
        # to the number of playbacks. Thus, they can be rearranged by:
        dims = len(list(x.shape))
        if x.shape[-3] > 1:
            if dims == 4:
                x = self.rearrange_batched_data(x)
            elif dims == 3:
                x = self.rearrange_data(x)


        en_feats = []
        id = 0
        t_dim = x[0].shape[-2]

        for i,x_i in enumerate(x):

            if i>0: # Only the first loop will not be required to be segmented
                if x_i.shape[-2] > t_dim:
                    start, end = self.scaled_region_estimator(id,length=data_length,ratio=2)
                    x_i = x_i[:, :, start:end,:]
            else: # get the temporal length of the scpectrogram
                data_length = x_i.shape[-2]

            # Transfer the data to the current GPU device.
            x_i = x_i.cuda(non_blocking=True)
            if self.cfg.MODEL.FREEZE_ENCODER :
                with torch.no_grad():
                    out = self.encoder(x_i)
            else:
                out = self.encoder(x_i)

            # get labels and features
            if self.cfg.ENCODER.RETURN_EMBD:
                feats = out[0]
                preds = out[1]
            elif self.cfg.ENCODER.RETURN_EMBD_ONLY:
                feats = out
                preds = None
            else:
                preds = out[1]
                feats = None

            assert not (self.cfg.MODEL.IGNORE_DECODER and preds==None), "Cannot get predictions if Encoder only returns features and no Decoder is used. Either use a Decoder or set `ENCODER.RETURN_EMBD_ONLY` to False."

            # Should only be used with no playback so first predictions that are made are also returned
            if self.cfg.MODEL.IGNORE_DECODER:
                return preds

            assert feats is not None, "Cannot use the Decoder without extracted features."

            # Get saliency
            closest_divisor = approx_divisor(t_dim//32, feats.shape[-2])
            _, id = self.get_salient_region_idx(feats,temporal_dim=closest_divisor,idx=i)
            feats = rearrange(feats, 'b (h w) d -> b d w h',w=closest_divisor)
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
    model = PlayItBack(cfg=cfg)

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
