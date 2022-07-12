#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""
import sys
import torch
import torch.nn.functional as F
from .tempr import TemPr
from .mvit import MViT

import numpy as np
from einops import rearrange, reduce

from einops.layers.torch import Rearrange, Reduce


class SlotAttention(torch.nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = torch.nn.Parameter(abs(torch.randn(1, 1, dim)))
        self.slots_sigma = torch.nn.Parameter(abs(torch.randn(1, 1, dim)))

        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)

        self.gru = torch.nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, dim)

        self.norm_input  = torch.nn.LayerNorm(dim)
        self.norm_slots  = torch.nn.LayerNorm(dim)
        self.norm_pre_ff = torch.nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(torch.nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = torch.nn.Linear(2, hidden_size, bias=True)
        self.grid = self.build_grid(resolution)

    def build_grid(self,resolution):
        ranges = [np.linspace(0., 1., num=resolution)]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution, -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

    def set_grid_device(self,target):
        self.grid = self.grid.to(target)

    def forward(self, inputs):
        self.set_grid_device(inputs.device)
        grid = self.embedding(self.grid)
        inputs = rearrange(inputs,'b c t -> b t c')
        return inputs + grid




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

        res = 400
        self.saliency_slots = torch.nn.ModuleList([])

        for i in range(cfg.DECODER.DEPTH):
            self.saliency_slots.append(
                torch.nn.Sequential(
                SoftPositionEmbed(hidden_size=self.cfg.DECODER.INPUT_CHANNELS, resolution=res),
                Rearrange('b t c -> b c t'),
                SlotAttention(num_slots=2,dim=res)#,
                #Mlp(in_features=self.cfg.DECODER.INPUT_CHANNELS,
                #hidden_features=int(self.cfg.DECODER.INPUT_CHANNELS * 2),
                #out_features=self.cfg.DECODER.INPUT_CHANNELS)
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


    def reginal_reg(self,act,idx,size):
        tmp = torch.arange(start=0, end=size)
        tmp = tmp.cuda(act.get_device()) if act.is_cuda else tmp
        tmp = abs(tmp-idx)*-1
        tmp += abs(tmp.min())
        v_min, v_max = tmp.min(), tmp.max()
        ratio = (act.max().item() - act.min().item()) / act.max().item()
        tmp_p = (tmp - v_min)/(v_max - v_min) * (ratio) + (1.-ratio)
        new_act = tmp_p * act

        return new_act,tmp_p.unsqueeze(1)


    def get_indices(self,emb):
        no_zeros = emb.nonzero()
        indices = {}
        for p in no_zeros:
            if p[0].item() in indices.keys():
                if len(indices[p[0].item()]['start']) > len(indices[p[0].item()]['end']):
                    indices[p[0].item()]['end'].append(p[1].item())
                else:
                    indices[p[0].item()]['start'].append(p[1].item())
            else:
                indices
                indices[p[0].item()] = {'start':[p[1].item()], 'end':[]}
        # re-iterate to assign end positions
        for k in indices.keys():
            if len(indices[k]['start']) > len(indices[k]['end']):
                 indices[k]['end'].append(emb.shape[-1]-1)

        return indices



    def find_start_and_end(self,data_length,ids,mask):

        b_ar = torch.where(mask <= mask.mean(dim=-1,keepdim=True), torch.tensor(0.,device=mask.device), mask)

        start_ids = torch.tensor([0 for _ in range(b_ar.shape[0])], device=b_ar.device)
        end_ids = torch.tensor([b_ar.shape[-1] for _ in range(b_ar.shape[0])], device=b_ar.device)

        for b in range(b_ar.shape[0]):

            idcs = ids[b]
            idcs[idcs < b_ar.shape[-1]-1] += 1

            # Find start indices
            start = (b_ar.squeeze(1)[b,...,:idcs] == 0).int().argmin(dim=-1,keepdim=True)

            # Find end indices
            end = ids[b] + (b_ar.squeeze(1)[b,...,idcs:] == 0).int().argmax(dim=-1,keepdim=True)
            # cases that `end`==`ids[b]` correspond to argmax returning `0`, thus they should be changed
            # to the sequence length
            end[end==ids[b]] = b_ar.shape[-1]-1

            start_ids[b] = start
            end_ids[b] = end

        return start_ids, end_ids



    def get_salient_region_idx(self,emb,temporal_dim=13,idx=0,data_length=400):

        # Change the flattened dimension to time x frequency
        emb = rearrange(emb, 'b (h w) d -> b d w h',w=temporal_dim)
        # Reduce the frequency dimension
        emb = reduce(emb, 'b d w h -> b d w', 'mean')

        emb_s = self.saliency_mlps[idx](rearrange(emb, 'b d w -> b w d'))

        emb_m = torch.sum(emb, dim=1, keepdim=True)
        emb_m = self.mask_mlps[idx](rearrange(emb_m, 'b d w -> b w d'))
        emb_m = rearrange(emb_m, 'b w d -> b d w')


        emb_s = emb + rearrange(emb_s, 'b w d -> b d w')

        # Calculate the channel-wise min and max
        min = torch.min(emb_s, dim=-1, keepdim=True)[0]
        max = torch.max(emb_s, dim=-1, keepdim=True)[0]
        temporal_saliency = (emb_s - min) / (max - min)
        temporal_saliency = torch.sum(temporal_saliency, dim=1, keepdim=True)
        temporal_saliency = F.interpolate(temporal_saliency, size=(data_length), mode='linear', align_corners=False)

        id = torch.argmax(temporal_saliency, dim=-1)

        emb_m = F.interpolate(emb_m, size=(data_length), mode='linear', align_corners=False)

        _, dist = self.reginal_reg(emb_m,id,data_length)

        mask = temporal_saliency * dist

        min = torch.min(mask, dim=-1, keepdim=True)[0]
        max = torch.max(mask, dim=-1, keepdim=True)[0]
        mask = (mask - min) / (max - min)

        start_idx, end_idx = self.find_start_and_end(data_length,id,mask)


        return(temporal_saliency, id, mask, start_idx, end_idx)


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
        # assume `x` size: [(playbacks x c=1) x t x f]
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

        # Iterate over playbacks: x [playbacks x b x c=1 x t x f]
        for i,x_i in enumerate(x):

            if i>0: # Only the first loop will not be required to be segmented
                if x_i.shape[-2] > t_dim:
                    ratio = int(x_i.shape[-2]/x[i-1].shape[-2])
                    batched_new_x = []
                    for b in indices.keys():
                        tmp = x[i-1][b,:,:indices[b]['start'][0]]
                        if len(tmp.shape) < 3:
                            tmp = tmp.unsqueeze(1)
                        new_x = [tmp]
                        for indx in range(len(indices[b]['start'])):

                            slice = x_i[b,:,indices[b]['start'][indx]*ratio:indices[b]['end'][indx]*ratio,:]
                            if len(slice.shape) < 3:
                                slice = slice.unsqueeze(1)
                            new_x.append(slice)

                            if indx+1 < len(indices[b]['start']):
                                prev = x[i-1][b,:,indices[b]['end'][indx]:indices[b]['start'][indx+1],:]
                            else:
                                prev = x[i-1][b,:,indices[b]['end'][indx]:,:]

                            if len(prev.shape) < 3:
                                prev = prev.unsqueeze(1)
                            new_x.append(prev)


                        new_x = torch.cat(new_x, dim=-2).unsqueeze(0)
                        new_x = F.interpolate(new_x, size=(t_dim,x_i.shape[-1]))
                        batched_new_x.append(new_x)
                    x_i = torch.stack(batched_new_x).squeeze(1)
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
            #_, id, mask, start, end = self.get_salient_region_idx(feats,temporal_dim=closest_divisor,idx=i,data_length=data_length)
            feats = rearrange(feats, 'b (h w) d -> b d w h',w=closest_divisor)
            en_feats.append(feats)

            # Reduce the frequency dimension
            feats_reduced = reduce(feats, 'b d w h -> b d w', 'mean')
            feats_reduced = F.interpolate(feats_reduced, size=(t_dim), mode='linear', align_corners=False)

            # slot-attention
            s_slots = self.saliency_slots[i](feats_reduced) # B x SLOTS x t_dim
            pos_slot, neg_slot = torch.split(s_slots,1,1)

            # shifting
            pos_slot = pos_slot + abs(pos_slot.min())
            neg_slot = neg_slot + abs(neg_slot.min())

            # inverse
            inverse_neg_slot = neg_slot - 1.
            inverse_neg_slot = abs(inverse_neg_slot)

            inverse_neg_slot = inverse_neg_slot.unsqueeze(1) # B x 1 x T
            pos_slot = pos_slot.unsqueeze(2) # B x T x 1

            comp = inverse_neg_slot - pos_slot
            diag = torch.diagonal(comp, dim1=1, dim2=2)
            diag = diag.squeeze(-1)

            # normalise
            min = torch.min(diag, dim=-1, keepdim=True)[0]
            max = torch.max(diag, dim=-1, keepdim=True)[0]
            diag = (diag - min) / (max - min)

            diag_mask = torch.where(diag <= diag.mean(dim=-1,keepdim=True), torch.tensor(0.,device=diag.device), torch.tensor(1.,device=diag.device))

            indices = self.get_indices(diag_mask)

            tmp_f = rearrange(feats, 'b d w h -> b d (h w)')
            pos_slot = pos_slot.squeeze(1)
            pos_slot = F.interpolate(pos_slot, size=(tmp_f.shape[-1])).permute(0,2,1)
            inverse_neg_slot = inverse_neg_slot.squeeze(-2)
            inverse_neg_slot = F.interpolate(inverse_neg_slot, size=(tmp_f.shape[-1])).permute(0,2,1)

            tmp_f = rearrange(tmp_f, 'b h d -> b d h')
            tmp_f_pos = tmp_f * pos_slot
            tmp_f_neg = tmp_f * inverse_neg_slot

            with torch.no_grad():
                pred_pos = self.encoder.head(tmp_f_pos)
                pred_neg = self.encoder.head(tmp_f_neg)




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
