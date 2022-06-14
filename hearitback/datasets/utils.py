#!/usr/bin/env python3

import logging
import torch
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def pack_pathway_output(cfg, spectrogram):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        spectrogram (tensor): frames of spectrograms sampled from the complete spectrogram. The
            dimension is `channel` x `num frames` x `num frequencies`.
    Returns:
        spectrogram_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `num frequencies`.
    """
    #print('packed:',spectrogram.shape)
    spectrogram_list = [spectrogram]
    '''
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        spectrogram_list = [spectrogram]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = spectrogram
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            spectrogram,
            1,
            torch.linspace(
                0, spectrogram.shape[1] - 1, spectrogram.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        spectrogram_list = [slow_pathway, fast_pathway]

    if not cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH and not cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    '''
    return spectrogram_list


def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            hearitback/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None
