import os, sys
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data
import os
import random

import playitback.utils.logging as logging

from .build import DATASET_REGISTRY

from .spec_augment import combined_transforms
from . import utils as utils
from .audio_loader_audioset import pack_audio

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Audioset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Audioset".format(mode)
        self.cfg = cfg
        self.mode = mode
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        logger.info("Constructing Audioset {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        if self.mode == "train":
            path_annotations_pickle = os.path.join(self.cfg.AUDIOSET.ANNOTATIONS_DIR, self.cfg.AUDIOSET.TRAIN_LIST)
        elif self.mode == "val":
            path_annotations_pickle = os.path.join(self.cfg.AUDIOSET.ANNOTATIONS_DIR, self.cfg.AUDIOSET.VAL_LIST)
        else:
            path_annotations_pickle = os.path.join(self.cfg.AUDIOSET.ANNOTATIONS_DIR, self.cfg.AUDIOSET.TEST_LIST)

        assert os.path.exists(path_annotations_pickle), "{} dir not found".format(
            path_annotations_pickle
        )

        self._audio_records = []
        self._temporal_idx = []
        for tup in pd.read_pickle(path_annotations_pickle).iterrows():
            for idx in range(self._num_clips):
                self._audio_records.append(tup[1])
                self._temporal_idx.append(idx)
        assert (
                len(self._audio_records) > 0
        ), "Failed to load Audioset split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing audioset dataloader (size: {}) from {}".format(
                len(self._audio_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, and audio
        index.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """

        import warnings
        warnings.filterwarnings('ignore')

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        spectrograms = pack_audio(self.cfg, self._audio_records[index], temporal_sample_index)

        sps = []
        for i,spectrogram in enumerate(spectrograms):
            # Normalization.
            spectrogram = spectrogram.float()
            if self.mode in ["train"]:
                # Data augmentation.
                # C T F -> C F T
                spectrogram = spectrogram.permute(0, 2, 1)
                # SpecAugment
                spectrogram = combined_transforms(spectrogram)
                # C F T -> C T F
                spectrogram = spectrogram.permute(0, 2, 1)
            if spectrogram.shape[-2] != self.cfg.DATA_LOADER.TRAIN_CROP_SIZE[-2]:
                spectrogram = F.interpolate(spectrogram.unsqueeze(0),size=self.cfg.DATA_LOADER.TRAIN_CROP_SIZE).squeeze(0)
            sps.append(utils.pack_pathway_output(self.cfg, spectrogram)[0])
        label = self._audio_records[index]['class_ids']
        if not isinstance(label,list):
            label = [label]

        spectrograms = sps
        # pad spectrogram along temporal dimension
        if len(spectrograms) > 0:
            for i,s in enumerate(spectrograms):
                pad_2d = (0,0,spectrograms[-1].shape[-2] - s[-1].shape[-2], 0)
                spectrograms[i] = torch.nn.functional.pad(s,pad_2d,'constant',0)

            spectrograms = torch.stack(spectrograms).permute(1,0,2,3).squeeze(0)

        multi_hot = torch.zeros(self.cfg.MODEL.NUM_CLASSES, dtype=torch.float32)
        multi_hot[[l-1 for l in label]] = 1.

        return spectrograms, multi_hot, index, {}

    def __len__(self):
        return len(self._audio_records)
