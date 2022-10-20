#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

_C.BN.FREEZE = False

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm2d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "vggsound"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = True

# ---------------------------------------------------------------------------- #
# MixUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.3

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 0.3

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "vggsound"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from an audio uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.MODEL_NAME = "PlayItBackX2"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# If multi-task is used (enables overwritting `NUM_CLASSES` when creating the model).
_C.MODEL.MULTITASK = False

# Multi-task number of classes per task
_C.MODEL.MULTITASK_CLASSES = [97,300]

# Loss function.
_C.MODEL.LOSS_FUNC = "soft_cross_entropy"

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Only use the encoder as a feature extractor.
_C.MODEL.FREEZE_ENCODER = True

# Only use the encoder part of the network.
_C.MODEL.IGNORE_DECODER = False

# PlayItBack loops.
_C.MODEL.PLAYBACK = 3

# PlayItBack weights.
_C.MODEL.PLAYBACK_WEIGHTS = .3


# -----------------------------------------------------------------------------
# DECODER/TEMPR options
# -----------------------------------------------------------------------------

_C.DECODER = CfgNode()

# Number of freq bands, with original value (2 * K + 1)
_C.DECODER.NUM_FREQ_BANDS = 6

# Maximum frequency, hyperparameter depending on how fine the data is.
_C.DECODER.MAX_FREQ = 10

# Number of channels for each token of the input. (1 for audio 3 for images)
_C.DECODER.INPUT_CHANNELS = 1

# Number of latents, or induced set points, or centroids.
_C.DECODER.NUM_LATENTS = 256

# Latent dimension.
_C.DECODER.LATENT_DIM = 512

# Number of heads for cross attention. Perceiver paper uses 1.
_C.DECODER.CROSS_HEADS = 1

# Number of heads for latent self attention, 8.
_C.DECODER.LATENT_HEADS = 8

# Number of dimensions per cross attention head.
_C.DECODER.CROSS_DIM_HEAD = 64

# Number of dimensions per latent self attention head.
_C.DECODER.LATENT_DIM_HEAD = 64

# Attention dropout
_C.DECODER.ATTN_DROPOUT = 0.

# Feedforward dropout
_C.DECODER.FF_DROPOUT = 0.

# Whether to weight tie layers (optional).
_C.DECODER.WEIGHT_TIE_LAYERS = False

# Whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off
_C.DECODER.FOURIER_ENCODE_DATA = True

# Number of self attention blocks per cross attn.
_C.DECODER.SELF_PER_CROSS_ATTN = 3

# mean pool and project embeddings to number of classes (num_classes) at the end
_C.DECODER.FINAL_CLASSIFIER_HEAD = True

# Aggregation method for the tower predictors could be set to `mean` or `adaptive`
_C.DECODER.FUSION = 'adaptive'

# Aggregation of features or predictions, could be set to `features` to fuse playback features and
# use a single predictor or `predictions` to use multiple predictions and fuse them together.
_C.DECODER.FUSION_LOC = 'features'

# Number of attention towers
_C.DECODER.DEPTH = 2

# -----------------------------------------------------------------------------
# ENCODER/MViT options
# -----------------------------------------------------------------------------
_C.ENCODER = CfgNode()

# Options include `conv`, `max`.
_C.ENCODER.MODE = "conv"

# If True, perform pool before projection in attention.
_C.ENCODER.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.ENCODER.CLS_EMBED_ON = False

# Kernel size for patchtification.
_C.ENCODER.PATCH_KERNEL = [7, 7]

# Stride size for patchtification.
_C.ENCODER.PATCH_STRIDE = [4, 4]

# Padding size for patchtification.
_C.ENCODER.PATCH_PADDING = [3, 3]

# Base embedding dimension for the transformer.
_C.ENCODER.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.ENCODER.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.ENCODER.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.ENCODER.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.ENCODER.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.ENCODER.DEPTH = 16

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.ENCODER.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.ENCODER.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.ENCODER.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the ratio of ENCODER.DIM_MUL. If will overwrite ENCODER.POOL_KV_STRIDE if not None.
_C.ENCODER.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.ENCODER.POOL_Q_STRIDE = []

# Kernel size for Q, K, V pooling.
_C.ENCODER.POOL_KVQ_KERNEL = (3, 3)

# If True, perform no decay on positional embedding and cls embedding.
_C.ENCODER.ZERO_DECAY_POS_CLS = False

# If True, use absolute positional embedding.
_C.ENCODER.USE_ABS_POS = False

# If True, use relative positional embedding for spatial dimentions
_C.ENCODER.REL_POS_SPATIAL = True

# If True, init rel with zero
_C.ENCODER.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.ENCODER.RESIDUAL_POOLING = True

# Dim mul in qkv linear layers of attention block instead of MLP
_C.ENCODER.DIM_MUL_IN_ATT = True

# Use MVIT as backbone
_C.ENCODER.RETURN_EMBD_ONLY = False

# Get classes and embd
_C.ENCODER.RETURN_EMBD = True

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# List of input spectrogram channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [1, 1]

# If True, calculate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# Using multi-label data
_C.DATA.MULTI_LABEL = False

# -----------------------------------------------------------------------------
# Audio data options
# -----------------------------------------------------------------------------
_C.AUDIO_DATA = CfgNode()

# Sampling rate of audio (in kHz)
_C.AUDIO_DATA.SAMPLING_RATE = 24000

# Duration of audio clip from which to extract the spectrogram
_C.AUDIO_DATA.CLIP_SECS = 1.279

_C.AUDIO_DATA.WINDOW_LENGTH = 10

_C.AUDIO_DATA.HOP_LENGTH = 5

# Number of timesteps of the input spectrogram
_C.AUDIO_DATA.NUM_FRAMES = 256

# Number of frequencies of the input spectrogram
_C.AUDIO_DATA.NUM_FREQUENCIES = 128


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 1e-4

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 1e-8

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = True

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = True

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# The spatial crop size for training.
_C.DATA_LOADER.TRAIN_CROP_SIZE = (500, 128)

#The spatial crop size for testing.
_C.DATA_LOADER.TEST_CROP_SIZE = (500, 128)




# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

_C.EPICKITCHENS.AUDIO_DATA_FILE = ""

_C.EPICKITCHENS.ANNOTATIONS_DIR = ""

_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TRAIN_PLUS_VAL = False

_C.EPICKITCHENS.TEST_SPLIT = "validation"


# -----------------------------------------------------------------------------
# VGG-Sound Dataset options
# -----------------------------------------------------------------------------
_C.VGGSOUND = CfgNode()

_C.VGGSOUND.AUDIO_DATA_DIR = ""

_C.VGGSOUND.ANNOTATIONS_DIR = ""

_C.VGGSOUND.TRAIN_LIST = "train.pkl"

_C.VGGSOUND.VAL_LIST = "test.pkl"

_C.VGGSOUND.TEST_LIST = "test.pkl"


# -----------------------------------------------------------------------------
# Audioset Dataset options
# -----------------------------------------------------------------------------
_C.AUDIOSET = CfgNode()

_C.AUDIOSET.AUDIO_DATA_DIR = ""

_C.AUDIOSET.ANNOTATIONS_DIR = ""

_C.AUDIOSET.TRAIN_LIST = "as500k_train_flac.pkl"

_C.AUDIOSET.VAL_LIST = "test_flac.pkl"

_C.AUDIOSET.TEST_LIST = "test_flac.pkl"


# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]


# -----------------------------------------------------------------------------
# WANDB Visualization Options
# -----------------------------------------------------------------------------
_C.WANDB = CfgNode()
_C.WANDB.ENABLE = False
_C.WANDB.RUN_ID = ""


# Add custom config with default values.
custom_config.add_custom_config(_C)


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
