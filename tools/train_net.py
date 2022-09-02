#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train an audio classification model."""

import numpy as np
from scipy.stats import gmean
import pprint
import wandb
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from sklearn import metrics as skmetrics

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import playitback.models.losses as losses
import playitback.models.optimizer as optim
import playitback.utils.checkpoint as cu
import playitback.utils.distributed as du
import playitback.utils.logging as logging
import playitback.utils.metrics as metrics
import playitback.utils.misc as misc
import playitback.visualization.tensorboard_vis as tb
from playitback.datasets import loader
from playitback.datasets.mixup import MixUp
from playitback.models import build_model
from playitback.utils.meters import TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter

from einops import reduce

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None, wandb_log=False, scaler=None
):
    """
    Perform the audio training for one epoch.
    Args:
        train_loader (loader): audio training loader.
        model (model): the audio model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            playitback/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    if cfg.BN.FREEZE:
        model.module.freeze_fn('bn_statistics') if cfg.NUM_GPUS > 1 else model.freeze_fn('bn_statistics')

    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for cur_iter, (inputs, labels, _, _) in enumerate(train_loader):

        cached_labels = labels

        # Transfer the labels to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            optimizer.zero_grad()
            preds = model(inputs)

            if not cfg.MODEL.IGNORE_DECODER:
                pos_pred = preds[1]
                neg_pred = preds[2]

                playback_preds = preds[0][1]
                preds = preds[0][0]

            if isinstance(labels, (dict,)):
                # Explicitly declare reduction to mean.
                loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

                # Compute the (per-playback) loss.


                loss_verb = torch,mean([loss_fun(playback_preds[0][i], labels['verb']) for i in range(playback_preds[0].shape[0])])
                loss_noun = torch,mean([loss_fun(preds[1][i], labels['noun']) for i in range(playback_preds[0].shape[0])])
                loss = 0.5 * (loss_verb + loss_noun)

                # check Nan Loss.
                misc.check_nan_losses(loss)
            else:
                # Explicitly declare reduction to mean.
                loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
                if not cfg.MODEL.IGNORE_DECODER:
                    loss_fun_pb = losses.get_loss_func('rank')(reduction="mean")

                # Compute the loss.
                loss = 0.
                if not cfg.MODEL.IGNORE_DECODER:
                    for i in range(playback_preds.shape[0]):
                        loss_i = loss_fun(playback_preds[i],labels)
                        loss = loss_i + loss
                else:
                    print(preds.shape,labels.shape)
                    loss = loss_fun(preds,labels)

                if cfg.MODEL.PLAYBACK > 0:
                    loss = loss_fun(preds,labels) + 0.1 * loss
                if not cfg.MODEL.IGNORE_DECODER:
                    if cfg.MODEL.LOSS_FUNC != 'cross_entropy':
                        use_multilabel = True
                    else:
                        use_multilabel = False
                    pos_loss = loss_fun_pb(pos_pred, cached_labels, multilabel=use_multilabel)
                    neg_loss = loss_fun_pb(neg_pred, cached_labels, multilabel=use_multilabel)

                    pb_loss = 0.1 * (pos_loss + neg_loss)
                    loss = loss + pb_loss

                # check Nan Loss.
                misc.check_nan_losses(loss)

        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if isinstance(labels, (dict,)):
            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce(
                    [loss_verb, verb_top1_acc, verb_top5_acc]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce(
                    [loss_noun, noun_top1_acc, noun_top5_acc]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (
                loss_noun.item(),
                noun_top1_acc.item(),
                noun_top5_acc.item(),
            )

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                 (labels['verb'], labels['noun']),
                                                                                 (1, 5))
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce(
                    [loss, action_top1_acc, action_top5_acc]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            # Update and log stats.
            train_meter.update_stats(
                (verb_top1_acc, noun_top1_acc, action_top1_acc),
                (verb_top5_acc, noun_top5_acc, action_top5_acc),
                (loss_verb, loss_noun, loss),
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": action_top1_acc,
                        "Train/Top5_acc": action_top5_acc,
                        "Train/verb/loss": loss_verb,
                        "Train/noun/loss": loss_noun,
                        "Train/verb/Top1_acc": verb_top1_acc,
                        "Train/verb/Top5_acc": verb_top5_acc,
                        "Train/noun/Top1_acc": noun_top1_acc,
                        "Train/noun/Top5_acc": noun_top5_acc,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": action_top1_acc,
                        "Train/Top5_acc": action_top5_acc,
                        "Train/verb/loss": loss_verb,
                        "Train/noun/loss": loss_noun,
                        "Train/verb/Top1_acc": verb_top1_acc,
                        "Train/verb/Top5_acc": verb_top5_acc,
                        "Train/noun/Top1_acc": noun_top1_acc,
                        "Train/noun/Top5_acc": noun_top5_acc,
                        "train_step": data_size * cur_epoch + cur_iter,
                    },
                )
        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
                # Average precision over multi-label
                avg_pr = 0.
                for b in range(labels.shape[0]):
                    avg_pr += skmetrics.average_precision_score(labels[b].cpu().detach().numpy(), preds[b].cpu().detach().numpy())
                avg_pr /= labels.shape[0]
            else:
                avg_pr = None
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                avg_pr,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                if cfg.DATA.MULTI_LABEL:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/mAP": avg_pr
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
                else:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )

            if wandb_log:
                if cfg.DATA.MULTI_LABEL:
                    wandb.log(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/mAP": avg_pr,
                            "train_step": data_size * cur_epoch + cur_iter,
                        },
                    )
                else:
                    wandb.log(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                            "train_step": data_size * cur_epoch + cur_iter,
                        },
                    )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, wandb_log=False):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            playitback/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    data_size = len(val_loader)

    for cur_iter, (inputs, labels, _, _) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

        val_meter.data_toc()

        preds = model(inputs)
        top1_err, top5_err = None, None
        pos_pred = preds[1]
        neg_pred = preds[2]

        playback_preds = preds[0][1]
        preds = preds[0][0]

        if isinstance(labels, (dict,)):
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss_verb = loss_fun(preds[0], labels['verb'])
            loss_noun = loss_fun(preds[1], labels['noun'])
            loss = 0.5 * (loss_verb + loss_noun)

            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce(
                    [loss_verb, verb_top1_acc, verb_top5_acc]
                )

            # Copy the errors from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce(
                    [loss_noun, noun_top1_acc, noun_top5_acc]
                )

            # Copy the errors from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (
                loss_noun.item(),
                noun_top1_acc.item(),
                noun_top5_acc.item(),
            )

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                 (labels['verb'], labels['noun']),
                                                                                 (1, 5))
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce(
                    [loss, action_top1_acc, action_top5_acc]
                )

            # Copy the errors from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            # Update and log stats.
            val_meter.update_stats(
                (verb_top1_acc, noun_top1_acc, action_top1_acc),
                (verb_top5_acc, noun_top5_acc, action_top5_acc),
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                    },
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                        "val_step": len(val_loader) * cur_epoch + cur_iter,
                    },
                )

            val_meter.update_predictions((preds[0], preds[1]), (labels['verb'], labels['noun']))

        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            loss_fun_pb = losses.get_loss_func('rank')(reduction="mean")

            # Compute the loss.
            loss = 0.
            if not cfg.MODEL.IGNORE_DECODER:
                for i in range(playback_preds.shape[0]):
                    loss_i = loss_fun(playback_preds[i],labels)
                    loss = loss_i + loss
            else:
                print(preds.shape,labels.shape)
                loss = loss_fun(preds,labels)

            if cfg.MODEL.PLAYBACK > 0:
                loss = loss_fun(preds,labels) + 0.1 * loss
            if not cfg.MODEL.IGNORE_DECODER:
                if cfg.MODEL.LOSS_FUNC != 'cross_entropy':
                    use_multilabel = True
                else:
                    use_multilabel = False
                pos_loss = loss_fun_pb(pos_pred, labels, multilabel=use_multilabel)
                neg_loss = loss_fun_pb(neg_pred, labels, multilabel=use_multilabel)

                pb_loss = 0.1 * (pos_loss + neg_loss)
                loss = loss + pb_loss

            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
                # Average precision over multi-label
                avg_pr = 0.
                for b in range(labels.shape[0]):
                    avg_pr += skmetrics.average_precision_score(labels[b].cpu().detach().numpy(), preds[b].cpu().detach().numpy())
                avg_pr /= labels.shape[0]

            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the errors from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                avg_pr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                if cfg.DATA.MULTI_LABEL:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/mAP": avg_pr
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
                else:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )

            if wandb_log:
                if cfg.DATA.MULTI_LABEL:
                    wandb.log(
                        {
                            "Train/loss": loss,
                            "Train/mAP": avg_pr,
                            "train_step": data_size * cur_epoch + cur_iter,
                        },
                    )
                else:
                    wandb.log(
                        {
                            "Train/loss": loss,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                            "train_step": data_size * cur_epoch + cur_iter,
                        },
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch, top1_dict = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )

    if writer is not None and not wandb_log:
        if "top1_acc" in top1_dict.keys():
            if not cfg.DATA.MULTI_LABEL:
                writer.add_scalars(
                    {
                        "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                        "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                        "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                    },
                    global_step=cur_epoch,
                )
            else:
                writer.add_scalars(
                    {"Val/epoch/mAP": top1_dict["map"]},
                    global_step=cur_epoch,
                )

        else:
            if not cfg.DATA.MULTI_LABEL:
                writer.add_scalars(
                    {"Val/epoch/Top1_err": top1_dict["top1_err"]},
                    global_step=cur_epoch,
                )
            else:
                writer.add_scalars(
                    {"Val/epoch/mAP": top1_dict["map"]},
                    global_step=cur_epoch,
                )

    if wandb_log:
        if "top1_acc" in top1_dict.keys():
            wandb.log(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                    "epoch": cur_epoch,
                },
            )

        else:
            if not cfg.DATA.MULTI_LABEL:
                wandb.log(
                    {"Val/epoch/Top1_err": top1_dict["top1_err"], "epoch": cur_epoch}
                )
            else:
                wandb.log(
                    {"Val/epoch/mAP": top1_dict["map"], "epoch": cur_epoch}
                )
    if not cfg.DATA.MULTI_LABEL:
        top1 = top1_dict["top1_acc"] if "top1_acc" in top1_dict.keys() else top1_dict["top1_err"]
    else:
        top1 = top1_dict["map"]
    val_meter.reset()
    return is_best_epoch, top1


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train an audio model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            playitback/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the audio model and print model statistics.
    model = build_model(cfg)
    #if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #    misc.log_model_info(model, cfg)

    if cfg.BN.FREEZE:
        model.module.freeze_fn('bn_parameters') if cfg.NUM_GPUS > 1 else model.freeze_fn('bn_parameters')

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None)

    # Create the audio train and val loaders.
    if cfg.TRAIN.DATASET != 'epickitchens' or not cfg.EPICKITCHENS.TRAIN_PLUS_VAL:
        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")
        precise_bn_loader = (
            loader.construct_loader(cfg, "train")
            if cfg.BN.USE_PRECISE_STATS
            else None
        )
    else:
        train_loader = loader.construct_loader(cfg, "train+val")
        val_loader = loader.construct_loader(cfg, "val")
        precise_bn_loader = (
            loader.construct_loader(cfg, "train+val")
            if cfg.BN.USE_PRECISE_STATS
            else None
        )

    # Create meters.
    if cfg.TRAIN.DATASET == 'epickitchens':
        train_meter = EPICTrainMeter(len(train_loader), cfg)
        val_meter = EPICValMeter(len(val_loader), cfg)
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    if cfg.WANDB.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        wandb_log = True
        if cfg.TRAIN.AUTO_RESUME and cfg.WANDB.RUN_ID != "":
            wandb.init(project='playitback', config=cfg, sync_tensorboard=True, resume=cfg.WANDB.RUN_ID)
        else:
            wandb.init(project='playitback', config=cfg, sync_tensorboard=True)
        wandb.watch(model)

    else:
        wandb_log = False

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer, wandb_log, scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch,
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            is_best_epoch, _ = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer, wandb_log)
            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=is_best_epoch, scaler=scaler if cfg.TRAIN.MIXED_PRECISION else None)

    if writer is not None:
        writer.close()
