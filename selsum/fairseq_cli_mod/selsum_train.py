# SelSum training script where one batch is used for separately computing
# gradients for the summarizer and posterior

import math
import os
from copy import deepcopy
import random

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from selsum.fairseq_mod.trainer import Trainer

from fairseq_cli.train import logger, tpu_data_loader,\
    get_training_stats, validate_and_save, validate
from selsum.utils.constants.model import SEL_TRAIN_INNER, SUM_TRAIN_INNER, \
    SEL_STEP_NUM, SUM_STEP_NUM, SEL_STEP


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"
    metrics.reset()

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu and not getattr(args, "tpu", False):
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info(
        "model {}, criterion {}".format(args.arch, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.max_sentences
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    if args.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers
        xm.mark_step()

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, "data", "")),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Trains the model for one epoch and return validation losses.

    It is modified to optimize the posterior (selector) and summarizer
    separately.
    """
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = tpu_data_loader(args, itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    sel_step_num = getattr(task, SEL_STEP_NUM, 1)
    sum_step_num = getattr(task, SUM_STEP_NUM, 1)

    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    for samples in progress:

        # TODO: investigate why empty samples are passed!
        samples = [s for s in samples if s is not None and len(s) != 0]
        if len(samples) == 0:
            logger.info("Empty sample; skipping")
            continue

        # posterior step
        sel_samples = []
        for s in samples:
            s = deepcopy(s)
            s[SEL_STEP] = True
            sel_samples.append(s)

        with metrics.aggregate(SEL_TRAIN_INNER):
            for _ in range(sel_step_num):
                log_output = trainer.train_step(sel_samples)
                if log_output is None:  # OOM, overflow, ...
                    continue

        with metrics.aggregate(SUM_TRAIN_INNER):
            for _ in range(sum_step_num):
                log_output = trainer.train_step(samples)
                if log_output is None:  # OOM, overflow, ...
                    continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:

            if sel_step_num > 0:
                stats = get_training_stats(metrics.get_smoothed_values(SEL_TRAIN_INNER))
                progress.log(stats, tag=SEL_TRAIN_INNER, step=num_updates)
                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters(SEL_TRAIN_INNER)

            if sum_step_num > 0:
                stats = get_training_stats(metrics.get_smoothed_values(SUM_TRAIN_INNER))
                progress.log(stats, tag=SUM_TRAIN_INNER, step=num_updates)
                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters(SUM_TRAIN_INNER)

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
        if should_stop:
            break

    # performing validation in the end of epoch
    if len(valid_losses) == 1 and valid_losses[0] is None:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        if not getattr(args, "tpu", False):
            # fallback for single node with multiple GPUs
            assert args.distributed_world_size <= torch.cuda.device_count()
            port = random.randint(10000, 20000)
            args.distributed_init_method = "tcp://localhost:{port}".format(port=port)
            args.distributed_rank = None  # set based on device id
            torch.multiprocessing.spawn(
                fn=distributed_main, args=(args,), nprocs=args.distributed_world_size
            )
        else:
            import torch_xla.distributed.xla_multiprocessing as xmp

            torch.multiprocessing.set_sharing_strategy("file_system")
            xmp.spawn(
                fn=distributed_main, args=(args,), nprocs=8  # use all 8 TPU cores
            )
    else:
        # single GPU training
        main(args)


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


if __name__ == "__main__":
    cli_main()
