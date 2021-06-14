r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ...

    Example:
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --use_env \
        -m lib.training.train TADAM_MOT17 --config TADAM_MOT17

    Note:
        If conducting multiple mgpu training on the same machine, add --master_port=$any_number in first line
        Use different numbers for trainings. This is to avoid setup conflict between different trainings
"""

import os
import sys
import time
import datetime
import random
import math
import numpy as np
import torch

from ..modules.detector import Detector
from ..dataset.mot import MOTDetection, collate_fn
from .train_utils import init_distributed_mode, get_rank, get_transform, \
    MetricLogger, SmoothedValue, reduce_dict, save_on_master
from ..configs.config import load_config
from ..utils.log import get_logger, log_or_print


def train_mot(training_name, save_dir, config, logger, is_distributed=False):
    # Deterministic
    seed = config.TRAINING.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Get current device
    # Operations other than training should only happen on device 0
    current_device = get_rank()

    # Dataset
    dataset_train = MOTDetection(
        root=config.PATHS.DATASET_ROOT,
        dataset=config.NAMES.DATASET if config.NAMES.DATASET != "MOT17" else "MOT17Det",
        transforms=get_transform(train=True),
        vis_threshold=config.TRAINING.VIS_THRESHOLD,
    )
    # Dataloader
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, config.TRAINING.BATCH_SIZE, drop_last=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=train_batch_sampler,
        num_workers=config.TRAINING.WORKERS,
        collate_fn=collate_fn
    )

    # Create model and load checkpoint
    log_or_print(logger, f"Creating model on device #{current_device}")
    model = Detector(
        config,
        num_classes=2,
        num_ids=dataset_train.num_ids,
        tracking=False,
        logger=logger
    )
    device = "cuda"
    model.to(device)
    # For distributed training
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    # Warmup epochs
    total_warmup_epochs = config.TRAINING.WARMUP_EPOCHS
    remaining_warmup_epochs = total_warmup_epochs

    # Optimizer
    if config.TRAINING.WARMUP_EPOCHS > 0:
        # Set lr for id warmup components separately
        id_warmup_lr_list = ["roi_heads.id_module", "memory_net"]
        base_params = []
        warmup_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                in_warmup_list = False
                for w_n in id_warmup_lr_list:
                    if name.startswith(w_n):
                        in_warmup_list = True
                if in_warmup_list:
                    warmup_params.append(p)
                else:
                    base_params.append(p)
        params = [
            {"params": base_params},
            {"params": warmup_params, "lr": config.TRAINING.WARMUP_LR}
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.TRAINING.LR,
        momentum=config.TRAINING.MOMENTUM,
        weight_decay=config.TRAINING.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config.TRAINING.LR_STEP_SIZE, gamma=config.TRAINING.LR_GAMMA)

    # Ready for training
    if is_distributed:
        log_or_print(logger, f"Multiple GPU training on device #{current_device}")
    else:
        log_or_print(logger, f"Single GPU training on device #{current_device}")

    # Train
    start_time = time.time()
    # Epoch starts from 1 for easier understanding
    for epoch in range(1, 1 + config.TRAINING.EPOCHS + total_warmup_epochs):
        # Sync
        if is_distributed:
            train_sampler.set_epoch(epoch)
            torch.cuda.synchronize()
        # Reset lr for formal training
        if epoch == total_warmup_epochs + 1:
            for g in optimizer.param_groups:
                g["lr"] = config.TRAINING.LR

        # Initialize tracklets for the epoch, for identity training
        if is_distributed:
            model.module.all_tracklets_dict = {}
        else:
            model.all_tracklets_dict = {}

        model.train()
        metric_logger = MetricLogger(delimiter="  ", logger=logger)
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"device: [{current_device}] {'warmup ' if remaining_warmup_epochs > 0 else ''}" + \
            f"epoch: [{epoch if remaining_warmup_epochs > 0 else epoch - total_warmup_epochs:2d}/" + \
            f"{total_warmup_epochs if remaining_warmup_epochs > 0 else config.TRAINING.EPOCHS:2d}]"

        for step, (images, targets) in metric_logger.log_every(data_loader_train, config.TRAINING.PRINT_FREQ, header):
            step += 1  # Starts with 1 for easier understanding

            # Move to cuda
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Loss
            if is_distributed:
                loss_dict = model.module.custom_train(images, targets,
                    warmup=remaining_warmup_epochs > 0)
            else:
                loss_dict = model.custom_train(images, targets,
                    warmup=remaining_warmup_epochs > 0)
            losses = sum(loss for loss in loss_dict.values())

            # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)

            # Overall loss
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            # Detect loss explosion
            if not math.isfinite(loss_value):
                logger.info(f"Loss is {loss_value}, stopping training")
                logger.info(f"Last loss {loss_dict_reduced}")
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            # Clip gradients in warmup only, may happen if lr is large
            if remaining_warmup_epochs > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # Update log
            if remaining_warmup_epochs > 0:
                metric_logger.update(**{"lr_warmup": optimizer.param_groups[1]["lr"]})
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # No lr_scheduler stepping in warmup
        if remaining_warmup_epochs > 0:
            # Do not step lr during warmup
            remaining_warmup_epochs -= 1
            # Save checkpoint at end of warmup
            if remaining_warmup_epochs == 0:
                # Save checkpoint, only in main process
                save_on_master(
                    {"state_dict": model.module.state_dict() if is_distributed else model.state_dict()},
                    os.path.join(save_dir, f"checkpoint_{training_name}_warmup_epoch_{total_warmup_epochs}.pth")
                )
        # End of warmup
        else:
            # Move forward in lr_scheduler
            lr_scheduler.step()
            # Save checkpoints, at SAVE_FREQ or last epoch
            if (epoch - total_warmup_epochs) % config.TRAINING.SAVE_FREQ == 0 or \
                    (epoch - total_warmup_epochs) == config.TRAINING.EPOCHS:
                # Save checkpoint, only in main process
                save_on_master(
                    {"state_dict": model.module.state_dict() if is_distributed else model.state_dict()},
                    os.path.join(save_dir, f"checkpoint_{training_name}_epoch_{epoch - total_warmup_epochs}.pth")
                )

    # Clean up
    if is_distributed:
        torch.distributed.destroy_process_group()

    # Save final model to model root
    model_path = os.path.join(config.PATHS.MODEL_ROOT, f"{training_name}.pth")
    save_on_master({"state_dict": model.module.state_dict() if is_distributed else model.state_dict()}, model_path)
    log_or_print(logger, f"Saved trained model to '{model_path}'")

    # Log total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_or_print(logger, f"Training time {total_time_str}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="train on mot")
    parser.add_argument("name", help="name for training, required")
    parser.add_argument("--config", default="TADAM_MOT17", type=str, help="config file to load")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    args = parser.parse_args()

    # Load config
    config, cfg_msg = load_config(args.config)
    # Create folder for output
    save_dir = os.path.join(config.PATHS.MODEL_ROOT, "checkpoints", args.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Change default url in case multiple training in progress to avoid conflict
    if args.dist_url == "env://":
        args.dist_url += f"{args.name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    # Setup distributed training
    init_distributed_msg = init_distributed_mode(args)

    # Logger
    logger = get_logger(name="global", save_file=True, overwrite_file=True,
        log_dir=save_dir, log_name=f"{args.name}")
    log_or_print(logger, cfg_msg)
    log_or_print(logger, init_distributed_msg)

    train_mot(args.name, save_dir, config, logger, is_distributed=args.distributed)
