# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import (
    normalize_camera_extrinsics_and_points_batch,
    normalize_camera_extrinsics_points_and_3djoints_batch,
)
from train_utils.optimizer import construct_optimizers
from training.temporal import flatten_temporal_batch_for_framewise_model


NORMALIZE_CAM = False


def first_nonfinite(x, name=""):
    import torch
    if torch.is_tensor(x):
        if x.numel() and not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).nonzero()[0].tolist()
            raise FloatingPointError(f"nonfinite: {name} shape={tuple(x.shape)} idx={bad} val={x[tuple(bad)].item()}")
    elif isinstance(x, dict):
        for k,v in x.items(): first_nonfinite(v, f"{name}.{k}" if name else k)
    elif isinstance(x, (list, tuple)):
        for i,v in enumerate(x): first_nonfinite(v, f"{name}[{i}]")

class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        enable_val: bool = True,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        scale_by_extrinsics: bool = True,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.enable_val = bool(enable_val)
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        self.scale_by_extrinsics = bool(scale_by_extrinsics)
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

        smpl_loss_cfg = getattr(loss, "smpl", None)
        ball_loss_cfg = getattr(loss, "ball", None)
        if smpl_loss_cfg is not None and hasattr(smpl_loss_cfg, "normalize_cam"):
            self.normalize_cam = smpl_loss_cfg.normalize_cam
        elif ball_loss_cfg is not None and hasattr(ball_loss_cfg, "normalize_cam"):
            self.normalize_cam = ball_loss_cfg.normalize_cam
        else:
            self.normalize_cam = True

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        load_aggregator_only = getattr(self.checkpoint_conf, "load_aggregator_only", False)
        if load_aggregator_only:
            aggregator_state = {
                key: value for key, value in model_state_dict.items() if key.startswith("aggregator.")
            }
            if not aggregator_state:
                logging.warning(
                    "No aggregator.* keys found in checkpoint while load_aggregator_only=True; skipping model load."
                )
            else:
                model_state_dict = aggregator_state
                logging.info(
                    f"Filtered checkpoint to {len(model_state_dict)} aggregator parameters (load_aggregator_only=True)."
                )

        # Optional warm start for the pixel-level (DPT) person-mask head: copy a
        # pretrained depth head's DPT trunk (depth_head.*) into
        # person_mask_head.trunk.* when the model itself has no such weights yet.
        # The depth trunk already encodes body/depth boundaries, which speeds up
        # mask convergence a lot vs. training the trunk from scratch.
        #   True  -> take depth_head.* from THIS resume checkpoint;
        #   <str> -> load depth_head.* from that checkpoint path instead (use when
        #            resuming from an SMPL-only checkpoint that has no depth head,
        #            e.g. base VGGT ckpt/model.pt keeps the depth weights).
        mask_trunk_src = getattr(self.checkpoint_conf, "init_mask_trunk_from_depth", False)
        if mask_trunk_src:
            if isinstance(mask_trunk_src, str):
                with g_pathmgr.open(mask_trunk_src, "rb") as f:
                    depth_ckpt = torch.load(f, map_location="cpu")
                depth_state = depth_ckpt["model"] if "model" in depth_ckpt else depth_ckpt
            else:
                depth_state = model_state_dict
            own_state = self.model.state_dict()
            n_copied = 0
            for key, value in depth_state.items():
                if not key.startswith("depth_head."):
                    continue
                target_key = "person_mask_head.trunk." + key[len("depth_head."):]
                if target_key in own_state and target_key not in model_state_dict:
                    if own_state[target_key].shape == value.shape:
                        model_state_dict[target_key] = value.clone()
                        n_copied += 1
            if self.rank == 0:
                src_name = mask_trunk_src if isinstance(mask_trunk_src, str) else "resume checkpoint"
                logging.info(
                    f"init_mask_trunk_from_depth: copied {n_copied} depth_head.* tensors "
                    f"from {src_name} into person_mask_head.trunk.*"
                )
                if n_copied == 0:
                    logging.warning(
                        "init_mask_trunk_from_depth found no copyable depth_head.* weights -- "
                        "the source checkpoint has no depth head (or shapes mismatch). "
                        "Point it at a checkpoint that has one (e.g. the base VGGT model.pt)."
                    )

        strict_flag = self.checkpoint_conf.strict and not load_aggregator_only
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=strict_flag
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        if getattr(self.checkpoint_conf, "load_model_only", False):
            if self.rank == 0:
                logging.info("load_model_only=True: skipping optimizer/scaler/epoch/steps/time_elapsed restore.")
            # Reset training progress explicitly
            self.epoch = 0
            self.steps = {"train": 0, "val": 0}
            self.ckpt_time_elapsed = 0
            return
        
        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint and getattr(self, "optims", None) is not None:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            opt_state = checkpoint["optimizer"]
            if isinstance(opt_state, list):
                if len(opt_state) != len(self.optims):
                    logging.warning(
                        "Checkpoint has %d optimizer states but trainer has %d optimizers; loading the common prefix.",
                        len(opt_state),
                        len(self.optims),
                    )
                for optim, state in zip(self.optims, opt_state):
                    try:
                        optim.optimizer.load_state_dict(state)
                    except Exception:
                        logging.exception("Failed to load optimizer state; continuing without it.")
                        break
            elif isinstance(opt_state, dict):
                if len(self.optims) == 1:
                    try:
                        self.optims[0].optimizer.load_state_dict(opt_state)
                    except Exception:
                        logging.exception("Failed to load optimizer state; continuing without it.")
                else:
                    logging.warning(
                        "Checkpoint optimizer state is a single dict but trainer has %d optimizers; "
                        "loading into optims[0] only.",
                        len(self.optims),
                    )
                    try:
                        self.optims[0].optimizer.load_state_dict(opt_state)
                    except Exception:
                        logging.exception("Failed to load optimizer state; continuing without it.")
            else:
                logging.warning("Unknown optimizer state type in checkpoint: %s", type(opt_state))

            # Ensure optimizer state tensors are on the right device.
            # torch.optim.Optimizer.load_state_dict keeps state tensors on their original device.
            for optim in self.optims:
                for state in optim.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device, non_blocking=True)

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        elif "prev_epoch" in checkpoint:
            self.epoch = checkpoint["prev_epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        if bool(getattr(self.logging_conf, "use_wandb", False)):
            self.tb_writer = instantiate(self.logging_conf.wandb_writer, _recursive_=False)
        else:
            self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        # enable_val=False skips building the val dataset entirely (avoids the
        # expensive raw-mamma pyd cold-read); run_val() then no-ops since
        # self.val_dataset stays None.
        if self.mode in ["train", "val"] and self.enable_val:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value
        elif self.mode == "val" and not self.enable_val:
            logging.warning("mode='val' but enable_val=False; nothing to run.")

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            logging.info(f"self.checkpoint_conf.save_dir: {self.checkpoint_conf.save_dir}")

            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)

            # Save a checkpoint every save_freq epochs (was: every epoch
            # unconditionally, which on a 1-sample overfit = a full save every
            # step). Set save_freq: 1 to keep the old save-every-epoch behavior.
            _save_freq = getattr(self.checkpoint_conf, "save_freq", 0)
            if (
                _save_freq > 0
                and int(self.epoch) % _save_freq == 0
                and (int(self.epoch) > 0 or _save_freq == 1)
            ):
                self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        if self.rank == 0:
            prefix = f"Loss/{phase}_"
            for meter_name, meter in loss_meters.items():
                if meter.count <= 0:
                    continue
                key = meter_name[len(prefix):] if meter_name.startswith(prefix) else meter_name
                self.tb_writer.log(f"Values/{phase}_epoch/{key}", meter.avg, self.epoch)

        return True

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        profiled_sample_times = []
        collate_pin_overheads = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            profile_sample_seconds = batch.pop("_profile_sample_seconds", None)
            if profile_sample_seconds is not None and data_iter > 0:
                profiled_seconds = float(profile_sample_seconds.sum().item())
                profiled_sample_times.append(profiled_seconds)
                collate_pin_overheads.append(max(0.0, data_time.val - profiled_seconds))
                if len(profiled_sample_times) >= 10:
                    logging.info(
                        "Data profile (avg over %d batches): dataset_samples=%.3fs | "
                        "collate_pin_overhead=%.3fs | measured_data_time=%.3fs",
                        len(profiled_sample_times),
                        sum(profiled_sample_times) / len(profiled_sample_times),
                        sum(collate_pin_overheads) / len(collate_pin_overheads),
                        sum(profiled_sample_times[i] + collate_pin_overheads[i]
                            for i in range(len(profiled_sample_times)))
                        / len(profiled_sample_times),
                    )
                    profiled_sample_times.clear()
                    collate_pin_overheads.clear()

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    if not optim.schedulers:
                        continue  # no schedulers configured for this optimizer
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                nonfinite_grad_name = None
                nonfinite_grad_max = None
                for n,p in self.model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        nonfinite_grad_name = n
                        nonfinite_grad_max = p.grad.detach().abs().amax().item()
                        break

                if nonfinite_grad_name is not None:
                    logging.error(
                        "Skipping optimizer step due to nonfinite grad: %s max=%s",
                        nonfinite_grad_name,
                        nonfinite_grad_max,
                    )
                    for optim in self.optims:
                        optim.zero_grad(set_to_none=True)
                    self.scaler.update()

                    batch_time.update(time.time() - end)
                    end = time.time()
                    self.time_elapsed_meter.update(
                        time.time() - self.start_time + self.ckpt_time_elapsed
                    )
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)

                    if data_iter % self.logging_conf.log_freq == 0:
                        progress.display(data_iter)
                    continue

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            save_step_freq = getattr(self.checkpoint_conf, "save_step_freq", 0)
            if (
                phase == "train"
                and save_step_freq
                and self.steps[phase] > 0
                and self.steps[phase] % save_step_freq == 0
            ):
                checkpoint_names = [
                    "checkpoint",
                    f"checkpoint_step_{self.steps[phase]}",
                ]
                self.save_checkpoint(self.epoch, checkpoint_names=checkpoint_names)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        if self.normalize_cam:
            # Normalize camera extrinsics, points, and 3D joints together.
            # joints3d_world 可能不存在於部分資料集，因此使用 get 以保持相容性。
            (
                normalized_extrinsics,
                normalized_cam_points,
                normalized_world_points,
                normalized_joints3d_world,
                normalized_depths,
                avg_scale,
            ) = normalize_camera_extrinsics_points_and_3djoints_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch.get("cam_points"),
                world_points=batch.get("world_points"),
                joints3d_world=batch.get("smpl_joints3d_world"),
                depths=batch.get("depths"),
                scale_by_extrinsics=self.scale_by_extrinsics,
                point_masks=batch.get("point_masks"),
            )

            # Replace the original values in the batch with the normalized ones.
            # 同時保留 avg_scale 供後續需要時使用。
            batch["avg_scale"] = avg_scale
            batch["raw_extrinsics"] = batch["extrinsics"].clone()
            batch["extrinsics"] = normalized_extrinsics
            if normalized_cam_points is not None:
                batch["cam_points"] = normalized_cam_points
            if normalized_world_points is not None:
                batch["world_points"] = normalized_world_points
            if normalized_joints3d_world is not None:
                batch["smpl_joints3d_world"] = normalized_joints3d_world
            if normalized_depths is not None:
                batch["depths"] = normalized_depths
        else:
            B, S, _, _ = batch["extrinsics"].shape
            device = batch["extrinsics"].device
            assert device == torch.device("cpu")
            batch["avg_scale"] = torch.ones(B, device=device)
            batch["raw_extrinsics"] = batch["extrinsics"].clone()

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        model_module = model.module if hasattr(model, "module") else model
        use_temporal_model = bool(
            getattr(model_module, "use_temporal_smpl_head", False)
        )
        # Preserve the legacy framewise-temporal configs: only the new temporal
        # head needs the clip axis during forward.
        if not use_temporal_model:
            batch = flatten_temporal_batch_for_framewise_model(batch)

        if batch["images"].dtype == torch.uint8:
            batch["images"] = batch["images"].to(
                dtype=torch.get_default_dtype()
            ).div_(255)
        smpl_inputs = {}
        for key in (
            "views_per_frame", "temporal_num_frames", "num_groups",
            "frame_ids", "view_ids",
        ):
            if key in batch:
                smpl_inputs[key] = batch[key]

        # Keep [B,T*V,...] intact for the model.  VGGT folds it to [B*T,V,...]
        # only around the per-timestep aggregator, then the temporal SMPL head
        # jointly decodes all T frames.  Flatten the GT AFTER forward so the
        # existing framewise SMPL/camera/mask losses retain their [B*T,...] API.
        y_hat = model(images=batch["images"], smpl_inputs=smpl_inputs)
        if use_temporal_model:
            batch = flatten_temporal_batch_for_framewise_model(batch)

        # first_nonfinite(y_hat, "y_hat")
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)

        # for key in (
        #     "loss_smpl_losses",
        #     "loss_smpl_joints2d",
        #     "loss_smpl_joints3d",
        #     "loss_smpl_vertices",
        #     "loss_smpl",
        # ):
        #     if key in loss_dict:
        #         first_nonfinite(loss_dict[key], f"loss.{key}")
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        log_step = self.steps[phase]

        self._update_and_log_scalars(log_data, phase, log_step, loss_meters)
        self._log_tb_visuals(log_data, phase, log_step)

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
