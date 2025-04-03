#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import asdict

from accelerate import Accelerator
from accelerate.utils import (
    TERecipeKwargs,
    MSAMPRecipeKwargs, 
    AORecipeKwargs,
    set_seed
)
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    save_dir: str,
    args: Any,
    is_best: bool = False,
    prefix: str = "",
    save_state_dict_only: bool = False,
) -> str:
    """
    Save a checkpoint of the training state.
    
    Args:
        accelerator: The Accelerator instance
        model: The model being trained
        tokenizer: The tokenizer
        optimizer: The optimizer
        lr_scheduler: The learning rate scheduler
        epoch: Current epoch number
        step: Current step number
        save_dir: Directory to save the checkpoint
        args: Training arguments
        is_best: Whether this is the best model so far
        prefix: Optional prefix for the checkpoint directory
        save_state_dict_only: If True, only save model state dict instead of full model
        
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory
    if prefix:
        checkpoint_dir = os.path.join(save_dir, f"{prefix}_epoch{epoch}_step{step}")
    else:
        checkpoint_dir = os.path.join(save_dir, f"epoch{epoch}_step{step}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Wait until all processes are ready
    accelerator.wait_for_everyone()
    
    # Get unwrapped model
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Save model
    if save_state_dict_only:
        # Save only the state dict
        model_state = accelerator.get_state_dict(model)
        accelerator.save(model_state, os.path.join(checkpoint_dir, "model.safetensors"))
    else:
        # Save the entire model
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            save_format="safetensors"
        )
    
    # Save tokenizer
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimization state
        torch.save({
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'step': step,
        }, os.path.join(checkpoint_dir, "optimizer.pt"))
        
        # Save training arguments
        with open(os.path.join(checkpoint_dir, "training_args.json"), 'w') as f:
            # Convert args to dict if it's not already
            if hasattr(args, "__dict__"):
                args_dict = vars(args)
            else:
                args_dict = args
            json.dump(args_dict, f, indent=2)
        
        # Save special copy if this is the best model
        if is_best:
            best_dir = os.path.join(save_dir, "best_model")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            logger.info(f"Saved best model to {best_dir}")
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    strict: bool = True,
) -> Tuple[int, int]:
    """
    Load a checkpoint.
    
    Args:
        accelerator: The Accelerator instance
        model: The model to load weights into
        optimizer: The optimizer to load state into
        lr_scheduler: The learning rate scheduler to load state into
        checkpoint_path: Path to the checkpoint directory or file
        strict: Whether to strictly enforce that the keys in state_dict match the model
        
    Returns:
        Tuple of (epoch, step) loaded from the checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Handle both file and directory paths
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        model_path = checkpoint_path
        optimizer_path = checkpoint_path / "optimizer.pt"
    else:
        model_path = checkpoint_path.parent
        optimizer_path = model_path / "optimizer.pt"
    
    # Load model with accelerator to ensure correct device placement
    accelerator.load_state(str(model_path))
    
    # Load optimizer state
    if optimizer_path.exists():
        opt_state = torch.load(optimizer_path, map_location='cpu')
        
        if optimizer is not None and 'optimizer' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer'])
            
        if lr_scheduler is not None and 'lr_scheduler' in opt_state and opt_state['lr_scheduler'] is not None:
            lr_scheduler.load_state_dict(opt_state['lr_scheduler'])
        
        epoch = opt_state.get('epoch', 0)
        step = opt_state.get('step', 0)
    else:
        logger.warning(f"Optimizer state not found at {optimizer_path}")
        # Try to extract epoch/step info from the directory name
        dir_name = model_path.name
        epoch = 0
        step = 0
        
        if "epoch" in dir_name:
            try:
                # Extract epoch from format like "epoch5_step100"
                epoch_str = dir_name.split("epoch")[1].split("_")[0]
                epoch = int(epoch_str)
            except (IndexError, ValueError):
                pass
                
        if "step" in dir_name:
            try:
                # Extract step from format like "epoch5_step100"
                step_str = dir_name.split("step")[1].split("_")[0]
                step = int(step_str)
            except (IndexError, ValueError):
                pass
    
    logger.info(f"Loaded checkpoint from epoch {epoch}, step {step}")
    return epoch, step


def find_latest_checkpoint(base_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory based on epoch and step numbers.
    
    Args:
        base_dir: Base directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint directory or None if no checkpoint found
    """
    if not os.path.exists(base_dir):
        return None
    
    checkpoint_dirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d)) and 
                      ("epoch" in d or "step" in d)]
    
    if not checkpoint_dirs:
        return None
    
    def extract_numbers(dirname):
        epoch = 0
        step = 0
        
        if "epoch" in dirname:
            try:
                epoch_parts = dirname.split("epoch")[1].split("_")[0]
                epoch = int(epoch_parts)
            except (IndexError, ValueError):
                pass
                
        if "step" in dirname:
            try:
                step_parts = dirname.split("step")[1].split("_")[0]
                step = int(step_parts)
            except (IndexError, ValueError):
                pass
                
        return (epoch, step)
    
    # Sort by epoch, then step
    latest_dir = sorted(checkpoint_dirs, key=extract_numbers, reverse=True)[0]
    return os.path.join(base_dir, latest_dir)


def initialize_accelerator(precision: str, fp8_backend: Optional[str] = None, tracking: bool = False):
    """
    Initialize the accelerator with appropriate precision settings.
    
    Args:
        precision: One of "fp8", "fp16", "bf16"
        fp8_backend: Backend for FP8 training ("te", "msamp", "ao")
        tracking: Whether to enable tracking with wandb
        
    Returns:
        Configured Accelerator instance
    """
    if precision == "fp8":
        if fp8_backend == "te":
            fp8_kwargs = TERecipeKwargs(
                fp8_format="HYBRID",  # E4M3 during forward pass, E5M2 during backward pass
                amax_history_len=16,
                amax_compute_algo="max"
            )
        elif fp8_backend == "msamp":
            fp8_kwargs = MSAMPRecipeKwargs()
        else:  # fp8_backend == "ao" or None
            fp8_kwargs = AORecipeKwargs()
            
        accelerator = Accelerator(
            mixed_precision=precision,
            kwargs_handlers=[fp8_kwargs],
            log_with="wandb" if tracking else None,
        )
    else:
        # For fp16 or bf16, use standard mixed precision
        accelerator = Accelerator(
            mixed_precision=precision,
            log_with="wandb" if tracking else None,
        )
    
    return accelerator


def get_checkpoint_info(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint from the training_args.json file.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        Dictionary with checkpoint information or empty dict if not found
    """
    args_path = os.path.join(checkpoint_dir, "training_args.json")
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            return json.load(f)
    return {}