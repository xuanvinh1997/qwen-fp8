#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import math
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    Qwen2MoeForCausalLM,
    Qwen2MoeConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
import wandb
from accelerate import Accelerator
from accelerate.utils import (
    TERecipeKwargs,
    MSAMPRecipeKwargs,
    AORecipeKwargs,
    DummyOptim,
    DummyScheduler,
    find_executable_batch_size,
)
from accelerate.logging import get_logger
from datasets import load_dataset
from tqdm.auto import tqdm

from dataset import TextDataset, prepare_datasets, create_dataloaders

logger = get_logger(__name__)

def calculate_num_parameters(model: nn.Module) -> int:
    """
    Calculate the number of parameters in a PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model.
        
    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen 2.5 0.5B Instruct model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen2",
        help="Model type to use. Should be one of ['qwen2', 'qwen2_moe']",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--sub_dataset_percent",
        type=float,
        default=1.0,
        help="The percentage of the dataset to use for training. 1.0 means 100%.",
    )

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Where to store the final model."
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp8", "fp16", "bf16"],
        default="bf16",
        help="Precision for training: fp8, fp16, or bf16",
    )
    parser.add_argument(
        "--fp8_backend",
        type=str,
        choices=["te", "msamp", "ao"],
        default="te",
        help="Backend for FP8 training (Transformer Engine, MS-AMP, or torch.ao)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation. Will use training batch size if not set.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers (Weights & Biases).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-finetune",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name. If None, will be automatically generated.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--drop_last", 
        action="store_true", 
        help="Whether to drop the last incomplete batch from the training dataloader."
    )
    parser.add_argument(
        "--distributed", 
        action="store_true", 
        help="Whether to use distributed training."
    )
    
    args = parser.parse_args()

    # Sanity checks for dataset arguments
    if args.dataset_name is None and (args.train_file is None or args.validation_file is None):
        raise ValueError("Need either a dataset name or a training/validation file.")
    
    return args


def main():
    args = parse_args()
    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        ShardingStrategy
    )
    from torch.distributed.fsdp.wrap import (
        enable_wrap,
        wrap,
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
    
    # Initialize the FSDP plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",  # or ShardingStrategy.SHARD_GRAD_OP for ZeRO-2
        cpu_offload=CPUOffload(offload_params=False),  # Set to True if still running out of memory
        # auto_wrap_policy=size_based_auto_wrap_policy,
        # or for transformer based models:
        auto_wrap_policy=transformer_auto_wrap_policy,
        # transformer_cls_to_wrap=(transformers.PreTrainedModel,),
    )
    
    
    # Initialize accelerator with the appropriate mixed precision settings
    if args.precision == "fp8":
        if args.fp8_backend == "te":
            fp8_kwargs = TERecipeKwargs(
                fp8_format="HYBRID",  # E4M3 during forward pass, E5M2 during backward pass
                amax_history_len=16,
                amax_compute_algo="max"
            )
        elif args.fp8_backend == "msamp":
            fp8_kwargs = MSAMPRecipeKwargs()
        else:  # args.fp8_backend == "ao"
            fp8_kwargs = AORecipeKwargs()
            
        # accelerator = Accelerator(
        #     mixed_precision=args.precision,
        #     kwargs_handlers=[fp8_kwargs, fsdp_config],
        #     log_with="wandb" if args.with_tracking else None,
        # )
        # Then incorporate into your Accelerator setup
        accelerator = Accelerator(
            mixed_precision=args.precision,
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=[fp8_kwargs] if args.precision == "fp8" else None,
            log_with="wandb" if args.with_tracking else None,
        )
    else:
        # For fp16 or bf16, use standard mixed precision
        accelerator = Accelerator(
            mixed_precision=args.precision,
            kwargs_handlers=[fsdp_config],
            log_with="wandb" if args.with_tracking else None,
        )
        # Then incorporate into your Accelerator setup
        accelerator = Accelerator(
            mixed_precision=args.precision,
            fsdp_plugin=fsdp_plugin,
            log_with="wandb" if args.with_tracking else None,
        )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Setup logging and set seed
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    set_seed(args.seed)
    
    # Initialize Weights & Biases
    if args.with_tracking and accelerator.is_main_process:
        run_name = args.wandb_name if args.wandb_name else f"qwen-{args.precision}-{args.fp8_backend if args.precision == 'fp8' else ''}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model: {args.model_name_or_path}")
    if args.model_type == "qwen2_moe":

        # clone config
        config = Qwen2MoeConfig.from_pretrained(args.model_name_or_path)
        config.num_experts = 15
        config.num_experts_per_tok = 4

        model = Qwen2MoeForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
        )
    logger.info(f"Model loaded with {calculate_num_parameters(model)/ 1_000_000_000:,}B parameters.")
    # Prepare datasets and dataloaders
    logger.info("Preparing datasets")
    train_dataset, val_dataset = prepare_datasets(args, tokenizer)
    
    train_dataloader, val_dataloader, train_sampler = create_dataloaders(
        args,
        train_dataset,
        val_dataset,
        world_size=accelerator.num_processes if args.distributed else 1,
        rank=accelerator.process_index if args.distributed else 0,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Calculate steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
    # Learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare the optimizer, model, dataloaders with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Train!
    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Precision = {args.precision}")
    if args.precision == "fp8":
        logger.info(f"  FP8 backend = {args.fp8_backend}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_val_loss = float("inf")
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
            
            # Extract the step or epoch from the checkpoint folder name
            training_difference = 0
            if "step" in path:
                step_or_epoch = path.replace("step_", "")
                try:
                    step_or_epoch = int(step_or_epoch)
                    completed_steps = step_or_epoch
                    training_difference = step_or_epoch
                except ValueError:
                    pass
            elif "epoch" in path:
                step_or_epoch = path.replace("epoch_", "")
                try:
                    epoch = int(step_or_epoch)
                    starting_epoch = epoch
                    training_difference = epoch * num_update_steps_per_epoch
                    completed_steps = training_difference
                except ValueError:
                    pass
            
            progress_bar.update(training_difference)
    
    # Training loop
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Log and save
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                # Log loss and learning rate
                train_loss += loss.detach().float()
                if step % args.gradient_accumulation_steps == 0:
                    avg_loss = train_loss.item() / (step + 1)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    if args.with_tracking and accelerator.is_main_process:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/step": completed_steps,
                        })
                
                # Save checkpoint
                if args.checkpointing_steps == "step":
                    if completed_steps % args.save_every == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                
                if completed_steps >= args.max_train_steps:
                    break
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.detach().float()
        
        val_loss = accelerator.gather(val_loss).mean().item() / len(val_dataloader)
        perplexity = math.exp(val_loss)
        
        logger.info(f"Epoch {epoch+1}: validation loss: {val_loss:.4f}, perplexity: {perplexity:.4f}")
        
        # Log validation metrics
        if args.with_tracking and accelerator.is_main_process:
            wandb.log({
                "validation/loss": val_loss,
                "validation/perplexity": perplexity,
                "epoch": epoch + 1,
            })
        
        # Save checkpoint at the end of each epoch
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch+1}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
            
            # Save best model
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(args.output_dir, "best_model"),
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
    
    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(args.output_dir, "final_model"),
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # End wandb run
    if args.with_tracking and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()