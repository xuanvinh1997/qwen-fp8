#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import io
import os
import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
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
    find_executable_batch_size,
)
from accelerate.logging import get_logger
from datasets import load_dataset
from tqdm.auto import tqdm

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with FP8 precision")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-350m",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=-1,
        help="Subset",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use",
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        type=int,
        help="The percentage of the train set used as validation set if validation set is not provided.",
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
        default="fp8",
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
        "--fp8_format",
        type=str,
        choices=["e4m3", "e5m2", "hybrid"],
        default="hybrid",
        help="FP8 format for Transformer Engine - e4m3, e5m2, or hybrid (e4m3 during forward, e5m2 during backward)",
    )
    parser.add_argument(
        "--amax_history_len",
        type=int,
        default=16,
        help="History length for amax in Transformer Engine's delayed scaling algorithm",
    )
    parser.add_argument(
        "--amax_compute_algo",
        type=str,
        choices=["max", "most_recent"],
        default="max",
        help="Algorithm to compute amax from history",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length to use for training",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers (Weights & Biases).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fp8-training",
        help="wandb project name to use."
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for multi-GPU loading. Options: 'auto', 'balanced', 'balanced_low_0', 'sequential'",
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Whether to use DeepSpeed ZeRO optimization.",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="ZeRO stage to use (if using DeepSpeed).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Whether to use Fully Sharded Data Parallelism (FSDP).",
    )
    parser.add_argument(
        "--fsdp_offload_params",
        action="store_true",
        help="Whether to offload parameters to CPU with FSDP.",
    )
    
    args = parser.parse_args()

    # Sanity checks for dataset arguments
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    
    return args

class TextDataset(Dataset):
    """Dataset for language model training"""

    def __init__(self, data, tokenizer, max_length=1024, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        
        # Tokenize and prepare input
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]

        # For causal language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Set padding tokens to -100 to ignore them in the loss computation
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def prepare_datasets(args, tokenizer):
    """Prepare train and validation datasets"""
    if args.dataset_name:
        # Load from Hugging Face datasets hub
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            # No validation split, create one from train data
            if args.subset != -1:
                raw_datasets['train'] = raw_datasets['train'].take(args.subset)
            
            split = raw_datasets["train"].train_test_split(
                test_size=args.validation_split_percentage / 100
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # Load from local files
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file
            
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)
        
        if "validation" not in raw_datasets.keys():
            # No validation split, create one from train data
            split = raw_datasets["train"].train_test_split(
                test_size=args.validation_split_percentage / 100
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]

    train_dataset = TextDataset(
        raw_datasets["train"], tokenizer, max_length=args.max_length, is_train=True
    )

    eval_dataset = TextDataset(
        raw_datasets["validation"], tokenizer, max_length=args.max_length, is_train=False
    )

    return train_dataset, eval_dataset

def main():
    args = parse_args()
     # Configure process group for FSDP if needed
    kwargs = {}
    if args.use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        kwargs["fsdp_plugin"] = fsdp_plugin
        kwargs["kwargs_handlers"] = [process_group_kwargs]
    
    # Initialize accelerator with the appropriate mixed precision settings
    if args.precision == "fp8":
        if args.fp8_backend == "te":
            fp8_kwargs = TERecipeKwargs(
                fp8_format=args.fp8_format,
                amax_history_len=args.amax_history_len,
                amax_compute_algo=args.amax_compute_algo
            )
        elif args.fp8_backend == "msamp":
            fp8_kwargs = MSAMPRecipeKwargs()
        else:  # args.fp8_backend == "ao"
            fp8_kwargs = AORecipeKwargs()
        # Add fp8_kwargs to the kwargs list
        kwargs.setdefault("kwargs_handlers", []).append(fp8_kwargs)
        
    # Configure DeepSpeed if requested
    if args.use_deepspeed:
        # import json
        # with open("ds_config.json", 'r') as f:
        #     ds_config = json.load(f)
        # # You can override some values from the command line if needed
        # if ds_config["zero_optimization"]["stage"] != args.zero_stage:
        #     ds_config["zero_optimization"]["stage"] = args.zero_stage
        # kwargs["deepspeed_config"] = ds_config
        from accelerate import DeepSpeedPlugin
        # kwargs["deepspeed_plugin"] = DeepSpeedPlugin()
        kwargs["deepspeed_plugin"] = DeepSpeedPlugin(
            hf_ds_config="ds_config.json"
            # zero_stage=args.zero_stage,
            # gradient_accumulation_steps=args.gradient_accumulation_steps,
            # gradient_clipping=1.0,
            # zero3_save_16bit_model=True,
            # offload_optimizer_device="cpu" if args.zero_stage >= 2 else "none",
            # offload_param_device="cpu" if args.zero_stage >= 3 else "none",
        )
    
    # Initialize accelerator with all the configured options
    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with=args.report_to if args.with_tracking else None,
        **kwargs
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize Weights & Biases
    if args.with_tracking and accelerator.is_main_process:
        run_name = f"{args.model_name_or_path.split('/')[-1]}-{args.precision}-{args.fp8_backend if args.precision == 'fp8' else ''}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Make sure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(torch.bfloat16 if args.precision == "bf16" else torch.float16)
    )
    # from utils import convert_and_setup_moe_model
    # model = convert_and_setup_moe_model(
    #     model,
    #     num_experts=12,
    #     top_k=2,
    #     router_jitter=0.01,
    #     load_balancing_weight=0.01,
    #     expert_diversity_weight=0.005,
    #     convert_layers=None,  # None = all layers
    #     shared_expert_size=None  # None = same as intermediate_size
    # )

    # Prepare datasets and dataloaders
    logger.info("Preparing datasets")
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Optimizer
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info("Using standard AdamW optimizer")
    
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
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # Log additional distributed training information
    logger.info("***** Running distributed training *****")
    if args.use_fsdp:
        logger.info("  Using Fully Sharded Data Parallelism (FSDP)")
        if args.fsdp_offload_params:
            logger.info("  With parameter offloading to CPU")
    
    if args.use_deepspeed:
        logger.info(f"  Using DeepSpeed ZeRO Stage-{args.zero_stage}")
    
    if args.device_map and not (args.use_fsdp or args.use_deepspeed):
        logger.info(f"  Using device map: {args.device_map}")
    
    if args.gradient_checkpointing:
        logger.info("  Using gradient checkpointing")
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Precision = {args.precision}")
    if args.precision == "fp8":
        logger.info(f"  FP8 backend = {args.fp8_backend}")
        if args.fp8_backend == "te":
            logger.info(f"  FP8 format = {args.fp8_format}")
            logger.info(f"  Amax history length = {args.amax_history_len}")
            logger.info(f"  Amax compute algorithm = {args.amax_compute_algo}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_eval_loss = float("inf")
    
    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        
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
                
            # Log and update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                # Log loss and learning rate
                total_train_loss += loss.detach().float()
                
                if completed_steps % 100 == 0:
                    avg_loss = total_train_loss.item() / (step + 1)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/step": completed_steps,
                    }, step=completed_steps)
                
                if completed_steps >= args.max_train_steps:
                    break
        
        # Evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
        
        eval_loss = accelerator.gather(eval_loss).mean().item()
        eval_loss = eval_loss / len(eval_dataloader)
        perplexity = math.exp(eval_loss)
        
        logger.info(f"Epoch {epoch+1}: validation loss: {eval_loss:.4f}, perplexity: {perplexity:.4f}")
        
        # Log validation metrics
        accelerator.log({
            "eval/loss": eval_loss,
            "eval/perplexity": perplexity,
            "epoch": epoch + 1,
        }, step=completed_steps)
        
        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            logger.info(f"New best model with validation loss: {best_eval_loss:.4f}")
            
            # Save best model
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = accelerator.get_state_dict(model)
                for key, value in list(state_dict.items()):
                    if isinstance(value, io.BytesIO):
                        # Either load the BytesIO into a proper tensor or remove it
                        try:
                            state_dict[key] = torch.load(value)
                        except:
                            del state_dict[key]
                # Save the model
                unwrapped_model.save_pretrained(
                    os.path.join(args.output_dir, "best_model"),
                    save_function=accelerator.save,
                    state_dict=state_dict
                )
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        for key, value in list(state_dict.items()):
            if isinstance(value, io.BytesIO):
                # Either load the BytesIO into a proper tensor or remove it
                try:
                    state_dict[key] = torch.load(value)
                except:
                    del state_dict[key]
        # Save the model
        unwrapped_model.save_pretrained(
            os.path.join(args.output_dir, "final_model"),
            save_function=accelerator.save,
            state_dict=state_dict
        )
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # End wandb run
    if args.with_tracking and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()