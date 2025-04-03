import json
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Dataset for language model fine-tuning"""

    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = "Bạn là một trợ lý ảo hỗ trợ tài xế trên xe"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        # conversation = [
        #     {"role": "system", "content": self.system_prompt}
        # ] + conversation
        # tools = json.loads(item["tools"])

        # text = self.tokenizer.apply_chat_template(
        #     conversation, tools=tools, add_generation_prompt=False, tokenize=False
        # )

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]

        # Shift labels for causal language modeling
        labels = input_ids.clone()
        # Set padding tokens to -100 to ignore them in the loss computation
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
        raw_datasets = load_dataset(args.dataset_name, split="train")
        if args.sub_dataset_percent > 0:
            # shuffle and select a subset of the dataset
            raw_datasets = raw_datasets.shuffle(seed=args.seed)
            # Select a subset of the dataset
            raw_datasets = raw_datasets.select(
                range(int(len(raw_datasets) * args.sub_dataset_percent))
            )
    elif args.train_file and args.validation_file:
        # Load from local files
        data_files = {"train": args.train_file, "validation": args.validation_file}
        raw_datasets = load_dataset("text", data_files=data_files)
    else:
        raise ValueError(
            "Either dataset_name or train_file and validation_file must be provided"
        )

    if "validation" not in raw_datasets:
        # Use test set as validation set
        raw_datasets = raw_datasets.shuffle().train_test_split(test_size=0.2)

    train_dataset = TextDataset(
        raw_datasets["train"], tokenizer, max_length=args.max_length
    )

    val_dataset = TextDataset(
        raw_datasets["validation" if "validation" in raw_datasets else "test"],
        tokenizer,
        max_length=args.max_length,
    )

    return train_dataset, val_dataset


def create_dataloaders(args, train_dataset, val_dataset, world_size=1, rank=0):
    """Create data loaders for training and validation"""
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.drop_last,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size or args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, train_sampler
