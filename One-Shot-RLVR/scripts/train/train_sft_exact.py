import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import LoraConfig, get_peft_model, TaskType


class ExactSFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=3072, max_target_length=1024):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        print(f"[SFT] Loaded rows: {len(self.rows)}")
        print(f"[SFT] max_length={self.max_length}, max_target_length={self.max_target_length}")

    def __len__(self):
        return len(self.rows)

    def _encode(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = row["messages"]

        user_text = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=True,
        )

        assistant_text = messages[1]["content"]

        eos_token = self.tokenizer.eos_token or ""
        if eos_token and not assistant_text.endswith(eos_token):
            assistant_text = assistant_text + eos_token

        prompt_ids = self._encode(user_text)
        target_ids = self._encode(assistant_text)

        eos_id = self.tokenizer.eos_token_id

        # Keep assistant target as much as possible.
        if len(target_ids) > self.max_target_length:
            target_ids = target_ids[: self.max_target_length]
            if eos_id is not None:
                target_ids[-1] = eos_id

        # Truncate prompt first, never truncate all target.
        max_prompt_len = self.max_length - len(target_ids)
        if max_prompt_len < 1:
            target_ids = target_ids[: self.max_length - 1]
            if eos_id is not None:
                target_ids.append(eos_id)
            max_prompt_len = 1

        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        # Hard safety: ensure at least one supervised token.
        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def count_trainable_params(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--merged_output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--wandb_project", type=str, default="exact_sft_logicality")
    parser.add_argument("--wandb_name", type=str, default="exact-qwen0p5b-sft")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_name)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        os.environ.setdefault("WANDB_WATCH", "false")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # IMPORTANT:
    # Avoid device_map="auto" with Trainer + LoRA on Colab.
    # Trainer/Accelerate will place model on GPU.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    # CRITICAL FIX for:
    # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()
    model.print_trainable_parameters()

    trainable, total = count_trainable_params(model)
    print(f"[SFT] Trainable params: {trainable:,} / {total:,}")

    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found. LoRA was not attached correctly."
        )

    train_dataset = ExactSFTDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
    )

    # Quick sample sanity check
    sample = train_dataset[0]
    valid_label_count = int((sample["labels"] != -100).sum().item())
    print("[SFT] First sample input length:", len(sample["input_ids"]))
    print("[SFT] First sample supervised label count:", valid_label_count)

    if valid_label_count == 0:
        raise RuntimeError(
            "First training sample has zero supervised labels. Check dataset formatting."
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.wandb_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[] if args.no_wandb else ["wandb"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )

    # IMPORTANT with PEFT + gradient checkpointing
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved LoRA adapter:", args.output_dir)

    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()

    Path(args.merged_output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_output_dir)

    print("Saved merged SFT model:", args.merged_output_dir)


if __name__ == "__main__":
    main()
    

