#!/usr/bin/env python
"""LoRA SFT training for Qwen2.5-3B-Instruct on judge distillation data."""

from pathlib import Path
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = Path(os.getenv("SFT_TRAIN_PATH", str(PROJECT_ROOT / "data" / "sft_train.jsonl")))
VAL_PATH = Path(os.getenv("SFT_VAL_PATH", str(PROJECT_ROOT / "data" / "sft_val.jsonl"))
)
OUTPUT_DIR = Path(
    os.getenv("SFT_OUTPUT_DIR", str(PROJECT_ROOT / "artifacts" / "qwen25_3b_lora_judge"))
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 3072


def build_text(example: dict, tokenizer: AutoTokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing SFT files. train={TRAIN_PATH.exists()} ({TRAIN_PATH}), "
            f"val={VAL_PATH.exists()} ({VAL_PATH})"
        )

    use_wandb = bool(os.getenv("WANDB_PROJECT"))
    report_to = "wandb" if use_wandb else "none"
    run_name = os.getenv("WANDB_RUN_NAME", OUTPUT_DIR.name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    ds = load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    )
    train_ds = ds["train"].map(lambda ex: build_text(ex, tokenizer))
    val_ds = ds["validation"].map(lambda ex: build_text(ex, tokenizer))

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        run_name=run_name,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=report_to,
    )

    train_ds = train_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Training complete. Model saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
