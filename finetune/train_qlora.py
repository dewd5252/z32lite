"""Colab-first QLoRA fine-tuning entrypoint for Z32LITE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROFILES = {
    "balanced": {
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 8,
    },
    "tool_heavy": {
        "learning_rate": 1.8e-4,
        "num_train_epochs": 3,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 8,
    },
    "light_regularization": {
        "learning_rate": 2.2e-4,
        "num_train_epochs": 2,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.02,
        "gradient_accumulation_steps": 8,
    },
}


def default_output_root() -> Path:
    drive_root = Path("/content/drive/MyDrive/z32lite_runs")
    if drive_root.exists():
        return drive_root
    return Path("/content/z32lite_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    run_dir = Path(args.output_root) / args.profile
    run_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=profile["lora_r"],
        lora_alpha=profile["lora_alpha"],
        lora_dropout=profile["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        learning_rate=profile["learning_rate"],
        num_train_epochs=profile["num_train_epochs"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=False,
        fp16=True,
        report_to="none",
        lr_scheduler_type="cosine",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    merged_dir = run_dir / "final_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    summary = {
        "model_id": args.model_id,
        "profile": args.profile,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "output_root": str(run_dir),
        "profile_config": profile,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ training finished: {run_dir}")
    print(f"✅ merged model saved to: {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
