"""Colab-first QLoRA fine-tuning entrypoint for Z32LITE."""

from __future__ import annotations

import argparse
import inspect
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
        "precision": "fp16",
    },
    "tool_heavy": {
        "learning_rate": 1.8e-4,
        "num_train_epochs": 3,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 8,
        "precision": "fp16",
    },
    "light_regularization": {
        "learning_rate": 2.2e-4,
        "num_train_epochs": 2,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.02,
        "gradient_accumulation_steps": 8,
        "precision": "fp16",
    },
    "colab_safe": {
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 4,
        "precision": "fp16",
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
    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16", "auto"],
        default="fp16",
        help="Precision mode for training. Default is fp16 (best for Colab T4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model + trainer only, skip trainer.train() for compatibility checks.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback (slow). By default training requires CUDA.",
    )
    return parser.parse_args()


def resolve_precision(
    requested_precision: str,
    profile: dict[str, float | int | str | bool],
    *,
    cuda_available: bool,
    bf16_supported: bool,
) -> tuple[str, str, bool, bool, bool]:
    if "bf16" in profile:
        print("[warn] Found deprecated profile key 'bf16'. It will be ignored.")

    profile_precision = str(profile.get("precision", "fp16")).lower()
    if profile_precision not in {"fp16", "bf16", "auto"}:
        print(f"[warn] Invalid profile precision={profile_precision!r}; falling back to fp16.")
        profile_precision = "fp16"

    requested = (requested_precision or profile_precision).lower()
    resolved = requested

    if requested == "auto":
        resolved = "bf16" if bf16_supported else "fp16"
    elif requested == "bf16" and not bf16_supported:
        print("[warn] bf16 requested but unsupported on this GPU. Falling back to fp16.")
        resolved = "fp16"

    if not cuda_available:
        return requested, "cpu", False, False, False

    return requested, resolved, resolved == "bf16", resolved == "fp16", bf16_supported


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    run_dir = Path(args.output_root) / args.profile
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datasets import load_dataset
    from peft import LoraConfig
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Connect to a GPU runtime in Colab, "
            "or pass --allow-cpu (not recommended due very slow training)."
        )
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    requested_precision, resolved_precision, use_bf16, use_fp16, bf16_supported = resolve_precision(
        args.precision,
        profile,
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
    )

    print(
        "precision:",
        {
            "requested_precision": requested_precision,
            "resolved_precision": resolved_precision,
            "cuda_name": cuda_name,
            "bf16_supported": bf16_supported,
        },
    )

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=profile["lora_r"],
        lora_alpha=profile["lora_alpha"],
        lora_dropout=profile["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_kwargs = {
        "output_dir": str(run_dir / "trainer"),
        "learning_rate": profile["learning_rate"],
        "num_train_epochs": profile["num_train_epochs"],
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": profile["gradient_accumulation_steps"],
        "eval_steps": args.save_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "warmup_ratio": args.warmup_ratio,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"
    else:
        raise RuntimeError(
            "Unsupported transformers TrainingArguments signature: "
            "missing evaluation strategy argument."
        )

    training_args = TrainingArguments(**training_kwargs)

    sft_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "peft_config": peft_config,
    }

    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sft_params:
        trainer_kwargs["formatting_func"] = lambda example: example["text"]

    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_params:
        trainer_kwargs["max_length"] = args.max_seq_length

    if "packing" in sft_params:
        trainer_kwargs["packing"] = False

    print("SFTTrainer signature compatibility:", sorted(sft_params))
    trainer = SFTTrainer(**trainer_kwargs)

    summary = {
        "model_id": args.model_id,
        "profile": args.profile,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "output_root": str(run_dir),
        "profile_config": profile,
        "requested_precision": requested_precision,
        "resolved_precision": resolved_precision,
        "cuda_name": cuda_name,
        "bf16_supported": bf16_supported,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✅ dry-run passed: {run_dir}")
        return 0

    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    merged_dir = run_dir / "final_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    summary["merged_model_dir"] = str(merged_dir)
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ training finished: {run_dir}")
    print(f"✅ merged model saved to: {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
