"""SFT training loop for the subprocess.

This module is only imported inside the training subprocess — never by
the main lqh process.  All torch/transformers/trl imports happen here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from lqh.train.data_utils import (
    chatml_to_sft_dataset,
    load_chatml_dataset,
    split_train_eval,
)
from lqh.train.progress import write_eval_request, write_progress, write_status


# ---------------------------------------------------------------------------
# Callback: writes progress.jsonl and handles checkpoint eval
# ---------------------------------------------------------------------------


class ProgressCallback(TrainerCallback):
    """HF Trainer callback that writes structured progress to the filesystem."""

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        tokenizer: Any,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.tokenizer = tokenizer

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        # HF Trainer emits eval logs (with "eval_loss") on a separate
        # on_log call from training logs. Forward whichever metric is
        # present so progress.jsonl carries both train and eval signal.
        extra: dict[str, Any] = {}
        if "eval_loss" in logs:
            extra["eval_loss"] = logs["eval_loss"]
        if "eval_runtime" in logs:
            extra["eval_runtime"] = logs["eval_runtime"]
        write_progress(
            self.run_dir,
            step=state.global_step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            epoch=state.epoch,
            extra=extra or None,
        )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self.config.get("eval_on_checkpoints", False):
            return
        if not self.config.get("eval_dataset"):
            return

        checkpoint_dir = self.run_dir / "checkpoints" / f"step_{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model = kwargs.get("model")
        if model is None:
            return

        # Run inference on eval dataset and write predictions
        _run_checkpoint_eval(
            model=model,
            tokenizer=self.tokenizer,
            config=self.config,
            checkpoint_dir=checkpoint_dir,
        )


# ---------------------------------------------------------------------------
# Checkpoint evaluation (inline inference, no subprocess)
# ---------------------------------------------------------------------------


def _run_checkpoint_eval(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    """Generate predictions on the eval dataset and signal for scoring."""
    eval_dataset_path = config.get("eval_dataset")
    if not eval_dataset_path:
        return

    eval_convos = load_chatml_dataset(eval_dataset_path)
    max_seq = config.get("training", {}).get("max_seq_length", 2048)

    predictions: list[dict[str, Any]] = []
    model.eval()

    for i, conv in enumerate(eval_convos):
        # Strip trailing assistant turn if present (we want the model to generate)
        prompt_msgs = conv
        if conv and conv[-1].get("role") == "assistant":
            prompt_msgs = conv[:-1]

        try:
            inputs = tokenizer.apply_chat_template(
                prompt_msgs,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=min(max_seq, 1024),
                    do_sample=False,
                )
            response = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
        except Exception as exc:
            response = f"[generation error: {exc}]"

        # Store the full conversation including model response
        full_conv = prompt_msgs + [{"role": "assistant", "content": response}]
        predictions.append(
            {
                "sample_index": i,
                "messages": json.dumps(full_conv),
            }
        )

    # Write predictions as parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "sample_index": [p["sample_index"] for p in predictions],
            "messages": [p["messages"] for p in predictions],
        }
    )
    pq.write_table(table, checkpoint_dir / "predictions.parquet")

    # Signal the main process to score
    write_eval_request(checkpoint_dir)


# ---------------------------------------------------------------------------
# Main SFT loop
# ---------------------------------------------------------------------------


def sft_loop(run_dir: Path, config: dict[str, Any]) -> None:
    """Run supervised fine-tuning.

    Called from ``lqh.train.__main__`` — this is the subprocess entry point.
    """
    base_model = config["base_model"]
    dataset_path = config["dataset"]
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    print(f"Loading model: {base_model}")

    # GPU info
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Determine dtype
    dtype = torch.bfloat16 if training_cfg.get("bf16", True) else torch.float32

    # device_map="auto" is REQUIRED on recent transformers/accelerate to
    # put the model on GPU. Without it the model lands on CPU and the HF
    # Trainer no longer silently relocates it (v5 behavior change).
    # Symptom: training "runs" but at ~100× CPU speed.
    device_map = "auto" if torch.cuda.is_available() else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.02),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"],
            ),
            task_type="CAUSAL_LM",
        )

    # Load dataset and split off a held-out eval slice for in-training
    # validation loss. eval_split_ratio=0 disables; default 0.1.
    print(f"Loading dataset: {dataset_path}")
    conversations = load_chatml_dataset(dataset_path)
    eval_split_ratio = float(training_cfg.get("eval_split_ratio", 0.1))
    if eval_split_ratio > 0:
        train_convos, eval_convos = split_train_eval(
            conversations, eval_split_ratio, seed=0
        )
    else:
        train_convos, eval_convos = conversations, []
    print(
        f"  train={len(train_convos)} eval={len(eval_convos)} "
        f"(eval_split_ratio={eval_split_ratio})"
    )

    train_dataset = Dataset.from_list(chatml_to_sft_dataset(train_convos))
    eval_dataset: Dataset | None = None
    if eval_convos:
        eval_dataset = Dataset.from_list(chatml_to_sft_dataset(eval_convos))

    # Checkpoints dir
    checkpoint_output = str(run_dir / "checkpoints")

    # Eval cadence is shared with save cadence so load_best_model_at_end
    # has matching checkpoints to choose from. When eval_dataset is
    # absent we fall back to the pre-existing behaviour (save_steps
    # only, no eval).
    eval_steps = int(training_cfg.get("eval_steps", 50))
    has_eval = eval_dataset is not None
    sft_kwargs: dict[str, Any] = dict(
        output_dir=checkpoint_output,
        num_train_epochs=training_cfg.get("num_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        logging_steps=training_cfg.get("logging_steps", 50),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        bf16=training_cfg.get("bf16", True),
        max_length=training_cfg.get("max_seq_length", 2048),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )
    if has_eval:
        # load_best_model_at_end with metric=eval_loss is SAFE for SFT
        # (unlike DPO, where we disabled it — see lqh/train/dpo.py).
        # SFT cross-entropy directly measures the absolute probability
        # the policy assigns to the gold continuation; there is no
        # hackable chosen-vs-rejected ratio. The proxy-validation run
        # on ar_to_de (2026-05-11) confirmed Pearson r = -0.90 between
        # eval_loss and judge_mean, with top-1 picked correctly.
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            per_device_eval_batch_size=training_cfg.get(
                "per_device_eval_batch_size",
                training_cfg.get("per_device_batch_size", 4),
            ),
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        sft_kwargs["save_steps"] = training_cfg.get("save_steps", 500)
    sft_config = SFTConfig(**sft_kwargs)

    # Progress callback
    progress_cb = ProgressCallback(run_dir, config, tokenizer)

    # Trainer
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "callbacks": [progress_cb],
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    # Dump the full log history (one entry per logging step, including
    # eval rows) for downstream correlation analysis. Filter to
    # JSON-serialisable scalars only — log_history sometimes carries
    # tensors when callbacks misbehave.
    try:
        log_history = []
        for entry in trainer.state.log_history:
            row = {k: v for k, v in entry.items() if isinstance(v, (int, float, str, bool, type(None)))}
            log_history.append(row)
        (run_dir / "eval_history.json").write_text(
            json.dumps(log_history, indent=2) + "\n"
        )
    except Exception as exc:
        print(f"  WARNING: failed to dump eval_history.json: {exc}")

    # Save final model
    final_model_dir = run_dir / "model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    if peft_config is not None:
        # Merge LoRA adapters into base model
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(final_model_dir))
    else:
        trainer.model.save_pretrained(str(final_model_dir))

    tokenizer.save_pretrained(str(final_model_dir))

    print(f"Model saved to {final_model_dir}")

    # Final eval if requested
    if config.get("eval_on_checkpoints") and config.get("eval_dataset"):
        final_checkpoint = run_dir / "checkpoints" / "final"
        final_checkpoint.mkdir(parents=True, exist_ok=True)

        # Reload for clean inference
        del trainer
        torch.cuda.empty_cache()

        eval_model = AutoModelForCausalLM.from_pretrained(
            str(final_model_dir),
            dtype=dtype,
            device_map="auto",
        )
        _run_checkpoint_eval(
            model=eval_model,
            tokenizer=tokenizer,
            config=config,
            checkpoint_dir=final_checkpoint,
        )
        del eval_model
        torch.cuda.empty_cache()

    write_status(run_dir, "completed")
    print("Training completed.")
