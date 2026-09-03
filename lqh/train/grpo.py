"""GRPO training loop for the subprocess.

Only runs inside the ``grpo`` cloud image (vLLM base + TRL): this module
is imported by ``lqh.train.__main__`` on ``type == "grpo"`` and never by
the main lqh process. The grpo image is distinct because vLLM comes from
its own base; its TRL is the same release lqh_py pins (``trl>=1.12,<1.13``,
whose vllm extra bounds the base at 0.27.1 — see the backend's grpo image
build, GRPO_RUNTIME_DEPS). Code here targets the TRL >= 1.10 GRPOTrainer
API (no ``max_prompt_length``, ``warmup_steps`` not ``warmup_ratio``).

Structure mirrors ``sft_loop`` deliberately: same adapter-continuation
semantics, same checkpoint fallback, same final-eval shape
(``checkpoints/final`` + inline scoring), same lineage sidecars — so
publish, retention, resume, and the progress pipeline all work unchanged.
What is new is the on-policy middle: TRL's GRPOTrainer generates G
completions per prompt with colocated vLLM, the reward functions in
``lqh.train.reward`` judge them through the backend API, and the policy
updates on group-relative advantages. There is no DPO-style
laptop-sandbox ping-pong — rewards are computed inline, synchronously
with training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

from lqh.progress import (
    ProgressEvent,
    ProgressReporter,
    has_final_scoring,
    training_end_for,
    write_progress_event,
)
from lqh.train.data_utils import (
    chatml_to_grpo_rows,
    load_chatml_datasets_with_tools,
)
from lqh.train.defaults import (
    GRPO_BETA,
    GRPO_BETA_CONTINUATION,
    GRPO_LOSS_TYPE,
    GRPO_MAX_COMPLETION_LENGTH,
    GRPO_MAX_STEPS,
    GRPO_MIN_HEALTHY_OPTIMIZER_STEPS,
    GRPO_MIN_P,
    GRPO_MIN_P_FROM_BASE,
    GRPO_NUM_GENERATIONS,
    GRPO_REPETITION_PENALTY,
    GRPO_REPETITION_PENALTY_FROM_BASE,
    GRPO_SCALE_REWARDS,
    GRPO_TEMPERATURE,
    GRPO_TEMPERATURE_FROM_BASE,
    GRPO_TOP_P,
    fill_missing_hyperparameters,
)
from lqh.train.progress import write_progress, write_status
from lqh.train.resume import train_with_checkpoint_fallback
from lqh.train.reward import build_reward_funcs
from lqh.train.sft import (
    _persist_resolved_config,
    _run_checkpoint_eval,
    _write_checkpoint_lineage,
)

# Metrics copied from TRL's log rows into grpo_diagnostics.jsonl when
# present. Per-reward-function means arrive as ``rewards/<name>/mean``
# and are matched by prefix.
_DIAG_KEYS = (
    "reward",
    "reward_std",
    "frac_reward_zero_std",
    "kl",
    "clip_ratio",
    "entropy",
    "completions/mean_length",
    "completions/clipped_ratio",
    "loss",
    "learning_rate",
)


class _GRPOProgressCallback(TrainerCallback):
    """Progress + diagnostics: the headline event stream the TUI/backend
    read, plus a per-log diagnostics artifact (grpo_diagnostics.jsonl) in
    the spirit of DPO_PLAN.md Phase 0 — the reward/KL/length record that
    makes a flat or hacked run diagnosable after the fact."""

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        reward_engine: Any,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.reward_engine = reward_engine
        self.training_end = training_end_for(config)
        self.reporter = ProgressReporter(
            task_kind="grpo", label=run_dir.name, run_dir=run_dir,
        )
        self._warned_all_clipped = False

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
        max_steps = getattr(state, "max_steps", None)
        reward = logs.get("reward")
        # A step where EVERY completion hit max_completion_length is a
        # zero-gradient step: mask_truncated_completions (correctly)
        # masks them all, so the run trains on nothing while looking
        # alive. Observed on the first e2e run (translation outputs
        # longer than the configured cap). Loud, once.
        clipped = logs.get("completions/clipped_ratio")
        if (
            not self._warned_all_clipped
            and isinstance(clipped, (int, float)) and clipped >= 0.999
        ):
            self._warned_all_clipped = True
            print(
                "  WARNING: every completion this step hit "
                "max_completion_length — truncated completions are masked "
                "from the loss, so fully-clipped steps apply NO gradient. "
                "If completions/clipped_ratio stays at 1.0, raise "
                "grpo.max_completion_length above the task's typical "
                "output length."
            )
        write_progress(
            self.run_dir,
            step=state.global_step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            epoch=state.epoch,
            extra={
                k: logs[k] for k in ("reward", "reward_std", "kl") if k in logs
            } or None,
            emit_cloud=False,  # the reporter below is the cloud headline
        )
        diag = {"step": state.global_step}
        for key in _DIAG_KEYS:
            if key in logs and isinstance(logs[key], (int, float)):
                diag[key.replace("/", "_")] = logs[key]
        for key, value in logs.items():
            if key.startswith("rewards/") and isinstance(value, (int, float)):
                diag[key.replace("/", "_")] = value
        diag.update({f"judge_{k}": v for k, v in self.reward_engine.stats.items()})
        try:
            with (self.run_dir / "grpo_diagnostics.jsonl").open("a") as fh:
                fh.write(json.dumps(diag) + "\n")
        except OSError:
            pass  # diagnostics must never take down training

        if isinstance(max_steps, int) and max_steps > 0:
            detail = None
            if isinstance(reward, (int, float)):
                detail = f"reward {reward:.3f}"
            self.reporter.update(
                phase="training",
                phase_label="training GRPO",
                completed=state.global_step,
                total=max_steps,
                unit="steps",
                overall_fraction=self.training_end * state.global_step / max_steps,
                step=state.global_step,
                loss=(
                    float(logs["loss"])
                    if isinstance(logs.get("loss"), (int, float)) else None
                ),
                lr=(
                    float(logs["learning_rate"])
                    if isinstance(logs.get("learning_rate"), (int, float))
                    else None
                ),
                detail=detail,
                force=state.global_step >= max_steps,
            )


def _filter_overlong_prompts(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
) -> list[dict[str, Any]]:
    """Drop prompts that exceed the prompt budget.

    We filter ourselves rather than relying on the trainer:
    ``max_prompt_length`` does not truncate conversational datasets under
    vLLM (trl#4358), and truncating ChatML mid-message would corrupt the
    template anyway. Dropping is loud and honest.
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        try:
            ids = tokenizer.apply_chat_template(
                row["prompt"], add_generation_prompt=True,
            )
            length = len(ids["input_ids"] if isinstance(ids, dict) else ids)
        except Exception:  # noqa: BLE001 — a template failure means "too weird to train on"
            dropped += 1
            continue
        if length <= max_prompt_length:
            kept.append(row)
        else:
            dropped += 1
    if dropped:
        print(
            f"  dropped {dropped}/{len(rows)} prompts over the "
            f"{max_prompt_length}-token prompt budget (trl#4358: the trainer "
            "does not truncate conversational prompts under vLLM)"
        )
    return kept


def grpo_loop(run_dir: Path, config: dict[str, Any]) -> None:
    """Run GRPO on-policy training.

    Called from ``lqh.train.__main__`` — this is the subprocess entry point.
    """
    base_model = config["base_model"]
    dataset_path = config["dataset"]
    # setdefault, not get — same reasoning as sft_loop: resolved values
    # written into a detached dict would be invisible to lineage/status.
    training_cfg = config.setdefault("training", {})
    grpo_cfg = config.setdefault("grpo", {})
    lora_cfg = config.get("lora", {})

    if config.get("modality") == "vision":
        raise ValueError(
            "GRPO v1 is text-only — vision rollouts and rewards are a "
            "separate project (use type='sft' for VLM bases)"
        )
    config["modality"] = "text"

    from lqh.train.load_model import detect_kind, display_model_ref, load_for_training

    print(f"Loading model: {display_model_ref(base_model, run_dir)} (GRPO)")
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    if num_gpus > 1:
        # trl#3671: PEFT + vLLM colocate hangs multi-GPU. Cloud planning
        # only leases one GPU for train_grpo; this guards hand-run configs.
        print(
            "  WARNING: GRPO v1 is validated single-GPU only (PEFT + vLLM "
            "colocate is unstable multi-GPU); using device 0"
        )

    dtype = torch.bfloat16 if training_cfg.get("bf16", True) else torch.float32

    # Adapter continuation — identical contract to sft_loop/dpo_loop:
    # continuing the SFT adapter in place keeps the artifact adapter-only
    # and deployable.
    lora_enabled = lora_cfg.get("enabled", True)
    base_model_kind = detect_kind(base_model)
    continuing_adapter = bool(
        lora_enabled
        and base_model_kind == "adapter"
        and lora_cfg.get("continue_existing_adapter", True)
    )
    fresh_adapter_on_merged_parent = bool(
        lora_enabled and base_model_kind == "adapter" and not continuing_adapter
    )
    model, tokenizer, _effective_base = load_for_training(
        base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        merge_before_attach=not continuing_adapter,
        adapter_trainable=continuing_adapter,
        modality="text",
    )
    if continuing_adapter:
        print(
            "Continuing the existing LoRA adapter in place "
            "(adapter-only output remains deployable)."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if lora_enabled and not continuing_adapter:
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.02),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "in_proj", "out_proj", "w1", "w2", "w3"],
            ),
            task_type="CAUSAL_LM",
        )

    # --- GRPO knobs, resolved into the config for lineage/status ----------
    _filled = fill_missing_hyperparameters(
        training_cfg, run_type="grpo", lora=lora_enabled,
    )
    # Sampling profile by starting point (measured — see the constants'
    # provenance comment in defaults.py): a fresh policy gets full
    # exploration; continuing a converged SFT adapter keeps the
    # conservative LFM profile, where hot sampling measured negative.
    # A merged-SFT model passed as base gets the exploration profile too
    # (indistinguishable from a raw base here) — set grpo.temperature
    # explicitly for that case.
    exploration = not continuing_adapter
    grpo_defaults = {
        "num_generations": GRPO_NUM_GENERATIONS,
        "max_steps": GRPO_MAX_STEPS,
        "max_completion_length": GRPO_MAX_COMPLETION_LENGTH,
        "temperature": (
            GRPO_TEMPERATURE_FROM_BASE if exploration else GRPO_TEMPERATURE
        ),
        "top_p": GRPO_TOP_P,
        "min_p": GRPO_MIN_P_FROM_BASE if exploration else GRPO_MIN_P,
        "repetition_penalty": (
            GRPO_REPETITION_PENALTY_FROM_BASE if exploration
            else GRPO_REPETITION_PENALTY
        ),
        # KL leash only when the adapter-disabled reference IS the
        # starting policy (from-base). Continuing an SFT adapter, TRL's
        # PEFT reference is the raw base — the wrong anchor (defaults.py
        # provenance comment on GRPO_BETA_CONTINUATION).
        "beta": GRPO_BETA if exploration else GRPO_BETA_CONTINUATION,
        "loss_type": GRPO_LOSS_TYPE,
        "scale_rewards": GRPO_SCALE_REWARDS,
        # TRL calls this num_iterations (the paper's mu). DPO's
        # num_iterations means outer rollout cycles — deliberately NOT
        # reusing that name (see GRPO_CODEX.md §Dispatch).
        "policy_updates_per_rollout": 1,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.3,
    }
    for key, value in grpo_defaults.items():
        if grpo_cfg.get(key) is None:
            grpo_cfg[key] = value
            _filled[f"grpo.{key}"] = value
    # Training-judge default (MEASURED — grpo_value exploration probes,
    # RESULTS.md 2026-08-20): post-SFT, judge:small's reward signal is
    # exhausted (it filtered the SFT data; 3-seed null) while
    # judge:medium passes the value gate 3/3 seeds (+0.52..+0.68 under
    # the independent judge). From-base the reverse holds: medium adds
    # nothing over small on the independent judge (3-seed null) at
    # higher cost and latency.
    reward_cfg = config.setdefault("reward", {})
    if not (
        reward_cfg.get("judge_size")
        or grpo_cfg.get("judge_size")
        or config.get("judge_size")
    ):
        reward_cfg["judge_size"] = "medium" if continuing_adapter else "small"
        _filled["reward.judge_size"] = reward_cfg["judge_size"]
    if _filled:
        print(
            "  resolved missing hyperparameters from lqh.train.defaults: "
            + ", ".join(f"{k}={v}" for k, v in _filled.items())
        )
        _persist_resolved_config(run_dir, config)

    num_generations = int(grpo_cfg["num_generations"])
    max_steps = int(grpo_cfg["max_steps"])
    micro_batch = int(training_cfg.get("per_device_batch_size", 8))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 8))
    completions_per_step = micro_batch * grad_accum
    if num_generations < 2:
        raise ValueError("grpo.num_generations must be >= 2 (group statistic)")
    if completions_per_step % num_generations:
        raise ValueError(
            f"per_device_batch_size × gradient_accumulation_steps "
            f"({completions_per_step}) must be divisible by num_generations "
            f"({num_generations}) — each optimizer step must hold whole groups"
        )
    prompts_per_step = completions_per_step // num_generations
    print(
        f"  GRPO batch: {completions_per_step} completions/step = "
        f"{prompts_per_step} prompt groups × G={num_generations}; "
        f"{max_steps} optimizer steps planned"
    )
    if max_steps < GRPO_MIN_HEALTHY_OPTIMIZER_STEPS:
        # The DPO post-mortem's headline bug was 10 optimizer steps for a
        # whole run. Warn, don't refuse — smoke tests legitimately run short.
        print(
            f"  WARNING: only {max_steps} optimizer steps planned (healthy is "
            f">= {GRPO_MIN_HEALTHY_OPTIMIZER_STEPS}). A flat result after this "
            "is far more likely to mean too few updates than a bad reward."
        )

    # --- dataset ----------------------------------------------------------
    print(f"Loading dataset: {dataset_path}")
    conversations, tools_per_sample = load_chatml_datasets_with_tools(dataset_path)
    if any(tools_per_sample):
        # `has_tools` reaches the reward guard, but the tool *definitions*
        # never reach the prompt: trl's GRPOTrainer(tools=...) takes
        # executable callables shared by the whole run, not the per-sample
        # schemas a dataset carries, so there is nowhere to put them. The
        # policy would be graded on calling tools it was never shown.
        print(
            "  WARNING: this dataset carries tool definitions, but GRPO does not "
            "render them into the prompt — the policy generates tool calls "
            "without seeing the tool list. Train tool use with SFT for now."
        )
    rows = chatml_to_grpo_rows(conversations, tools_per_sample)
    max_seq_length = int(training_cfg.get("max_seq_length", 2048))
    max_completion_length = int(grpo_cfg["max_completion_length"])
    max_prompt_length = max(256, max_seq_length - max_completion_length)
    rows = _filter_overlong_prompts(rows, tokenizer, max_prompt_length)
    if not rows:
        raise ValueError(
            "no usable GRPO prompts after stripping assistant turns and "
            "filtering over-long prompts"
        )
    unique_prompts = len({row["sample_id"] for row in rows})
    print(
        f"  prompts: {len(rows)} rows, {unique_prompts} unique "
        f"(duplicates re-judge the same group and add cost, not signal)"
    )
    train_dataset = Dataset.from_list(rows)

    # --- rewards ----------------------------------------------------------
    reward_funcs, reward_weights, reward_engine = build_reward_funcs(
        run_dir, config, num_generations=num_generations,
    )

    # --- trainer ----------------------------------------------------------
    checkpoint_output = str(run_dir / "checkpoints")
    default_logging_steps = max(1, min(10, math.ceil(max_steps / 50)))
    save_steps = int(grpo_cfg.get("save_steps", max(10, max_steps // 10)))
    grpo_args = GRPOConfig(
        output_dir=checkpoint_output,
        max_steps=max_steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["learning_rate"],
        # warmup_steps, not warmup_ratio: transformers 5.x removed the
        # ratio argument (sft.py makes the same conversion). With max_steps
        # always set, steps express the same 3% schedule.
        warmup_steps=int(
            training_cfg.get("warmup_steps", max(1, round(max_steps * 0.03)))
        ),
        logging_steps=training_cfg.get("logging_steps", default_logging_steps),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        bf16=training_cfg.get("bf16", True),
        seed=training_cfg.get("seed", 42),
        # GRPO core
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        # No max_prompt_length: TRL 1.10 removed it from GRPOConfig —
        # prompt truncation is the caller's job (_filter_overlong_prompts
        # above enforces the budget before the trainer sees the data).
        beta=float(grpo_cfg["beta"]),
        loss_type=str(grpo_cfg["loss_type"]),
        scale_rewards=grpo_cfg["scale_rewards"],
        num_iterations=int(grpo_cfg["policy_updates_per_rollout"]),
        # A truncated ramble must neither train nor score.
        mask_truncated_completions=True,
        reward_weights=reward_weights,
        # Sampling: defaults follow the LFM low-temperature discipline
        # (MODELS.md); all four knobs are configurable so production-RL
        # sampling (T=1.0, top_p=1.0, no min_p/rep-penalty) is expressible
        # (grpo_value exploration study, 2026-08-17).
        temperature=float(grpo_cfg["temperature"]),
        top_p=float(grpo_cfg["top_p"]),
        min_p=float(grpo_cfg["min_p"]),
        repetition_penalty=float(grpo_cfg["repetition_penalty"]),
        # Rollout engine: colocated vLLM on the same GPU, weights slept
        # during the optimizer step.
        use_vllm=bool(grpo_cfg.get("use_vllm", True)),
        vllm_mode=str(grpo_cfg["vllm_mode"]),
        vllm_gpu_memory_utilization=float(grpo_cfg["vllm_gpu_memory_utilization"]),
        vllm_enable_sleep_mode=bool(grpo_cfg.get("vllm_enable_sleep_mode", True)),
        save_steps=save_steps,
        save_total_limit=2,
        report_to=[],
    )

    progress_cb = _GRPOProgressCallback(run_dir, config, reward_engine)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": grpo_args,
        "reward_funcs": reward_funcs,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "callbacks": [progress_cb],
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = GRPOTrainer(**trainer_kwargs)

    print("Starting GRPO training...")
    train_with_checkpoint_fallback(trainer, run_dir / "checkpoints", label="grpo")

    judge_stats = dict(reward_engine.stats)
    print(f"  judge usage: {judge_stats}")

    # Full log history for post-hoc analysis (same artifact SFT writes).
    try:
        log_history = [
            {k: v for k, v in entry.items()
             if isinstance(v, (int, float, str, bool, type(None)))}
            for entry in trainer.state.log_history
        ]
        (run_dir / "eval_history.json").write_text(
            json.dumps(log_history, indent=2) + "\n"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: failed to dump eval_history.json: {exc}")

    # --- save final model (same adapter/merge contract as sft_loop) --------
    merge_lora = bool(lora_cfg.get("merge", False) or fresh_adapter_on_merged_parent)
    has_lora_model = peft_config is not None or continuing_adapter
    saving_adapter = has_lora_model and not merge_lora
    final_dir_name = "model-lora" if saving_adapter else "model"
    final_model_dir = run_dir / final_dir_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    if has_lora_model and merge_lora:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(final_model_dir))
    else:
        trainer.model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    _write_checkpoint_lineage(
        final_model_dir,
        config=config,
        training_method="lora" if saving_adapter else "full",
    )
    print(f"Model saved to {display_model_ref(final_model_dir, run_dir)}")

    # --- final held-out eval (SFT's exact shape: checkpoints/final +
    # inline judge scoring in cloud mode) ----------------------------------
    if config.get("eval_on_checkpoints") and config.get("eval_dataset"):
        final_checkpoint = run_dir / "checkpoints" / "final"
        final_checkpoint.mkdir(parents=True, exist_ok=True)
        del trainer
        torch.cuda.empty_cache()

        from lqh.train.load_model import load_for_inference

        eval_model, _ = load_for_inference(
            str(final_model_dir), dtype=dtype, device_map="auto",
            modality="text",
            # A run whose adapter came out all-zero still has to publish
            # its checkpoint: here that means degenerate training, not a
            # failed load, and it is the eval's job to report it.
            verify_adapter=False,
        )
        _run_checkpoint_eval(
            model=eval_model,
            tokenizer=tokenizer,
            config=config,
            checkpoint_dir=final_checkpoint,
        )
        del eval_model
        torch.cuda.empty_cache()

    if not has_final_scoring(config):
        write_progress_event(
            run_dir,
            ProgressEvent(
                task_kind="grpo", label=run_dir.name,
                phase="completed", phase_label="training complete",
                completed=1, total=1, unit="run", overall_fraction=1.0,
                result_ready=True,
            ),
        )
    write_status(run_dir, "completed")
    print("GRPO training completed.")
