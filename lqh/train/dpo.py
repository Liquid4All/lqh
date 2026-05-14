"""On-policy DPO training loop for the subprocess.

This module is only imported inside the training subprocess — never by
the main lqh process.  All torch/transformers/trl imports happen here.

The DPO loop generates predictions on the preference dataset (defaults to
the training dataset), waits for the main process to score and assemble
preferences, then runs a DPO optimization step.  This ping-pong repeats
for ``num_iterations``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

from lqh.train.data_utils import (
    load_chatml_dataset,
    load_preferences_parquet,
    split_train_eval,
)
from lqh.train.progress import (
    wait_for_file,
    write_iter_request,
    write_progress,
    write_status,
)


class _DPOProgressCallback(TrainerCallback):
    """Forwards DPO Trainer log entries (train+eval) into progress.jsonl.

    DPO emits eval_loss + eval_rewards/{chosen,rejected,margins,accuracies}
    when ``eval_dataset`` is passed. We route the eval keys through
    write_progress so the harness/main process can read them without
    having to parse trainer.state.log_history.
    """

    _EVAL_KEYS = (
        "eval_loss",
        "eval_rewards/chosen",
        "eval_rewards/rejected",
        "eval_rewards/margins",
        "eval_rewards/accuracies",
        "eval_runtime",
    )

    def __init__(self, run_dir: Path, iteration: int) -> None:
        self.run_dir = run_dir
        self.iteration = iteration

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
        extra: dict[str, Any] = {"iteration": self.iteration, "phase": "dpo_train"}
        for key in self._EVAL_KEYS:
            if key in logs:
                extra[key] = logs[key]
        write_progress(
            self.run_dir,
            step=state.global_step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            epoch=state.epoch,
            extra=extra,
        )


def _per_example_ce(
    model: Any,
    tokenizer: Any,
    prompts: list[list[dict[str, str]]],
    responses: list[str],
    *,
    batch_size: int = 4,
    max_length: int = 2048,
) -> list[float]:
    """Cross-entropy on the response tokens for each (prompt, response) pair.

    Returns one mean-CE-per-response-token value per pair. The prompt
    tokens are masked out via -100 labels so they don't contribute to
    the loss; only the assistant-response tokens count.

    This is the SFT-style "how likely would the model have written
    this response" — the absolute generation-quality signal that DPO
    loss does NOT capture (DPO loss is the relative margin between
    chosen and rejected).
    """
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token or ""

    # Tokenize once up front so we can batch.
    examples: list[dict[str, Any]] = []
    for prompt_msgs, response in zip(prompts, responses):
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fall back to raw concat if the template rejects this shape.
            prompt_text = "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}" for m in prompt_msgs
            ) + "\n"
        full_text = prompt_text + response + eos

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # Truncate from the LEFT (drop prompt prefix) when too long, so
        # the response tail is always present and labels are meaningful.
        if len(full_ids) > max_length:
            drop = len(full_ids) - max_length
            full_ids = full_ids[drop:]
            prompt_ids = prompt_ids[max(0, len(prompt_ids) - drop):]

        prompt_len = min(len(prompt_ids), len(full_ids))
        # If the response would have zero tokens (e.g. truncated away),
        # skip with a NaN sentinel — the aggregator drops those.
        if prompt_len >= len(full_ids):
            examples.append({"input_ids": None})
        else:
            examples.append({"input_ids": full_ids, "prompt_len": prompt_len})

    results: list[float] = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            valid_idx = [j for j, e in enumerate(batch) if e["input_ids"] is not None]
            if not valid_idx:
                results.extend([float("nan")] * len(batch))
                continue

            max_len = max(len(batch[j]["input_ids"]) for j in valid_idx)
            B = len(valid_idx)
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long)
            labels = torch.full((B, max_len), -100, dtype=torch.long)
            for new_j, orig_j in enumerate(valid_idx):
                e = batch[orig_j]
                ids = e["input_ids"]
                L = len(ids)
                input_ids[new_j, :L] = torch.tensor(ids, dtype=torch.long)
                attention_mask[new_j, :L] = 1
                start = e["prompt_len"]
                # Labels only on response tokens (prompt masked with -100).
                labels[new_j, start:L] = torch.tensor(ids[start:], dtype=torch.long)

            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels_t = labels.to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous().float()
            shift_labels = labels_t[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            mask = (shift_labels != -100).float()
            denom = mask.sum(dim=-1).clamp(min=1)
            per_example = (per_tok * mask).sum(dim=-1) / denom
            ces = per_example.cpu().tolist()

            # Re-interleave with NaN for invalid examples in the batch.
            out_batch = [float("nan")] * len(batch)
            for new_j, orig_j in enumerate(valid_idx):
                out_batch[orig_j] = ces[new_j]
            results.extend(out_batch)

    return results


def _aggregate_ce(values: list[float]) -> dict[str, float]:
    """Return mean, std, p50, p90, p95 of a per-example CE list (NaNs dropped)."""
    clean = [v for v in values if v == v]  # NaN != NaN
    if not clean:
        return {}
    clean_sorted = sorted(clean)
    n = len(clean_sorted)

    def _quantile(q: float) -> float:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return clean_sorted[idx]

    import statistics as _st
    return {
        "mean": float(_st.fmean(clean)),
        "std": float(_st.pstdev(clean)) if n > 1 else 0.0,
        "p50": _quantile(0.50),
        "p90": _quantile(0.90),
        "p95": _quantile(0.95),
        "max": clean_sorted[-1],
        "n": n,
    }


class _ChosenCECallback(TrainerCallback):
    """Per-eval-step CE on chosen and rejected responses.

    Unlike DPO eval_loss (which is the relative chosen-vs-rejected
    margin), this measures the absolute probability the policy
    assigns to the chosen/rejected continuations — directly
    proportional to how likely the policy is to GENERATE that text.

    When DPO collapses (model degrades to garbage while still
    correctly ranking chosen > rejected on held-out pairs), DPO
    eval_loss drops toward 0 but CE on chosen explodes. This is the
    canonical "DPO reward hacking" failure mode and is invisible to
    the margin-based metric.

    See ``lqh/train/sweep.py`` for the empirical correlation results
    that motivate using ``eval_ce_chosen_mean`` (not DPO eval_loss) as
    the selector across hyperparameter configs.

    For each held-out pair we record:
      - eval_ce_chosen_{mean,std,p90,p95,max}
      - eval_ce_rejected_{mean,std,p90,p95,max}
      - eval_ce_abs_margin = mean(CE_rejected) - mean(CE_chosen)
        (in CE units; analogous to DPO margins but on absolute logits)
      - eval_ce_chosen_delta_ref = mean(CE_chosen) - mean(ref_CE_chosen)
        (positive = worse than reference, i.e. collapse signal)
    """

    def __init__(
        self,
        run_dir: Path,
        iter_dir: Path,
        iteration: int,
        eval_prompts: list[list[dict[str, str]]],
        eval_chosens: list[str],
        eval_rejecteds: list[str],
        tokenizer: Any,
        ref_ce_chosen: list[float],
        ref_ce_rejected: list[float],
        max_length: int = 2048,
        batch_size: int = 4,
    ) -> None:
        self.run_dir = run_dir
        self.iter_dir = iter_dir
        self.iteration = iteration
        self.eval_prompts = eval_prompts
        self.eval_chosens = eval_chosens
        self.eval_rejecteds = eval_rejecteds
        self.tokenizer = tokenizer
        self.ref_ce_chosen = ref_ce_chosen
        self.ref_ce_rejected = ref_ce_rejected
        self.max_length = max_length
        self.batch_size = batch_size
        self.history_path = iter_dir / "chosen_ce_history.jsonl"
        self.last_row: dict[str, Any] = {}

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if model is None:
            return
        try:
            ce_chosen = _per_example_ce(
                model, self.tokenizer,
                self.eval_prompts, self.eval_chosens,
                batch_size=self.batch_size, max_length=self.max_length,
            )
            ce_rejected = _per_example_ce(
                model, self.tokenizer,
                self.eval_prompts, self.eval_rejecteds,
                batch_size=self.batch_size, max_length=self.max_length,
            )
        except Exception as exc:
            print(f"  WARNING: chosen-CE eval failed at step {state.global_step}: {exc}")
            return

        ch = _aggregate_ce(ce_chosen)
        rj = _aggregate_ce(ce_rejected)
        if not ch or not rj:
            return

        row: dict[str, Any] = {
            "step": state.global_step,
            "iteration": self.iteration,
            "phase": "chosen_ce_eval",
            "eval_ce_chosen_mean": ch["mean"],
            "eval_ce_chosen_std": ch["std"],
            "eval_ce_chosen_p90": ch["p90"],
            "eval_ce_chosen_p95": ch["p95"],
            "eval_ce_chosen_max": ch["max"],
            "eval_ce_rejected_mean": rj["mean"],
            "eval_ce_rejected_p90": rj["p90"],
            "eval_ce_abs_margin": rj["mean"] - ch["mean"],
            "eval_ce_n": ch["n"],
        }
        if self.ref_ce_chosen:
            ref_clean = [v for v in self.ref_ce_chosen if v == v]
            if ref_clean:
                import statistics as _st
                ref_mean = _st.fmean(ref_clean)
                row["eval_ce_chosen_delta_ref"] = ch["mean"] - ref_mean
                row["ref_ce_chosen_mean"] = ref_mean

        self.last_row = row
        # Forward into progress.jsonl alongside the other eval metrics.
        write_progress(self.run_dir, step=state.global_step, extra=row)
        # Per-iter history file for the proxy-validation harness.
        try:
            with open(self.history_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as exc:
            print(f"  WARNING: could not append to {self.history_path}: {exc}")


def dpo_loop(run_dir: Path, config: dict[str, Any]) -> None:
    """Run on-policy DPO training.

    Called from ``lqh.train.__main__`` — this is the subprocess entry point.

    The loop:
    1. Generate predictions on preference prompts
    2. Write predictions + signal file
    3. Wait for main process to write preferences.parquet
    4. Run DPO step on preference pairs
    5. Repeat
    """
    base_model = config["base_model"]
    # The preference dataset provides prompts for on-policy generation.
    # Defaults to the training dataset when not explicitly provided.
    # Chained .get so we never index a missing key — the proxy
    # harness passes only `preference_dataset`.
    preference_dataset_path = (
        config.get("preference_dataset")
        or config.get("eval_dataset")
        or config.get("dataset")
    )
    # Optional held-out eval dataset — when set, the subprocess runs
    # inference on it after each DPO iter's checkpoint and writes
    # eval_predictions.parquet alongside the iter dir. The harness
    # scores those predictions and can write run_dir/early_abort.json
    # to ask the subprocess to stop on regression.
    held_out_eval_path = config.get("held_out_eval_dataset")
    # When set, iters whose preferences.parquet already exists on disk
    # skip the (expensive) on-policy generation step entirely. Used by
    # the proxy-validation harness, which pre-seeds preferences so it
    # can sweep training hyperparams on a frozen dataset without
    # paying for rollouts per config.
    skip_generation_if_preferences_exist = bool(
        config.get("skip_generation_if_preferences_exist", False)
    )
    num_iterations = config.get("num_iterations", 5)
    dpo_beta = config.get("dpo_beta", 0.1)
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    print(f"Loading model: {base_model}")

    # GPU info
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    dtype = torch.bfloat16 if training_cfg.get("bf16", True) else torch.float32

    # device_map="auto" is REQUIRED to put the model on GPU on recent
    # transformers/accelerate. Without it the model lands on CPU and
    # the HF Trainer no longer silently relocates it (v5 behavior
    # change). Symptom: training "runs" but at ~100× CPU speed.
    device_map = "auto" if torch.cuda.is_available() else None

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a reference model for DPO (frozen copy). Also on GPU; trl's
    # DPOTrainer expects model and ref_model to be co-located.
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
    )

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

    # Load preference dataset — prompts for on-policy generation + scoring.
    # If the caller pre-seeds preferences and skips generation, the
    # parquet may not have a "messages" column (it's the prefs file
    # itself). Defer the load to lazy-on-use in that case.
    system_prompt = config.get("system_prompt")
    max_new_tokens = training_cfg.get("max_seq_length", 2048)
    preference_convos: list[list[dict[str, str]]] = []
    if not skip_generation_if_preferences_exist:
        print(f"Loading preference dataset: {preference_dataset_path}")
        preference_convos = load_chatml_dataset(preference_dataset_path)

    # Load held-out eval prompts once (cheap; just a parquet read).
    held_out_convos: list[list[dict[str, str]]] | None = None
    if held_out_eval_path:
        print(f"Loading held-out eval dataset: {held_out_eval_path}")
        try:
            held_out_convos = load_chatml_dataset(held_out_eval_path)
            print(f"  Held-out eval has {len(held_out_convos)} samples")
        except Exception as exc:
            print(f"  WARNING: failed to load held-out eval dataset: {exc}")
            held_out_convos = None

    iterations_dir = run_dir / "iterations"

    # --- Resume logic: check for completed iterations from a previous run ---
    start_iteration = 0
    model_has_peft = False

    if iterations_dir.exists():
        last_completed = _find_last_completed_iteration(iterations_dir)
        if last_completed is not None:
            if last_completed >= num_iterations - 1:
                print(f"All {num_iterations} iterations already completed. Saving final model.")
                # Load the last checkpoint for final merge-and-save
                ckpt_dir = iterations_dir / f"iter_{last_completed:03d}" / "checkpoint"
                model = PeftModel.from_pretrained(model, str(ckpt_dir))
                model_has_peft = True
                start_iteration = num_iterations  # skip the loop entirely
            else:
                ckpt_dir = iterations_dir / f"iter_{last_completed:03d}" / "checkpoint"
                print(f"Resuming from iteration {last_completed + 1}, loading checkpoint from {ckpt_dir}")
                model = PeftModel.from_pretrained(model, str(ckpt_dir))
                model_has_peft = True
                start_iteration = last_completed + 1

    interrupted = False

    for iteration in range(start_iteration, num_iterations):
        iter_name = f"iter_{iteration:03d}"
        iter_dir = iterations_dir / iter_name
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Step 1: Generate predictions — unless caller pre-seeded
        # preferences.parquet and asked us to skip generation (used by
        # the proxy-validation harness on a frozen dataset).
        preseeded_pref_path = iter_dir / "preferences.parquet"
        if (
            skip_generation_if_preferences_exist
            and preseeded_pref_path.exists()
            and preseeded_pref_path.stat().st_size > 0
        ):
            print(
                f"Pre-seeded preferences detected at {preseeded_pref_path}; "
                f"skipping on-policy generation."
            )
            # Still write iter_request so any harness watching gets the signal.
            write_iter_request(iter_dir)
        else:
            print("Generating predictions...")
            predictions = _generate_predictions(
                model, tokenizer, preference_convos, max_new_tokens, system_prompt, run_dir
            )
            _write_predictions(predictions, iter_dir)
            write_iter_request(iter_dir)

        write_progress(
            run_dir,
            step=iteration,
            extra={"phase": "waiting_for_preferences", "iteration": iteration},
        )

        # Step 2: Wait for preferences from main process
        print("Waiting for preferences from main process...")
        try:
            preferences_path = wait_for_file(
                iter_dir / "preferences.parquet",
                poll_interval=3.0,
                timeout=7200.0,  # 2 hours max
            )
        except TimeoutError:
            print("Timeout waiting for preferences. Saving progress and exiting.")
            interrupted = True
            break

        # Step 3: Load preferences and run DPO step
        preferences = load_preferences_parquet(preferences_path)
        if not preferences:
            print("No preference pairs received — model may have converged. Stopping.")
            write_progress(
                run_dir,
                step=iteration,
                extra={"phase": "converged", "iteration": iteration},
            )
            break

        print(f"Running DPO step with {len(preferences)} preference pairs...")

        # Build per-pair tuples (prompt_msgs, chosen_str, rejected_str)
        # so we can split once and derive both the DPO-trainer wrapping
        # and the chosen-CE-callback inputs from the same shuffle.
        raw_pairs: list[dict[str, Any]] = []
        for pref in preferences:
            prompt_msgs = pref["prompt"]
            if isinstance(prompt_msgs, str):
                prompt_msgs = json.loads(prompt_msgs)
            raw_pairs.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": pref["chosen"],
                    "rejected": pref["rejected"],
                }
            )

        # Split off a held-out eval slice for in-training val_loss.
        eval_split_ratio = float(training_cfg.get("eval_split_ratio", 0.1))
        if eval_split_ratio > 0:
            train_raw, eval_raw = split_train_eval(
                raw_pairs, eval_split_ratio, seed=iteration
            )
        else:
            train_raw, eval_raw = raw_pairs, []
        print(
            f"  train pairs={len(train_raw)} eval pairs={len(eval_raw)} "
            f"(eval_split_ratio={eval_split_ratio})"
        )

        def _to_dpo(p: dict[str, Any]) -> dict[str, Any]:
            return {
                "prompt": p["prompt"],
                "chosen": [{"role": "assistant", "content": p["chosen"]}],
                "rejected": [{"role": "assistant", "content": p["rejected"]}],
            }

        dpo_dataset = Dataset.from_list([_to_dpo(p) for p in train_raw])
        eval_dataset = (
            Dataset.from_list([_to_dpo(p) for p in eval_raw]) if eval_raw else None
        )

        # Eval cadence: DPO iters are short (50–300 steps), so eval more
        # frequently than SFT. Default ~10 steps; user-overridable.
        eval_steps = int(training_cfg.get("dpo_eval_steps", training_cfg.get("eval_steps", 10)))
        has_eval = eval_dataset is not None

        # DPO training config
        dpo_kwargs: dict[str, Any] = dict(
            output_dir=str(iter_dir / "dpo_output"),
            num_train_epochs=training_cfg.get("dpo_num_epochs", 1),
            per_device_train_batch_size=training_cfg.get("per_device_batch_size", 2),
            learning_rate=training_cfg.get("learning_rate", 5e-6),
            beta=dpo_beta,
            gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
            bf16=training_cfg.get("bf16", True),
            max_length=training_cfg.get("max_seq_length", 2048),
            logging_steps=10,
            remove_unused_columns=False,
        )
        if has_eval:
            # Run eval at each eval_steps step (gives us per-step CE
            # values via _ChosenCECallback) and write checkpoints at the
            # same cadence (so we have artefacts on disk for inspection).
            #
            # IMPORTANT: load_best_model_at_end is deliberately OFF here.
            # HF Trainer would pick its best checkpoint by DPO eval_loss,
            # but that metric is BROKEN as a quality signal in DPO: lower
            # eval_loss can correspond to policy collapse (driving the
            # rejected log-prob down faster than the chosen log-prob).
            # The proxy-validation experiment on ar_to_de (2026-05-11)
            # showed eval_loss correlates with judge_mean in the WRONG
            # direction (Pearson r = +0.92). Cross-config selection by
            # the validated proxy (eval_ce_chosen_mean) happens at the
            # sweep level in lqh/train/sweep.py. Within a single iter we
            # keep the final-step weights.
            dpo_kwargs.update(
                eval_strategy="steps",
                eval_steps=eval_steps,
                per_device_eval_batch_size=training_cfg.get(
                    "per_device_eval_batch_size",
                    training_cfg.get("per_device_batch_size", 2),
                ),
                save_strategy="steps",
                save_steps=eval_steps,
                save_total_limit=2,
                # load_best_model_at_end=False (default) — see comment above.
            )
        dpo_config = DPOConfig(**dpo_kwargs)

        callbacks: list[TrainerCallback] = [_DPOProgressCallback(run_dir, iteration)]

        # If we have an eval split, precompute reference-model CE on
        # both chosen and rejected once — this is the absolute baseline
        # the policy starts from, useful for the "delta vs ref" signal
        # in the chosen-CE callback. The ref model is the same across
        # iters (frozen), so the values are stable.
        if eval_raw:
            eval_prompts_only = [p["prompt"] for p in eval_raw]
            eval_chosens_only = [p["chosen"] for p in eval_raw]
            eval_rejecteds_only = [p["rejected"] for p in eval_raw]
            print(f"  computing ref-model CE on {len(eval_raw)} held-out pairs...")
            try:
                ref_ce_chosen = _per_example_ce(
                    ref_model, tokenizer,
                    eval_prompts_only, eval_chosens_only,
                    batch_size=training_cfg.get("ce_eval_batch_size", 4),
                    max_length=training_cfg.get("max_seq_length", 2048),
                )
                ref_ce_rejected = _per_example_ce(
                    ref_model, tokenizer,
                    eval_prompts_only, eval_rejecteds_only,
                    batch_size=training_cfg.get("ce_eval_batch_size", 4),
                    max_length=training_cfg.get("max_seq_length", 2048),
                )
                ref_agg = _aggregate_ce(ref_ce_chosen)
                if ref_agg:
                    print(
                        f"  ref CE(chosen): mean={ref_agg['mean']:.4f} "
                        f"p90={ref_agg['p90']:.4f} max={ref_agg['max']:.4f}"
                    )
                callbacks.append(_ChosenCECallback(
                    run_dir=run_dir,
                    iter_dir=iter_dir,
                    iteration=iteration,
                    eval_prompts=eval_prompts_only,
                    eval_chosens=eval_chosens_only,
                    eval_rejecteds=eval_rejecteds_only,
                    tokenizer=tokenizer,
                    ref_ce_chosen=ref_ce_chosen,
                    ref_ce_rejected=ref_ce_rejected,
                    max_length=training_cfg.get("max_seq_length", 2048),
                    batch_size=training_cfg.get("ce_eval_batch_size", 4),
                ))
            except Exception as exc:
                print(f"  WARNING: ref-CE precompute failed, skipping chosen-CE eval: {exc}")

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "ref_model": ref_model,
            "args": dpo_config,
            "train_dataset": dpo_dataset,
            "processing_class": tokenizer,
            "callbacks": callbacks,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        # Only pass peft_config on the first iteration when the model doesn't
        # already have LoRA adapters (from a previous run's checkpoint).
        if peft_config is not None and iteration == 0 and not model_has_peft:
            trainer_kwargs["peft_config"] = peft_config

        trainer = DPOTrainer(**trainer_kwargs)
        train_result = trainer.train()

        # Pull final eval metrics out of the log history. Scan
        # backwards for the last eval row (one with "eval_loss").
        final_eval: dict[str, Any] = {}
        for entry in reversed(trainer.state.log_history):
            if "eval_loss" in entry:
                final_eval = {
                    k: v for k, v in entry.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                }
                break

        # Dump per-iter eval history (all train+eval log rows) so the
        # proxy-validation harness can read them without re-loading
        # progress.jsonl.
        try:
            log_history = []
            for entry in trainer.state.log_history:
                row = {k: v for k, v in entry.items() if isinstance(v, (int, float, str, bool, type(None)))}
                log_history.append(row)
            (iter_dir / "eval_history.json").write_text(
                json.dumps(log_history, indent=2) + "\n"
            )
        except Exception as exc:
            print(f"  WARNING: failed to dump eval_history.json for iter {iteration}: {exc}")

        # Pull the last chosen-CE eval row (if the callback ran).
        final_chosen_ce: dict[str, Any] = {}
        ce_cb: _ChosenCECallback | None = None
        for cb in callbacks:
            if isinstance(cb, _ChosenCECallback):
                ce_cb = cb
                if cb.last_row:
                    final_chosen_ce = cb.last_row
                break

        # If load_best_model_at_end reloaded an earlier checkpoint, the
        # callback's last row was measured against weights that have
        # since been replaced. Re-measure on the current trainer model
        # so the summary matches the saved final checkpoint.
        if ce_cb is not None and has_eval:
            try:
                ce_chosen_post = _per_example_ce(
                    trainer.model, tokenizer,
                    ce_cb.eval_prompts, ce_cb.eval_chosens,
                    batch_size=ce_cb.batch_size, max_length=ce_cb.max_length,
                )
                ce_rejected_post = _per_example_ce(
                    trainer.model, tokenizer,
                    ce_cb.eval_prompts, ce_cb.eval_rejecteds,
                    batch_size=ce_cb.batch_size, max_length=ce_cb.max_length,
                )
                ch_post = _aggregate_ce(ce_chosen_post)
                rj_post = _aggregate_ce(ce_rejected_post)
                if ch_post and rj_post:
                    post_row: dict[str, Any] = {
                        "step": trainer.state.global_step,
                        "iteration": iteration,
                        "phase": "chosen_ce_eval_final",
                        "eval_ce_chosen_mean": ch_post["mean"],
                        "eval_ce_chosen_std": ch_post["std"],
                        "eval_ce_chosen_p90": ch_post["p90"],
                        "eval_ce_chosen_p95": ch_post["p95"],
                        "eval_ce_chosen_max": ch_post["max"],
                        "eval_ce_rejected_mean": rj_post["mean"],
                        "eval_ce_rejected_p90": rj_post["p90"],
                        "eval_ce_abs_margin": rj_post["mean"] - ch_post["mean"],
                        "eval_ce_n": ch_post["n"],
                    }
                    if ce_cb.ref_ce_chosen:
                        ref_clean = [v for v in ce_cb.ref_ce_chosen if v == v]
                        if ref_clean:
                            import statistics as _st
                            ref_mean = _st.fmean(ref_clean)
                            post_row["eval_ce_chosen_delta_ref"] = ch_post["mean"] - ref_mean
                            post_row["ref_ce_chosen_mean"] = ref_mean
                    final_chosen_ce = post_row
                    print(
                        f"  final CE(chosen): mean={ch_post['mean']:.4f} "
                        f"p90={ch_post['p90']:.4f} "
                        f"Δref={post_row.get('eval_ce_chosen_delta_ref', float('nan')):+.4f}"
                    )
            except Exception as exc:
                print(f"  WARNING: post-train chosen-CE eval failed: {exc}")

        if final_chosen_ce:
            (iter_dir / "chosen_ce_summary.json").write_text(
                json.dumps(final_chosen_ce, indent=2) + "\n"
            )

        # Record DPO step metrics
        dpo_metrics = {
            "iteration": iteration,
            "num_preferences": len(preferences),
            "train_pairs": len(train_raw),
            "eval_pairs": len(eval_raw),
            "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
            "final_eval": final_eval,
            "final_chosen_ce": final_chosen_ce,
        }
        (iter_dir / "dpo_result.json").write_text(
            json.dumps(dpo_metrics, indent=2) + "\n"
        )

        # Cheap collapse detection — based on the chosen-CE delta from the
        # frozen reference model.
        #
        # An earlier version of this check used eval_rewards/margins <= 0,
        # but the proxy-validation experiment on ar_to_de (2026-05-11)
        # showed that's the wrong failure mode: at policy collapse,
        # margins are LARGE and positive (the model learns to reject
        # rejected very aggressively while ALSO making chosen less
        # likely). Margins never went negative in any observed collapse.
        #
        # The reliable signal is the absolute generation likelihood of
        # chosen, anchored to the frozen reference. Calm configs had
        # delta_ref ≈ -0.01 (slight improvement over ref). The first
        # collapsed config jumped to delta_ref = +1.14. Threshold 0.5
        # cleanly separates them.
        delta_ref = final_chosen_ce.get("eval_ce_chosen_delta_ref")
        if delta_ref is not None and delta_ref > 0.5:
            abort_payload = {
                "iteration": iteration,
                "reason": (
                    f"chosen-CE degraded {delta_ref:+.4f} nats above the "
                    f"reference model at iter {iteration} — policy is "
                    f"becoming less likely to generate the chosen response "
                    f"(threshold 0.5)."
                ),
                "eval_ce_chosen_delta_ref": delta_ref,
                "eval_ce_chosen_mean": final_chosen_ce.get("eval_ce_chosen_mean"),
                "ref_ce_chosen_mean": final_chosen_ce.get("ref_ce_chosen_mean"),
                "source": "dpo_subprocess_chosen_ce_check",
            }
            (run_dir / "early_abort.json").write_text(
                json.dumps(abort_payload, indent=2) + "\n"
            )
            print(
                f"  ⚠ chosen-CE Δref={delta_ref:+.4f} > 0.5 — likely "
                f"collapse. Will abort after this iter."
            )

        write_progress(
            run_dir,
            step=iteration,
            loss=dpo_metrics.get("train_loss"),
            extra={"phase": "dpo_complete", "iteration": iteration},
        )

        # DPOTrainer wraps the model with PEFT internally when peft_config
        # is provided. Grab the trained model back from the trainer so
        # subsequent iterations (and the final save) use the updated weights.
        model = trainer.model
        model_has_peft = True
        del trainer
        torch.cuda.empty_cache()

        # Save iteration checkpoint so we can resume if interrupted
        ckpt_dir = iter_dir / "checkpoint"
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        (iter_dir / "iter_complete.json").write_text(
            json.dumps({
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )

        print(f"DPO step complete. Loss: {dpo_metrics.get('train_loss')}")

        # Per-iter held-out eval: regenerate predictions on the eval set
        # using the just-trained model so the harness can score it and
        # decide whether to early-abort. Uses the same in-memory model
        # to avoid loading the LoRA adapter from disk again.
        if held_out_convos:
            print(f"Running held-out eval ({len(held_out_convos)} samples)...")
            try:
                eval_preds = _generate_predictions(
                    model, tokenizer, held_out_convos,
                    max_new_tokens, system_prompt, run_dir,
                )
                _write_predictions(
                    eval_preds, iter_dir,
                    filename="eval_predictions.parquet",
                )
                (iter_dir / "eval_predictions_ready.json").write_text(
                    json.dumps({
                        "iteration": iteration,
                        "samples": len(eval_preds),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }) + "\n"
                )
                print("  Held-out eval predictions written.")
            except Exception as exc:
                print(f"  WARNING: held-out eval inference failed: {exc}")

        # Check for early-abort signal from the harness. The harness
        # writes run_dir/early_abort.json if the per-iter eval shows a
        # regression past --dpo-early-abort-delta. We respect it before
        # starting the next iter's expensive prediction run.
        abort_signal = run_dir / "early_abort.json"
        if abort_signal.exists():
            try:
                payload = json.loads(abort_signal.read_text())
            except Exception:
                payload = {"reason": "unparseable"}
            print(f"Early abort signaled by harness at iter {iteration}: {payload}")
            interrupted = True
            break

    # Restore the best-scoring iter before the final save. Without
    # this, an early-abort at iter N saves iter N's LoRA, but iter
    # N-1 may have had the better held-out score (e.g. ar_to_de
    # 2026-05-05 run: iter 0 held-out 6.99, iter 1 dropped to 6.53,
    # trend-abort fired — but the saved model was iter 1's). We
    # restore the best LoRA before the final merge-and-save so the
    # checkpoint we ship matches the peak we observed.
    best_iter, best_mean = _find_best_held_out_iter(iterations_dir)
    if best_iter is not None:
        completed_iters = [
            int(d.name.split("_")[1])
            for d in iterations_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("iter_")
            and (d / "iter_complete.json").exists()
        ]
        current_iter = max(completed_iters) if completed_iters else None
        if current_iter is not None and best_iter != current_iter:
            best_ckpt = iterations_dir / f"iter_{best_iter:03d}" / "checkpoint"
            if best_ckpt.exists():
                print(
                    f"Restoring iter {best_iter} (held-out mean={best_mean:.3f}) "
                    f"for final save — current weights are from iter "
                    f"{current_iter}."
                )
                del model
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, dtype=dtype, device_map=device_map,
                )
                model = PeftModel.from_pretrained(model, str(best_ckpt))
                model_has_peft = True
            else:
                print(
                    f"Best iter {best_iter} has no checkpoint dir; "
                    f"keeping current iter {current_iter} weights."
                )
        elif current_iter is not None:
            print(
                f"Best iter is {best_iter} (current weights); no reload needed."
            )

    # Save final model
    final_model_dir = run_dir / "model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    if model_has_peft or peft_config is not None:
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(str(final_model_dir))
        except Exception:
            model.save_pretrained(str(final_model_dir))
    else:
        model.save_pretrained(str(final_model_dir))

    tokenizer.save_pretrained(str(final_model_dir))

    if interrupted:
        write_status(run_dir, "interrupted", error="Timeout waiting for preferences")
        print(f"DPO training interrupted. Partial model saved to {final_model_dir}")
    else:
        write_status(run_dir, "completed")
        print(f"DPO training completed. Model saved to {final_model_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_last_completed_iteration(iterations_dir: Path) -> int | None:
    """Scan for the highest iteration with a valid completion marker."""
    last = None
    for d in sorted(iterations_dir.iterdir()):
        marker = d / "iter_complete.json"
        ckpt = d / "checkpoint"
        if marker.exists() and ckpt.exists():
            try:
                data = json.loads(marker.read_text())
                iteration = data["iteration"]
                if last is None or iteration > last:
                    last = iteration
            except (json.JSONDecodeError, KeyError):
                continue
    return last


def _find_best_held_out_iter(
    iterations_dir: Path,
) -> tuple[int | None, float | None]:
    """Find the iter with the highest held-out mean.

    Reads ``iter_NNN/held_out_eval.json`` files (written by the harness
    after the per-iter eval) and returns ``(iter_number, mean)`` for
    the iter with the highest mean, or ``(None, None)`` if no held-out
    evals exist (e.g. eval was disabled, or run aborted before any
    iter scored). Ties resolve to the earliest iter — preferring less
    drift when the held-out scores are equal.
    """
    if not iterations_dir.exists():
        return None, None
    best_iter: int | None = None
    best_mean: float | None = None
    for d in sorted(iterations_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("iter_"):
            continue
        held_path = d / "held_out_eval.json"
        if not held_path.exists():
            continue
        try:
            mean = json.loads(held_path.read_text()).get("mean")
            iter_num = int(d.name.split("_")[1])
        except (json.JSONDecodeError, OSError, IndexError, ValueError):
            continue
        if mean is None:
            continue
        if best_mean is None or mean > best_mean:
            best_mean = float(mean)
            best_iter = iter_num
    return best_iter, best_mean


def _generate_predictions(
    model: Any,
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    max_new_tokens: int,
    system_prompt: str | None = None,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate model responses for each conversation prompt."""
    model.eval()
    predictions = []

    for i, conv in enumerate(conversations):
        # Strip trailing assistant turn
        prompt_msgs = conv
        if conv and conv[-1].get("role") == "assistant":
            prompt_msgs = conv[:-1]

        # Prepend system prompt if provided and not already present
        if system_prompt and (not prompt_msgs or prompt_msgs[0].get("role") != "system"):
            prompt_msgs = [{"role": "system", "content": system_prompt}] + prompt_msgs

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
                    max_new_tokens=min(max_new_tokens, 1024),
                    do_sample=False,
                )
            response = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
        except Exception as exc:
            response = f"[generation error: {exc}]"

        full_conv = prompt_msgs + [{"role": "assistant", "content": response}]
        predictions.append(
            {
                "sample_index": i,
                "messages": json.dumps(full_conv),
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(conversations)}")
            if run_dir is not None:
                write_progress(
                    run_dir,
                    step=i + 1,
                    extra={"phase": "generating_predictions", "total": len(conversations)},
                )

    return predictions


def _write_predictions(
    predictions: list[dict[str, Any]],
    iter_dir: Path,
    *,
    filename: str = "predictions.parquet",
) -> None:
    """Write predictions to parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "sample_index": [p["sample_index"] for p in predictions],
            "messages": [p["messages"] for p in predictions],
        }
    )
    pq.write_table(table, iter_dir / filename)
