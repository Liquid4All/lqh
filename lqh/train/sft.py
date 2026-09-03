"""SFT training loop for the subprocess.

This module is only imported inside the training subprocess — never by
the main lqh process.  All torch/transformers/trl imports happen here.
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
from trl import SFTConfig, SFTTrainer

from lqh.progress import (
    FINAL_INFERENCE_END,
    ProgressEvent,
    ProgressReporter,
    TRAINING_END,
    checkpoint_eval_band,
    has_final_inference,
    has_final_scoring,
    training_end_for,
    write_progress_event,
)

from lqh.train.assistant_mask import (
    drop_rows_without_assistant_labels,
    require_assistant_mask_support,
)
from lqh.train.data_utils import (
    chatml_to_sft_dataset,
    load_chatml_datasets_with_tools,
    load_eval_sources,
    split_train_eval,
)
from lqh.train.deadline import DeadlineStopCallback, past_deadline
from lqh.train.progress import write_eval_request, write_progress, write_status
from lqh.train.resume import train_with_checkpoint_fallback


# ---------------------------------------------------------------------------
# Lineage sidecars
# ---------------------------------------------------------------------------


def _write_checkpoint_lineage(
    model_dir: Path,
    *,
    config: dict[str, Any],
    training_method: str,
    stopped_at_step: int | None = None,
) -> None:
    """Write metadata consumed by lqh.remote.publish for checkpoint artifacts."""
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    # Batch fields are recorded post-calibration: the probe can lower the
    # micro-batch and _apply rewrites the effective target to what it realized,
    # and config.json still carries the *requested* values. Without these, what
    # a checkpoint actually trained at is not recoverable from any artifact —
    # and the optimizer-step count (hence "did this run get enough updates?")
    # follows from them.
    hyperparams: dict[str, Any] = {
        "learning_rate": training_cfg.get("learning_rate"),
        "num_epochs": training_cfg.get("num_epochs"),
        "max_seq_length": training_cfg.get("max_seq_length"),
        "per_device_batch_size": training_cfg.get("per_device_batch_size"),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
        "effective_batch_size": training_cfg.get("effective_batch_size"),
        # Not a knob: text SFT always trains on assistant tokens only
        # (lqh.train.assistant_mask). Recorded so checkpoints from before the
        # switch, whose lineage lacks the key, stay distinguishable.
        "assistant_only_loss": config.get("modality", "text") != "vision",
    }
    if lora_cfg.get("enabled", True):
        hyperparams.update(
            {
                "lora_r": lora_cfg.get("r", 32),
                "lora_alpha": lora_cfg.get("alpha", 64),
                "lora_dropout": lora_cfg.get("dropout", 0.02),
                "lora_base": config.get("base_model"),
            }
        )
    lineage = {
        "artifact_kind": "checkpoint",
        "training_method": training_method,
        "base_model": config.get("base_model"),
        "hyperparams": hyperparams,
        "parent_ids": [],
    }
    # Spec provenance (R6): submission injects the spec hash into the
    # config so every published checkpoint records which spec revision
    # it was trained against — including SSH-direct publications with
    # no cloud job to inherit from.
    if config.get("spec_sha256"):
        lineage["spec_sha256"] = config["spec_sha256"]
    # A checkpoint from a deadline-truncated run is usable but did not
    # train the schedule its hyperparams describe. Without this flag no
    # artifact distinguishes it from a completed run, and the LR schedule
    # never reached its end — which is exactly what a weak eval score
    # would otherwise be blamed on the data for.
    if stopped_at_step is not None:
        lineage["stopped_early"] = True
        lineage["stopped_at_step"] = stopped_at_step
    (model_dir / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")


def _persist_resolved_config(run_dir: Path, config: dict[str, Any]) -> None:
    """Rewrite ``run_dir/config.json`` after filling in resolved defaults.

    Only called when something was actually resolved, so a config that already
    carried every value is left untouched byte-for-byte.

    Why it has to reach disk: ``training_status``'s Training-health line reads
    the on-disk config for the learning rate (``handlers._run_config_training_
    block``) — the subprocess's in-memory dict is invisible to it. A run that
    resolved its rate to 1e-4 would otherwise train correctly and still show no
    LR on the one line whose job is to explain a flat run. The published
    config.json and any resume of this run pick the values up for the same
    reason.

    Best-effort: a run must never die because its config could not be rewritten.
    """
    target = run_dir / "config.json"
    if not target.exists():
        # Unusual layout (the config was loaded from another name/place). Don't
        # invent an artifact readers would then trust.
        return
    try:
        target.write_text(json.dumps(config, indent=2) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        print(f"  WARNING: could not persist resolved config: {exc}")


def _resolve_text_seq_length(
    run_dir: Path,
    config: dict[str, Any],
    training_cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact data-derived sequence length (lqh.train.seq_length) plus the
    run-directory bookkeeping that belongs to the trainer: the dropped-row
    counts land in ``dataset_rows`` and the resolved config reaches disk so
    training_status, lineage, publish and a resume all report the length this
    run actually trained at."""
    from lqh.train.seq_length import resolve_text_seq_length

    model_cfg = getattr(model, "config", None)
    max_positions = getattr(model_cfg, "max_position_embeddings", None)
    res = resolve_text_seq_length(
        training_cfg, tokenizer, train_rows, eval_rows,
        model_max_positions=int(max_positions) if max_positions else None,
    )
    if res.changed:
        rows = config.setdefault("dataset_rows", {})
        if res.dropped_train or res.dropped_eval:
            rows["train_dropped_too_long"] = res.dropped_train
            rows["eval_dropped_too_long"] = res.dropped_eval
        _persist_resolved_config(run_dir, config)
    return res.train_rows, res.eval_rows


def _drop_rows_without_assistant_labels(
    run_dir: Path,
    config: dict[str, Any],
    training_cfg: dict[str, Any],
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A row whose assistant turn sits entirely past ``max_seq_length`` would
    train on nothing (every label masked): drop it and count it in
    ``dataset_rows``, next to the too-long counts. A no-op when the length was
    derived from the data (over-long rows are already gone); it matters for a
    pinned ``max_seq_length``. A row with no assistant tokens at all (no
    assistant turn, or a template that does not mark this row's shape) raises
    instead: that is a data or template problem, not a length one."""
    max_length = training_cfg.get("max_seq_length")
    train_rows, dropped_train = drop_rows_without_assistant_labels(
        train_rows, tokenizer, max_length
    )
    eval_rows, dropped_eval = drop_rows_without_assistant_labels(
        eval_rows, tokenizer, max_length
    )
    if dropped_train or dropped_eval:
        print(
            f"  skipped {dropped_train} train / {dropped_eval} eval conversations "
            f"whose assistant turn starts past the training limit "
            f"({max_length} tokens): nothing to learn from them"
        )
        rows = config.setdefault("dataset_rows", {})
        rows["train_dropped_no_assistant_labels"] = dropped_train
        rows["eval_dropped_no_assistant_labels"] = dropped_eval
        _persist_resolved_config(run_dir, config)
    if not train_rows:
        raise ValueError(
            "No trainable conversations: every assistant turn starts past the "
            f"training limit ({max_length} tokens). Raise max_seq_length or "
            "shorten the prompts."
        )
    return train_rows, eval_rows


def _no_fit_message(training_cfg: dict[str, Any], base_model: str) -> str:
    """Plain-language explanation for a probe that could not fit micro-batch 1."""
    import os

    gpu = os.environ.get("LQH_GPU_TYPE") or (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "this GPU"
    )
    longest = training_cfg.get("longest_row_tokens") or training_cfg.get(
        "max_seq_length", 0
    )
    return (
        f"The longest conversation in the dataset (~{int(longest):,} tokens) does "
        f"not fit on {gpu} with {base_model}, even one at a time. Shorten or "
        "split the longest conversations, or use a smaller model."
    )


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
        self.training_end = training_end_for(config)
        self.reporter = ProgressReporter(
            task_kind="sft", label=run_dir.name, run_dir=run_dir,
        )

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
        max_steps = getattr(state, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            extra["max_steps"] = max_steps
        write_progress(
            self.run_dir,
            step=state.global_step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            epoch=state.epoch,
            extra=extra or None,
            # The v1 reporter immediately below is the cloud headline event;
            # keep this metric row local instead of doubling backend events.
            emit_cloud=not (isinstance(max_steps, int) and max_steps > 0),
        )
        if isinstance(max_steps, int) and max_steps > 0:
            self.reporter.update(
                phase="training",
                phase_label="training SFT",
                completed=state.global_step,
                total=max_steps,
                unit="steps",
                overall_fraction=(
                    self.training_end * state.global_step / max_steps
                ),
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
                epoch=(
                    float(state.epoch)
                    if isinstance(state.epoch, (int, float)) else None
                ),
                detail=(f"loss {logs['loss']:.4f}" if isinstance(logs.get('loss'), (int, float)) else None),
                force=state.global_step >= max_steps,
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

        # Run inference on eval dataset and write predictions.
        # reporter+band are what keep this pass visible: it can run longer
        # than the training it interrupts, and without progress rows the
        # stall watchdog reports a healthy run as a wedged sandbox.
        max_steps = getattr(state, "max_steps", 0)
        _run_checkpoint_eval(
            model=model,
            tokenizer=self.tokenizer,
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            reporter=self.reporter,
            band=checkpoint_eval_band(
                self.training_end,
                state.global_step,
                max_steps if isinstance(max_steps, int) else 0,
            ),
        )


# ---------------------------------------------------------------------------
# Checkpoint evaluation (inline inference, no subprocess)
# ---------------------------------------------------------------------------

# A mid-run checkpoint eval generates one sample at a time over the eval set.
# On an L4 a 149-row eval set took 72 minutes — longer than the ~17-minute
# training run it interrupted (feedback #80), and nothing SELECTS on it:
# checkpoint selection is `load_best_model_at_end` on the teacher-forced
# `eval_loss` (34 seconds, a separate pass), the sweep proxy reads
# `eval_history.json`, and the published `selected_checkpoint` is always
# "final". So the mid-run pass is a mid-training sanity curve, and a sample of
# the eval set draws the same curve for a fraction of the GPU time.
#
# It is not invisible, though: a LOCAL run's `training_status` prints every
# `checkpoints/*/eval_result.json` under "Eval scores:", step rows directly
# above the final one (`handlers._format_status`). A 24-row mean is not
# comparable to a full-set one, so `_write_eval_sampling` records the counts
# and that renderer labels the sampled rows. The final eval stays uncapped —
# that one is the reported number, and on a cloud run it is the only one
# `handlers._hydrate_run_eval_artifacts` pulls back at all.
MID_RUN_CHECKPOINT_EVAL_SAMPLES = 24

# Sidecar the status renderer reads to label a thinned checkpoint score. Kept
# out of eval_result.json because that file is written much later by whichever
# scorer ran (the laptop watcher, or `cloud_score` inline in the sandbox) and
# neither one can see how many rows generation started from.
EVAL_SAMPLING_FILE = "eval_sampling.json"


def mid_run_checkpoint_eval_samples(config: dict[str, Any]) -> int:
    """How many eval rows a MID-RUN checkpoint eval may generate.

    ``checkpoint_eval_samples: 0`` (or any non-positive value) restores the
    old behaviour of generating over the whole eval set. It is an escape
    hatch for a hand-written ``config.json`` run through ``python -m
    lqh.train`` — ``train_start`` builds its config from typed arguments and
    does not pass this through.
    """
    raw = config.get("checkpoint_eval_samples", MID_RUN_CHECKPOINT_EVAL_SAMPLES)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return MID_RUN_CHECKPOINT_EVAL_SAMPLES


def cap_eval_sources(
    eval_srcs: list[tuple[str, list[Any]]],
    limit: int,
) -> list[tuple[str, list[Any]]]:
    """Keep about *limit* conversations across the sources, proportionally.

    Sources stay DISTINCT and every non-empty one keeps at least one
    conversation, so the per-source macro-average the judge computes still has
    a row from each source it had before. That floor wins over *limit*: with
    more sources than *limit* the result is one row each, above it. Otherwise
    rounding lands the total a sample or two either side, which costs seconds.

    The rows are an evenly spaced stride, not a random sample: every
    checkpoint of a run then evaluates the SAME rows, so the score curve
    compares like with like. A non-positive *limit*, or a set already at or
    under it, is returned untouched.
    """
    total = sum(len(samples) for _, samples in eval_srcs)
    if limit <= 0 or total <= limit:
        return eval_srcs
    capped: list[tuple[str, list[Any]]] = []
    for label, samples in eval_srcs:
        if not samples:
            capped.append((label, samples))
            continue
        keep = min(len(samples), max(1, round(limit * len(samples) / total)))
        stride = len(samples) / keep
        capped.append(
            (
                label,
                [
                    samples[min(len(samples) - 1, int(i * stride))]
                    for i in range(keep)
                ],
            )
        )
    return capped


def _write_eval_sampling(
    checkpoint_dir: Path, generated: int, eval_rows: int
) -> None:
    """Record whether this checkpoint's score came from a sample of the set.

    Writes the sidecar when the set was thinned and REMOVES it when it was
    not. Both directions matter: a relaunch re-runs the same ``step_N`` into
    the same directory, so a resume that opts out of the cap (or whose eval
    set shrank below it) would otherwise inherit the previous attempt's file
    and label a full-set score as sampled.

    Best-effort: a run must never die because the sidecar could not be
    written. Its only consumer is the status renderer, which falls back to an
    unlabelled line.
    """
    sidecar = checkpoint_dir / EVAL_SAMPLING_FILE
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if generated < eval_rows:
            sidecar.write_text(
                json.dumps({"generated": generated, "eval_rows": eval_rows})
                + "\n"
            )
        else:
            sidecar.unlink(missing_ok=True)
    except OSError as exc:
        print(f"  WARNING: could not record checkpoint eval sampling: {exc}")


def _run_checkpoint_eval(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    checkpoint_dir: Path,
    *,
    reporter: ProgressReporter | None = None,
    band: tuple[float, float] | None = None,
) -> None:
    """Generate predictions on the eval dataset and signal for scoring.

    The final eval (``checkpoint_dir.name == "final"``) builds its own
    reporter over the reserved final-inference band. A MID-RUN checkpoint
    instead passes the trainer callback's own *reporter* plus a *band* from
    ``checkpoint_eval_band`` — reusing that one reporter is what keeps the
    overall fraction monotonic across the training/eval interleave.

    With neither, the pass runs silently, which is what made a healthy run
    look like a stalled sandbox for the whole generation loop.
    """
    eval_dataset_path = config.get("eval_dataset")
    if not eval_dataset_path:
        return

    # Eval sources are kept DISTINCT and each prediction row is tagged with
    # its source, so the judge can score every source separately and combine
    # them into a macro-average (see score_predictions_by_source).
    eval_srcs = load_eval_sources(eval_dataset_path)
    if checkpoint_dir.name != "final":
        full_rows = sum(len(samples) for _, samples in eval_srcs)
        eval_srcs = cap_eval_sources(
            eval_srcs, mid_run_checkpoint_eval_samples(config),
        )
        kept_rows = sum(len(samples) for _, samples in eval_srcs)
        if kept_rows < full_rows:
            print(
                f"  checkpoint eval ({checkpoint_dir.name}): generating over "
                f"{kept_rows} of {full_rows} eval rows (the final eval uses "
                f"all of them)"
            )
        _write_eval_sampling(checkpoint_dir, kept_rows, full_rows)
    max_seq = config.get("training", {}).get("max_seq_length", 2048)
    is_vision = config.get("modality") == "vision"

    predictions: list[dict[str, Any]] = []
    model.eval()

    eval_reporter = None
    phase = "inference"
    phase_label = "evaluating final model"
    frac_start = 0.0
    frac_end = 0.0
    total_eval = sum(len(samples) for _, samples in eval_srcs)
    if checkpoint_dir.name == "final" and has_final_inference(config):
        frac_start = training_end_for(config)
        frac_end = FINAL_INFERENCE_END if has_final_scoring(config) else 1.0
        # grpo_loop reuses this final-eval path; keep its progress events
        # labelled with the run's own kind rather than "sft".
        task_kind = (
            "grpo" if config.get("type") in ("grpo", "on_policy_grpo") else "sft"
        )
        eval_reporter = ProgressReporter(
            task_kind=task_kind, label=checkpoint_dir.parent.parent.name,
            run_dir=checkpoint_dir.parent.parent,
        )
    elif reporter is not None and band is not None:
        # Mid-run checkpoint: reuse the caller's reporter (a second instance
        # would carry its own _last_fraction and let the reported fraction
        # jump around) and stay inside the training band it was given.
        eval_reporter = reporter
        frac_start, frac_end = band
        phase = "checkpoint_eval"
        phase_label = (
            "evaluating checkpoint "
            + checkpoint_dir.name.replace("step_", "step ")
        )
    if eval_reporter is not None:
        eval_reporter.update(
            phase=phase, phase_label=phase_label,
            completed=0, total=total_eval, unit="samples",
            overall_fraction=frac_start, force=True,
        )

    idx = 0
    out_of_time = False
    for source_label, eval_samples in eval_srcs:
        if out_of_time:
            break
        for conv, sample_tools in eval_samples:
            # Generation over the eval set is the least bounded thing in a
            # run (a 159-row set took 72 minutes on an L4) and it sits
            # between the saved model and the launcher's publish. Spending
            # the publish reserve here is what gets the sandbox killed with
            # a checkpoint nobody ever uploads — the run then has to be
            # trained again from scratch just to be evaluated. The caller
            # checks the deadline before starting; this checks it again per
            # sample because training can finish with time for only part of
            # the pass.
            if past_deadline():
                out_of_time = True
                print(
                    f"  deadline stop: ending this eval after {idx}/"
                    f"{total_eval} samples so the run can publish its "
                    f"checkpoint. Score that checkpoint without retraining: "
                    f"push lqh:<artifact_id> to HF, then eval_hf_model. Or "
                    f"resubmit with a larger timeout_minutes.",
                    flush=True,
                )
                break

            # Strip trailing assistant turn if present (we want the model to generate)
            prompt_msgs = conv
            if conv and conv[-1].get("role") == "assistant":
                prompt_msgs = conv[:-1]

            try:
                if is_vision:
                    # Vision path: `tokenizer` is the AutoProcessor here;
                    # vlm_generate decodes the data-URL image parts and
                    # moves the full inputs dict (pixel_values included)
                    # to the model device.
                    from lqh.train.vlm_data import vlm_generate

                    response = vlm_generate(
                        model,
                        tokenizer,
                        prompt_msgs,
                        max_new_tokens=min(max_seq, 1024),
                    )
                else:
                    # tools= is the ONLY way the tool list reaches the
                    # prompt: LFM chat templates read the template
                    # variable, never a per-message "tools" key. Generating
                    # without it grades the model on calling tools it was
                    # never shown.
                    template_kwargs: dict[str, Any] = {
                        "return_tensors": "pt",
                        "add_generation_prompt": True,
                        "return_dict": True,
                    }
                    if sample_tools is not None:
                        template_kwargs["tools"] = sample_tools
                    inputs = tokenizer.apply_chat_template(
                        prompt_msgs,
                        **template_kwargs,
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
            pred_entry = {
                "sample_index": idx,
                "messages": json.dumps(full_conv),
                "source": source_label,
            }
            # The judge reads this column (scoring._load_samples) to show the
            # available tools alongside the calls it is grading.
            if sample_tools is not None:
                pred_entry["tools"] = json.dumps(sample_tools)
            # Gold answer for the judge (see lqh.scoring._load_references) —
            # prompt_msgs dropped it above so the model would generate.
            if conv and conv[-1].get("role") == "assistant":
                pred_entry["reference"] = json.dumps([conv[-1]])
            predictions.append(pred_entry)
            idx += 1
            if eval_reporter is not None:
                eval_reporter.update(
                    phase=phase, phase_label=phase_label,
                    completed=idx, total=total_eval, unit="samples",
                    overall_fraction=(
                        frac_start
                        + (frac_end - frac_start) * idx / max(total_eval, 1)
                    ),
                    force=idx == total_eval,
                )

    if out_of_time:
        # Predictions over part of the set would be scored as if they
        # covered it, and the judge pass is itself unbounded. What is left
        # of the cap belongs to the publish.
        return

    # Write predictions as parquet (single combined file, source-tagged)
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns: dict[str, list] = {
        "sample_index": [p["sample_index"] for p in predictions],
        "messages": [p["messages"] for p in predictions],
        "source": [p["source"] for p in predictions],
    }
    if any(p.get("reference") for p in predictions):
        columns["reference"] = [p.get("reference") for p in predictions]
    if any(p.get("tools") for p in predictions):
        columns["tools"] = [p.get("tools") for p in predictions]
    table = pa.table(columns)
    pq.write_table(table, checkpoint_dir / "predictions.parquet")

    # Signal the main process to score
    write_eval_request(checkpoint_dir)

    # Cloud path: the laptop-side watcher is bypassed (the sandbox
    # has no laptop), so the eval_request signal would never be
    # picked up. Run the judge inline using the scoped
    # LQH_API_TOKEN — same shape sweep eval-of-best and DPO already
    # use. No-op in local / SSH-direct training where the watcher
    # does the work.
    from lqh.train.cloud_score import is_cloud_mode, score_run_eval_inline

    if is_cloud_mode():
        try:
            score_run_eval_inline(checkpoint_dir, config)
        except Exception as exc:  # noqa: BLE001
            # Inline scoring is best-effort — a judge failure must
            # never crash a training run. The predictions file is
            # still on disk so a re-score is possible offline.
            import logging

            logging.getLogger(__name__).warning(
                "inline scoring failed for %s: %s", checkpoint_dir, exc
            )


# ---------------------------------------------------------------------------
# Main SFT loop
# ---------------------------------------------------------------------------


def sft_loop(run_dir: Path, config: dict[str, Any]) -> None:
    """Run supervised fine-tuning.

    Called from ``lqh.train.__main__`` — this is the subprocess entry point.
    """
    base_model = config["base_model"]
    dataset_path = config["dataset"]
    # setdefault, not get: with `get` a config that carries no training block at
    # all leaves training_cfg DETACHED, so everything written into it here —
    # resolved defaults, the calibrated batch — is invisible to anything reading
    # `config` (checkpoint lineage recorded nulls for exactly this reason).
    training_cfg = config.setdefault("training", {})
    lora_cfg = config.get("lora", {})

    # Modality: normally set by handle_start_training; the AutoConfig
    # fallback covers configs written by hand or by older callers.
    modality = config.get("modality")
    if modality not in ("text", "vision"):
        from lqh.train.load_model import detect_modality

        modality = detect_modality(base_model)
        config["modality"] = modality
    is_vision = modality == "vision"

    from lqh.train.load_model import display_model_ref

    print(
        f"Loading model: {display_model_ref(base_model, run_dir)} "
        f"(modality={modality})"
    )

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

    # Continue an adapter by updating its existing weights. Stacking a new
    # LoRA on an adapter-loaded model is ambiguous in PEFT and can save only
    # the outer adapter, silently losing the SFT starting point. Updating the
    # existing adapter preserves an adapter-only, multi-tenant deployable
    # artifact. Callers can opt out with continue_existing_adapter=false,
    # which merges the old adapter before attaching a fresh one.
    from lqh.train.load_model import detect_kind, load_for_training

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
    model, processing, _effective_base = load_for_training(
        base_model,
        dtype=dtype,
        device_map=device_map,
        merge_before_attach=not continuing_adapter,
        adapter_trainable=continuing_adapter,
        modality=modality,
        max_image_tokens=int(training_cfg.get("max_image_tokens", 256)),
    )
    processor = processing if is_vision else None
    tokenizer = processing.tokenizer if is_vision else processing
    if continuing_adapter:
        print(
            "Continuing the existing LoRA adapter in place "
            "(adapter-only output remains deployable)."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not is_vision:
        # Fail quickly, before the dataset is touched: text SFT trains on
        # assistant tokens only, and a chat template without a
        # {% generation %} block cannot mark them (lqh.train.assistant_mask).
        require_assistant_mask_support(tokenizer, base_model)

    # LoRA config. Defaults are per-modality: the vision recipe
    # (docs.liquid.ai/lfm/fine-tuning/trl) uses a lower rank and targets
    # the attention + feed-forward + projector linears; hitting the
    # vision-tower MLPs (fc1/fc2) and the multimodal projector (linear)
    # is intentional. handle_start_training writes these explicitly, so
    # the fallbacks here only matter for hand-written configs.
    if is_vision:
        _lora_defaults = {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "v_proj", "fc1", "fc2", "linear",
                "gate_proj", "up_proj", "down_proj",
            ],
        }
    else:
        _lora_defaults = {
            "r": 32,
            "alpha": 64,
            "dropout": 0.02,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj", "out_proj", "w1", "w2", "w3",
            ],
        }
    peft_config = None
    if lora_enabled and not continuing_adapter:
        peft_config = LoraConfig(
            r=lora_cfg.get("r", _lora_defaults["r"]),
            lora_alpha=lora_cfg.get("alpha", _lora_defaults["alpha"]),
            lora_dropout=lora_cfg.get("dropout", _lora_defaults["dropout"]),
            target_modules=lora_cfg.get(
                "target_modules", _lora_defaults["target_modules"],
            ),
            task_type="CAUSAL_LM",
        )

    # TRL >= 1.7 defaults the SFT loss to chunked_nll: the same NLL, with the
    # logits chunked so a 32k-token row does not materialize a full-vocab
    # logit tensor. Pinned explicitly so a TRL default change cannot alter
    # training silently. chunked_nll rejects lm_head as a LoRA target (it
    # hooks the output embedding), so such a config falls back to plain nll.
    sft_loss_type = "chunked_nll"
    _targets = lora_cfg.get("target_modules") or []
    if isinstance(_targets, str):
        _targets = [_targets]
    if lora_enabled and "lm_head" in _targets:
        print("  lm_head is a LoRA target: loss_type=nll (chunked_nll rejects it)")
        sft_loss_type = "nll"

    # Load training set. The eval set is resolved with this order
    # of precedence:
    #   1. explicit ``config["eval_dataset"]`` — separate parquet,
    #      used verbatim. Honoured even when small (no min_eval gate)
    #      because the caller knows it's the eval they want.
    #   2. internal train/eval split, controlled by
    #      ``training.eval_split_ratio`` (default 0.1). Falls back
    #      to "no eval" when the resulting slice is below
    #      split_train_eval's min_eval threshold.
    #   3. eval_split_ratio=0 disables entirely.
    #
    # Reason precedence 1 exists: the sweep harness ships dataset +
    # eval_dataset in the bundle (the SAME files the laptop watcher
    # would have scored locally). Without honouring eval_dataset
    # here, a small training set + a small eval set silently runs
    # train-only and the sweep's proxy metric stays NaN, marking the
    # config "failed". That bit us on the before/after test.
    print(f"Loading dataset: {dataset_path}")
    conversations, tools_per_sample = load_chatml_datasets_with_tools(dataset_path)
    eval_dataset_path = config.get("eval_dataset")
    eval_split_ratio = float(training_cfg.get("eval_split_ratio", 0.1))
    if eval_dataset_path:
        # The in-training eval set (drives eval_loss / load_best_model_at_end
        # / the sweep proxy) is the CONCATENATION of all eval sources — a
        # single scalar selection signal. Per-source judge scoring happens
        # separately in _run_checkpoint_eval, which keeps sources distinct.
        eval_srcs = load_eval_sources(eval_dataset_path)
        print(
            "Loading explicit eval_dataset: "
            + ", ".join(f"{label}({len(smp)})" for label, smp in eval_srcs)
        )
        train_convos, train_tools = conversations, tools_per_sample
        eval_pairs = [pair for _label, samples in eval_srcs for pair in samples]
    elif eval_split_ratio > 0:
        # Split the (conversation, tools) pairs, not the conversations, so a
        # sample keeps its tool definitions on whichever side it lands.
        train_pairs, eval_pairs = split_train_eval(
            list(zip(conversations, tools_per_sample)), eval_split_ratio, seed=0
        )
        train_convos = [conv for conv, _tools in train_pairs]
        train_tools = [tools for _conv, tools in train_pairs]
    else:
        train_convos, train_tools = conversations, tools_per_sample
        eval_pairs = []
    eval_convos = [conv for conv, _tools in eval_pairs]
    eval_tools = [tools for _conv, tools in eval_pairs]
    print(
        f"  train={len(train_convos)} eval={len(eval_convos)} "
        f"(eval_dataset={'explicit' if eval_dataset_path else f'split:{eval_split_ratio}'})"
    )
    n_with_tools = sum(1 for t in train_tools if t)
    if n_with_tools:
        # Silence here used to mean the tool definitions were dropped on the
        # floor: they only reach the model through the dataset's tools column.
        print(f"  tool definitions: {n_with_tools}/{len(train_convos)} train samples")

    if is_vision:
        # Vision rows carry compressed image bytes in a parallel column;
        # the VLMCollator decodes them to PIL lazily per batch.
        from lqh.train.vlm_data import chatml_to_vlm_dataset

        train_dataset = Dataset.from_list(chatml_to_vlm_dataset(train_convos))
        eval_dataset: Dataset | None = None
        if eval_convos:
            eval_dataset = Dataset.from_list(chatml_to_vlm_dataset(eval_convos))
    else:
        train_rows_sft = chatml_to_sft_dataset(train_convos, train_tools)
        eval_rows_sft = (
            chatml_to_sft_dataset(eval_convos, eval_tools) if eval_convos else []
        )
        # Exact sequence length from the data, BEFORE the calibration probe
        # below reads training_cfg["max_seq_length"].
        train_rows_sft, eval_rows_sft = _resolve_text_seq_length(
            run_dir, config, training_cfg, tokenizer, model,
            train_rows_sft, eval_rows_sft,
        )
        train_rows_sft, eval_rows_sft = _drop_rows_without_assistant_labels(
            run_dir, config, training_cfg, tokenizer,
            train_rows_sft, eval_rows_sft,
        )
        train_dataset = Dataset.from_list(train_rows_sft)
        eval_dataset = Dataset.from_list(eval_rows_sft) if eval_rows_sft else None

    # Safe batch-size auto-tuning (GPU_TYPE.md §6). Mutates training_cfg
    # in place (per_device_batch_size + gradient_accumulation_steps) so
    # the sft_kwargs below pick up the calibrated values. No-op when
    # auto_batch is off, no GPU, or the backend is unreachable.
    #
    # The probe must see the model in its TRAINING configuration: for
    # LoRA we wrap with the adapter first (frozen base + tiny trainable
    # adapter) and unload right after — SFTTrainer re-wraps via
    # peft_config exactly as before, so the training path is unchanged.
    # The pre-fix probe ran on the raw model, measured roughly
    # full-FT-without-checkpointing memory, and discovered micro-batches
    # ~10x too small (GPU_TYPE_2.md).
    from lqh.train.calibrate import ensure_batch_defaults, maybe_autotune_batch_size

    if is_vision:
        # The calibration probe builds synthetic TEXT batches, which would
        # under-measure the vision encoder's activation peak — skip it and
        # start from conservative defaults. auto_batch stays on in the
        # config, so an OOM still triggers report_oom_downgrade, which
        # writes a halved, modality="vision"-keyed batch profile.
        ensure_batch_defaults(
            training_cfg,
            default_micro_batch=2,
            default_effective_batch=16,
        )
    else:
        # Fallback for a config that carries no batch fields at all (older
        # bundles, hand-written configs). Submission normally derives the LoRA
        # target from the dataset (lqh.train.defaults.sft_effective_batch) and
        # writes both fields, in which case these values are unused.
        ensure_batch_defaults(
            training_cfg,
            default_micro_batch=256 if lora_enabled else 1,
            default_effective_batch=256 if lora_enabled else 16,
        )
        probe_model = model
        if peft_config is not None:
            from peft import get_peft_model

            probe_model = get_peft_model(model, peft_config)
        calibration = maybe_autotune_batch_size(
            training_cfg,
            model=probe_model,
            tokenizer=tokenizer,
            base_model=base_model,
            method="lora" if lora_enabled else "full",
            lora_rank=int(lora_cfg.get("r", 32)) if lora_enabled else 0,
        )
        if calibration == "no_fit":
            # Even micro-batch 1 exceeded the budget at this run's sequence
            # length. Proceeding would train on the configured default
            # (256 for LoRA) and OOM after the model load — fail now, with
            # the reason, instead.
            raise RuntimeError(_no_fit_message(training_cfg, base_model))
        from lqh.train.defaults import MAX_SEQ_LENGTH

        if (
            calibration == "skipped"
            and torch.cuda.is_available()
            and bool(training_cfg.get("auto_batch", True))
            and int(training_cfg.get("max_seq_length", 0)) > MAX_SEQ_LENGTH
        ):
            # Auto-tuning was on but produced no measurement (probe or
            # backend error). The configured micro-batch is the dataset-
            # derived throughput value (up to 256), which is only known to
            # be safe at the default length. Above it, micro-batch 1 is the
            # one assumption that cannot OOM more than the probe would have;
            # the effective batch is preserved through accumulation.
            from lqh.train.calibrate import _apply

            target = int(training_cfg.get("effective_batch_size", 1))
            _apply(training_cfg, 1, target)
            print(
                "  calibrate: no measurement at "
                f"max_seq_length={training_cfg['max_seq_length']}; using "
                f"micro-batch 1 x {training_cfg['gradient_accumulation_steps']} "
                "accumulation as the safe default"
            )
        if probe_model is not model:
            model = probe_model.unload()
            # PEFT leaves these markers on some transformers versions after
            # unload(), causing the real trainer wrap to be treated as a
            # second adapter attachment.
            if hasattr(model, "peft_config"):
                delattr(model, "peft_config")
            if hasattr(model, "_hf_peft_config_loaded"):
                model._hf_peft_config_loaded = False

    # How many optimizer updates this run will actually take. Two consumers:
    # a warning when that number is too small for the adapter to move (the
    # failure that reads as "training doesn't work on this dataset" —
    # submission derives the batch to avoid it, but an explicit batch or a
    # tiny dataset can still land here), and the log cadence below.
    from lqh.train.defaults import (
        SFT_MIN_HEALTHY_OPTIMIZER_STEPS,
        fill_missing_hyperparameters,
        optimizer_steps,
    )

    # A config with no learning_rate (or num_epochs) reaches here from an older
    # bundle, a hand-written config, or a direct `python -m lqh.train`
    # invocation. Fallbacks must come from the same source of truth the tool
    # uses, not hardcoded literals — the learning-rate one was 2e-5, the value
    # item 47 was about, so a config missing the field trained at the known-bad
    # rate.
    #
    # Resolve them INTO training_cfg rather than at each read site: the config
    # dict is what _write_checkpoint_lineage publishes and what the rest of this
    # function reads, so a value left implicit shows up as `learning_rate: null`
    # in a checkpoint's lineage and as a missing LR on the training_status
    # health line — for a run that did have one.
    _filled = fill_missing_hyperparameters(
        training_cfg,
        run_type="sft",
        lora=lora_enabled,
        modality="vision" if is_vision else "text",
    )
    if _filled:
        print(
            "  resolved missing hyperparameters from lqh.train.defaults: "
            + ", ".join(f"{k}={v}" for k, v in _filled.items())
        )
        # Reaches the on-disk config so training_status, publish and a resume
        # all report what this run actually trained at.
        _persist_resolved_config(run_dir, config)

    # micro x accumulation, never the config's effective_batch_size field: the
    # calibration probe above may have lowered the micro-batch for memory, and
    # these two are what HF Trainer actually uses. Reading the field would
    # report a step count the run does not do — same arithmetic dpo.py uses for
    # its own step-starvation warning.
    effective_batch = max(
        1,
        int(training_cfg.get("per_device_batch_size", 1))
        * int(training_cfg.get("gradient_accumulation_steps", 1)),
    )
    total_steps = optimizer_steps(
        train_rows=len(train_dataset),
        num_epochs=int(training_cfg.get("num_epochs", 3)),
        effective_batch_size=effective_batch,
    )
    if total_steps:
        print(
            f"  optimizer steps: ~{total_steps} "
            f"({len(train_dataset)} rows x {training_cfg.get('num_epochs', 3)} "
            f"epochs / effective batch {effective_batch})"
        )
        if total_steps < SFT_MIN_HEALTHY_OPTIMIZER_STEPS:
            print(
                f"  WARNING: only ~{total_steps} optimizer updates for the whole "
                f"run (healthy is >= {SFT_MIN_HEALTHY_OPTIMIZER_STEPS}). A flat "
                f"eval score after this is far more likely to mean too few "
                f"updates than a bad dataset — more rows or more epochs before "
                f"blaming the data."
            )

    # Checkpoints dir
    checkpoint_output = str(run_dir / "checkpoints")

    # Log often enough to see a loss *curve* rather than a single point: at
    # ~20 log rows per run, capped at the old fixed cadence of 50 steps. A
    # short run (54 total steps at logging_steps=50) printed loss once, which
    # is exactly the diagnostic a flat run needs and cannot get.
    default_logging_steps = 50
    if total_steps:
        default_logging_steps = max(1, min(50, math.ceil(total_steps / 20)))

    # Eval cadence is shared with save cadence so load_best_model_at_end
    # has matching checkpoints to choose from. When eval_dataset is
    # absent we fall back to the pre-existing behaviour (save_steps
    # only, no eval).
    eval_steps = int(training_cfg.get("eval_steps", 50))
    has_eval = eval_dataset is not None
    sft_kwargs: dict[str, Any] = dict(
        output_dir=checkpoint_output,
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg["learning_rate"],
        # warmup_steps, not warmup_ratio: transformers 5.x removed the ratio
        # argument (grpo.py made the same switch). warmup_ratio stays the
        # user-facing knob and is turned into steps from the optimizer-step
        # estimate above; an explicit warmup_steps wins. 0 when the step
        # count is unknown.
        warmup_steps=int(
            training_cfg.get(
                "warmup_steps",
                round(total_steps * float(training_cfg.get("warmup_ratio", 0.1)))
                if total_steps
                else 0,
            )
        ),
        logging_steps=training_cfg.get("logging_steps", default_logging_steps),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        bf16=training_cfg.get("bf16", True),
        max_length=training_cfg.get("max_seq_length", 2048),
        loss_type=sft_loss_type,
        # Always on for text. TRL (through 1.12) refuses it for vision
        # datasets (rows with an images column): multimodal assistant masks
        # are still broken upstream (transformers #44521), so VLM runs keep
        # the full-sequence loss. Not configurable (lqh.train.assistant_mask).
        assistant_only_loss=not is_vision,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        seed=training_cfg.get("seed", 42),
        data_seed=training_cfg.get(
            "data_seed", training_cfg.get("seed", 42)
        ),
    )
    if is_vision:
        # The VLMCollator owns tokenization; TRL must not try to prepare
        # (tokenize/truncate) the dataset itself — its truncation could
        # sever an image-token span. Length enforcement happens in the
        # collator (over-long samples are dropped, not truncated).
        #
        # dataloader_num_workers MUST be 0: the collator runs torch/
        # torchvision image-processing ops, and forked DataLoader workers
        # deadlock on those after the parent has initialized CUDA (first
        # cloud smoke run hung indefinitely at the first batch fetch).
        # Text runs are unaffected — their collation is plain tokenizer
        # work. PIL decode + preprocess is cheap next to the VLM forward,
        # so in-process collation costs little.
        sft_kwargs.update(
            max_length=None,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_num_workers=0,
        )
    if has_eval:
        # load_best_model_at_end with metric=eval_loss is SAFE for SFT
        # (unlike DPO, where we disabled it — see lqh/train/dpo.py).
        # SFT cross-entropy directly measures the absolute probability
        # the policy assigns to the gold continuation; there is no
        # hackable chosen-vs-rejected ratio. eval_loss tracks judge score
        # well enough to pick a checkpoint: Pearson r = -0.90 on the
        # ar_to_de proxy validation (2026-05-11), mean Spearman -0.787
        # across the hp_defaults cells. It is a good *ranker* and a poor
        # *picker* — that study's top-1 agreement was 0/6 (see the module
        # docstring in lqh/train/sweep.py), which matters for choosing
        # between configs, not for choosing between this run's own
        # checkpoints, where the alternative is "take the last one".
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
        # A run killed part-way (wall-clock cap, preemption, OOM) can only
        # be salvaged from a checkpoint on disk, and the old default of 500
        # writes none at all for the short runs this trainer usually does —
        # a small dataset at an auto-tuned batch is ~14 optimizer steps.
        # Same shape as logging_steps above: about four saves per run,
        # never coarser than the old 500.
        default_save_steps = 500
        if total_steps:
            default_save_steps = max(1, min(500, math.ceil(total_steps / 4)))
        sft_kwargs["save_steps"] = training_cfg.get("save_steps", default_save_steps)
        # Checkpoints are full copies of the model and the project volume is
        # shared with every sibling job, so keep the tail bounded (the eval
        # branch above already does).
        sft_kwargs["save_total_limit"] = training_cfg.get("save_total_limit", 2)
    sft_config = SFTConfig(**sft_kwargs)

    # Progress callback. For vision it gets the processor (checkpoint eval
    # needs apply_chat_template with images), not the bare tokenizer.
    progress_cb = ProgressCallback(run_dir, config, processor if is_vision else tokenizer)

    # Stops training before the sandbox's wall-clock cap so the final save
    # and the launcher's publish step still happen. No-op when the backend
    # passed no deadline (SSH-direct runs, local runs).
    deadline_cb = DeadlineStopCallback(run_dir, label="sft")

    # Trainer
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "processing_class": processor if is_vision else tokenizer,
        "callbacks": [progress_cb, deadline_cb],
    }
    if is_vision:
        from lqh.train.vlm_data import VLMCollator

        trainer_kwargs["data_collator"] = VLMCollator(
            processor, max_length=training_cfg.get("max_seq_length", 2048),
        )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    train_with_checkpoint_fallback(
        trainer,
        run_dir / "checkpoints",
        label="sft",
    )

    # Always evaluate the final model once after training. Two reasons:
    #   1. The in-training eval cadence (eval_steps, default 50) can
    #      exceed the total number of optimizer steps entirely — small
    #      dataset x large auto-tuned batch means a 2-epoch run may have
    #      ~14 steps. Then HF Trainer never evaluates, eval_history.json
    #      carries no eval_loss, and the sweep proxy is missing for
    #      every config (sweep fails with "wrote no proxy metric").
    #   2. Even when step evals did fire, this measures the model that
    #      is actually saved (post load_best_model_at_end), which is
    #      what the sweep is selecting between. Trainer.evaluate() logs
    #      its metrics into state.log_history, so the dump below picks
    #      it up as the LAST eval_loss entry — exactly the one
    #      _read_sft_proxy in sweep.py uses.
    #
    # Skipped on a deadline stop: everything after this point shares the
    # publish reserve with the save, the tar and a multi-GB upload, and an
    # eval pass is not bounded by it. A run long enough to hit the cap has
    # fired its step evals many times over, so eval_history still carries an
    # eval_loss for the sweep proxy — the case above is a short run, which
    # never reaches the deadline.
    if eval_dataset is not None and not deadline_cb.triggered:
        try:
            final_metrics = trainer.evaluate()
            print(f"Final eval: eval_loss={final_metrics.get('eval_loss')}")
        except Exception as exc:  # noqa: BLE001 — eval must not kill a finished train
            print(f"  WARNING: final evaluation failed: {exc}")
    elif eval_dataset is not None:
        print("  deadline stop: skipping the final evaluation so the run can publish.")

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

    # Save final model. For LoRA runs the default is now to save the
    # adapter alone (tens of MB) rather than merging into the base
    # (multi-GB) — set ``lora.merge=True`` to opt into the merged
    # artifact. The adapter-only layout writes to ``run_dir/model-lora``
    # so the artifact in R2 (``model-lora.tar.gz``) is visually
    # distinct from a merged checkpoint (``model.tar.gz``). Downstream
    # consumers go through :func:`lqh.train.load_model.load_for_inference`
    # which transparently handles both layouts.
    # A new adapter trained on top of an in-memory merged parent cannot be
    # loaded later from the original hub base by itself. Force a merged
    # artifact for that explicit opt-out path; the default continuation path
    # remains adapter-only.
    merge_lora = bool(
        lora_cfg.get("merge", False) or fresh_adapter_on_merged_parent
    )
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

    # Vision: save the full processor (tokenizer + image processor + chat
    # template) so downstream loads (merge, serving) get the preprocessor
    # config, not just the tokenizer.
    (processor if is_vision else tokenizer).save_pretrained(str(final_model_dir))
    _write_checkpoint_lineage(
        final_model_dir,
        config=config,
        training_method="lora" if saving_adapter else "full",
        stopped_at_step=deadline_cb.stopped_at_step if deadline_cb.triggered else None,
    )

    print(f"Model saved to {display_model_ref(final_model_dir, run_dir)}")

    # Final eval if requested. Generation + scoring over the eval set, so
    # the least bounded thing in the run — never inside the publish reserve.
    # `triggered` alone is not enough: training that ends on its own with
    # only the reserve left never reaches the callback, and loading a fresh
    # model for inference at that point spends the reserve before the pass
    # even starts.
    out_of_time = deadline_cb.triggered or past_deadline()
    if out_of_time and config.get("eval_on_checkpoints"):
        print(
            "  deadline stop: skipping checkpoint eval so the run can publish. "
            "Score the published checkpoint without retraining: push "
            "lqh:<artifact_id> to HF, then eval_hf_model."
        )
    if (
        config.get("eval_on_checkpoints")
        and config.get("eval_dataset")
        and not out_of_time
    ):
        final_checkpoint = run_dir / "checkpoints" / "final"
        final_checkpoint.mkdir(parents=True, exist_ok=True)

        # Free the trainer's GPU memory before loading a fresh copy for
        # inference. The loader auto-detects adapter vs merged.
        del trainer
        torch.cuda.empty_cache()

        from lqh.train.load_model import load_for_inference

        eval_model, _ = load_for_inference(
            str(final_model_dir),
            dtype=dtype,
            device_map="auto",
            modality=modality,
            max_image_tokens=int(training_cfg.get("max_image_tokens", 256)) if is_vision else None,
            # See grpo.py: a degenerate adapter must not fail the run
            # before its checkpoint is published.
            verify_adapter=False,
        )
        _run_checkpoint_eval(
            model=eval_model,
            tokenizer=processor if is_vision else tokenizer,
            config=config,
            checkpoint_dir=final_checkpoint,
        )
        del eval_model
        torch.cuda.empty_cache()

    if not has_final_scoring(config):
        write_progress_event(
            run_dir,
            ProgressEvent(
                task_kind="sft", label=run_dir.name,
                phase="completed", phase_label="training complete",
                completed=1, total=1, unit="run", overall_fraction=1.0,
                result_ready=True,
            ),
        )
    write_status(run_dir, "completed")
    print("Training completed.")
