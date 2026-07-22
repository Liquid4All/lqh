"""Entry point for ``python -m lqh.infer <config.json>``.

One-shot local inference: loads a model, runs it on a dataset, writes
predictions.parquet + eval_request.json, then exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.infer <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    run_dir = config_path.parent

    # Write PID file
    (run_dir / "pid").write_text(str(__import__("os").getpid()))
    from lqh.train.progress import begin_run_attempt, write_status
    begin_run_attempt(run_dir)

    try:
        _run_inference(run_dir, config)
    except Exception as exc:
        write_status(run_dir, "failed", error=str(exc))
        raise


# Incremental predictions: each completed sample is appended to this
# JSONL in the run dir (the durable Modal volume for cloud jobs), so a
# worker continuation after a SIGKILL/timeout resumes from the last
# completed sample instead of regenerating everything. Mirrors the
# data_gen partial pattern in lqh/engine.py. Deleted once the canonical
# predictions.parquet is written; never published (publish.py allowlist).
PREDICTIONS_PARTIAL = "predictions.partial.jsonl"


def _predictions_digest(config: dict) -> str:
    """Identity hash binding a partial file to the exact model, dataset,
    and decoding settings that produced it. Resume only happens on an
    exact match; anything else restarts clean rather than absorbing
    predictions from a different configuration.
    """
    import hashlib

    ident = {
        k: config.get(k)
        for k in (
            "base_model", "base_override", "dataset", "max_new_tokens",
            "system_prompt", "response_format", "spec_sha256",
        )
    }
    blob = json.dumps(ident, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _append_prediction_partial(path: Path, index: int, row: dict) -> None:
    line = json.dumps({"index": index, **row}, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()


def _load_prediction_partial(
    path: Path, total: int, digest: str,
) -> dict[int, dict] | None:
    """Parse a partial file into ``{index: pred_entry}``.

    Returns None (→ full restart) when the header is missing or bound to
    a different total/digest. A truncated final line (killed mid-write)
    is tolerated; out-of-range indices are ignored; duplicate indices
    keep the last write.
    """
    rows: dict[int, dict] = {}
    header_ok = False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("_meta"):
                    if obj.get("total") != total or obj.get("digest") != digest:
                        return None
                    header_ok = True
                    continue
                idx = obj.pop("index", None)
                if isinstance(idx, int) and 0 <= idx < total:
                    rows[idx] = obj
    except OSError:
        return None
    return rows if header_ok else None


def _init_prediction_partial(
    run_dir: Path, total: int, digest: str,
) -> dict[int, dict]:
    """Load resumable predictions from a prior attempt, or start a fresh
    partial file. A file bound to a different run identity is preserved
    as ``predictions.partial.stale.jsonl`` (already-paid-for GPU output,
    kept for the operator) and excluded from this run.
    """
    path = run_dir / PREDICTIONS_PARTIAL
    if path.exists():
        rows = _load_prediction_partial(path, total, digest)
        if rows is not None:
            return rows
        try:
            path.replace(run_dir / "predictions.partial.stale.jsonl")
        except OSError:
            path.unlink(missing_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": True, "total": total, "digest": digest}) + "\n")
    return {}


def _normalize_inner_schema(response_format: Any) -> Any:
    """Unwrap a ``response_format`` config value to the bare JSON schema.

    Accepts either the bare schema or the OpenAI-style envelope
    ({"type":"json_schema","json_schema":{"schema": {...}}}) so the same
    prompts/<task>.schema.json file works for both API and local eval.
    Shared by both engines: the HF loop feeds it to lm-format-enforcer,
    the sglang engine re-wraps it into the server's json_schema
    response_format.
    """
    inner_schema: Any = response_format
    if isinstance(response_format, dict):
        if "json_schema" in response_format:
            js = response_format["json_schema"]
            if isinstance(js, dict) and "schema" in js:
                inner_schema = js["schema"]
            else:
                inner_schema = js
    return inner_schema


def _prompt_messages(conv: list[dict], system_prompt: str | None) -> list[dict]:
    """Prompt-side messages for one eval sample: the trailing assistant
    turn (the reference answer, when present) is stripped, and the
    configured system prompt is prepended unless the conversation
    already opens with one. Shared by the HF and sglang engines so the
    prompts they judge are identical.
    """
    prompt_msgs = list(conv)
    if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
        prompt_msgs = prompt_msgs[:-1]
    if system_prompt and (not prompt_msgs or prompt_msgs[0].get("role") != "system"):
        prompt_msgs = [{"role": "system", "content": system_prompt}] + prompt_msgs
    return prompt_msgs


def _finalize_predictions(
    run_dir: Path,
    results: list[dict | None],
    config: dict,
    reporter: Any,
    progress_end: float,
) -> None:
    """Assemble predictions.parquet from per-sample results, signal
    scoring readiness, and (unless the caller owns terminal status via
    ``defer_terminal_status``) clean up the partial and write completed.
    Shared by both generation engines — the parquet schema here IS the
    scoring contract (sample_index/messages/source[/tools]).
    """
    from lqh.train.progress import write_eval_request, write_status

    predictions = [r for r in results if r is not None]
    import pyarrow as pa
    import pyarrow.parquet as pq

    has_tools_col = any("tools" in p for p in predictions)
    columns: dict[str, list] = {
        "sample_index": [p["sample_index"] for p in predictions],
        "messages": [p["messages"] for p in predictions],
        "source": [p["source"] for p in predictions],
    }
    fields = [
        pa.field("sample_index", pa.int64()),
        pa.field("messages", pa.string()),
        pa.field("source", pa.string()),
    ]
    if has_tools_col:
        columns["tools"] = [p.get("tools") for p in predictions]
        fields.append(pa.field("tools", pa.string()))

    table = pa.table(columns, schema=pa.schema(fields))
    pq.write_table(table, run_dir / "predictions.parquet")
    # The partial has served its purpose once the canonical parquet
    # exists (delete only after the write succeeds) — EXCEPT when a
    # scoring step still follows (defer_terminal_status): a sandbox
    # killed mid-scoring would then find neither partial nor resume
    # state and regenerate every sample. The status-owning caller
    # (eval_hf) deletes it after scoring succeeds instead.
    if not config.get("defer_terminal_status"):
        (run_dir / PREDICTIONS_PARTIAL).unlink(missing_ok=True)

    # Signal for scoring
    write_eval_request(run_dir)
    if progress_end >= 1.0:
        reporter.update(
            phase="completed", phase_label="predictions ready",
            completed=len(predictions), total=len(predictions), unit="samples",
            overall_fraction=1.0, result_ready=True, force=True,
        )
    # eval_hf sets defer_terminal_status: its inline scoring step is
    # load-bearing, so the caller owns the terminal status and writes
    # completed/failed only after scoring resolves. Plain infer runs
    # never set the flag and keep the historical behavior.
    if not config.get("defer_terminal_status"):
        write_status(run_dir, "completed")
    print(f"Inference complete: {len(predictions)} predictions written")


def _run_inference(run_dir: Path, config: dict) -> None:
    """Engine dispatcher. The sglang engine is used when the sglang
    package is importable (i.e. the sandbox runs the gpu_eval image);
    everywhere else — training images, dev machines — this stays the
    historical HF transformers loop. ``force_hf_engine`` is the debug /
    parity-run escape hatch. Both engines share the partial-file format
    and digest, so a continuation may switch engines mid-run safely.
    """
    from lqh.infer.engine_sglang import run_inference_sglang, sglang_available

    if not config.get("force_hf_engine") and sglang_available():
        run_inference_sglang(run_dir, config)
        return
    _run_inference_hf(run_dir, config)


def _run_inference_hf(run_dir: Path, config: dict) -> None:
    import torch

    from lqh.progress import ProgressReporter
    from lqh.train.data_utils import load_eval_sources_with_tools
    from lqh.train.load_model import load_for_inference
    from lqh.train.progress import write_eval_request, write_progress, write_status
    from lqh.train.tool_format import get_tool_formatter

    base_model = config["base_model"]
    dataset_path = config["dataset"]
    # Optional explicit base override — used by eval_hf when a LoRA
    # adapter's `adapter_config.json["base_model_name_or_path"]` is
    # missing, points at a renamed/moved repo, or the caller wants to
    # pin a specific base revision regardless of what the adapter
    # declares. Forwarded to load_for_inference; ignored for non-
    # adapter `base_model` values.
    base_override = config.get("base_override")
    progress_dir = Path(config.get("progress_run_dir", run_dir))
    progress_start = float(config.get("progress_start", 0.0))
    progress_end = float(config.get(
        "progress_end", 0.5 if config.get("scorer") else 1.0,
    ))
    reporter = ProgressReporter(
        task_kind=str(config.get("progress_task_kind", "evaluation")),
        label=str(config.get("progress_label", "Model evaluation")),
        run_dir=progress_dir,
    )
    reporter.update(
        phase="setup", phase_label="loading model",
        overall_fraction=progress_start, unit="samples", force=True,
    )

    print(f"Loading model: {base_model}")
    # load_for_inference transparently handles hub ids, merged dirs,
    # and adapter dirs (the latter via base+PeftModel+merge_and_unload).
    # For vision (LFM-VL) models it returns the AutoProcessor in the
    # tokenizer slot — the raw tokenizer is at ``.tokenizer``.
    model, tokenizer = load_for_inference(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        base_override=base_override,
    )
    is_vision = hasattr(tokenizer, "image_processor")
    if is_vision:
        print("  vision model detected — processor-based generation")

    print(f"Loading dataset: {dataset_path}")
    # dataset_path may name one or more sources (eval-of-best passes the
    # eval_dataset list here). Each prediction is tagged with its source so
    # the judge can score sources separately and macro-average them.
    conversations, tools_per_sample, sources_per_sample = (
        load_eval_sources_with_tools(dataset_path)
    )
    reporter.update(
        phase="inference", phase_label="running inference", completed=0,
        total=len(conversations), unit="samples", overall_fraction=progress_start,
        force=True,
    )

    max_new_tokens = config.get("max_new_tokens", 4096)
    system_prompt = config.get("system_prompt")
    response_format = config.get("response_format")

    partial_path = run_dir / PREDICTIONS_PARTIAL
    resumed = _init_prediction_partial(
        run_dir, len(conversations), _predictions_digest(config),
    )
    results: list[dict | None] = [None] * len(conversations)
    for idx, entry in resumed.items():
        results[idx] = entry
    completed_count = len(resumed)
    if resumed:
        print(
            f"Resuming: {completed_count}/{len(conversations)} "
            "predictions already done"
        )
        reporter.update(
            phase="inference", phase_label="running inference",
            completed=completed_count, total=len(conversations),
            unit="samples",
            overall_fraction=(
                progress_start
                + (progress_end - progress_start)
                * completed_count / max(len(conversations), 1)
            ),
            force=True,
        )

    # Get the tool formatter for this model (if applicable)
    tool_formatter = get_tool_formatter(base_model)

    if system_prompt:
        print(f"System prompt: {system_prompt[:80]}...")

    # JSON-schema constrained decoding via lm-format-enforcer.
    #
    # If ``response_format`` is set in the config, we hard-fail on setup
    # errors rather than silently falling back to free-form decoding —
    # silent fallback once produced 200 invalid-JSON predictions on a
    # constrained eval before anyone noticed.
    schema_prefix_fn = None
    if response_format:
        inner_schema = _normalize_inner_schema(response_format)

        # lm-format-enforcer ≤0.11.3 imports ``PreTrainedTokenizerBase`` from
        # ``transformers.tokenization_utils`` (the v4 path). In transformers
        # v5 the class moved to ``transformers.tokenization_utils_base``, so
        # the integration import fails and lmfe's shim re-raises a misleading
        # "transformers is not installed" error. Patch the old path before
        # the integration module is imported.
        import transformers.tokenization_utils as _ttu
        from transformers.tokenization_utils_base import (
            PreTrainedTokenizerBase as _PTTB,
        )
        if not hasattr(_ttu, "PreTrainedTokenizerBase"):
            _ttu.PreTrainedTokenizerBase = _PTTB  # type: ignore[attr-defined]

        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(inner_schema)
        schema_prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer.tokenizer if is_vision else tokenizer, parser,
        )
        print(
            f"  JSON-schema constrained decoding enabled "
            f"(keys: {list(inner_schema.get('properties', {}).keys())})"
        )

    has_any_tools = any(t is not None for t in tools_per_sample)
    if has_any_tools:
        print(f"  Tool-calling dataset detected (formatter: {type(tool_formatter).__name__ if tool_formatter else 'none'})")

    model.eval()
    for i, conv in enumerate(conversations):
        if i in resumed:
            continue
        sample_tools = tools_per_sample[i]
        prompt_msgs = _prompt_messages(conv, system_prompt)

        try:
            if is_vision:
                # Vision path: the processor handles image (data-URL)
                # decoding and multimodal inputs. Tool parsing is not
                # supported for vision datasets (no VL tool-calling data
                # exists yet); the shared tail below is unchanged.
                from lqh.train.vlm_data import vlm_generate

                vlm_kwargs: dict[str, Any] = {}
                if schema_prefix_fn is not None:
                    vlm_kwargs["prefix_allowed_tokens_fn"] = schema_prefix_fn
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": vlm_generate(
                        model,
                        tokenizer,
                        prompt_msgs,
                        max_new_tokens=max_new_tokens,
                        **vlm_kwargs,
                    ),
                }
            else:
                # Build chat template kwargs — pass tools when available
                template_kwargs: dict = {
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
                    generate_kwargs: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                    }
                    if schema_prefix_fn is not None:
                        generate_kwargs["prefix_allowed_tokens_fn"] = schema_prefix_fn
                    output_ids = model.generate(input_ids, **generate_kwargs)
                # Decode without skipping special tokens so we can parse
                # tool call markers if present
                raw_response = tokenizer.decode(
                    output_ids[0][input_ids.shape[-1]:],
                    skip_special_tokens=False,
                )

                # Parse tool calls from model output if formatter available
                assistant_msg = {"role": "assistant"}
                if tool_formatter and sample_tools:
                    content, tool_calls = tool_formatter.parse_assistant_output(raw_response)
                    assistant_msg["content"] = content
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                else:
                    # Fallback: clean decode with special tokens skipped
                    response = tokenizer.decode(
                        output_ids[0][input_ids.shape[-1]:],
                        skip_special_tokens=True,
                    )
                    assistant_msg["content"] = response

        except Exception as exc:
            assistant_msg = {"role": "assistant", "content": f"[generation error: {exc}]"}

        full_conv = prompt_msgs + [assistant_msg]
        pred_entry: dict[str, Any] = {
            "sample_index": i,
            "messages": json.dumps(full_conv),
            "source": sources_per_sample[i],
        }
        if sample_tools is not None:
            pred_entry["tools"] = json.dumps(sample_tools)
        results[i] = pred_entry
        _append_prediction_partial(partial_path, i, pred_entry)
        completed_count += 1

        if completed_count % 10 == 0 or completed_count == len(conversations):
            print(f"  {completed_count}/{len(conversations)} samples done")
            write_progress(
                run_dir,
                step=completed_count,
                extra={"phase": "inference", "total": len(conversations)},
                emit_cloud=False,
            )
            reporter.update(
                phase="inference", phase_label="running inference",
                completed=completed_count, total=len(conversations),
                unit="samples",
                overall_fraction=(
                    progress_start
                    + (progress_end - progress_start)
                    * completed_count / max(len(conversations), 1)
                ),
                force=completed_count == len(conversations),
            )

    _finalize_predictions(run_dir, results, config, reporter, progress_end)


if __name__ == "__main__":
    main()
