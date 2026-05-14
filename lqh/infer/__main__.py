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

    try:
        _run_inference(run_dir, config)
    except Exception as exc:
        from lqh.train.progress import write_status

        write_status(run_dir, "failed", error=str(exc))
        raise


def _run_inference(run_dir: Path, config: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lqh.train.data_utils import load_chatml_dataset_with_tools
    from lqh.train.progress import write_eval_request, write_progress, write_status
    from lqh.train.tool_format import get_tool_formatter

    base_model = config["base_model"]
    dataset_path = config["dataset"]

    print(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading dataset: {dataset_path}")
    conversations, tools_per_sample = load_chatml_dataset_with_tools(dataset_path)

    max_new_tokens = config.get("max_new_tokens", 4096)
    system_prompt = config.get("system_prompt")
    response_format = config.get("response_format")
    predictions: list[dict] = []

    # Get the tool formatter for this model (if applicable)
    tool_formatter = get_tool_formatter(base_model)

    if system_prompt:
        print(f"System prompt: {system_prompt[:80]}...")

    # JSON-schema constrained decoding via lm-format-enforcer.
    # We accept either the bare JSON schema or the OpenAI-style envelope
    # ({"type":"json_schema","json_schema":{"schema": {...}}}) so the same
    # prompts/<task>.schema.json file works for both API and local eval.
    #
    # If ``response_format`` is set in the config, we hard-fail on setup
    # errors rather than silently falling back to free-form decoding —
    # silent fallback once produced 200 invalid-JSON predictions on a
    # constrained eval before anyone noticed.
    schema_prefix_fn = None
    if response_format:
        inner_schema: Any = response_format
        if isinstance(response_format, dict):
            if "json_schema" in response_format:
                js = response_format["json_schema"]
                if isinstance(js, dict) and "schema" in js:
                    inner_schema = js["schema"]
                else:
                    inner_schema = js

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
            tokenizer, parser,
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
        sample_tools = tools_per_sample[i]

        # Strip trailing assistant turn
        prompt_msgs = list(conv)
        if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
            prompt_msgs = prompt_msgs[:-1]

        # Prepend system prompt if configured and not already present
        if system_prompt and (not prompt_msgs or prompt_msgs[0].get("role") != "system"):
            prompt_msgs = [{"role": "system", "content": system_prompt}] + prompt_msgs

        try:
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
            assistant_msg: dict[str, Any] = {"role": "assistant"}
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
        }
        if sample_tools is not None:
            pred_entry["tools"] = json.dumps(sample_tools)
        predictions.append(pred_entry)

        if (i + 1) % 10 == 0 or i == len(conversations) - 1:
            print(f"  {i + 1}/{len(conversations)} samples done")
            write_progress(
                run_dir,
                step=i + 1,
                extra={"phase": "inference", "total": len(conversations)},
            )

    # Write predictions
    import pyarrow as pa
    import pyarrow.parquet as pq

    has_tools_col = any("tools" in p for p in predictions)
    columns: dict[str, list] = {
        "sample_index": [p["sample_index"] for p in predictions],
        "messages": [p["messages"] for p in predictions],
    }
    fields = [
        pa.field("sample_index", pa.int64()),
        pa.field("messages", pa.string()),
    ]
    if has_tools_col:
        columns["tools"] = [p.get("tools") for p in predictions]
        fields.append(pa.field("tools", pa.string()))

    table = pa.table(columns, schema=pa.schema(fields))
    pq.write_table(table, run_dir / "predictions.parquet")

    # Signal for scoring
    write_eval_request(run_dir)
    write_status(run_dir, "completed")
    print(f"Inference complete: {len(predictions)} predictions written")


if __name__ == "__main__":
    main()
