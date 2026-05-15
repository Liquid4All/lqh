"""Experiment: API vs local-HF JSON-constrained inference parity.

Recent evals show a quality gap on the same model and same JSON schema
between the lqh API (clean, ~7.8/10) and local HF inference with
lm-format-enforcer (~6.93/10 SFT, ~4.83/10 base). This script runs the
same inputs through both paths and surfaces *where* they diverge so we
can chase the root cause (sampling params? chat template? schema
interpretation?).

For each sample it prints:
  • the exact prompt string both paths see (apply_chat_template result)
  • API output, local output, and the first-divergence character index
  • JSON parse status of each
  • (with --score) per-sample judge scores

At the end: aggregate identical-output rate, mean char-edit distance,
and (with --score) the mean judge-score gap.

Usage::

    python -m tests.remote.experiment_inference_parity \
        --dataset example_project/datasets/business_chat_translation_v1_eval/data.parquet \
        --schema example_project/prompts/business_chat_translation.schema.json \
        --system-prompt example_project/prompts/business_chat_translation_v0.md \
        --num-samples 10 \
        --score

Run on a machine with a GPU (or CPU + patience) and a valid lqh token
in ``~/.lqh/config.json`` or ``LQH_API_KEY``. The HF model load needs
``HF_TOKEN`` if the model is gated.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


@dataclass
class SampleResult:
    sample_index: int
    prompt_text: str
    api_output: str
    local_output: str
    api_parse_ok: bool
    local_parse_ok: bool
    first_diverge: int  # char index of first differing character, or -1 if identical
    api_score: float | None = None
    local_score: float | None = None


def _load_samples(dataset_path: Path, n: int) -> list[list[dict[str, Any]]]:
    """Read N user-side message lists from a parquet eval dataset.

    The eval parquet has a ``messages`` column where each row is a JSON
    string of the full conversation (user + assistant). We strip the
    trailing assistant turn — that's what the model is supposed to
    produce.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path)
    rows: list[list[dict[str, Any]]] = []
    for i in range(min(n, table.num_rows)):
        raw = table["messages"][i].as_py()
        msgs = json.loads(raw) if isinstance(raw, str) else raw
        if msgs and msgs[-1].get("role") == "assistant":
            msgs = msgs[:-1]
        rows.append(msgs)
    return rows


def _load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text())


def _inner_schema(response_format: dict[str, Any]) -> dict[str, Any]:
    """Extract the bare JSON schema from the OpenAI envelope, if present."""
    if "json_schema" in response_format:
        js = response_format["json_schema"]
        if isinstance(js, dict) and "schema" in js:
            return js["schema"]
        return js
    return response_format


async def _run_api(
    messages: list[dict[str, Any]],
    *,
    model: str,
    response_format: dict[str, Any],
    max_tokens: int,
) -> str:
    """Run inference via the lqh API (OpenAI-compatible)."""
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    if not api_key:
        raise RuntimeError(
            "No lqh API key found. Set LQH_API_KEY or run `lqh /login`."
        )
    client = create_client(api_key, load_config().api_base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return response.choices[0].message.content or ""


def _build_local_runtime(model_id: str, schema: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Load HF model + tokenizer + lmfe prefix_allowed_tokens_fn.

    Returns ``(model, tokenizer, prefix_fn)``. Loading is heavy so we
    do it once per script invocation and reuse across samples.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Same lmfe shim as lqh/infer/__main__.py: lmfe ≤0.11.3 imports from
    # the v4 path. Patch before integration import.
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

    print(f"  Loading model: {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    parser = JsonSchemaParser(schema)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return model, tokenizer, prefix_fn


def _run_local(
    messages: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    prefix_fn: Any,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Run local HF inference with lmfe constraint. Returns (prompt_text, output)."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
        )
    output = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return prompt_text, output


def _first_diverge(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _try_parse_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


async def _score_sample(
    sample: list[dict[str, Any]],
    candidate: str,
    *,
    scorer_text: str,
    judge_model: str,
) -> float | None:
    """Run the API judge on a single (input, candidate) pair.

    Builds the same scoring prompt and uses the same SCORE_RESPONSE_SCHEMA
    that ``lqh.scoring`` uses for production evals, so scores are
    directly comparable to what users see in their eval reports.
    Returns the numeric score, or None if scoring failed.
    """
    from lqh.auth import get_token
    from lqh.client import create_client
    from lqh.config import load_config
    from lqh.scoring import (
        SCORE_RESPONSE_SCHEMA,
        _build_scoring_prompt,
        _parse_score_response,
    )

    api_key = os.environ.get("LQH_API_KEY") or get_token() or ""
    client = create_client(api_key, load_config().api_base_url)

    # Strip trailing assistant from the input sample (we'll add ours)
    msgs = [m for m in sample if m.get("role") != "assistant"]
    scored = msgs + [{"role": "assistant", "content": candidate}]

    prompt = _build_scoring_prompt(scorer_text, scored)
    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=prompt,
            temperature=0.0,
            response_format=SCORE_RESPONSE_SCHEMA,
        )
        raw = response.choices[0].message.content or ""
        score, _ = _parse_score_response(raw)
        return score
    except Exception as exc:
        print(f"    judge error: {exc}", flush=True)
        return None


async def _process_sample(
    idx: int,
    messages: list[dict[str, Any]],
    *,
    model_id: str,
    response_format: dict[str, Any],
    max_tokens: int,
    local_runtime: tuple[Any, Any, Any],
    scorer_text: str | None,
    judge_model: str,
) -> SampleResult:
    print(f"\n[sample {idx}] {len(messages)} messages", flush=True)

    # API path (no system prompt is prepended here; the caller already has it).
    api_output = await _run_api(
        messages,
        model=model_id,
        response_format=response_format,
        max_tokens=max_tokens,
    )

    # Local path
    model, tokenizer, prefix_fn = local_runtime
    prompt_text, local_output = _run_local(
        messages,
        model=model, tokenizer=tokenizer, prefix_fn=prefix_fn,
        max_new_tokens=max_tokens,
    )

    diverge = _first_diverge(api_output, local_output)
    result = SampleResult(
        sample_index=idx,
        prompt_text=prompt_text,
        api_output=api_output,
        local_output=local_output,
        api_parse_ok=_try_parse_json(api_output),
        local_parse_ok=_try_parse_json(local_output),
        first_diverge=diverge,
    )

    if scorer_text is not None:
        result.api_score = await _score_sample(
            messages, api_output,
            scorer_text=scorer_text, judge_model=judge_model,
        )
        result.local_score = await _score_sample(
            messages, local_output,
            scorer_text=scorer_text, judge_model=judge_model,
        )

    return result


def _print_sample(r: SampleResult) -> None:
    print("  prompt (chat-templated, last 240 chars):")
    print(f"    …{r.prompt_text[-240:]!r}")
    print(f"  API   ({'json✓' if r.api_parse_ok else 'json✗'}): {r.api_output!r}")
    print(f"  local ({'json✓' if r.local_parse_ok else 'json✗'}): {r.local_output!r}")
    if r.first_diverge < 0:
        print("  ⇒ outputs IDENTICAL")
    else:
        i = r.first_diverge
        before = r.api_output[max(0, i - 20):i]
        api_after = r.api_output[i:i + 20]
        loc_after = r.local_output[i:i + 20]
        print(f"  ⇒ diverge @ char {i}: …{before!r}")
        print(f"    API   continues: {api_after!r}")
        print(f"    local continues: {loc_after!r}")
    if r.api_score is not None or r.local_score is not None:
        print(
            f"  scores: API={r.api_score}  local={r.local_score}  "
            f"gap={(r.api_score or 0) - (r.local_score or 0):+.2f}"
        )


def _print_summary(results: list[SampleResult]) -> None:
    n = len(results)
    if n == 0:
        return
    identical = sum(1 for r in results if r.first_diverge < 0)
    api_parse = sum(1 for r in results if r.api_parse_ok)
    loc_parse = sum(1 for r in results if r.local_parse_ok)
    avg_edit = (
        sum(
            sum(d.size for d in difflib.SequenceMatcher(
                None, r.api_output, r.local_output,
            ).get_matching_blocks())
            / max(len(r.api_output), len(r.local_output), 1)
            for r in results
        )
        / n
    )
    print(f"\n=== summary over {n} samples ===")
    print(f"  identical outputs:   {identical}/{n}")
    print(f"  API json-parse OK:   {api_parse}/{n}")
    print(f"  local json-parse OK: {loc_parse}/{n}")
    print(f"  mean char similarity: {avg_edit:.3f} (1.0 = identical)")
    api_scored = [r.api_score for r in results if r.api_score is not None]
    loc_scored = [r.local_score for r in results if r.local_score is not None]
    if api_scored and loc_scored:
        api_mean = sum(api_scored) / len(api_scored)
        loc_mean = sum(loc_scored) / len(loc_scored)
        print(
            f"  judge: API mean={api_mean:.2f}  local mean={loc_mean:.2f}  "
            f"gap={api_mean - loc_mean:+.2f}"
        )


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, type=Path,
                   help="Path to eval parquet (with a 'messages' column)")
    p.add_argument("--schema", required=True, type=Path,
                   help="Path to JSON schema file (OpenAI envelope or bare schema)")
    p.add_argument("--system-prompt", type=Path, default=None,
                   help="Optional path to a system-prompt markdown file")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id for local inference (default {DEFAULT_MODEL})")
    p.add_argument("--api-model", default=None,
                   help="API model name (defaults to HF id lower-cased without "
                        "the 'LiquidAI/' prefix, e.g. lfm2.5-1.2b-instruct)")
    p.add_argument("--judge-model", default="judge:small",
                   help="Model name for the API judge when --score is set "
                        "(judge:small/medium/large). Matches lqh.scoring's "
                        "DEFAULT_JUDGE_MODEL_SIZE.")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Max generation length per sample. Default 8192 "
                        "because lmfe-constrained outputs are noticeably "
                        "longer than free-form for the same content.")
    p.add_argument("--score", action="store_true",
                   help="Also score each pair with the API judge")
    p.add_argument("--scorer", type=Path, default=None,
                   help="Required if --score is set: path to scorer markdown")
    args = p.parse_args()

    if args.score and args.scorer is None:
        print("--score requires --scorer", file=sys.stderr)
        return 2

    api_model = args.api_model or args.model.split("/", 1)[-1].lower()
    print(f"  HF model:  {args.model}")
    print(f"  API model: {api_model}")

    samples = _load_samples(args.dataset, args.num_samples)
    schema_envelope = _load_schema(args.schema)
    inner = _inner_schema(schema_envelope)

    system_prompt: str | None = None
    if args.system_prompt is not None:
        system_prompt = args.system_prompt.read_text()

    if system_prompt:
        for s in samples:
            if not s or s[0].get("role") != "system":
                s.insert(0, {"role": "system", "content": system_prompt})

    scorer_text: str | None = None
    if args.score:
        scorer_text = args.scorer.read_text()  # type: ignore[union-attr]

    print(f"loading local runtime ({args.model})…", flush=True)
    local_runtime = _build_local_runtime(args.model, inner)

    results: list[SampleResult] = []
    for i, msgs in enumerate(samples):
        try:
            r = await _process_sample(
                i, msgs,
                model_id=api_model,
                response_format=schema_envelope,
                max_tokens=args.max_tokens,
                local_runtime=local_runtime,
                scorer_text=scorer_text,
                judge_model=args.judge_model,
            )
        except Exception as exc:
            print(f"[sample {i}] error: {exc}", flush=True)
            continue
        _print_sample(r)
        results.append(r)

    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
