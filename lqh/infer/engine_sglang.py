"""sglang-backed eval generation engine (ISSUE 4 P1).

Runs an ephemeral ``sglang.launch_server`` inside the eval sandbox and
drives it with bounded-concurrency OpenAI-API requests, replacing the
sequential HF ``model.generate`` loop for cloud eval_hf jobs. Only
active on images where the sglang package is importable (the gpu_eval
image); every other environment keeps the HF loop via the dispatcher in
``lqh.infer.__main__._run_inference``.

Contract parity with the HF engine (both share the helpers in
``lqh.infer.__main__``):
  - identical prompt prep (``_prompt_messages``) and predictions
    output (``_finalize_predictions`` → sample_index/messages/source
    [/tools] parquet — the scoring contract),
  - identical partial-file format and digest, so a continuation may
    resume a partial written by either engine,
  - greedy decoding (temperature=0), same max_new_tokens,
  - response_format enforced server-side (xgrammar json_schema) on the
    same inner schema lm-format-enforcer would constrain locally; a
    request rejected by the server is FATAL, never N in-band error rows,
  - LFM tool calls come back from the server's ``--tool-call-parser
    lfm2`` as native tool_calls and are converted to the exact dict
    shape ``LFM2ToolFormatter.parse_tool_calls`` produces.

LoRA checkpoints are merged to disk first (sglang serves full weights
only — same merge-and-serve stance as the inference pods): a CPU child
process runs ``lqh.remote.merge_lora._merge`` so its memory is fully
released before the server claims the GPU.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SGLANG_PORT = 30000
SGLANG_BASE_URL = f"http://127.0.0.1:{SGLANG_PORT}/v1"
SERVED_MODEL_NAME = "lqh-eval"
BOOT_TIMEOUT_SEC = 600
# Generous per-request ceiling: one 8k-token greedy completion on a
# saturated L4 batch. Requests don't time out individually below this;
# the job-level wall clock is the real budget.
REQUEST_TIMEOUT_SEC = 900.0
DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 32


def sglang_available() -> bool:
    return importlib.util.find_spec("sglang") is not None


def _prepare_model_path(config: dict) -> tuple[str, tempfile.TemporaryDirectory | None]:
    """Resolve the on-disk weights sglang will serve.

    Full checkpoints (and hub ids) pass through untouched. A LoRA
    adapter dir is merged onto its base into a container-local temp dir
    — sglang has no use for a bare adapter, and merged weights are
    re-derivable so they don't belong on the durable volume. Returns
    the model path plus the TemporaryDirectory keeping it alive (None
    when nothing was merged).
    """
    base_model = str(config["base_model"])
    adapter_cfg = Path(base_model) / "adapter_config.json"
    if not adapter_cfg.exists():
        return base_model, None

    base = config.get("base_override")
    if not base:
        try:
            base = json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable adapter_config.json: {exc}") from exc
    if not base:
        raise RuntimeError(
            "LoRA eval needs a base model: adapter_config.json has no "
            "base_model_name_or_path and no base_override was configured"
        )

    tmp = tempfile.TemporaryDirectory(prefix="lqh-merged-")
    out_dir = Path(tmp.name) / "merged"
    out_dir.mkdir()
    print(f"Merging LoRA adapter onto {base} (cpu child process) ...")
    code = (
        "import sys; from pathlib import Path; "
        "from lqh.remote.merge_lora import _merge; "
        "_merge(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(base), base_model, str(out_dir)],
        check=False,
    )
    if proc.returncode != 0:
        tmp.cleanup()
        raise RuntimeError(
            f"LoRA merge failed (exit {proc.returncode}); see log above"
        )
    return str(out_dir), tmp


class _SglangServer:
    """Popen wrapper for the in-sandbox sglang server.

    stdout/stderr go to ``run_dir/logs/sglang_server.log`` — the
    launcher treats sandbox stdout as job progress, and sglang's
    per-request logging would drown it.
    """

    def __init__(self, model_path: str, run_dir: Path, extra_args: str = "") -> None:
        log_dir = run_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_path = log_dir / "sglang_server.log"
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "127.0.0.1",
            "--port", str(SGLANG_PORT),
            "--served-model-name", SERVED_MODEL_NAME,
            "--tool-call-parser", "lfm2",
        ]
        if extra_args:
            cmd += shlex.split(extra_args)
        print(f"Starting sglang server: {' '.join(cmd)}")
        self._log_fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=self._log_fh)

    def _log_tail(self, max_bytes: int = 8192) -> str:
        try:
            data = self.log_path.read_bytes()
            return data[-max_bytes:].decode(errors="replace")
        except OSError:
            return "<no server log>"

    def raise_if_dead(self) -> None:
        rc = self.proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"sglang server exited (code {rc}); log tail:\n{self._log_tail()}"
            )

    def wait_healthy(self, timeout: float = BOOT_TIMEOUT_SEC) -> None:
        """Poll /health_generate until 200 (the readiness contract the
        serve harness uses), failing fast if the child dies mid-boot.
        """
        import httpx

        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{SGLANG_PORT}/health_generate"
        last_err = ""
        while time.monotonic() < deadline:
            self.raise_if_dead()
            try:
                resp = httpx.get(url, timeout=10.0)
                if resp.status_code == 200:
                    print("sglang server healthy")
                    return
                last_err = f"HTTP {resp.status_code}"
            except Exception as exc:  # connection refused while booting
                last_err = str(exc)[:200]
            time.sleep(3)
        raise RuntimeError(
            f"sglang server not healthy within {timeout:.0f}s "
            f"(last: {last_err}); log tail:\n{self._log_tail()}"
        )

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        try:
            self._log_fh.close()
        except OSError:
            pass


def _tool_calls_to_dicts(tool_calls: Any) -> list[dict]:
    """Convert openai-SDK tool_call objects to the exact plain-dict
    shape ``LFM2ToolFormatter.parse_tool_calls`` emits, so scoring sees
    an identical predictions schema from both engines.
    """
    out: list[dict] = []
    for i, tc in enumerate(tool_calls):
        fn = tc.function
        out.append({
            "id": getattr(tc, "id", None) or f"call_{fn.name}_{i}",
            "type": "function",
            "function": {"name": fn.name, "arguments": fn.arguments or ""},
        })
    return out


def _build_request_kwargs(
    prompt_msgs: list[dict],
    sample_tools: list | None,
    max_new_tokens: int,
    response_format: Any,
) -> dict:
    kwargs: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": prompt_msgs,
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    if sample_tools is not None:
        kwargs["tools"] = sample_tools
    if response_format:
        from lqh.infer.__main__ import _normalize_inner_schema

        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "lqh_schema",
                "schema": _normalize_inner_schema(response_format),
                "strict": True,
            },
        }
    return kwargs


class _FatalGenerationError(RuntimeError):
    """Raised when the server rejects a request shape (4xx) or dies —
    conditions where retrying per-sample would only manufacture N
    garbage rows. Fails the whole run instead (same stance as the HF
    loop's hard-fail on lm-format-enforcer setup errors).
    """


async def _generate_one(
    client: Any,
    server: _SglangServer,
    kwargs: dict,
) -> dict:
    """One sample → the assistant message dict. Transient errors retry
    twice; persistent transport errors check whether the server died
    (fatal) before degrading to an in-band error entry.
    """
    import openai

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = _tool_calls_to_dicts(msg.tool_calls)
            return assistant_msg
        except openai.BadRequestError as exc:
            raise _FatalGenerationError(
                f"sglang rejected the request shape (HTTP 400): {exc}"
            ) from exc
        except (openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as exc:
            last_exc = exc
            try:
                server.raise_if_dead()
            except RuntimeError as dead:
                raise _FatalGenerationError(str(dead)) from exc
            if attempt < 2:
                await asyncio.sleep(1 + 2 * attempt)
        except Exception as exc:  # per-sample, non-transport
            return {"role": "assistant", "content": f"[generation error: {exc}]"}
    return {"role": "assistant", "content": f"[generation error: {last_exc}]"}


def run_inference_sglang(run_dir: Path, config: dict) -> None:
    """Same contract as ``_run_inference_hf``: incremental partial
    appends, predictions.parquet + eval_request.json via the shared
    finalizer, terminal status deferred when the caller owns it.
    """
    from lqh.infer.__main__ import (
        PREDICTIONS_PARTIAL,
        _finalize_predictions,
        _init_prediction_partial,
        _predictions_digest,
        _prompt_messages,
    )
    from lqh.progress import ProgressReporter
    from lqh.train.data_utils import load_eval_sources_with_tools

    base_model = config["base_model"]
    dataset_path = config["dataset"]
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

    print(f"Loading dataset: {dataset_path}")
    conversations, tools_per_sample, sources_per_sample = (
        load_eval_sources_with_tools(dataset_path)
    )
    total = len(conversations)

    max_new_tokens = int(config.get("max_new_tokens", 4096))
    system_prompt = config.get("system_prompt")
    response_format = config.get("response_format")
    concurrency = max(1, min(
        int(config.get("generation_concurrency", DEFAULT_CONCURRENCY)),
        MAX_CONCURRENCY,
    ))

    partial_path = run_dir / PREDICTIONS_PARTIAL
    resumed = _init_prediction_partial(run_dir, total, _predictions_digest(config))
    results: list[dict | None] = [None] * total
    for idx, entry in resumed.items():
        results[idx] = entry
    if resumed:
        print(f"Resuming: {len(resumed)}/{total} predictions already done")
        reporter.update(
            phase="inference", phase_label="running inference",
            completed=len(resumed), total=total, unit="samples",
            overall_fraction=(
                progress_start
                + (progress_end - progress_start) * len(resumed) / max(total, 1)
            ),
            force=True,
        )

    pending = [i for i in range(total) if i not in resumed]
    if not pending:
        _finalize_predictions(run_dir, results, config, reporter, progress_end)
        return

    model_path, merged_tmp = _prepare_model_path(config)
    server = _SglangServer(
        model_path, run_dir, extra_args=str(config.get("sglang_extra_args", "")),
    )
    try:
        server.wait_healthy()
        reporter.update(
            phase="inference", phase_label="running inference",
            completed=len(resumed), total=total, unit="samples",
            overall_fraction=(
                progress_start
                + (progress_end - progress_start) * len(resumed) / max(total, 1)
            ),
            force=True,
        )
        print(
            f"Generating {len(pending)}/{total} samples "
            f"(concurrency {concurrency}, greedy, max_tokens {max_new_tokens})"
        )
        _run_generation(
            server=server,
            config=config,
            run_dir=run_dir,
            pending=pending,
            conversations=conversations,
            tools_per_sample=tools_per_sample,
            sources_per_sample=sources_per_sample,
            results=results,
            resumed_count=len(resumed),
            partial_path=partial_path,
            reporter=reporter,
            progress_start=progress_start,
            progress_end=progress_end,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            response_format=response_format,
            concurrency=concurrency,
            prompt_messages=_prompt_messages,
        )
    finally:
        server.terminate()
        if merged_tmp is not None:
            merged_tmp.cleanup()

    _finalize_predictions(run_dir, results, config, reporter, progress_end)


def _first_leaf_exception(eg: BaseExceptionGroup) -> BaseException:
    for exc in eg.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            return _first_leaf_exception(exc)
        return exc
    return eg


def _run_generation(**kwargs: Any) -> None:
    """asyncio.run wrapper that unwraps the TaskGroup's ExceptionGroup —
    the terminal status text (and thus the job error the user sees)
    carries the actual failure, not an 'unhandled errors in a
    TaskGroup' envelope.
    """
    try:
        asyncio.run(_generate_all(**kwargs))
    except BaseExceptionGroup as eg:
        raise _first_leaf_exception(eg) from eg


async def _generate_all(
    *,
    server: _SglangServer,
    config: dict,
    run_dir: Path,
    pending: list[int],
    conversations: list,
    tools_per_sample: list,
    sources_per_sample: list,
    results: list[dict | None],
    resumed_count: int,
    partial_path: Path,
    reporter: Any,
    progress_start: float,
    progress_end: float,
    system_prompt: str | None,
    max_new_tokens: int,
    response_format: Any,
    concurrency: int,
    prompt_messages: Any,
) -> None:
    import openai

    from lqh.infer.__main__ import _append_prediction_partial
    from lqh.train.progress import write_progress

    total = len(conversations)
    completed_count = resumed_count
    client = openai.AsyncOpenAI(
        base_url=SGLANG_BASE_URL,
        api_key="unused",
        timeout=REQUEST_TIMEOUT_SEC,
        max_retries=0,
    )
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        nonlocal completed_count
        sample_tools = tools_per_sample[i]
        prompt_msgs = prompt_messages(conversations[i], system_prompt)
        kwargs = _build_request_kwargs(
            prompt_msgs, sample_tools, max_new_tokens, response_format,
        )
        async with sem:
            assistant_msg = await _generate_one(client, server, kwargs)

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

        if completed_count % 10 == 0 or completed_count == total:
            print(f"  {completed_count}/{total} samples done")
            write_progress(
                run_dir,
                step=completed_count,
                extra={"phase": "inference", "total": total},
                emit_cloud=False,
            )
            reporter.update(
                phase="inference", phase_label="running inference",
                completed=completed_count, total=total, unit="samples",
                overall_fraction=(
                    progress_start
                    + (progress_end - progress_start)
                    * completed_count / max(total, 1)
                ),
                force=completed_count == total,
            )

    try:
        # TaskGroup cancels every sibling on the first fatal error, so a
        # 400/server-death aborts the run instead of burning the budget.
        async with asyncio.TaskGroup() as tg:
            for i in pending:
                tg.create_task(one(i))
    finally:
        await client.close()
