"""sglang eval engine (ISSUE 4 P1): dispatcher selection, request
mapping parity with the HF loop, tool-call shape conversion, error
taxonomy, partial-resume across engines, and server lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from lqh.infer import engine_sglang
from lqh.infer.__main__ import (
    PREDICTIONS_PARTIAL,
    _append_prediction_partial,
    _init_prediction_partial,
    _predictions_digest,
    _run_inference,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path: Path, n: int, *, with_tools: bool = False) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = tmp_path / "evals" / "x"
    ds_dir.mkdir(parents=True)
    messages = [
        json.dumps([
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"reference {i}"},
        ])
        for i in range(n)
    ]
    cols: dict[str, list] = {"messages": messages}
    if with_tools:
        tool = [{"type": "function", "function": {"name": "get_weather",
                 "parameters": {"type": "object"}}}]
        cols["tools"] = [json.dumps(tool)] * n
    pq.write_table(pa.table(cols), ds_dir / "data.parquet")
    return str(ds_dir)


class _FakeServer:
    instances: list["_FakeServer"] = []

    def __init__(self, model_path: str, run_dir: Path, extra_args: str = "") -> None:
        self.model_path = model_path
        self.extra_args = extra_args
        _FakeServer.instances.append(self)

    def wait_healthy(self, timeout: float = 0) -> None:
        pass

    def raise_if_dead(self) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True


class _FakeClient:
    created: list["_FakeClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )
        _FakeClient.created.append(self)

    async def _create(self, **kwargs: Any):
        self.calls.append(kwargs)
        await asyncio.sleep(0)  # let completions interleave out of order
        user = kwargs["messages"][-1]["content"]
        msg = SimpleNamespace(content=f"echo:{user}", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeServer.instances = []
    _FakeClient.created = []
    yield


def _engine_config(dataset: str, **extra: Any) -> dict:
    return {
        "base_model": "org/model",
        "dataset": dataset,
        "max_new_tokens": 64,
        **extra,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_selects_sglang_when_available(tmp_path, monkeypatch) -> None:
    called = {}
    monkeypatch.setattr(engine_sglang, "sglang_available", lambda: True)
    monkeypatch.setattr(
        engine_sglang, "run_inference_sglang",
        lambda run_dir, config: called.setdefault("sglang", (run_dir, config)),
    )
    _run_inference(tmp_path, {"base_model": "m", "dataset": "d"})
    assert "sglang" in called


def test_dispatcher_hf_when_sglang_unavailable(tmp_path, monkeypatch) -> None:
    import lqh.infer.__main__ as infer_main

    called = {}
    monkeypatch.setattr(engine_sglang, "sglang_available", lambda: False)
    monkeypatch.setattr(
        infer_main, "_run_inference_hf",
        lambda run_dir, config: called.setdefault("hf", True),
    )
    _run_inference(tmp_path, {"base_model": "m", "dataset": "d"})
    assert called == {"hf": True}


def test_dispatcher_force_hf_engine_wins(tmp_path, monkeypatch) -> None:
    import lqh.infer.__main__ as infer_main

    called = {}
    monkeypatch.setattr(engine_sglang, "sglang_available", lambda: True)
    monkeypatch.setattr(
        infer_main, "_run_inference_hf",
        lambda run_dir, config: called.setdefault("hf", True),
    )
    _run_inference(tmp_path, {"base_model": "m", "dataset": "d",
                              "force_hf_engine": True})
    assert called == {"hf": True}


def test_sglang_available_reflects_find_spec(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "sglang" else None,
    )
    assert engine_sglang.sglang_available() is True
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert engine_sglang.sglang_available() is False


# ---------------------------------------------------------------------------
# Request mapping
# ---------------------------------------------------------------------------


def test_request_kwargs_greedy_no_tools() -> None:
    kwargs = engine_sglang._build_request_kwargs(
        [{"role": "user", "content": "q"}], None, 128, None,
    )
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 128
    assert kwargs["model"] == engine_sglang.SERVED_MODEL_NAME
    assert "tools" not in kwargs
    assert "response_format" not in kwargs


def test_request_kwargs_tools_only_when_present() -> None:
    tools = [{"type": "function", "function": {"name": "f"}}]
    kwargs = engine_sglang._build_request_kwargs(
        [{"role": "user", "content": "q"}], tools, 64, None,
    )
    assert kwargs["tools"] is tools


@pytest.mark.parametrize("rf", [
    # bare schema
    {"type": "object", "properties": {"a": {"type": "string"}}},
    # OpenAI envelope with schema key
    {"type": "json_schema",
     "json_schema": {"schema": {"type": "object",
                                "properties": {"a": {"type": "string"}}}}},
    # envelope where json_schema IS the schema
    {"json_schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
])
def test_request_kwargs_schema_normalization(rf: dict) -> None:
    kwargs = engine_sglang._build_request_kwargs(
        [{"role": "user", "content": "q"}], None, 64, rf,
    )
    sent = kwargs["response_format"]
    assert sent["type"] == "json_schema"
    js = sent["json_schema"]
    assert js["name"] == "lqh_schema" and js["strict"] is True
    assert js["schema"] == {"type": "object",
                            "properties": {"a": {"type": "string"}}}


def test_tool_calls_to_dicts_exact_shape() -> None:
    tcs = [
        SimpleNamespace(
            id="call_abc123",
            function=SimpleNamespace(name="get_weather",
                                     arguments='{"city": "Paris"}'),
        ),
        SimpleNamespace(id=None, function=SimpleNamespace(name="f2", arguments=None)),
    ]
    out = engine_sglang._tool_calls_to_dicts(tcs)
    # Exact LFM2ToolFormatter dict shape — the scoring contract.
    assert out[0] == {"id": "call_abc123", "type": "function",
                      "function": {"name": "get_weather",
                                   "arguments": '{"city": "Paris"}'}}
    assert out[1] == {"id": "call_f2_1", "type": "function",
                      "function": {"name": "f2", "arguments": ""}}


# ---------------------------------------------------------------------------
# Error taxonomy (_generate_one)
# ---------------------------------------------------------------------------


def _client_raising(exc_factory):
    class C:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kwargs):
            self.calls += 1
            raise exc_factory()

    return C()


def _bad_request() -> openai.BadRequestError:
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x/v1"))
    return openai.BadRequestError("bad schema", response=resp, body=None)


def _conn_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))


def test_generate_one_bad_request_is_fatal(monkeypatch) -> None:
    client = _client_raising(_bad_request)
    server = _FakeServer("m", Path("."))
    with pytest.raises(engine_sglang._FatalGenerationError):
        asyncio.run(engine_sglang._generate_one(client, server, {"messages": []}))
    assert client.calls == 1  # no retry on 400


def test_generate_one_transient_retries_then_inband(monkeypatch) -> None:
    async def instant(_delay):
        pass

    monkeypatch.setattr(engine_sglang.asyncio, "sleep", instant)
    client = _client_raising(_conn_error)
    server = _FakeServer("m", Path("."))
    msg = asyncio.run(engine_sglang._generate_one(client, server, {"messages": []}))
    assert client.calls == 3
    assert msg["role"] == "assistant"
    assert msg["content"].startswith("[generation error:")


def test_generate_one_dead_server_is_fatal(monkeypatch) -> None:
    client = _client_raising(_conn_error)
    server = _FakeServer("m", Path("."))
    monkeypatch.setattr(
        server, "raise_if_dead",
        lambda: (_ for _ in ()).throw(RuntimeError("sglang server exited (code 1)")),
    )
    with pytest.raises(engine_sglang._FatalGenerationError, match="exited"):
        asyncio.run(engine_sglang._generate_one(client, server, {"messages": []}))
    assert client.calls == 1


def test_generate_one_other_errors_go_inband() -> None:
    client = _client_raising(lambda: ValueError("weird payload"))
    server = _FakeServer("m", Path("."))
    msg = asyncio.run(engine_sglang._generate_one(client, server, {"messages": []}))
    assert "weird payload" in msg["content"]


# ---------------------------------------------------------------------------
# run_inference_sglang end-to-end (faked server + client)
# ---------------------------------------------------------------------------


def _run_engine(tmp_path, monkeypatch, config) -> Path:
    monkeypatch.setattr(engine_sglang, "_SglangServer", _FakeServer)
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    engine_sglang.run_inference_sglang(run_dir, config)
    return run_dir


def test_run_engine_writes_ordered_parquet(tmp_path, monkeypatch) -> None:
    import pyarrow.parquet as pq

    dataset = _make_dataset(tmp_path, 12)
    run_dir = _run_engine(tmp_path, monkeypatch, _engine_config(dataset))

    table = pq.read_table(run_dir / "predictions.parquet")
    assert table.column("sample_index").to_pylist() == list(range(12))
    row0 = json.loads(table.column("messages").to_pylist()[0])
    # Full conv: prompt (reference stripped) + generated turn.
    assert row0 == [
        {"role": "user", "content": "question 0"},
        {"role": "assistant", "content": "echo:question 0"},
    ]
    assert (run_dir / "eval_request.json").exists()
    # No defer flag → engine owns cleanup + completed status.
    assert not (run_dir / PREDICTIONS_PARTIAL).exists()
    assert len(_FakeClient.created) == 1
    assert len(_FakeClient.created[0].calls) == 12


def test_run_engine_resumes_partial_and_defers_status(tmp_path, monkeypatch) -> None:
    import pyarrow.parquet as pq

    dataset = _make_dataset(tmp_path, 6)
    config = _engine_config(dataset, defer_terminal_status=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A prior attempt (either engine — shared format) completed 0 and 3.
    digest = _predictions_digest(config)
    _init_prediction_partial(run_dir, 6, digest)
    for i in (0, 3):
        _append_prediction_partial(
            run_dir / PREDICTIONS_PARTIAL, i,
            {"sample_index": i,
             "messages": json.dumps([{"role": "user", "content": f"question {i}"},
                                     {"role": "assistant", "content": f"old {i}"}]),
             "source": "x"},
        )

    _run_engine(tmp_path, monkeypatch, config)

    assert len(_FakeClient.created[0].calls) == 4  # only the missing samples
    table = pq.read_table(run_dir / "predictions.parquet")
    msgs = [json.loads(m)[-1]["content"] for m in table.column("messages").to_pylist()]
    assert msgs[0] == "old 0" and msgs[3] == "old 3"  # resumed rows kept
    assert msgs[1] == "echo:question 1"
    # defer_terminal_status → the partial survives for the scoring phase.
    assert (run_dir / PREDICTIONS_PARTIAL).exists()


def test_run_engine_all_resumed_skips_server(tmp_path, monkeypatch) -> None:
    dataset = _make_dataset(tmp_path, 2)
    config = _engine_config(dataset)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    digest = _predictions_digest(config)
    _init_prediction_partial(run_dir, 2, digest)
    for i in range(2):
        _append_prediction_partial(
            run_dir / PREDICTIONS_PARTIAL, i,
            {"sample_index": i, "messages": json.dumps([]), "source": "x"},
        )

    _run_engine(tmp_path, monkeypatch, config)

    assert _FakeServer.instances == []  # no GPU server for a no-op attempt
    assert (run_dir / "predictions.parquet").exists()


def test_run_engine_tools_forwarded_and_stored(tmp_path, monkeypatch) -> None:
    import pyarrow.parquet as pq

    dataset = _make_dataset(tmp_path, 3, with_tools=True)
    run_dir = _run_engine(tmp_path, monkeypatch, _engine_config(dataset))

    calls = _FakeClient.created[0].calls
    assert all("tools" in c for c in calls)
    table = pq.read_table(run_dir / "predictions.parquet")
    assert "tools" in table.column_names


def test_run_engine_fatal_error_fails_run_keeps_partial(tmp_path, monkeypatch) -> None:
    dataset = _make_dataset(tmp_path, 4)

    class BadClient(_FakeClient):
        async def _create(self, **kwargs):
            raise _bad_request()

    monkeypatch.setattr(engine_sglang, "_SglangServer", _FakeServer)
    monkeypatch.setattr(openai, "AsyncOpenAI", BadClient)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _engine_config(dataset, defer_terminal_status=True)
    # The unwrapped fatal error (not an ExceptionGroup) reaches the
    # status-owning caller, and no predictions.parquet is written.
    with pytest.raises(engine_sglang._FatalGenerationError):
        engine_sglang.run_inference_sglang(run_dir, config)
    assert not (run_dir / "predictions.parquet").exists()
    assert (run_dir / PREDICTIONS_PARTIAL).exists()


# ---------------------------------------------------------------------------
# Model path preparation (LoRA merge-to-disk)
# ---------------------------------------------------------------------------


def test_prepare_model_path_full_checkpoint_passthrough(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    path, tmp = engine_sglang._prepare_model_path({"base_model": str(ckpt)})
    assert path == str(ckpt) and tmp is None


def test_prepare_model_path_merges_adapter(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base-from-adapter"})
    )
    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(engine_sglang.subprocess, "run", fake_run)

    # base_override wins over the adapter's recorded base.
    path, tmp = engine_sglang._prepare_model_path(
        {"base_model": str(adapter), "base_override": "org/override-base"},
    )
    try:
        assert "org/override-base" in recorded["cmd"]
        assert str(adapter) in recorded["cmd"]
        assert "merge_lora" in " ".join(recorded["cmd"])
        assert path.endswith("/merged") and Path(path).exists()
    finally:
        assert tmp is not None
        tmp.cleanup()

    # Without base_override the adapter's recorded base is used.
    path2, tmp2 = engine_sglang._prepare_model_path({"base_model": str(adapter)})
    try:
        assert "org/base-from-adapter" in recorded["cmd"]
    finally:
        assert tmp2 is not None
        tmp2.cleanup()


def test_prepare_model_path_merge_failure_raises(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({}))
    monkeypatch.setattr(
        engine_sglang.subprocess, "run",
        lambda cmd, check: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="base model"):
        # No recorded base and no override → clear config error.
        engine_sglang._prepare_model_path({"base_model": str(adapter)})

    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    with pytest.raises(RuntimeError, match="merge failed"):
        engine_sglang._prepare_model_path({"base_model": str(adapter)})


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def test_server_dead_during_boot_raises_with_log_tail(tmp_path, monkeypatch) -> None:
    class DeadProc:
        def __init__(self, cmd, stdout=None, stderr=None):
            self.cmd = cmd

        def poll(self):
            return 137

    monkeypatch.setattr(engine_sglang.subprocess, "Popen", DeadProc)
    server = engine_sglang._SglangServer("org/model", tmp_path)
    server.log_path.write_text("CUDA out of memory somewhere\n")
    with pytest.raises(RuntimeError) as exc_info:
        server.wait_healthy(timeout=5)
    assert "code 137" in str(exc_info.value)
    assert "CUDA out of memory" in str(exc_info.value)


def test_server_command_shape(tmp_path, monkeypatch) -> None:
    captured = {}

    class FakeProc:
        def __init__(self, cmd, stdout=None, stderr=None):
            captured["cmd"] = cmd

        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(engine_sglang.subprocess, "Popen", FakeProc)
    server = engine_sglang._SglangServer(
        "/weights/m", tmp_path, extra_args="--mem-fraction-static 0.8",
    )
    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    assert ["--model-path", "/weights/m"] == cmd[3:5]
    assert "--tool-call-parser" in cmd and "lfm2" in cmd
    assert cmd[-2:] == ["--mem-fraction-static", "0.8"]
    server.terminate()
    assert captured.get("terminated") is True
