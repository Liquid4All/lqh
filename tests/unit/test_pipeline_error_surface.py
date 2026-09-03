"""Regression guard for pipeline failure reporting (feedback #105).

``_execute_pipeline`` wraps the run in ``except asyncio.CancelledError``,
but ``lqh.tools.handlers`` did not import ``asyncio`` (fixed in 0.11.0).
Evaluating that except clause raised ``NameError: name 'asyncio' is not
defined``, which replaced the pipeline's real exception — every failing
pipeline reported the same bogus error and neither the user nor the agent
could see what actually went wrong.
"""

from __future__ import annotations

from pathlib import Path


async def test_pipeline_error_reaches_the_agent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / "data_gen").mkdir(parents=True)
    (project / "data_gen" / "pipeline.py").write_text("# test\n")
    monkeypatch.setattr("lqh.telemetry.active_telemetry", lambda: None)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "token")
    monkeypatch.setattr("lqh.client.create_client", lambda *_args: object())

    async def exploding_pipeline(**_kwargs):
        raise TypeError("hf_dataset() takes 1 positional argument but 2 were given")

    monkeypatch.setattr("lqh.engine.run_pipeline", exploding_pipeline)
    from lqh.tools.handlers import _execute_pipeline

    result = await _execute_pipeline(
        project, "data_gen/pipeline.py", 2, "out", None,
    )

    assert "NameError" not in result.content
    assert "hf_dataset() takes 1 positional argument" in result.content
