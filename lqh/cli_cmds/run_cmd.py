"""`lqh run` — delegate one task to lqh's agent, headlessly (CLI_PLAN §4).

Contract: exactly one JSON result document on stdout; NDJSON progress
events on stderr (schema_version, run_id, seq); exit code mirrors the
result status. The sub-agent runs under the SUBAGENT policy: no user,
task-implied permissions granted for the invocation, publishing gated
behind --allow-publish.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 1

_STATUS_EXIT = {
    "success": 0,
    "failure": 1,
    "needs_permission": 3,
    "auth_required": 4,
    "needs_configuration": 5,
    "limit_exceeded": 1,
    "interrupted": 6,
    "timed_out": 6,
}


class _EventStream:
    """NDJSON progress events on stderr."""

    def __init__(self, run_id: str, *, quiet: bool) -> None:
        self.run_id = run_id
        self.quiet = quiet
        self._seq = 0

    def emit(self, event: str, **fields: Any) -> None:
        if self.quiet:
            return
        self._seq += 1
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": self._seq,
            "event": event,
            **fields,
        }
        print(json.dumps(payload, default=str), file=sys.stderr, flush=True)


class _ArtifactLedger:
    """Deterministic artifact tracking from successful tool calls.

    This — not the model's exit_auto_mode claims — is the authoritative
    artifact list (CLI_PLAN §4.6). Best-effort by construction: entries
    are derived from tool arguments and result text of calls that
    succeeded.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.entries: list[dict[str, Any]] = []

    def _add(self, kind: str, **fields: Any) -> None:
        entry = {"kind": kind, **fields, "source": "ledger"}
        if entry not in self.entries:
            self.entries.append(entry)

    def observe(self, tool: str, args: dict, result: Any) -> None:
        ok = result.ok is True or (
            result.ok is None
            and not result.content.startswith(("Error", "❌", "Internal error"))
            and not result.requires_user_input
        )
        if not ok:
            return
        content = result.content
        if tool == "run_data_gen_pipeline":
            name = args.get("output_dataset")
            if name:
                self._add("dataset", path=f"datasets/{name}")
        elif tool == "run_data_filter":
            name = args.get("output_dataset") or args.get("output")
            if name:
                self._add("dataset", path=f"datasets/{name}")
        elif tool in ("start_training", "start_local_eval", "eval_hf_model"):
            run_name = args.get("run_name")
            if not run_name:
                match = re.search(r"runs/([A-Za-z0-9_\-\.]+)", content)
                run_name = match.group(1) if match else None
            if run_name:
                kind = "run" if tool == "start_training" else "eval_run"
                self._add(kind, path=f"runs/{run_name}")
        elif tool == "hf_push":
            match = re.search(r"[A-Za-z0-9\-_\.]+/[A-Za-z0-9\-_\.]+", str(args.get("repo_id") or "")) or re.search(
                r"huggingface\.co/([A-Za-z0-9\-_\.]+/[A-Za-z0-9\-_\.]+)", content
            )
            repo = args.get("repo_id") or (match.group(1) if match else None)
            if repo:
                self._add("hf_repo", id=str(repo))
        elif tool == "push_to_production":
            match = re.search(r"(dep-[A-Za-z0-9\-]+|https://\S+)", content)
            self._add("deployment", ref=match.group(1) if match else "created")
        elif tool == "gguf_convert":
            match = re.search(r"job ([A-Za-z0-9\-_]+)", content)
            if match:
                self._add("gguf_job", id=match.group(1))

    def finalize(self) -> None:
        """Reconcile entries against the filesystem at run end.

        A submission that succeeded may still never have produced its
        output (e.g. a cloud data-gen job that failed after submit) — a
        local path entry that does not exist is dropped rather than
        reported as an artifact. Training runs additionally gain their
        checkpoints directory when it materialized.
        """
        kept: list[dict[str, Any]] = []
        for entry in self.entries:
            path = entry.get("path")
            if path and not (self.project_dir / str(path)).exists():
                continue
            kept.append(entry)
            if (
                entry["kind"] == "run"
                and path
                and (self.project_dir / str(path) / "checkpoints").is_dir()
            ):
                checkpoint = {
                    "kind": "checkpoint",
                    "path": f"{path}/checkpoints",
                    "source": "ledger",
                }
                if checkpoint not in kept and checkpoint not in self.entries:
                    kept.append(checkpoint)
        self.entries = kept

    def merge_claims(self, claimed: list) -> None:
        """Validate and merge model-supplied artifact paths (claims)."""
        for item in claimed or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            kind = str(item.get("kind") or "file")
            if not path or path.startswith(("/", "..")) or ".." in path.split("/"):
                continue
            if not (self.project_dir / path).exists():
                continue
            entry = {"kind": kind, "path": path, "source": "reported"}
            if not any(
                e.get("path") == path for e in self.entries
            ):
                self.entries.append(entry)


class _LimitExceeded(Exception):
    def __init__(self, what: str) -> None:
        self.what = what


def _result_json(
    *,
    run_id: str,
    status: str,
    reason: str,
    summary: str,
    artifacts: list,
    metrics: dict,
    session_id: str | None,
    usage: dict,
    duration_s: float,
    secrets: list | None = None,
) -> dict:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "summary": summary,
        "artifacts": artifacts,
        "metrics": metrics,
        "session_id": session_id,
        "usage": usage,
        "duration_s": round(duration_s, 1),
    }
    if secrets:
        payload["secrets"] = secrets
    return payload


def _load_task(ns: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve the task prompt. Returns (task, error)."""
    task = ns.task
    if ns.prompt_file:
        if task:
            return None, "pass either a task string or --prompt-file, not both"
        try:
            raw = (
                sys.stdin.read()
                if ns.prompt_file == "-"
                else Path(ns.prompt_file).read_text(encoding="utf-8")
            )
        except OSError as e:
            return None, f"cannot read --prompt-file: {e}"
        return raw.strip() or None, None
    if task == "-":
        return sys.stdin.read().strip() or None, None
    return (task.strip() if task else None), None


def cmd_run(ns: argparse.Namespace) -> int:
    from lqh.cli_cmds.envelope import stdout_to_stderr

    # The whole run executes under an fd-level stdout redirect: pipeline
    # subprocesses and library prints land on stderr, and the ONLY bytes
    # on stdout are the final result JSON written to the saved fd.
    with stdout_to_stderr() as real_stdout:
        return _cmd_run_guarded(ns, real_stdout)


def _cmd_run_guarded(ns: argparse.Namespace, real_stdout: int) -> int:
    run_id = uuid.uuid4().hex[:12]
    start = time.monotonic()

    def _finish(payload: dict) -> int:
        import os

        code = _STATUS_EXIT.get(payload["status"], 1)
        os.write(real_stdout, (json.dumps(payload, default=str) + "\n").encode())
        return code

    def _early(status: str, reason: str, *, exit_code: int | None = None) -> int:
        code = _finish(_result_json(
            run_id=run_id, status=status, reason=reason, summary=reason,
            artifacts=[], metrics={}, session_id=None,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "turns": 0},
            duration_s=time.monotonic() - start,
        ))
        return exit_code if exit_code is not None else code

    def _usage_error(reason: str) -> int:
        # Even usage errors keep the one-JSON-document contract; exit 2
        # signals "usage" while status/reason stay machine-readable.
        print(f"error: {reason}", file=sys.stderr)
        return _early("failure", reason, exit_code=2)

    task, task_error = _load_task(ns)
    if task_error:
        return _usage_error(task_error)
    if not task and not ns.resume:
        return _usage_error(
            "a task is required (string, --prompt-file, or '-' for stdin) "
            "unless --resume is given"
        )

    if ns.project:
        project_dir = Path(ns.project).resolve()
        if not project_dir.is_dir():
            return _usage_error(f"--project is not a directory: {ns.project}")
    else:
        project_dir = Path.cwd()

    # Identity contract (CLI_PLAN §4.8) — fail closed before any cloud work.
    from lqh.headless import headless_boot

    boot = headless_boot(project_dir)
    if boot.identity_error:
        return _early(
            "needs_configuration",
            "Project identity file is corrupt and will NOT be auto-replaced: "
            f"{boot.identity_error}. Restore .lqh/project.json (e.g. from a "
            "backup or the original project directory) — do NOT delete it; "
            "a fresh identity would disconnect this project's cloud history.",
        )
    if boot.copy_status == "copied":
        return _early(
            "needs_configuration",
            "This project directory is an unresolved COPY of another lqh "
            "project. Resolve it first: `lqh project continue` or "
            "`lqh project fork`, then retry.",
        )

    from lqh.auth import get_token

    if not get_token():
        return _early("auth_required", "Not logged in. Run `lqh login` first.")

    return asyncio.run(
        _run_async(ns, task, project_dir, run_id, start, _finish, _usage_error)
    )


async def _run_async(
    ns: argparse.Namespace,
    task: str | None,
    project_dir: Path,
    run_id: str,
    start: float,
    _finish,
    _usage_error,
) -> int:
    from lqh.agent import Agent, AgentCallbacks
    from lqh.agent_policy import subagent_policy
    from lqh.headless import claim_loop, live_loop_owner, release_loop
    from lqh.jobs import JobSupervisor, SupervisorHooks
    from lqh.session import (
        STATE_COMPLETED,
        STATE_INTERRUPTED,
        Session,
    )

    events = _EventStream(run_id, quiet=ns.quiet)
    ledger = _ArtifactLedger(project_dir)

    # Advisory project-loop lock (CLI_PLAN §7): warn — don't block — when
    # another live agent loop (TUI or run) is working in this project.
    other_loop = live_loop_owner(project_dir)
    if other_loop is not None:
        print(
            f"warning: another lqh agent loop (pid {other_loop}) appears to "
            "be running in this project; concurrent mutating work is not "
            "supported and may interleave.",
            file=sys.stderr,
        )

    # Session: resume loads the prior conversation; the injected message
    # (the new task, or a default continue instruction) makes the resume
    # CONTEXTUAL — prepare_context below rebuilds fresh state regardless.
    if ns.resume:
        try:
            session = Session.load(project_dir, ns.resume)
        except Exception as e:
            return _usage_error(f"cannot resume session {ns.resume}: {e}")
        # Contention guard (CLI_PLAN §7): an ATOMIC check-and-mark under
        # the session lock — two resuming processes cannot both pass.
        # (headless_boot already repaired dead-owner sessions, so a
        # refusal means a genuinely live owner.)
        if not session.claim_active():
            return _usage_error(
                f"session {ns.resume} is active in another process; two "
                "agent loops must not share one conversation."
            )
        if not task:
            task = (
                "[Resumed] Continue the previously delegated task. If you "
                "were blocked on a permission, the capability may now be "
                "granted — retry the blocked action once."
            )
    else:
        session = Session.create(project_dir)

    claim_loop(project_dir)

    async def _on_agent_message(text: str) -> None:
        events.emit("agent_message", text=text)

    async def _on_tool_call(tool: str, args: dict) -> None:
        events.emit("tool_call", tool=tool, args=args)

    async def _on_tool_result(tool: str, content: str) -> None:
        events.emit(
            "tool_result",
            tool=tool,
            content=content[:2000] + ("…" if len(content) > 2000 else ""),
        )

    def _on_auto_stage(stage: str, note: str | None) -> None:
        events.emit("stage", stage=stage, note=note)

    def _on_pipeline_progress(event: Any) -> None:
        # ProgressEvent protocol (lqh/progress.py) — surface the fields a
        # harness can render; long generations get a heartbeat this way.
        events.emit(
            "progress",
            label=getattr(event, "label", None),
            phase=getattr(event, "phase", None),
            completed=getattr(event, "completed", None),
            total=getattr(event, "total", None),
            overall_fraction=getattr(event, "overall_fraction", None),
            step=getattr(event, "step", None),
        )

    supervisor = JobSupervisor(
        project_dir,
        poll_interval=30.0,
        hooks=SupervisorHooks(
            # Background-job heartbeat: one event per scan per live run,
            # so a parked multi-hour training wait is never silent.
            on_running=lambda run, remote: events.emit(
                "job_running", run=run, remote=remote,
            ),
        ),
    )

    callbacks = AgentCallbacks(
        on_agent_message=_on_agent_message,
        on_tool_call=_on_tool_call,
        on_tool_result=_on_tool_result,
        on_auto_stage=_on_auto_stage,
        on_pipeline_progress=_on_pipeline_progress,
        on_background_task_started=supervisor.register_started,
        on_await_background=supervisor.wait_for_runs,
        legacy_pipeline_progress_callback=False,
    )

    policy = subagent_policy(allow_publish=ns.allow_publish)

    status: str
    reason: str
    interrupted = False
    timed_out = False
    watch_task: asyncio.Task | None = None
    telemetry = None
    try:
        # ------------------------------------------------------------------
        # Setup — any failure here still yields the one JSON result doc.
        # ------------------------------------------------------------------
        try:
            agent = Agent(
                project_dir, session, callbacks,
                policy=policy, extra_spec=ns.spec,
            )
            # Deterministic limits, enforced by the agent BEFORE the next
            # API/tool call (a cooperative post-hoc cancel could not stop
            # a synchronous or shielded call from completing).
            agent.max_llm_calls = ns.max_turns
            agent.max_total_tool_calls = ns.max_tool_calls

            # Ledger observation rides the tool-result path via a wrapper
            # around the agent's tool executor — observe AFTER
            # interpretation so ok/sentinel handling has happened.
            original_handle = agent._handle_tool_call

            async def _observing_handle(tool_name: str, arguments: dict, **kw):
                result = await original_handle(tool_name, arguments, **kw)
                try:
                    ledger.observe(tool_name, arguments, result)
                except Exception:
                    pass
                return result

            agent._handle_tool_call = _observing_handle  # type: ignore[method-assign]

            # Fresh cloud signals when authenticated (same as TUI startup,
            # §4.8). fetch_and_cache_snapshot reports freshness itself — a
            # stale cached fallback must keep its "may be stale" warning.
            try:
                from lqh.snapshot import fetch_and_cache_snapshot

                snapshot, fresh = await asyncio.wait_for(
                    fetch_and_cache_snapshot(project_dir), 10.0
                )
                agent.set_startup_facts(snapshot=snapshot, snapshot_fresh=fresh)
            except Exception:
                pass

            # prepare_context BEFORE the supervisor starts scanning: its
            # finished-while-away diff consumes the job_seen.json baseline,
            # which the supervisor's first scan would otherwise overwrite.
            await agent.prepare_context()
            watch_task = asyncio.create_task(supervisor.watch_loop())
            # Wait for the first scan so a resumed agent's immediate
            # training_status sees pre-existing jobs in the registry
            # (otherwise parking would see nothing running and fall back
            # to LLM polling).
            await supervisor.wait_primed()

            # Telemetry (CLI_PLAN §4.7): start a session only when consent
            # was previously recorded (the TUI notice); else a one-line note.
            telemetry = _maybe_start_telemetry(project_dir)
        except Exception as e:  # noqa: BLE001 — result JSON is the contract
            import traceback

            traceback.print_exc(file=sys.stderr)
            session.mark_state(STATE_INTERRUPTED)
            return _finish(_result_json(
                run_id=run_id, status="failure",
                reason=f"run setup failed: {type(e).__name__}: {e}",
                summary=f"run setup failed: {e}", artifacts=[],
                metrics={}, session_id=session.id,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "turns": 0},
                duration_s=time.monotonic() - start,
            ))

        events.emit("start", task=task, session_id=session.id)

        try:
            agent_task = asyncio.create_task(agent.process_user_input(task or ""))
            if ns.timeout:
                # Wall-clock budget measured from command start (identity
                # boot, snapshot fetch, and priming already spent part of
                # it). Best-effort: an in-flight protected submission may
                # extend past it by its bounded grace.
                remaining = ns.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    agent_task.cancel()
                else:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(agent_task), remaining
                        )
                    except asyncio.TimeoutError:
                        timed_out = True
                        agent_task.cancel()
                if timed_out:
                    try:
                        await agent_task
                    except BaseException:
                        pass
            else:
                await agent_task
        except (asyncio.CancelledError, KeyboardInterrupt):
            interrupted = True
        except Exception as e:  # noqa: BLE001 — result JSON is the contract
            import traceback

            traceback.print_exc(file=sys.stderr)
            agent.abort_turn()
            session.mark_state(STATE_INTERRUPTED)
            ledger.finalize()
            await _end_telemetry(telemetry, "failure")
            return _finish(_result_json(
                run_id=run_id, status="failure",
                reason=f"{type(e).__name__}: {e}",
                summary=f"The run crashed: {e}", artifacts=ledger.entries,
                metrics={}, session_id=session.id,
                usage=_usage(agent), duration_s=time.monotonic() - start,
            ))
    finally:
        if watch_task is not None:
            watch_task.cancel()
            try:
                await watch_task
            except BaseException:
                pass
        await supervisor.stop_watchers()
        release_loop(project_dir)

    details = agent._auto_exit_details or {}
    summary = str(details.get("summary") or "")
    metrics_raw = details.get("metrics") or {}
    metrics = {
        str(k): {"value": v, "provenance": "reported"}
        for k, v in metrics_raw.items()
        if isinstance(v, (int, float))
    }
    ledger.merge_claims(details.get("artifacts") or [])
    ledger.finalize()

    if timed_out:
        agent.abort_turn()
        status, reason = "timed_out", f"--timeout {ns.timeout}s elapsed"
        session.mark_state(STATE_INTERRUPTED)
    elif interrupted:
        agent.abort_turn()
        status, reason = "interrupted", "the run was interrupted"
        session.mark_state(STATE_INTERRUPTED)
    elif agent._policy_halt is not None:
        status, reason = agent._policy_halt
        session.mark_state(STATE_COMPLETED)
    elif agent._auto_exit is not None:
        status, reason = agent._auto_exit
        summary = summary or reason
        session.mark_state(STATE_COMPLETED)
    else:
        status = "failure"
        reason = "the run ended without calling exit_auto_mode"
        session.mark_state(STATE_COMPLETED)

    # Secrets always ride the result payload; --save-secret ADDITIONALLY
    # persists them to the project's .env (CLI_PLAN §4.2).
    secrets = [
        {"env_var": s.env_var, "value": s.payload}
        for s in agent.delivered_secrets
    ]
    if ns.save_secret and agent.delivered_secrets:
        from lqh.env_secrets import append_env_secret

        for delivery in agent.delivered_secrets:
            try:
                append_env_secret(
                    project_dir, delivery.env_var, delivery.payload,
                    delivery.env_comment,
                )
            except Exception as e:
                print(f"warning: could not save secret to .env: {e}", file=sys.stderr)

    await _end_telemetry(telemetry, status)
    events.emit("end", status=status)
    return _finish(_result_json(
        run_id=run_id, status=status, reason=reason,
        summary=summary or reason,
        artifacts=ledger.entries, metrics=metrics,
        session_id=session.id, usage=_usage(agent),
        duration_s=time.monotonic() - start,
        secrets=secrets or None,
    ))


def _maybe_start_telemetry(project_dir: Path):
    """Start telemetry ONLY when consent was previously recorded (§4.7).

    The consent check must not call notice_needed() — it has a
    marker-writing side effect. A missing marker means the TUI notice was
    never shown, so telemetry stays off with a one-line stderr note.
    """
    try:
        from lqh.config import config_dir, load_config

        consent_shown = (config_dir() / "telemetry_notice_v1").exists()
        if not consent_shown or not load_config().telemetry_enabled:
            print("telemetry: disabled for this run", file=sys.stderr)
            return None
        from lqh.telemetry import TelemetryClient, set_active_telemetry

        telemetry = TelemetryClient(project_dir, auto_mode=True)
        set_active_telemetry(telemetry)
        telemetry.defer(telemetry.start_session)
        return telemetry
    except Exception:
        return None


async def _end_telemetry(telemetry, status: str) -> None:
    if telemetry is None:
        return
    outcome = {
        "success": "succeeded",
        "interrupted": "cancelled",
        "timed_out": "cancelled",
    }.get(status, "failed")
    try:
        # Await so the daemon worker flushes before process exit.
        await telemetry.run_deferred(telemetry.end_session, outcome)
    except Exception:
        pass


def _usage(agent) -> dict:
    return {
        "prompt_tokens": agent._run_prompt_tokens,
        "completion_tokens": agent._run_completion_tokens,
        "turns": agent._llm_calls_made,
    }
