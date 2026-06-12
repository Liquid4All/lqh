"""Cloud remote backend — runs jobs on api.lqh.ai.

The GPU provider is a backend-implemented detail; the user only sees
``api.lqh.ai``. This backend mirrors the SSH-direct ``RemoteBackend``
contract so ``RemoteRunWatcher`` doesn't need to know which path is
in use — it reads ``progress.jsonl`` / ``status.json`` / ``stdout.log``
either way.

Disconnect resilience is a first-class concern: the SSE stream carries
a monotonically increasing ``seq`` per job. We persist the last seen
``seq`` to ``<run_dir>/cloud_state.json`` after every event so that:

  * A laptop sleeping mid-fine-tune resumes from the gap on next wake.
  * A client crash and reopen sees the same.
  * Server-side history (``cloud_job_events`` table) replays missed
    events when the consumer reconnects with ``?last_seq=N``.

Cancellation is hooked to ``teardown`` and propagates to the cloud
runner; the server flips ``cloud_jobs.status='cancelled'`` and emits a
final status event that the consumer writes to ``status.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from lqh.auth import api_root, get_token, require_token
from lqh.project_meta import gather_project_meta
from lqh.remote.backend import JobStatus, RemoteBackend, RemoteConfig
from lqh.remote.bundle import build_bundle

logger = logging.getLogger(__name__)

__all__ = ["CloudBackend"]


# Idle timeout in sync_progress: how long to wait for a new event
# before returning so the watcher can re-tick. Picked tight enough
# that "submit then sit for a minute waiting for the first event"
# returns quickly; loose enough that we don't reconnect on every
# heartbeat (server emits one every ~25s).
_IDLE_RETURN_TIMEOUT_S = 5.0

# Total per-call ceiling. Even if events keep flowing, we hand
# control back to the watcher after this so it can do its work
# (scoring, eval requests). Returning is cheap — we resume from
# the persisted last_seq.
_MAX_SYNC_DURATION_S = 60.0

# Map cloud event status → JobStatus.state values used by the watcher.
_STATUS_MAP = {
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "failed",  # watcher treats cancelled like failed terminal
}


@dataclass
class _CloudState:
    """Persisted state for one cloud job, written to cloud_state.json.

    Holds enough to resume after the client process restarts:
    job_id, the last event seq we wrote to disk, and the latest
    status we observed. Updated atomically (tmp + rename)."""

    job_id: str
    last_seq: int = 0
    status: str = "pending"
    ended_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> "_CloudState | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return cls(
            job_id=data["job_id"],
            last_seq=int(data.get("last_seq", 0)),
            status=data.get("status", "pending"),
            ended_at=data.get("ended_at"),
        )

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "job_id": self.job_id,
                    "last_seq": self.last_seq,
                    "status": self.status,
                    "ended_at": self.ended_at,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(tmp, path)


class CloudBackend(RemoteBackend):
    """RemoteBackend backed by api.lqh.ai's /v1/cloud/jobs endpoints."""

    # The sandbox runs its own LLM-judge scoring + golden-trajectory
    # assembly using the scoped LQH_API_TOKEN injected by the backend.
    # RemoteRunWatcher checks this flag to skip _check_iter_requests /
    # _check_eval_requests / _push_results, because (a) it would race
    # the sandbox and (b) the push helpers raise NotImplementedError.
    # SSH backends don't set this; the laptop watcher does the work.
    inline_scoring = True

    def __init__(
        self,
        config: RemoteConfig,
        project_dir: Path,
        *,
        api_base: str | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(config)
        self.project_dir = project_dir
        # api_root() strips a trailing /v1 so we can build paths like
        # "/v1/cloud/jobs" cleanly.
        self._api_base = (api_base or api_root()).rstrip("/")
        # Token resolved lazily so tests can inject; production uses
        # the logged-in user's token from ~/.config/lqh/credentials.
        self._token = token

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        tok = self._token or get_token() or require_token()
        return {"Authorization": f"Bearer {tok}"}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> str:
        """No-op — cloud is provisioned server-side. The first submit is
        the only thing that exercises the backend; we don't pre-flight
        anything here."""
        return "Cloud ready — no setup needed."

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit_run(
        self,
        local_run_dir: str,
        config: dict[str, Any],
        *,
        module: str = "lqh.train",
    ) -> str:
        """Build the bundle, POST to /v1/cloud/jobs, persist state, return
        the job_id.

        The job_id (a UUID) plays the role of "PID" in the
        RemoteBackend contract — it's the handle for poll_status,
        teardown, and remote_job.json reconnection.
        """
        run_dir = Path(local_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Infer kind from module if the config doesn't say.
        kind = _infer_kind(config, module)

        # Build bundle in-memory.
        bundle = build_bundle(config, self.project_dir)

        meta = {
            "kind": kind,
            "project_id": self.project_dir.name,
            "module": module,
            "config": config,
        }
        # Project metadata: display name, spec hash, base/reward model.
        # The backend upserts the projects row on submit; missing
        # fields don't overwrite previously-recorded values.
        meta.update(gather_project_meta(self.project_dir, config).to_meta_dict())
        # HF token donate path: if the project binding asked us to
        # donate the local env var, attach it to meta.hf_token. The
        # backend forwards it as an ephemeral cloud secret and never
        # persists it.
        if getattr(self.config, "hf_token_configured", False):
            hf = os.environ.get("HF_TOKEN")
            if hf:
                meta["hf_token"] = hf

        # httpx multipart: meta is a string field, bundle is a file.
        files = [
            ("meta", (None, json.dumps(meta), "application/json")),
            ("bundle", ("bundle.tar.gz", bundle, "application/gzip")),
        ]
        async with httpx.AsyncClient(base_url=self._api_base, timeout=120.0) as client:
            resp = await client.post(
                "/v1/cloud/jobs",
                files=files,
                headers=self._auth_headers(),
            )
            _raise_for_cloud_error(resp)
            data = resp.json()
            job_id = data["job_id"]

        # Persist enough to reconnect after a client restart. We write
        # both remote_job.json (for parity with SSHDirect) and
        # cloud_state.json (for the SSE seq cursor).
        (run_dir / "remote_job.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "remote_name": self.config.name,
                    "remote_run_dir": f"cloud:{job_id}",
                    "module": module,
                    "kind": kind,
                    "backend": "cloud",
                },
                indent=2,
            )
            + "\n"
        )
        _CloudState(job_id=job_id, status=data.get("status", "pending")).save(
            run_dir / "cloud_state.json"
        )

        # Also stash the config locally so anything that reads the run
        # dir (eg. the watcher's eval-scoring loop) sees what was
        # submitted.
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        logger.info("cloud job submitted: %s (kind=%s)", job_id, kind)
        return job_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def poll_status(self, job_id: str) -> JobStatus:
        """Pull the snapshot endpoint and map to JobStatus.

        Cheap fallback if we don't have a fresh SSE event in
        cloud_state.json yet (e.g. very first poll, or after a long
        idle gap).
        """
        try:
            snap = await self._get_snapshot(job_id)
        except CloudError as exc:
            if _is_cloud_rate_limit_error(exc):
                raise
            logger.warning("poll_status snapshot failed: %s", exc)
            return JobStatus(state="unknown")
        except Exception as exc:  # network blip → unknown is the safe fallback
            logger.warning("poll_status snapshot failed: %s", exc)
            return JobStatus(state="unknown")
        return JobStatus(
            state=_STATUS_MAP.get(snap.get("status", ""), snap.get("status", "unknown")),
            error=snap.get("error"),
        )

    async def is_job_alive(self, job_id: str) -> bool:
        """The cloud equivalent of "is the PID still around." True iff
        the snapshot reports a non-terminal status."""
        try:
            snap = await self._get_snapshot(job_id)
        except Exception:
            # If we can't tell, assume alive — the watcher will retry
            # rather than prematurely declaring the run done.
            return True
        status = snap.get("status", "")
        return status not in {"completed", "failed", "cancelled"}

    async def _get_snapshot(self, job_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._api_base, timeout=30.0) as client:
            resp = await client.get(
                f"/v1/cloud/jobs/{job_id}",
                headers=self._auth_headers(),
            )
            _raise_for_cloud_error(resp)
            return resp.json()

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    async def sync_progress(
        self,
        remote_run_dir: str,
        local_run_dir: str,
    ) -> None:
        """Consume SSE events and translate them into local files.

        This is the disconnect-resilience workhorse. Reads
        cloud_state.json for the resume seq, opens
        /v1/cloud/jobs/{id}/stream?last_seq=N, writes each event into
        the appropriate local file (progress.jsonl, stdout.log,
        status.json, artifacts.json), and updates the persisted
        last_seq after each event.

        Returns when:
          - the stream emits a terminal status event;
          - the connection is idle for IDLE_RETURN_TIMEOUT_S;
          - MAX_SYNC_DURATION_S has elapsed;
          - the connection errors (network blip → watcher reconnects
            on the next tick).

        Idempotent and safe to call repeatedly — that's by design.
        """
        run_dir = Path(local_run_dir)
        state_path = run_dir / "cloud_state.json"
        state = _CloudState.load(state_path)
        if state is None:
            # No state to resume from — the submit must not have happened
            # yet (or remote_job.json was hand-deleted). Nothing to do.
            return
        if state.status in {"completed", "failed", "cancelled"}:
            # Already terminal — don't reconnect.
            return

        url = f"/v1/cloud/jobs/{state.job_id}/stream"
        params = {"last_seq": state.last_seq}

        deadline = asyncio.get_event_loop().time() + _MAX_SYNC_DURATION_S
        try:
            async with httpx.AsyncClient(
                base_url=self._api_base,
                timeout=httpx.Timeout(_IDLE_RETURN_TIMEOUT_S, connect=10.0),
            ) as client:
                async with client.stream(
                    "GET", url, params=params, headers=self._auth_headers()
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise CloudError(f"stream open failed ({resp.status_code}): {body[:200]!r}")
                    async for ev in _parse_sse(resp):
                        await self._apply_event(run_dir, state, ev)
                        state.save(state_path)
                        if state.status in {"completed", "failed", "cancelled"}:
                            return
                        if asyncio.get_event_loop().time() > deadline:
                            return
        except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            # ReadTimeout fires when the server sends no event for
            # _IDLE_RETURN_TIMEOUT_S (heartbeat-only intervals shouldn't
            # but if proxies strip the heartbeat comments, we'll see
            # this). Either way: return so the watcher reconnects.
            logger.debug("cloud stream idle/disconnect (%s); returning", exc)
            return
        except httpx.HTTPError as exc:
            logger.warning("cloud stream error: %s", exc)
            return

    # ------------------------------------------------------------------
    # File sync (push/pull)
    # ------------------------------------------------------------------

    async def sync_file_to_remote(
        self,
        local_path: str,
        remote_path: str,
    ) -> None:
        """Push a single file to the sandbox.

        Phase 1 ships with infer-only support and the GPU sandbox
        consumes the input bundle plus whatever's on the cloud volume
        — there's no mid-run client → remote sync. DPO's eval-result
        feedback loop will need this (Phase 2); for now raise.
        """
        raise NotImplementedError(
            "Cloud backend does not yet support mid-run file pushes; "
            "this is a Phase 2 deliverable."
        )

    async def sync_file_from_remote(
        self,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Pull a single file from the sandbox.

        Same Phase 1 limitation as sync_file_to_remote — pulls happen
        via ArtifactStore once the job ends. Watcher callers don't hit
        this path for infer.
        """
        raise NotImplementedError(
            "Cloud backend does not yet support arbitrary remote file pulls; "
            "use lqh.artifacts.ArtifactStore for post-job artifacts."
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def teardown(self, job_id: str) -> None:
        """DELETE /v1/cloud/jobs/{id}. Idempotent server-side."""
        async with httpx.AsyncClient(base_url=self._api_base, timeout=30.0) as client:
            resp = await client.delete(
                f"/v1/cloud/jobs/{job_id}",
                headers=self._auth_headers(),
            )
            if resp.status_code not in (204, 404):
                _raise_for_cloud_error(resp)

    # ------------------------------------------------------------------
    # Event → file translation
    # ------------------------------------------------------------------

    async def _apply_event(
        self, run_dir: Path, state: _CloudState, ev: "_SSEEvent"
    ) -> None:
        """Write one parsed SSE event into the run dir.

        Mapping is chosen so RemoteRunWatcher's existing readers work
        unchanged:
          status   → status.json + sentinel line in progress.jsonl
          log      → stdout.log or stderr.log
          progress → progress.jsonl row
          artifact → append to artifacts.json (JSON list)
        """
        # Track the highest seq we've actually written so disconnects
        # don't replay duplicates.
        seq = ev.payload.get("seq", state.last_seq + 1)
        if isinstance(seq, int) and seq <= state.last_seq:
            return
        if isinstance(seq, int):
            state.last_seq = seq

        kind = ev.kind
        payload = ev.payload.get("payload", {}) if isinstance(ev.payload, dict) else {}
        ts = ev.payload.get("ts") if isinstance(ev.payload, dict) else None

        if kind == "status":
            status = payload.get("status", "")
            if status:
                # Two kinds of "status=completed" events can reach us:
                # (1) the runner's final Wait() event, which always
                #     carries ``exit_code`` (cloud runner streamSandbox);
                # (2) the trainer subprocess's own end-of-training
                #     sentinel via lqh.train.progress.write_status, which
                #     does not carry exit_code.
                # Only (1) is actually terminal — the launcher continues
                # into a publish phase after (2). Treat (2) as a progress
                # signal so the SSE consumer keeps streaming.
                is_runner_terminal = (
                    status in {"completed", "failed", "cancelled"}
                    and "exit_code" in payload
                )
                if is_runner_terminal:
                    state.status = status
                    state.ended_at = ts
                status_path = run_dir / "status.json"
                status_path.write_text(
                    json.dumps(
                        {
                            "state": _STATUS_MAP.get(status, status),
                            "last_update": ts,
                            "error": payload.get("error"),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                # Mirror as a progress.jsonl row too so existing terminal
                # detection (lqh/remote/ssh_direct.py poll_status) works.
                _append_jsonl(run_dir / "progress.jsonl", {
                    "status": _STATUS_MAP.get(status, status),
                    "timestamp": ts,
                    "error": payload.get("error"),
                })

        elif kind == "log":
            stream = payload.get("stream", "stdout")
            line = payload.get("line", "")
            target = run_dir / ("stderr.log" if stream == "stderr" else "stdout.log")
            with target.open("a") as fh:
                fh.write(line + "\n")

        elif kind == "progress":
            entry = dict(payload)
            if ts and "timestamp" not in entry:
                entry["timestamp"] = ts
            _append_jsonl(run_dir / "progress.jsonl", entry)

        elif kind == "artifact":
            entry = dict(payload)
            if ts and "timestamp" not in entry:
                entry["timestamp"] = ts
            _append_artifact_manifest(run_dir / "artifacts.json", entry)


# ---------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------


@dataclass
class _SSEEvent:
    """One parsed SSE block: event-line + data-line."""

    kind: str
    payload: dict[str, Any]


async def _parse_sse(resp: httpx.Response) -> AsyncIterator[_SSEEvent]:
    """Iterate the response body, yielding one _SSEEvent per block.

    Implements the minimum subset of the SSE wire format we actually
    use: ``event: <kind>`` and ``data: <json>`` followed by a blank
    line. Comment lines (``:...``) — including our heartbeats — are
    ignored.
    """
    kind = ""
    data_lines: list[str] = []
    async for raw in resp.aiter_lines():
        line = raw.rstrip("\r")
        if line == "":
            if kind and data_lines:
                blob = "\n".join(data_lines)
                try:
                    payload = json.loads(blob)
                except json.JSONDecodeError:
                    payload = {"raw": blob}
                yield _SSEEvent(kind=kind, payload=payload)
            kind = ""
            data_lines = []
            continue
        if line.startswith(":"):
            # Comment / heartbeat.
            continue
        if line.startswith("event:"):
            kind = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        # Other SSE fields (id, retry) are not used by us.


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class CloudError(RuntimeError):
    """Raised on a non-2xx response from /v1/cloud/*."""


def _is_cloud_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _raise_for_cloud_error(resp: httpx.Response) -> None:
    if 200 <= resp.status_code < 300:
        return
    try:
        body = resp.json()
        msg = body.get("error", {}).get("message", resp.text[:200])
    except Exception:
        msg = resp.text[:200]
    # 402 is the "billing precondition" status the backend returns
    # for monthly-cap exhaustion, GPU min-balance floor violation,
    # and org deactivation (see OpenAPI CostLimitExceeded). The
    # backend message is human-readable; surface it with a clear
    # prefix so the TUI rendering of the tool failure makes the
    # "this is a billing issue, not a transient error" distinction
    # obvious to the user without parsing.
    if resp.status_code == 402:
        raise CloudError(f"insufficient balance: {msg}")
    raise CloudError(f"{resp.status_code}: {msg}")


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _append_artifact_manifest(path: Path, entry: dict[str, Any]) -> None:
    """Append one artifact descriptor to artifacts.json (a JSON object
    with an "artifacts" list)."""
    existing: dict[str, Any] = {"artifacts": [], "failed": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    existing.setdefault("artifacts", []).append(entry)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2) + "\n")
    os.replace(tmp, path)


def _infer_kind(config: dict[str, Any], module: str) -> str:
    """Best-effort kind detection from config + module name.

    The backend validates `kind` against its CHECK constraint, so we
    just need to produce one of the valid values. Explicit config
    field wins.
    """
    if isinstance(config.get("kind"), str):
        return config["kind"]
    cfg_type = (config.get("type") or "").lower()
    base_type = (
        (config.get("base_config") or {}).get("type", "").lower()
        if isinstance(config.get("base_config"), dict) else ""
    )
    # eval_hf has its own module + entrypoint; check first so it
    # doesn't get swallowed by the generic .infer test below.
    if module.endswith(".eval_hf") or cfg_type == "eval_hf":
        return "eval_hf"
    if module.endswith(".infer") or cfg_type == "infer":
        return "infer"
    # Sweep dispatch: prefer the DPO sweep module/base_config type
    # over the SFT default. ``module.endswith(".dpo_sweep")`` matches
    # the cloud-handler module whitelist (lqh.train.dpo_sweep).
    if module.endswith(".dpo_sweep") or base_type in ("dpo", "on_policy_dpo"):
        return "train_dpo_sweep"
    if cfg_type == "sweep" or module.endswith(".sweep"):
        return "train_sft_sweep"
    if cfg_type in ("dpo", "on_policy_dpo"):
        return "train_dpo"
    return "train_sft"


# Module-scope re-export for tests that want to exercise the parser in
# isolation without exposing the underscore name.
parse_sse = _parse_sse
SSEEvent = _SSEEvent
