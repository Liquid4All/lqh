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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from lqh.auth import api_root, get_token, require_token
from lqh.project_meta import gather_project_meta
from lqh.remote.backend import JobStatus, RemoteBackend, RemoteConfig
from lqh.remote.bundle import build_bundle_to_file

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

# Bundles above this ride the presigned-PUT staging path instead of the
# multipart submit form. The server tolerates larger forms (512 MiB
# default, for older CLIs), but direct-to-storage is strictly better
# for anything sizable: streamed from disk, no double buffering, and a
# signature-bound Content-Length. Typical training bundles stay far
# below; data_gen bring-your-own image folders are the case above.
_MULTIPART_BUNDLE_MAX = 32 * 2**20

# Backoff between submit retries after a transport failure (the retry is
# safe because every submit carries an idempotency key). Module-level so
# tests can zero it.
_SUBMIT_RETRY_BACKOFF_SECONDS = 2.0

# Map cloud event status → state values mirrored into status.json /
# progress.jsonl. "cancelled" folds to "failed" here on purpose: the
# run-watcher self-stop logic and ssh_direct terminal detection match on
# failed/interrupted. The SNAPSHOT path (poll_status → TUI job watcher)
# passes "cancelled" through instead, so user-initiated cancels are
# reported as cancelled rather than failed.
_STATUS_MAP = {
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "failed",
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
        telemetry_workflow_id: str | None = None,
    ) -> str:
        """Build the bundle, POST to /v1/cloud/jobs, persist state, return
        the job_id.

        The job_id (a UUID) plays the role of "PID" in the
        RemoteBackend contract — it's the handle for poll_status,
        teardown, and remote_job.json reconnection.

        ``telemetry_workflow_id`` lets a caller that already opened a
        client-side workflow (e.g. the data-gen handler's
        ``data_generation_started``) correlate the cloud_jobs row with
        it instead of this method minting an unrelated ID.
        """
        run_dir = Path(local_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Infer kind from module if the config doesn't say.
        kind = _infer_kind(config, module)
        from lqh.telemetry import active_telemetry
        telemetry = active_telemetry()
        telemetry_project_id = (
            # This identifier leaves the process in the cloud-job request, so
            # unlike best-effort event writes it must wait for persisted
            # consent to be checked instead of treating a detached waiter as
            # a negative result while the callback remains queued.
            await telemetry.run_deferred(
                telemetry.correlation_project_id, timeout=None,
            )
            if telemetry else None
        )
        if telemetry_project_id and telemetry_workflow_id is None:
            telemetry_workflow_id = str(uuid.uuid4())
        elif not telemetry_project_id:
            telemetry_workflow_id = None
        if telemetry and telemetry_workflow_id:
            if kind == "eval_hf":
                await telemetry.run_deferred(telemetry.event, "zero_shot_evaluation_started", {
                    "workflow_kind":"zero_shot_evaluation", "subtype":"zero_shot",
                    "execution_target":"cloud", "sample_count":int(config.get("num_samples", 0) or 0),
                }, telemetry_workflow_id)
            elif kind.startswith("train_"):
                subtype = {
                    "train_sft":"direct_sft", "train_dpo":"direct_dpo",
                    "train_sft_sweep":"sft_sweep", "train_dpo_sweep":"dpo_sweep",
                }.get(kind, "direct_sft")
                await telemetry.run_deferred(telemetry.event, "fine_tuning_started", {
                    "workflow_kind":"fine_tuning", "subtype":subtype,
                    "execution_target":"cloud",
                }, telemetry_workflow_id)

        async def emit_submit_terminal(outcome: str) -> None:
            if not telemetry or not telemetry_workflow_id:
                return
            if kind == "eval_hf":
                event_name = "zero_shot_evaluation_failed"
                workflow_kind = "zero_shot_evaluation"
                subtype = "zero_shot"
            elif kind.startswith("train_"):
                event_name = "fine_tuning_failed"
                workflow_kind = "fine_tuning"
                subtype = {
                    "train_sft":"direct_sft", "train_dpo":"direct_dpo",
                    "train_sft_sweep":"sft_sweep", "train_dpo_sweep":"dpo_sweep",
                }.get(kind, "direct_sft")
            else:
                return
            await telemetry.run_deferred(telemetry.event, event_name, {
                "workflow_kind":workflow_kind, "subtype":subtype,
                "execution_target":"cloud", "outcome":outcome,
            }, telemetry_workflow_id)

        # Bundle construction and submission can fail before a cloud job
        # exists. Always close the client-side workflow in those paths.
        bundle_tmp = run_dir / ".bundle.tar.gz.tmp"
        # Idempotency: the key is persisted to disk BEFORE the POST so a
        # lost response (timeout/disconnect after the backend launched
        # the sandbox) never leaves a billable job with no local marker —
        # resubmitting with the same key returns the existing job instead
        # of launching a duplicate. The intent file is removed once
        # remote_job.json is safely on disk; if it survives, it names the
        # key of a submit whose fate is unknown.
        idempotency_key = str(uuid.uuid4())
        intent_path = run_dir / "submit_intent.json"
        try:
            # Build to disk so bring-your-own seed data (image folders on
            # data_gen submits) never sits fully in RAM.
            bundle_size = build_bundle_to_file(config, self.project_dir, bundle_tmp)
            meta = {
                "kind": kind,
                "project_id": self.project_dir.name,
                "module": module,
                "config": config,
                "idempotency_key": idempotency_key,
            }
            intent_path.write_text(
                json.dumps(
                    {
                        "idempotency_key": idempotency_key,
                        "kind": kind,
                        "module": module,
                        "name": self.config.name,
                    },
                    indent=2,
                )
                + "\n"
            )
            if telemetry_project_id and telemetry_workflow_id:
                meta["telemetry_project_id"] = telemetry_project_id
                meta["telemetry_workflow_id"] = telemetry_workflow_id
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

            files = [("meta", (None, json.dumps(meta), "application/json"))]
            if bundle_size > _MULTIPART_BUNDLE_MAX:
                # Too big for the multipart submit path — stage via
                # presigned PUT and reference the key. The upload-url
                # endpoint enforces the server's size ceiling (default
                # 2 GiB) AND the kind-level submit gates, so a gated
                # data_gen submit fails before the upload, not after.
                meta["bundle_key"] = await self._stage_bundle(bundle_tmp, bundle_size, kind)
                files = [("meta", (None, json.dumps(meta), "application/json"))]
            else:
                files.append(
                    ("bundle", ("bundle.tar.gz", bundle_tmp.read_bytes(), "application/gzip"))
                )
            async with httpx.AsyncClient(base_url=self._api_base, timeout=120.0) as client:
                # Transport failures (timeout, dropped connection) are the
                # response-lost case: the backend may have already created
                # the job. The idempotency key makes the retry safe — the
                # server returns the existing job instead of a duplicate.
                # HTTP error *responses* are not retried; the server
                # answered and _raise_for_cloud_error reports it.
                for attempt in range(3):
                    try:
                        resp = await client.post(
                            "/v1/cloud/jobs",
                            files=files,
                            headers=self._auth_headers(),
                        )
                        break
                    except httpx.TransportError as exc:
                        if attempt == 2:
                            raise
                        logger.warning(
                            "cloud submit attempt %d failed (%s); retrying with "
                            "the same idempotency key",
                            attempt + 1, exc,
                        )
                        await asyncio.sleep(_SUBMIT_RETRY_BACKOFF_SECONDS * (attempt + 1))
                _raise_for_cloud_error(resp)
                data = resp.json()
                job_id = data["job_id"]
        except asyncio.CancelledError:
            # Fate unknown (cancelled mid-POST?) — keep submit_intent.json.
            await emit_submit_terminal("cancelled")
            raise
        except httpx.TransportError:
            # Response lost even after retries — the job may be running
            # server-side with no local marker. Keep submit_intent.json:
            # it names the idempotency key, and the job (if any) shows up
            # in the backend job list.
            logger.error(
                "cloud submit response lost; if the job launched it is visible "
                "in the job list (idempotency key recorded in %s)", intent_path,
            )
            await emit_submit_terminal("failed")
            raise
        except Exception:
            # The server answered with an error (or we failed before the
            # POST) — no orphaned job, so drop the intent marker.
            intent_path.unlink(missing_ok=True)
            await emit_submit_terminal("failed")
            raise
        finally:
            bundle_tmp.unlink(missing_ok=True)

        # Persist enough to reconnect after a client restart. We write
        # both remote_job.json (for parity with SSHDirect) and
        # cloud_state.json (for the SSE seq cursor). If any of these
        # writes fail (disk full, permissions), we'd otherwise report a
        # failed submission while a billable job runs with no local
        # watcher state — so cancel the job server-side before
        # re-raising.
        try:
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
        except Exception:
            logger.error(
                "cloud job %s accepted but local state could not be written — cancelling",
                job_id,
            )
            try:
                await self.teardown(job_id)
                intent_path.unlink(missing_ok=True)
            except Exception as terr:
                # Cancel failed too — keep submit_intent.json as the marker
                # for the still-running job.
                logger.warning("cancel after failed local persist also failed: %s", terr)
            await emit_submit_terminal("failed")
            raise

        intent_path.unlink(missing_ok=True)
        logger.info("cloud job submitted: %s (kind=%s)", job_id, kind)
        return job_id

    async def _stage_bundle(self, bundle_path: Path, size: int, kind: str) -> str:
        """Upload a large bundle via presigned PUT; return its R2 key.

        POST /v1/cloud/bundles/upload-url mints the target (and rejects
        sizes over the server ceiling with a message that tells the
        user to shrink the inputs or switch to lqh.sources.hf_dataset);
        the PUT streams the tarball straight to R2 so neither we nor
        the backend buffer it. Declaring the kind lets the server apply
        kind-level submit gates here, before the upload spends time.
        """
        async with httpx.AsyncClient(base_url=self._api_base, timeout=60.0) as client:
            resp = await client.post(
                "/v1/cloud/bundles/upload-url",
                json={
                    "project_id": self.project_dir.name,
                    "size_bytes": size,
                    "kind": kind,
                },
                headers=self._auth_headers(),
            )
            _raise_for_cloud_error(resp)
            data = resp.json()
        bundle_key = data["bundle_key"]
        upload_url = data["upload_url"]

        async def _chunks() -> AsyncIterator[bytes]:
            with bundle_path.open("rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        return
                    yield chunk

        # Timeout scales with size: assume a 1 MiB/s floor plus slack so
        # a maximum-size bundle on a slow uplink isn't killed mid-PUT
        # (2 GiB needs ~35 min at that floor; a flat 10 min would demand
        # ~27 Mbit/s).
        put_timeout = max(600.0, size / (1 << 20) + 120.0)
        # Explicit Content-Length: S3-style presigned PUTs reject
        # chunked transfer encoding, which httpx would otherwise use
        # for an iterable body. The URL's signature also binds this
        # exact length server-side.
        async with httpx.AsyncClient(timeout=httpx.Timeout(put_timeout, connect=30.0)) as client:
            resp = await client.put(
                upload_url,
                content=_chunks(),
                headers={
                    "Content-Type": "application/gzip",
                    "Content-Length": str(size),
                },
            )
            if resp.status_code < 200 or resp.status_code >= 300:
                raise CloudError(
                    f"bundle upload failed ({resp.status_code}): {resp.text[:200]}"
                )
        logger.info(
            "bundle staged via presigned PUT: %s (%.1f MiB)", bundle_key, size / 2**20
        )
        return bundle_key

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
        raw = snap.get("status", "")
        # Unlike the progress-file mirror (see _STATUS_MAP), the snapshot
        # feeds the TUI job watcher, which distinguishes cancelled from
        # failed for its terminal notifications.
        state = "cancelled" if raw == "cancelled" else _STATUS_MAP.get(raw, raw or "unknown")
        return JobStatus(state=state, error=snap.get("error"))

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
    # Cloud data generation (lqh.remote.data_gen) — before .infer so a
    # hypothetical *.data_gen_infer style module can't shadow it.
    if module.endswith(".data_gen") or cfg_type == "data_gen":
        return "data_gen"
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
