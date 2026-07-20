"""Best-effort, privacy-bounded first-party CLI telemetry.

Only the event names and metadata keys accepted by the backend can be queued.
Failures are swallowed by design: telemetry must never affect a workflow.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from lqh import __version__
from lqh.config import (
    config_dir,
    load_config,
    load_credential_account_id,
    load_credentials,
    save_credentials,
    telemetry_enabled,
)
from lqh.project_identity import ProjectIdentityError

QUEUE_MAX_BYTES = 1_048_576
QUEUE_TRIM_TARGET_BYTES = 786_432
BATCH_SIZE = 100
FLUSH_MAX_BATCHES = 20
FLUSH_TIME_BUDGET_SECONDS = 2.5
ACTIVE_GAP_CAP_SECONDS = 30 * 60
DEFERRED_WAIT_TIMEOUT_SECONDS = 0.25
_QUEUE_LOCK = threading.Lock()
_SEND_LOCK = threading.Lock()
_PROJECT_LOCK = threading.Lock()
_active: "TelemetryClient | None" = None

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows path
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX path
    msvcrt = None  # type: ignore[assignment]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _account_key(token: str | None) -> str | None:
    if not token:
        return None
    account_id = load_credential_account_id() if token == load_credentials() else None
    identity = f"account:{account_id}" if account_id else f"token:{token}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _root_from_api_base(base: str) -> str:
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-len("/v1")]
    return root


def _trusted_telemetry_root(root: str) -> bool:
    """Only send LQH credentials to an authenticated LQH or loopback origin."""
    parsed = urlsplit(root)
    host = (parsed.hostname or "").lower()
    if host == "api.lqh.ai" or host.endswith(".lqh.ai"):
        return parsed.scheme == "https"
    return host in {"localhost", "127.0.0.1", "::1"} and parsed.scheme in {"http", "https"}


def _telemetry_auth() -> tuple[str, str] | None:
    """Pair a telemetry bearer with the control-plane root it belongs to."""
    stored = load_credentials()
    if stored:
        from lqh.auth import api_root
        root = api_root()
        return (stored, root) if _trusted_telemetry_root(root) else None
    # The attributed endpoint rejects the unowned debug alias. In particular,
    # do not let it shadow a missing real credential and grow a retry queue.
    if os.environ.get("LQH_DEBUG_API_KEY"):
        return None
    from lqh.auth import get_token
    token = get_token()
    if not token:
        return None
    # Legacy config keys predate the device credential store. Only treat one
    # as telemetry-capable when it targets an LQH-owned or loopback control
    # plane. A third-party OpenAI-compatible key must never be sent either to
    # api.lqh.ai or to an unrelated provider's guessed telemetry endpoint.
    root = _root_from_api_base(load_config().api_base_url)
    if _trusted_telemetry_root(root):
        return token, root
    return None


@contextmanager
def _file_lock(path: Path, *, blocking: bool = True):
    """Cross-process lock; queue mutations remain additionally thread-locked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Some writable non-POSIX filesystems do not implement chmod. The
        # restrictive creation mode still applies where POSIX modes exist.
        pass
    handle = os.fdopen(fd, "a+b")
    acquired = True
    try:
        if fcntl is not None:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError:
                acquired = False
        elif msvcrt is not None:  # pragma: no cover - exercised on Windows
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            except OSError:
                acquired = False
        yield acquired
    finally:
        if acquired and fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif acquired and msvcrt is not None:  # pragma: no cover - Windows
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _meaningful_artifacts(project_dir: Path) -> bool:
    names = ("SPEC.md", "data_gen", "datasets", "runs", "other_specs", "evals", "prompts")
    return any((project_dir / name).exists() for name in names)


def ensure_project_identity(project_dir: Path) -> tuple[str, str]:
    """Return (stable UUID, new|pre_existing|reopened), creating it if needed.

    Identity ownership moved to ``lqh.project_identity`` (Phase 3):
    it is created unconditionally at startup, independent of telemetry
    consent or authentication. Telemetry only consumes the same UUID.
    """
    state_dir = project_dir / ".lqh"
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(state_dir, 0o700)
    except OSError:
        pass
    from lqh.project_identity import ensure_identity

    with _PROJECT_LOCK:
        identity, classification = ensure_identity(project_dir)
    try:
        os.chmod(state_dir / "project.json", 0o600)
    except OSError:
        pass
    return str(uuid.UUID(identity["project_id"])), classification


def get_project_id(project_dir: Path) -> str:
    return ensure_project_identity(project_dir)[0]


def set_active_telemetry(client: "TelemetryClient | None") -> None:
    global _active
    _active = client


def active_telemetry() -> "TelemetryClient | None":
    return _active


class TelemetryClient:
    def __init__(self, project_dir: Path, *, auto_mode: bool = False) -> None:
        self._state_lock = threading.RLock()
        self.project_dir = project_dir
        try:
            config = load_config()
            self.enabled = telemetry_enabled(config)
            self.consent_epoch = config.telemetry_consent_epoch
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            # Configuration/state storage is best-effort. A read-only home
            # must never prevent the CLI itself from starting.
            self.enabled = False
            self.consent_epoch = 0
        self._local_disabled = False
        self._needs_session_start = False
        self.session_id = str(uuid.uuid4())
        self.project_id: str | None = None
        try:
            self.project_state = "pre_existing" if _meaningful_artifacts(project_dir) else "new"
        except OSError:
            self.project_state = "pre_existing"
        self.mode = "auto" if auto_mode else "interactive"
        default_environment = "development" if os.environ.get("LQH_DEBUG_API_KEY") else "production"
        self.environment = os.environ.get("LQH_ENVIRONMENT", default_environment).lower()
        if self.environment not in {"production", "development", "test"}:
            self.environment = "development"
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.last_activity_mono = self.started_mono
        self.active_seconds = 0.0
        self.last_heartbeat_active_ms = 0
        self.user_turns = 0
        self.agent_turns = 0
        self.spec_workflow_id: str | None = None
        self.spec_started_wall = 0.0
        self.spec_started_mono = 0.0
        self.spec_started_active = 0.0
        self.spec_prior_active = 0.0
        self.spec_user_turns = 0
        self.spec_agent_turns = 0
        self.readiness_workflow_id: str | None = None
        self.readiness_started_wall = 0.0
        self.readiness_started_active = 0.0
        self.readiness_prior_active = 0.0
        self.generation_attempts = 0
        self.generation_succeeded = False
        self.generated_dataset_ids: set[str] = set()
        try:
            auth = _telemetry_auth() if self.enabled else None
            self.account_key = _account_key(auth[0]) if auth else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            self.account_key = None
            self.enabled = False
        if self.account_key is not None:
            try:
                self.project_id, self.project_state = ensure_project_identity(project_dir)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
                self.enabled = False
                self._local_disabled = True
        self.queue_path = self._queue_path_for(self.account_key)
        self._session_started = False
        if self.enabled:
            try:
                state_dir = config_dir()
                for path in state_dir.glob("telemetry*"):
                    if path.is_file():
                        os.chmod(path, 0o600)
                # Legacy global queues have no trustworthy account binding and
                # must never be submitted under whichever user happens to log in.
                legacy = state_dir / "telemetry_queue.jsonl"
                with _QUEUE_LOCK:
                    with _file_lock(legacy.with_suffix(".lock")):
                        legacy.unlink(missing_ok=True)
            except OSError:
                pass
            self._load_workflow_state()
        self._work_queue: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...], threading.Event, dict[str, Any]]] = queue.Queue(maxsize=1024)
        self._work_thread = threading.Thread(target=self._run_work_queue, name="lqh-telemetry", daemon=True)
        self._work_thread.start()

    def _run_work_queue(self) -> None:
        while True:
            callback, args, done, result = self._work_queue.get()
            try:
                result["value"] = callback(*args)
            except BaseException as exc:  # best-effort worker must survive every callback
                result["error"] = exc
            finally:
                done.set()
                self._work_queue.task_done()

    def defer(self, callback: Callable[..., Any], *args: Any) -> bool:
        """Queue ordered best-effort work without blocking the caller."""
        try:
            self._work_queue.put_nowait((callback, args, threading.Event(), {}))
            return True
        except queue.Full:
            return False

    async def run_deferred(
        self, callback: Callable[..., Any], *args: Any,
        timeout: float | None = DEFERRED_WAIT_TIMEOUT_SECONDS,
    ) -> Any:
        """Run ordered telemetry work without letting it stall product work.

        A timeout only detaches the waiter; the already-queued callback remains
        ordered and will still run. Privacy barriers may explicitly pass
        ``timeout=None`` when they must wait for queue capacity and completion.
        """
        done = threading.Event()
        result: dict[str, Any] = {}
        deadline = time.monotonic()+timeout if timeout is not None else None
        work = (callback, args, done, result)
        while True:
            try:
                self._work_queue.put_nowait(work)
                break
            except queue.Full:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                await asyncio.sleep(0.01)
        while not done.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.01)
        return result.get("value")

    def state_snapshot(self) -> tuple[bool, int, float, str | None]:
        """Return a coherent, non-mutating view safe for synchronous UI code."""
        with self._state_lock:
            return self.enabled, self.consent_epoch, self.active_seconds, self.account_key

    @staticmethod
    def _queue_path_for(account_key: str | None) -> Path:
        suffix = account_key or "unbound"
        # Constructing a path does not create state; writes remain guarded by
        # event() so a read-only home cannot fail client construction.
        return Path.home() / ".lqh" / f"telemetry_queue_{suffix}.jsonl"

    def _ensure_account_and_identity(self) -> bool:
        if not self._sync_consent():
            return False
        try:
            auth = _telemetry_auth()
            current_key = _account_key(auth[0]) if auth else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            return False
        if current_key is None:
            return False
        if self.account_key is None:
            with self._state_lock:
                self.account_key = current_key
                self.queue_path = self._queue_path_for(current_key)
        if current_key != self.account_key:
            return False
        if self.project_id is None:
            try:
                self.project_id, self.project_state = ensure_project_identity(self.project_dir)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
                with self._state_lock:
                    self.enabled = False
                    self._local_disabled = True
                return False
            self._load_workflow_state()
        return True

    def correlation_project_id(self) -> str | None:
        return self.project_id if self._ensure_account_and_identity() else None

    def consent_active(self, epoch: int) -> bool:
        """Return whether a workflow still belongs to the current consent era."""
        return self._sync_consent() and self.consent_epoch == epoch

    def cached_consent_active(self, epoch: int) -> bool:
        """Non-mutating consent check for synchronous UI callbacks.

        Event emission re-checks persisted consent on the ordered worker, so a
        stale positive result can at most schedule work that is later discarded.
        """
        enabled, consent_epoch, _active_seconds, _account_key = self.state_snapshot()
        return enabled and consent_epoch == epoch

    def is_enabled(self) -> bool:
        return self._sync_consent()

    def refresh_account_binding(self) -> None:
        """Adopt the account identity cached by a completed login."""
        if not self._sync_consent():
            return
        try:
            auth = _telemetry_auth()
            key = _account_key(auth[0]) if auth else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            return
        if key is None or key == self.account_key:
            return
        # Measurements and workflow state belong to the principal under which
        # they began. A login/account switch starts a fresh CLI session.
        with self._state_lock:
            self._reset_measurement_state()
            self.account_key = key
            self.queue_path = self._queue_path_for(key)
        if self.project_id is None:
            try:
                self.project_id, self.project_state = ensure_project_identity(self.project_dir)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
                with self._state_lock:
                    self.enabled = False
                    self._local_disabled = True
                return
        else:
            self.project_state = "reopened"
        self._load_workflow_state()
        with self._state_lock:
            self._needs_session_start = True

    def _sync_consent(self) -> bool:
        """Refresh persisted consent on the ordered telemetry worker.

        Callers outside the worker must use ``cached_consent_active`` or enqueue
        this operation with ``run_deferred``; this method can reset all measured
        state when another process changes the consent epoch.
        """
        try:
            config = load_config()
            enabled = telemetry_enabled(config)
            epoch = config.telemetry_consent_epoch
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            # Consent cannot be proven while its durable state is unreadable.
            # Fail closed until a later sync observes a valid config again.
            with self._state_lock:
                self.enabled = False
                self._reset_measurement_state()
            return False
        with self._state_lock:
            if epoch != self.consent_epoch:
                self.consent_epoch = epoch
                self._local_disabled = False
                self._reset_measurement_state()
                self.enabled = enabled
                self._needs_session_start = enabled
            elif self._local_disabled:
                return False
            elif not enabled and self.enabled:
                self.enabled = False
                self._reset_measurement_state()
            elif enabled and not self.enabled:
                self.enabled = True
                self._reset_measurement_state()
                self._needs_session_start = True
            return self.enabled

    @property
    def _project_state_path(self) -> Path:
        return self.project_dir / ".lqh" / "project.json"

    def _load_workflow_state(self) -> None:
        try:
            os.chmod(self._project_state_path, 0o600)
            state = json.loads(self._project_state_path.read_text())
            readiness = state.get("pipeline_readiness") or {}
            if readiness and readiness.get("account_key") == self.account_key and readiness.get("consent_epoch", 0) == self.consent_epoch:
                self.readiness_workflow_id = str(uuid.UUID(readiness["workflow_id"]))
                self.readiness_started_wall = float(readiness["started_at"])
                self.generation_attempts = int(readiness.get("generation_attempts", 0))
                self.generation_succeeded = bool(readiness.get("generation_succeeded", False))
                self.generated_dataset_ids = {
                    value for value in readiness.get("generated_dataset_ids", [])
                    if isinstance(value, str)
                }
                self.readiness_prior_active = float(readiness.get("active_seconds", 0))
            spec = state.get("spec_capture") or {}
            if spec and spec.get("account_key") == self.account_key and spec.get("consent_epoch", 0) == self.consent_epoch and not (self.project_dir / "SPEC.md").is_file():
                self.spec_workflow_id = str(uuid.UUID(spec["workflow_id"]))
                self.spec_started_wall = float(spec["started_at"])
                self.spec_started_mono = time.monotonic()
                self.spec_started_active = self.active_seconds
                self.spec_prior_active = float(spec.get("active_seconds", 0))
                self.spec_user_turns = int(spec.get("user_turns", 0))
                self.spec_agent_turns = int(spec.get("agent_turns", 0))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return

    def _claim_workflow_state(self, name: str, initial: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Atomically create or join a project-scoped workflow."""
        path = self._project_state_path
        try:
            with _PROJECT_LOCK:
                with _file_lock(path.with_suffix(".lock")):
                    state = json.loads(path.read_text())
                    existing = state.get(name)
                    if (isinstance(existing, dict) and existing.get("workflow_id")
                            and existing.get("account_key") == self.account_key
                            and existing.get("consent_epoch", 0) == self.consent_epoch):
                        return existing, False
                    state[name] = initial
                    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                    tmp.write_text(json.dumps(state, separators=(",", ":")) + "\n")
                    os.chmod(tmp, 0o600)
                    os.replace(tmp, path)
                    return initial, True
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            return initial, True

    def _clear_workflow_state(self, name: str, workflow_id: str) -> None:
        path = self._project_state_path
        try:
            with _PROJECT_LOCK:
                with _file_lock(path.with_suffix(".lock")):
                    state = json.loads(path.read_text())
                    existing = state.get(name)
                    if isinstance(existing, dict) and existing.get("workflow_id") == workflow_id:
                        state.pop(name, None)
                        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                        tmp.write_text(json.dumps(state, separators=(",", ":")) + "\n")
                        os.chmod(tmp, 0o600)
                        os.replace(tmp, path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            return

    def _save_workflow_state(self) -> None:
        try:
            path = self._project_state_path
            with _PROJECT_LOCK:
                with _file_lock(path.with_suffix(".lock")):
                    state = json.loads(path.read_text())
                    if self.readiness_workflow_id:
                        current = state.get("pipeline_readiness") or {}
                        if current.get("workflow_id") == self.readiness_workflow_id:
                            state["pipeline_readiness"] = {
                                "workflow_id": self.readiness_workflow_id,
                                "account_key": self.account_key,
                                "consent_epoch": self.consent_epoch,
                                "started_at": min(float(current.get("started_at", self.readiness_started_wall)), self.readiness_started_wall),
                                "generation_attempts": max(int(current.get("generation_attempts", 0)), self.generation_attempts),
                                "generation_succeeded": bool(current.get("generation_succeeded", False) or self.generation_succeeded),
                                "generated_dataset_ids": sorted(
                                    self.generated_dataset_ids | {
                                        value for value in current.get("generated_dataset_ids", [])
                                        if isinstance(value, str)
                                    }
                                ),
                                "active_seconds": max(float(current.get("active_seconds", 0)), self.readiness_prior_active + max(self.active_seconds-self.readiness_started_active, 0)),
                            }
                    if self.spec_workflow_id:
                        current = state.get("spec_capture") or {}
                        if current.get("workflow_id") == self.spec_workflow_id:
                            state["spec_capture"] = {
                                "workflow_id": self.spec_workflow_id,
                                "account_key": self.account_key,
                                "consent_epoch": self.consent_epoch,
                                "started_at": min(float(current.get("started_at", self.spec_started_wall)), self.spec_started_wall),
                                "active_seconds": max(float(current.get("active_seconds", 0)), self.spec_prior_active + max(self.active_seconds-self.spec_started_active, 0)),
                                "user_turns": max(int(current.get("user_turns", 0)), self.spec_user_turns),
                                "agent_turns": max(int(current.get("agent_turns", 0)), self.spec_agent_turns),
                            }
                    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                    tmp.write_text(json.dumps(state, separators=(",", ":")) + "\n")
                    os.chmod(tmp, 0o600)
                    os.replace(tmp, path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
            return

    def note_activity(self) -> None:
        if not self._sync_consent():
            return
        if self._needs_session_start:
            self.start_session()
        now = time.monotonic()
        with self._state_lock:
            self.active_seconds += min(max(now - self.last_activity_mono, 0), ACTIVE_GAP_CAP_SECONDS)
            self.last_activity_mono = now

    def event(self, event_name: str, metadata: dict[str, Any] | None = None, workflow_id: str | None = None) -> None:
        if not self._ensure_account_and_identity():
            return
        if self._needs_session_start and event_name not in {"session_started", "project_opened"}:
            self.start_session()
        consent_epoch = self.consent_epoch
        record = {
            "event_id": str(uuid.uuid4()),
            "occurred_at": _utcnow(),
            "cli_version": __version__,
            "environment": self.environment,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "event_name": event_name,
            "metadata": metadata or {},
            # Local-only durability marker. The sender removes this field
            # before building the server payload. Persisting it with every
            # queued event ensures a crash after an opt-out epoch is saved but
            # before the queue-clear barrier runs cannot resurrect old events
            # after telemetry is enabled again.
            "_consent_epoch": self.consent_epoch,
        }
        if workflow_id:
            record["workflow_id"] = workflow_id
        try:
            line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            # This machine-global lock closes the small race where another
            # process passed its consent check just before /telemetry off.
            with _file_lock(config_dir() / "telemetry_consent.lock"):
                if not self._sync_consent() or self.consent_epoch != consent_epoch:
                    return
                with _QUEUE_LOCK:
                    with _file_lock(self.queue_path.with_suffix(".lock")):
                        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
                        fd = os.open(self.queue_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                        os.chmod(self.queue_path, 0o600)
                        with os.fdopen(fd, "a", encoding="utf-8") as fh:
                            fh.write(line)
                        self._trim_queue_locked()
        except Exception:
            return

    def _trim_queue_locked(self) -> None:
        try:
            if self.queue_path.stat().st_size <= QUEUE_MAX_BYTES:
                return
            lines = self.queue_path.read_bytes().splitlines(keepends=True)
            decoded: list[dict[str, Any] | None] = []
            session_anchors: dict[str, int] = {}
            project_anchors: dict[tuple[str, str], int] = {}
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    record = None
                decoded.append(record if isinstance(record, dict) else None)
                if not isinstance(record, dict):
                    continue
                session_id = str(record.get("session_id") or "")
                project_id = str(record.get("project_id") or "")
                if record.get("event_name") == "session_started" and session_id:
                    session_anchors[session_id] = index
                elif record.get("event_name") == "project_opened" and session_id and project_id:
                    project_anchors[(session_id, project_id)] = index

            selected: set[int] = set()
            size = 0
            for index in range(len(lines)-1, -1, -1):
                required = {index}
                record = decoded[index]
                if record is not None:
                    session_id = str(record.get("session_id") or "")
                    project_id = str(record.get("project_id") or "")
                    if session_id in session_anchors:
                        required.add(session_anchors[session_id])
                    anchor = project_anchors.get((session_id, project_id))
                    if anchor is not None:
                        required.add(anchor)
                additions = required-selected
                added_size = sum(len(lines[item]) for item in additions)
                if size + added_size > QUEUE_TRIM_TARGET_BYTES:
                    break
                selected.update(additions)
                size += added_size
            tmp = self.queue_path.with_suffix(".tmp")
            tmp.write_bytes(b"".join(lines[index] for index in sorted(selected)))
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.queue_path)
        except OSError:
            pass

    async def flush(self) -> None:
        try:
            await self.run_deferred(self._flush_sync, timeout=3.5)
        except Exception:
            return

    def _flush_sync(self) -> None:
        if not _SEND_LOCK.acquire(blocking=False):
            return
        try:
            with _file_lock(self.queue_path.with_suffix(".send.lock"), blocking=False) as acquired:
                if acquired:
                    self._flush_sync_locked()
        finally:
            _SEND_LOCK.release()

    def _flush_sync_locked(self) -> None:
        import httpx
        deadline = time.monotonic() + FLUSH_TIME_BUDGET_SECONDS
        if not self._sync_consent():
            return
        auth = _telemetry_auth()
        if auth is None:
            return
        token, telemetry_root = auth
        current_key = _account_key(token)
        if current_key == self.account_key and token == load_credentials() and load_credential_account_id() is None:
            # One-time upgrade for credentials written by older CLIs. The
            # authenticated lookup proves the token's account, allowing its
            # token-keyed queue to move safely to a rotation-stable identity.
            try:
                me = httpx.get(
                    telemetry_root + "/api/auth/me",
                    headers={"Authorization": f"Bearer {token}"}, timeout=1.0,
                )
                if me.status_code == 200:
                    account_id = str((me.json().get("user") or {}).get("id") or "")
                    if account_id:
                        save_credentials(token, account_id)
                        new_key = _account_key(token)
                        if new_key and new_key != self.account_key:
                            self._migrate_queue_binding(new_key)
                            current_key = new_key
            except Exception:
                pass
        if current_key != self.account_key:
            return
        with _QUEUE_LOCK:
            with _file_lock(self.queue_path.with_suffix(".lock")):
                try:
                    lines = self.queue_path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    return
        consent_epoch = self.consent_epoch
        removed_lines: list[str] = []
        for offset in range(0, min(len(lines), BATCH_SIZE * FLUSH_MAX_BATCHES), BATCH_SIZE):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            batch_lines = lines[offset:offset+BATCH_SIZE]
            events = []
            sendable_lines: list[str] = []
            for line in batch_lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    removed_lines.append(line)
                    continue
                if not isinstance(event, dict):
                    removed_lines.append(line)
                    continue
                # Queues written before consent epochs existed implicitly
                # belong to epoch zero. Once the persisted epoch advances,
                # those and all explicitly older records are permanently
                # stale and must never be uploaded.
                queued_epoch = event.pop("_consent_epoch", 0)
                if (not isinstance(queued_epoch, int) or isinstance(queued_epoch, bool)
                        or queued_epoch != consent_epoch):
                    removed_lines.append(line)
                    continue
                events.append(event)
                sendable_lines.append(line)
            status_code = 202
            daily_quota_reached = False
            if events:
                response = httpx.post(
                    telemetry_root + "/v1/telemetry/events",
                    json={"schema_version": 1, "events": events},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=max(min(remaining, 1.5), 0.1),
                )
                status_code = response.status_code
                if status_code == 429:
                    try:
                        error = response.json().get("error") or {}
                        daily_quota_reached = error.get("message") == "telemetry daily admission limit reached"
                    except (AttributeError, TypeError, ValueError):
                        pass
            # A per-event 202 reports rejected poison records while accepting
            # the rest. Batch-shape 400/413 responses are permanently
            # incompatible. A normal rate-limit response remains queued.
            if status_code not in (200, 202, 204, 400, 413) and not daily_quota_reached:
                break
            removed_lines.extend(sendable_lines)
            if daily_quota_reached:
                break
        if not removed_lines:
            return
        with _QUEUE_LOCK:
            with _file_lock(self.queue_path.with_suffix(".lock")):
                try:
                    current = self.queue_path.read_text(encoding="utf-8").splitlines()
                    remaining_to_remove: dict[str, int] = {}
                    for line in removed_lines:
                        remaining_to_remove[line] = remaining_to_remove.get(line, 0) + 1
                    kept: list[str] = []
                    for line in current:
                        if remaining_to_remove.get(line, 0) > 0:
                            remaining_to_remove[line] -= 1
                        else:
                            kept.append(line)
                    tmp = self.queue_path.with_suffix(".tmp")
                    tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
                    os.chmod(tmp, 0o600)
                    os.replace(tmp, self.queue_path)
                except OSError:
                    pass

    def _migrate_queue_binding(self, new_key: str) -> None:
        old_path = self.queue_path
        new_path = self._queue_path_for(new_key)
        with _QUEUE_LOCK:
            lock_paths = sorted([old_path.with_suffix(".lock"), new_path.with_suffix(".lock")], key=str)
            with _file_lock(lock_paths[0]):
                with _file_lock(lock_paths[1]):
                    try:
                        old_lines = old_path.read_bytes() if old_path.exists() else b""
                        new_lines = new_path.read_bytes() if new_path.exists() else b""
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = new_path.with_name(f".{new_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                        tmp.write_bytes(new_lines + old_lines)
                        os.chmod(tmp, 0o600)
                        os.replace(tmp, new_path)
                        old_path.unlink(missing_ok=True)
                    except OSError:
                        return
        with self._state_lock:
            self.account_key = new_key
            self.queue_path = new_path
        with _QUEUE_LOCK:
            with _file_lock(new_path.with_suffix(".lock")):
                self._trim_queue_locked()

    def start_session(self) -> bool:
        if self._session_started or not self._ensure_account_and_identity():
            return False
        self._session_started = True
        self._needs_session_start = False
        self.event("session_started", {"mode": self.mode, "project_state": self.project_state})
        self.event("project_opened", {"project_state": self.project_state})
        return True

    def record_workflow_command(self, command: str) -> None:
        """Record a slash-command interaction and start its timed workflow."""
        self.note_activity()
        self.event("workflow_command", {"command": command})
        if command == "spec":
            self.start_spec_capture()
        elif command == "datagen":
            self.start_readiness()

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            # Serialize with the sender: once this returns, no earlier flush
            # can still upload and all measurements from the opted-out period
            # have been discarded.
            with _file_lock(config_dir() / "telemetry_consent.lock"):
                with _SEND_LOCK:
                    send_locks = {path.with_suffix(".send.lock") for path in config_dir().glob("telemetry_queue_*.jsonl")}
                    send_locks.add(self.queue_path.with_suffix(".send.lock"))
                    with ExitStack() as stack:
                        for path in sorted(send_locks, key=str):
                            stack.enter_context(_file_lock(path))
                        with self._state_lock:
                            self.enabled = False
                            self._local_disabled = True
                        self.clear_queue()
                        if self.spec_workflow_id:
                            self._clear_workflow_state("spec_capture", self.spec_workflow_id)
                        if self.readiness_workflow_id:
                            self._clear_workflow_state("pipeline_readiness", self.readiness_workflow_id)
                        with self._state_lock:
                            self._reset_measurement_state()
                            self.consent_epoch = load_config().telemetry_consent_epoch
            return
        if self.enabled and self._session_started:
            return
        with self._state_lock:
            self.enabled = True
            self._local_disabled = False
            self.consent_epoch = load_config().telemetry_consent_epoch
            self._reset_measurement_state()
        self.start_session()

    def _reset_measurement_state(self) -> None:
        with self._state_lock:
            self.session_id = str(uuid.uuid4())
            self.started_wall = time.time()
            self.started_mono = time.monotonic()
            self.last_activity_mono = self.started_mono
            self.active_seconds = 0.0
            self.last_heartbeat_active_ms = 0
            self.user_turns = 0
            self.agent_turns = 0
            self.spec_workflow_id = None
            self.spec_prior_active = 0.0
            self.spec_user_turns = 0
            self.spec_agent_turns = 0
            self.readiness_workflow_id = None
            self.readiness_prior_active = 0.0
            self.generation_attempts = 0
            self.generation_succeeded = False
            self.generated_dataset_ids = set()
            self._session_started = False
            self._needs_session_start = False

    def heartbeat(self) -> bool:
        if not self._sync_consent():
            return False
        if self._needs_session_start:
            self.start_session()
        if not self._session_started:
            return False
        active_ms = int(self.active_seconds * 1000)
        if active_ms <= self.last_heartbeat_active_ms:
            return False
        self.event("session_heartbeat", {"active_duration_ms": active_ms})
        self.last_heartbeat_active_ms = active_ms
        return True

    def record_user_turn(self, source: str = "message") -> None:
        if not self._sync_consent():
            return
        self.note_activity(); self.user_turns += 1
        self.event("user_turn", {"source": source})
        if self.spec_workflow_id is None:
            self.start_spec_capture(initial_user_turns=1)
        elif self.spec_workflow_id is not None:
            self.spec_user_turns += 1
            self._save_workflow_state()

    def start_spec_capture(self, *, initial_user_turns: int = 0) -> bool:
        """Start initial SPEC.md capture without classifying a slash command as a turn."""
        if (not self._ensure_account_and_identity() or self.spec_workflow_id is not None
                or (self.project_dir / "SPEC.md").is_file()):
            return False
        candidate = {
            "workflow_id": str(uuid.uuid4()), "started_at": time.time(),
            "account_key": self.account_key, "consent_epoch": self.consent_epoch,
            "active_seconds": 0.0, "user_turns": max(initial_user_turns, 0), "agent_turns": 0,
        }
        claimed, created = self._claim_workflow_state("spec_capture", candidate)
        self.spec_workflow_id = str(uuid.UUID(claimed["workflow_id"]))
        self.spec_started_wall = float(claimed["started_at"])
        self.spec_started_mono = time.monotonic()
        self.spec_started_active = self.active_seconds
        self.spec_prior_active = float(claimed.get("active_seconds", 0))
        self.spec_user_turns = int(claimed.get("user_turns", 0))
        self.spec_agent_turns = int(claimed.get("agent_turns", 0))
        if created:
            self.event("spec_capture_started", {"workflow_kind": "spec_capture"}, workflow_id=self.spec_workflow_id)
        self._save_workflow_state()
        return True

    def record_agent_turn(self) -> None:
        if not self._sync_consent():
            return
        self.note_activity(); self.agent_turns += 1; self.event("agent_turn")
        if self.spec_workflow_id is not None:
            self.spec_agent_turns += 1
            self._save_workflow_state()

    def maybe_spec_completed(self, path: str, succeeded: bool) -> None:
        if not succeeded or Path(path) != Path("SPEC.md") or self.spec_workflow_id is None:
            return
        self.note_activity()
        self.event("spec_capture_completed", {
            "workflow_kind":"spec_capture", "outcome":"succeeded",
            "wall_duration_ms":int(max(time.time()-self.spec_started_wall, 0)*1000),
            "active_duration_ms":int((self.spec_prior_active+max(self.active_seconds-self.spec_started_active, 0))*1000),
            "user_turns":self.spec_user_turns,"agent_turns":self.spec_agent_turns,
        }, workflow_id=self.spec_workflow_id)
        completed_id = self.spec_workflow_id
        self._clear_workflow_state("spec_capture", completed_id)
        self.spec_workflow_id = None

    def start_readiness(self) -> None:
        if not self._sync_consent():
            return
        if self.readiness_workflow_id is not None:
            return
        candidate = {"workflow_id": str(uuid.uuid4()), "account_key": self.account_key,
                     "consent_epoch": self.consent_epoch, "started_at": time.time(),
                     "generation_attempts": 0, "generation_succeeded": False,
                     "generated_dataset_ids": [],
                     "active_seconds": 0.0}
        claimed, created = self._claim_workflow_state("pipeline_readiness", candidate)
        self.readiness_workflow_id = str(uuid.UUID(claimed["workflow_id"]))
        self.readiness_started_wall = float(claimed["started_at"])
        self.readiness_started_active = self.active_seconds
        self.readiness_prior_active = float(claimed.get("active_seconds", 0))
        self.generation_attempts = int(claimed.get("generation_attempts", 0))
        self.generation_succeeded = bool(claimed.get("generation_succeeded", False))
        self.generated_dataset_ids = {
            value for value in claimed.get("generated_dataset_ids", [])
            if isinstance(value, str)
        }
        if created:
            self.event("pipeline_readiness_started", {
                "workflow_kind":"pipeline_readiness", "attempt_count":0,
            }, workflow_id=self.readiness_workflow_id)
        self._save_workflow_state()

    def record_generation_attempt(self) -> None:
        if not self._sync_consent():
            return
        self.start_readiness()
        self.generation_attempts += 1
        self._save_workflow_state()

    def _dataset_identity(self, candidate: str | Path) -> str | None:
        try:
            project = self.project_dir.resolve()
            path = Path(candidate)
            if not path.is_absolute():
                path = self.project_dir / path
            relative = path.resolve().relative_to(project)
            return hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
        except (OSError, ValueError):
            return None

    def record_generation_succeeded(self, output_path: str | Path) -> None:
        """Mark that the readiness workflow produced at least one usable sample."""
        if not self._sync_consent():
            return
        self.start_readiness()
        dataset_id = self._dataset_identity(output_path)
        if dataset_id is None:
            return
        self.generation_succeeded = True
        self.generated_dataset_ids.add(dataset_id)
        self._save_workflow_state()

    def _readiness_uses_current_data(self, arguments: dict[str, Any]) -> bool:
        """Prove the accepted launch references project data created in this workflow."""
        candidates: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        for key in ("dataset", "eval_dataset"):
            if key in arguments:
                collect(arguments[key])
        for candidate in candidates:
            dataset_id = self._dataset_identity(candidate)
            if dataset_id is not None and dataset_id in self.generated_dataset_ids:
                return True
        return False

    def complete_readiness(self, arguments: dict[str, Any] | None = None) -> None:
        if (not self._sync_consent() or self.readiness_workflow_id is None
                or self.generation_attempts < 1 or not self.generation_succeeded
                or not self._readiness_uses_current_data(arguments or {})):
            return
        self.note_activity()
        self.event("pipeline_readiness_completed", {
            "workflow_kind":"pipeline_readiness", "outcome":"succeeded",
            "attempt_count":self.generation_attempts,
            "wall_duration_ms":int(max(time.time()-self.readiness_started_wall, 0)*1000),
            "active_duration_ms":int((self.readiness_prior_active+max(self.active_seconds-self.readiness_started_active, 0))*1000),
        }, workflow_id=self.readiness_workflow_id)
        completed_id = self.readiness_workflow_id
        self._clear_workflow_state("pipeline_readiness", completed_id)
        self.readiness_workflow_id = None

    def end_session(self, outcome: str = "succeeded") -> None:
        if not self._session_started:
            return
        self.note_activity()
        if self.spec_workflow_id is not None:
            self.spec_prior_active += max(self.active_seconds-self.spec_started_active, 0)
            self.spec_started_active = self.active_seconds
            self._save_workflow_state()
        if self.readiness_workflow_id is not None:
            self.readiness_prior_active += max(self.active_seconds-self.readiness_started_active, 0)
            self.readiness_started_active = self.active_seconds
            self._save_workflow_state()
        self.event("session_ended", {"outcome":outcome,"wall_duration_ms":int(max(time.time()-self.started_wall, 0)*1000),"active_duration_ms":int(self.active_seconds*1000)})

    def clear_queue(self) -> None:
        try:
            with _QUEUE_LOCK:
                paths = list(config_dir().glob("telemetry_queue_*.jsonl"))
                paths.append(config_dir() / "telemetry_queue.jsonl")
                for path in paths:
                    with _file_lock(path.with_suffix(".lock")):
                        path.unlink(missing_ok=True)
        except OSError: pass


def notice_needed() -> bool:
    try:
        marker = config_dir() / "telemetry_notice_v1"
        if marker.exists() or not telemetry_enabled():
            return False
        marker.write_text("shown\n")
        os.chmod(marker, 0o600)
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ProjectIdentityError):
        return False
