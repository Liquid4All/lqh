from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows path
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX path
    msvcrt = None  # type: ignore[assignment]

_DEFAULT_API_BASE_URL = "https://api.lqh.ai/v1"
_CONFIG_LOCK = threading.RLock()


def default_api_base_url() -> str:
    """Resolve the API base URL.

    Honours ``LQH_BASE_URL`` so a staging environment or a third-party
    OpenAI-compatible API can be used without code changes.
    """
    return os.environ.get("LQH_BASE_URL", _DEFAULT_API_BASE_URL)


@dataclass
class LqhConfig:
    api_key: str | None = None
    api_base_url: str = field(default_factory=default_api_base_url)
    # Default compute target, used when neither the tool invocation nor
    # a per-project ``.lqh/compute.json`` overrides it. One of:
    # ``"cloud"`` (LQH Cloud), ``"ssh:<name>"`` (a configured SSH remote),
    # or ``None`` (not yet chosen → first-run picker fires).
    default_compute: str | None = None
    # First-party, operational CLI telemetry. Environment variable
    # LQH_TELEMETRY has final precedence; see telemetry_enabled().
    telemetry_enabled: bool = True
    # Monotonically changes whenever consent is explicitly changed. Running
    # CLI processes use this to discard measurements captured under an older
    # consent state instead of uploading them after a later re-enable.
    telemetry_consent_epoch: int = 0


def config_dir() -> Path:
    path = Path.home() / ".lqh"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        # Read-only/sandboxed homes still need to support read-only commands;
        # individual telemetry files are independently created as 0600.
        pass
    return path


def config_path() -> Path:
    return config_dir() / "config.json"


def _load_config_unlocked() -> LqhConfig:
    path = config_path()
    if not path.exists():
        return LqhConfig()
    data: dict[str, object] = json.loads(path.read_text())
    return LqhConfig(
        api_key=data.get("api_key"),  # type: ignore[arg-type]
        api_base_url=data.get("api_base_url", default_api_base_url()),  # type: ignore[arg-type]
        default_compute=data.get("default_compute"),  # type: ignore[arg-type]
        telemetry_enabled=bool(data.get("telemetry_enabled", True)),
        telemetry_consent_epoch=max(int(data.get("telemetry_consent_epoch", 0)), 0),
    )


@contextmanager
def _config_file_lock():
    """Serialize config read-modify-write operations across CLI processes."""
    path = config_dir() / "config.lock"
    with _CONFIG_LOCK:
        with path.open("a+b") as handle:
            try:
                path.chmod(0o600)
            except OSError:
                pass
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows path
                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows path
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def load_config() -> LqhConfig:
    # Writers publish by atomic replacement, so readers see either the old or
    # new complete document without taking the cross-process writer lock.
    return _load_config_unlocked()


def _save_config_unlocked(config: LqhConfig) -> None:
    path = config_path()
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            json.dump(asdict(config), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.chmod(0o600)
        os.replace(tmp, path)
        path.chmod(0o600)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def save_config(config: LqhConfig) -> None:
    """Atomically replace the full config with private file permissions."""
    with _config_file_lock():
        current = _load_config_unlocked()
        if current.telemetry_consent_epoch > config.telemetry_consent_epoch:
            # A caller may have loaded the whole config before another process
            # changed consent. The epoch is monotonic: never let that stale
            # object resurrect an older telemetry state.
            config = replace(
                config,
                telemetry_enabled=current.telemetry_enabled,
                telemetry_consent_epoch=current.telemetry_consent_epoch,
            )
        _save_config_unlocked(config)


def update_config(mutator: Callable[[LqhConfig], None]) -> LqhConfig:
    """Atomically update selected fields without clobbering concurrent writes."""
    with _config_file_lock():
        config = _load_config_unlocked()
        mutator(config)
        _save_config_unlocked(config)
        return config


def telemetry_enabled(config: LqhConfig | None = None) -> bool:
    """Resolve telemetry opt-in. LQH_TELEMETRY overrides persisted config."""
    raw = os.environ.get("LQH_TELEMETRY")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return (config or load_config()).telemetry_enabled


def credentials_path() -> Path:
    return Path.home() / ".config" / "lqh" / "credentials"


def load_credentials() -> str | None:
    path = credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    token = data.get("token")
    return token if isinstance(token, str) else None


def load_credential_account_id() -> str | None:
    path = credentials_path()
    try:
        value = json.loads(path.read_text()).get("account_id")
        return value if isinstance(value, str) and value else None
    except (json.JSONDecodeError, OSError):
        return None


def save_credentials(token: str, account_id: str | None = None) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    payload = {"token": token}
    if account_id:
        payload["account_id"] = account_id
    tmp: Path | None = None
    try:
        # NamedTemporaryFile creates the bearer-containing file as 0600 rather
        # than exposing it under the process umask before a later chmod.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as fh:
            json.dump(payload, fh)
            tmp = Path(fh.name)
        tmp.chmod(0o600)
        tmp.replace(path)
        path.chmod(0o600)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
