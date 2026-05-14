from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_DEFAULT_API_BASE_URL = "https://api.lqh.ai/v1"


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


def config_dir() -> Path:
    path = Path.home() / ".lqh"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> LqhConfig:
    path = config_path()
    if not path.exists():
        return LqhConfig()
    data: dict[str, object] = json.loads(path.read_text())
    return LqhConfig(
        api_key=data.get("api_key"),  # type: ignore[arg-type]
        api_base_url=data.get("api_base_url", default_api_base_url()),  # type: ignore[arg-type]
    )


def save_config(config: LqhConfig) -> None:
    path = config_path()
    path.write_text(json.dumps(asdict(config), indent=2) + "\n")


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


def save_credentials(token: str) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"token": token}))
    tmp.chmod(0o600)
    tmp.replace(path)
