"""A donated HF token must reach the wire and nothing else.

Specifically it must never appear in a tool result, a prompt, an
exception message, the run directory, the uploaded bundle, or the
session JSONL — because everything in that list ends up either in the
model's context or in the backend's payload capture.

Every test here pairs an absence assertion with a *presence* assertion on
the request body. Without that negative control the whole file would
still pass if token resolution silently broke.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from lqh.remote.backend import RemoteConfig
from lqh.remote.cloud import CloudBackend

SENTINEL = "hf_SENTINEL_DO_NOT_LEAK_0123456789"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class _CapturingBackend:
    """Records the submit body; optionally echoes it back in an error."""

    def __init__(self, *, fail_status: int | None = None, echo_body: bool = False):
        self.last_meta: dict | None = None
        self.last_bundle: bytes | None = None
        self.fail_status = fail_status
        self.echo_body = echo_body

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        self.last_meta = _extract_part(body, b'name="meta"', as_json=True)
        self.last_bundle = _extract_part(body, b'name="bundle"', as_json=False)
        if self.fail_status is not None:
            text = body.decode("utf-8", "replace")[:2000] if self.echo_body else "bad"
            return httpx.Response(
                self.fail_status, json={"error": {"message": f"validation failed: {text}"}}
            )
        return httpx.Response(201, json={"job_id": "job-leak-test", "status": "pending"})


def _extract_part(body: bytes, marker: bytes, *, as_json: bool):
    i = body.find(marker)
    if i < 0:
        return None
    start = body.index(b"\r\n\r\n", i) + 4
    end = body.index(b"\r\n--", start)
    chunk = body[start:end]
    if not as_json:
        return chunk
    try:
        return json.loads(chunk.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


@pytest.fixture
def capturing_cloud(monkeypatch):
    def _make(**kwargs):
        be = _CapturingBackend(**kwargs)
        transport = httpx.MockTransport(be.handler)
        real = httpx.AsyncClient

        def _patched(*a, **kw):
            kw.setdefault("transport", transport)
            return real(*a, **kw)

        monkeypatch.setattr("lqh.remote.cloud.httpx.AsyncClient", _patched)
        return be

    return _make


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "runs" / "r1").mkdir(parents=True)
    (project / "datasets").mkdir()
    (project / "datasets" / "train.parquet").write_bytes(b"rows")
    # The token lives in a project .env — the case this feature is for.
    (project / ".env").write_text(f"HF_TOKEN={SENTINEL}\n")
    return project


def _backend(project: Path) -> CloudBackend:
    return CloudBackend(
        RemoteConfig(
            name="cloud", type="cloud",
            hostname="api.mock.lqh.test", remote_root="cloud:test",
        ),
        project,
        api_base="https://mock.lqh.test",
        token="test-token",
    )


def _config() -> dict:
    return {"manifest": ["dataset"], "dataset": "datasets/train.parquet"}


# ---------------------------------------------------------------------------
# the token reaches the wire (negative control for everything below)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_donated_token_lands_in_meta(tmp_path, capturing_cloud):
    project = _project(tmp_path)
    be = capturing_cloud()

    await _backend(project).submit_run(
        str(project / "runs" / "r1"), _config(), donate_hf_token=True,
    )

    assert be.last_meta["hf_token"] == SENTINEL


@pytest.mark.asyncio
async def test_no_donation_means_no_token_field(tmp_path, capturing_cloud):
    project = _project(tmp_path)
    be = capturing_cloud()

    await _backend(project).submit_run(str(project / "runs" / "r1"), _config())

    assert "hf_token" not in be.last_meta


@pytest.mark.asyncio
async def test_opt_out_wins_over_the_donate_flag(tmp_path, capturing_cloud, monkeypatch):
    project = _project(tmp_path)
    be = capturing_cloud()
    monkeypatch.setenv("LQH_HF_DONATE", "0")

    await _backend(project).submit_run(
        str(project / "runs" / "r1"), _config(), donate_hf_token=True,
    )

    assert "hf_token" not in be.last_meta


# ---------------------------------------------------------------------------
# ...and nowhere else
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_not_in_bundle_or_config(tmp_path, capturing_cloud):
    """config.json is written to the run dir AND tarred into the upload."""
    project = _project(tmp_path)
    be = capturing_cloud()

    await _backend(project).submit_run(
        str(project / "runs" / "r1"), _config(), donate_hf_token=True,
    )

    assert SENTINEL in json.dumps(be.last_meta)  # control
    with tarfile.open(fileobj=io.BytesIO(be.last_bundle), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            assert SENTINEL.encode() not in tar.extractfile(m).read(), m.name


@pytest.mark.asyncio
async def test_token_not_in_run_dir_state_files(tmp_path, capturing_cloud):
    project = _project(tmp_path)
    run_dir = project / "runs" / "r1"
    capturing_cloud()

    await _backend(project).submit_run(str(run_dir), _config(), donate_hf_token=True)

    for path in run_dir.rglob("*"):
        if path.is_file():
            assert SENTINEL not in path.read_text(errors="replace"), path.name


@pytest.mark.asyncio
async def test_token_not_echoed_back_through_a_server_error(tmp_path, capturing_cloud):
    """The one path an external system can put the token back in our hands.

    A backend that echoes the request body in a validation error would
    otherwise land the token in a tool result, and from there in the
    session log and the payload capture.
    """
    project = _project(tmp_path)
    be = capturing_cloud(fail_status=400, echo_body=True)

    with pytest.raises(Exception) as excinfo:
        await _backend(project).submit_run(
            str(project / "runs" / "r1"), _config(), donate_hf_token=True,
        )

    assert be.last_meta["hf_token"] == SENTINEL  # control: it was sent
    assert SENTINEL not in str(excinfo.value)
    assert SENTINEL not in repr(excinfo.value)


@pytest.mark.asyncio
async def test_transfer_error_scrubs_the_token(tmp_path, monkeypatch):
    """Same property for the transfer client, which has its own error path."""
    from lqh.remote import transfer

    project = _project(tmp_path)
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["body"] = json.loads(request.content)
        # Echo the request back, the shape that leaks.
        return httpx.Response(400, text=f"bad request: {request.content.decode()}")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda *a, **kw: real(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )
    monkeypatch.setattr(transfer, "require_token", lambda: "t", raising=False)
    monkeypatch.setattr("lqh.auth.require_token", lambda: "t")

    with pytest.raises(RuntimeError) as excinfo:
        await transfer.submit_transfer(
            project_id="p",
            source_artifact_id="a",
            target_hf_repo="me/model",
            project_dir=project,
            donate_hf_token=True,
        )

    assert sent["body"]["hf_token"] == SENTINEL  # control
    assert SENTINEL not in str(excinfo.value)


# ---------------------------------------------------------------------------
# handler surface: prompts and tool results
# ---------------------------------------------------------------------------


def test_donation_prompt_never_shows_the_token(tmp_path):
    from lqh.tools.handlers import _resolve_hf_donation
    from lqh.tools.permissions import PermissionContext

    project = _project(tmp_path)
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "training")

    assert donate is False
    assert prompt is not None
    assert SENTINEL not in (prompt.question or "")
    assert SENTINEL not in prompt.content
    assert prompt.secret is None
    # It should still say enough to be a real consent prompt: where the
    # token came from, how long we keep it, and what declining does.
    q = prompt.question
    assert ".env" in q
    assert "deleted" in q
    assert "not added to your LQH account" in q
    assert "Declining still runs the job" in q
    # And it must not overclaim in either direction.
    assert "not stored" not in q.lower()
    assert "won't have HuggingFace access" not in q


def test_disclosure_line_never_shows_the_token(tmp_path):
    from lqh.hf_token import hf_disclosure_line

    project = _project(tmp_path)
    line = hf_disclosure_line(project)

    assert SENTINEL not in line
    assert ".env" in line


@pytest.mark.asyncio
async def test_token_split_by_truncation_is_still_scrubbed(tmp_path, capturing_cloud):
    """Redaction has to run before truncation.

    Slicing first can cut a token in half, leaving a prefix that exact
    replacement no longer matches — and a credential prefix in a
    transcript is still a credential leak.
    """
    project = _project(tmp_path)
    # Push the echoed token across the [:200]/[:300] cutoff.
    be = capturing_cloud(fail_status=400, echo_body=True)

    with pytest.raises(Exception) as excinfo:
        await _backend(project).submit_run(
            str(project / "runs" / "r1"), _config(), donate_hf_token=True,
        )

    assert be.last_meta["hf_token"] == SENTINEL  # control
    msg = str(excinfo.value)
    # No full token, and no recognizable fragment of one either.
    assert SENTINEL not in msg
    for size in (32, 24, 16, 12, 8):
        assert SENTINEL[:size] not in msg, f"{size}-char prefix survived: {msg}"
