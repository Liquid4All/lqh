"""Durable artifact store client.

Phase 0 of cloud fine-tuning introduces R2 as the canonical store for
post-job artifacts (checkpoints, predictions, metrics, eval results).
Bytes go client ↔ R2 directly via presigned URLs issued by
``api.lqh.ai``; we never proxy them through the backend.

Both the cloud backend and the SSH backends use this same store, so
the artifact-handle is the universal handoff between "thing produced"
and "thing consumed" regardless of where the producer ran.

Wire format mirrors the OpenAPI schemas declared in
``backend/api/openapi.yaml`` (``ArtifactRegisterRequest``,
``ArtifactUploadURLRequest`` etc.). Bumping either side without the
other will surface as a 400 from the backend — there's no codegen
on the Python side.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable

import httpx

from lqh.auth import api_root, require_token

__all__ = [
    "ArtifactHandle",
    "ArtifactKind",
    "ArtifactStore",
    "BackendArtifactStore",
    "ArtifactError",
    "ArtifactNotConfigured",
]


# The kinds here mirror the CHECK constraint in migration 0006_artifacts.
# Keep in sync with backend/internal/db/artifacts.go ArtifactKind.
ArtifactKind = str  # one of: checkpoint predictions metrics logs eval_result dataset bundle gguf other

def _augment_lineage_from_env(lineage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stamp environment-derived image provenance into a lineage payload.

    The cloud-job launcher injects ``LQH_IMAGE_ID`` and
    ``LQH_IMAGE_PURPOSE`` for every sandbox (see
    handler/cloud_jobs.go). When a caller hands us a lineage dict
    without those fields set, we fill them in here so the registered
    ``artifact_lineage`` row records which container image produced
    the artifact — needed to reproduce an old checkpoint against its
    exact image build (image registry, migration 0016).

    Caller-supplied values always win: a trainer that knows better
    than the env (e.g. an image override for a one-off rebuild)
    won't be overwritten.

    Returns the (possibly augmented) dict, or None if input is None.
    Idempotent on already-stamped inputs.
    """
    if lineage is None:
        return None
    out = dict(lineage)  # shallow copy so we don't mutate the caller's dict
    if not out.get("image_id"):
        if v := os.environ.get("LQH_IMAGE_ID"):
            out["image_id"] = v
    if not out.get("image_purpose"):
        if v := os.environ.get("LQH_IMAGE_PURPOSE"):
            out["image_purpose"] = v
    return out


_VALID_KINDS = frozenset(
    {
        "checkpoint",
        "predictions",
        "metrics",
        "logs",
        "eval_result",
        "dataset",
        "bundle",
        "gguf",
        "other",
    }
)


class ArtifactError(RuntimeError):
    """Raised when the backend rejects an artifact operation."""


class ArtifactNotConfigured(ArtifactError):
    """Raised when the backend has no R2 configured (503)."""


@dataclass(frozen=True)
class ArtifactHandle:
    """Reference to a registered artifact.

    The ``id`` is the canonical handle — pass it between agents,
    runs, and future eval steps. ``r2_key`` is informational; clients
    should always go through ``signed_url`` to fetch bytes (the key
    by itself is not a download URL).
    """

    id: str
    kind: ArtifactKind
    project_id: str
    size_bytes: int
    r2_key: str
    job_id: str | None = None
    sha256: str | None = None
    hf_repo: str | None = None
    created_at: str | None = None
    pinned: bool = False
    # When the retention engine may auto-expire this artifact. None =
    # never (pinned, or a never-expire kind like metrics/eval_result).
    expires_at: str | None = None
    # "final" / "intermediate" for checkpoints; None otherwise.
    checkpoint_role: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ArtifactHandle":
        return cls(
            id=data["id"],
            kind=data.get("kind", "other"),
            project_id=data.get("project_id", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            r2_key=data.get("r2_key", ""),
            job_id=data.get("job_id"),
            sha256=data.get("sha256"),
            hf_repo=data.get("hf_repo"),
            created_at=data.get("created_at"),
            pinned=bool(data.get("pinned", False)),
            expires_at=data.get("expires_at"),
            checkpoint_role=data.get("checkpoint_role"),
        )


@runtime_checkable
class ArtifactStore(Protocol):
    """Abstract store. ``BackendArtifactStore`` is the only impl today;
    a ``LocalArtifactStore`` may follow for tests."""

    async def upload_file(
        self,
        path: Path,
        *,
        project_id: str,
        kind: ArtifactKind,
        job_id: str | None = None,
        sha256: str | None = None,
    ) -> ArtifactHandle: ...

    async def signed_url(self, handle: ArtifactHandle | str) -> str: ...

    async def download(self, handle: ArtifactHandle | str, dest: Path) -> None: ...

    async def list_for_project(
        self,
        project_id: str,
        *,
        kind: ArtifactKind | None = None,
        limit: int = 100,
    ) -> list[ArtifactHandle]: ...

    async def delete(self, handle: ArtifactHandle | str) -> None: ...


# Bytes per chunk for streaming uploads/downloads. Big enough to keep
# the syscall rate low; small enough that progress reporting (later)
# can update at a reasonable cadence.
_CHUNK = 1 << 20  # 1 MiB


class BackendArtifactStore:
    """Default ArtifactStore — talks to api.lqh.ai with a bearer token.

    Uploads go: ``POST /v1/artifacts/upload-url`` → PUT to R2 →
    ``POST /v1/artifacts/register``. Downloads go:
    ``GET /v1/artifacts/{id}/url`` → GET from R2. Bytes never traverse
    api.lqh.ai.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        # api_root() returns the host without the /v1 segment; artifact
        # endpoints live under /v1/ so we keep the segment explicit in
        # each request path.
        self._base = (api_base or api_root()).rstrip("/")
        self._token = token  # resolved lazily so tests can construct without auth
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        tok = self._token or require_token()
        return {"Authorization": f"Bearer {tok}"}

    async def _request_upload_url(
        self,
        client: httpx.AsyncClient,
        *,
        project_id: str,
        kind: ArtifactKind,
        job_id: str | None,
        filename: str | None,
        content_type: str | None,
    ) -> tuple[str, str]:
        body: dict[str, Any] = {"project_id": project_id, "kind": kind}
        if job_id:
            body["job_id"] = job_id
        if filename:
            body["filename"] = filename
        if content_type:
            body["content_type"] = content_type
        r = await client.post(
            "/v1/artifacts/upload-url",
            json=body,
            headers=self._auth_headers(),
        )
        _raise_for_artifact_error(r)
        data = r.json()
        return data["r2_key"], data["upload_url"]

    async def _register(
        self,
        client: httpx.AsyncClient,
        *,
        project_id: str,
        kind: ArtifactKind,
        r2_key: str,
        size_bytes: int,
        job_id: str | None,
        sha256: str | None,
        hf_repo: str | None,
        lineage: dict[str, Any] | None = None,
        checkpoint_role: str | None = None,
    ) -> ArtifactHandle:
        body: dict[str, Any] = {
            "project_id": project_id,
            "kind": kind,
            "r2_key": r2_key,
            "size_bytes": size_bytes,
        }
        if job_id:
            body["job_id"] = job_id
        if sha256:
            body["sha256"] = sha256
        if hf_repo:
            body["hf_repo"] = hf_repo
        if lineage:
            body["lineage"] = lineage
        if checkpoint_role:
            body["checkpoint_role"] = checkpoint_role
        r = await client.post(
            "/v1/artifacts/register",
            json=body,
            headers=self._auth_headers(),
        )
        _raise_for_artifact_error(r)
        data = r.json()
        return ArtifactHandle(
            id=data["id"],
            kind=kind,
            project_id=project_id,
            size_bytes=size_bytes,
            r2_key=data["r2_key"],
            job_id=job_id,
            sha256=sha256,
            hf_repo=hf_repo,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        path: Path,
        *,
        project_id: str,
        kind: ArtifactKind,
        job_id: str | None = None,
        sha256: str | None = None,
        lineage: dict[str, Any] | None = None,
        checkpoint_role: str | None = None,
        hf_repo: str | None = None,
    ) -> ArtifactHandle:
        """Upload ``path`` to R2, then register it as an artifact.

        sha256 is computed locally if not supplied — checksum protects
        against silent corruption on PUT.

        ``lineage`` is an optional dict matching ArtifactLineageInput
        in the OpenAPI spec — set when the caller knows the artifact's
        provenance (training method, base model, parents, metrics, ...).

        ``hf_repo`` records that the artifact bytes were also published to
        this Hugging Face repo (set it only after the push succeeds).
        """
        if kind not in _VALID_KINDS:
            raise ValueError(f"invalid artifact kind: {kind!r}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if sha256 is None:
            sha256 = _sha256_file(path)

        # Stamp image / purpose from env when not pinned by caller.
        # Idempotent + safe outside the sandbox (env vars absent → no-op).
        lineage = _augment_lineage_from_env(lineage)

        async with httpx.AsyncClient(base_url=self._base, timeout=self._timeout) as client:
            r2_key, upload_url = await self._request_upload_url(
                client,
                project_id=project_id,
                kind=kind,
                job_id=job_id,
                filename=path.name,
                content_type=None,
            )
            # httpx.AsyncClient requires an async byte stream; we open
            # the file in binary mode and yield chunks via an async
            # generator so multi-GB checkpoints don't load into memory.
            put = await client.put(
                upload_url,
                content=_iter_file_async(path, _CHUNK),
                headers={"Content-Length": str(size)},
                timeout=httpx.Timeout(self._timeout * 10),  # large for big files
            )
            if put.status_code not in (200, 201):
                raise ArtifactError(
                    f"R2 upload failed ({put.status_code}): {put.text[:200]}"
                )
            return await self._register(
                client,
                project_id=project_id,
                kind=kind,
                r2_key=r2_key,
                size_bytes=size,
                job_id=job_id,
                sha256=sha256,
                hf_repo=hf_repo,
                lineage=lineage,
                checkpoint_role=checkpoint_role,
            )

    async def signed_url(self, handle: ArtifactHandle | str) -> str:
        artifact_id = handle.id if isinstance(handle, ArtifactHandle) else handle
        async with httpx.AsyncClient(base_url=self._base, timeout=self._timeout) as client:
            r = await client.get(
                f"/v1/artifacts/{artifact_id}/url",
                headers=self._auth_headers(),
            )
            _raise_for_artifact_error(r)
            return r.json()["url"]

    async def download(self, handle: ArtifactHandle | str, dest: Path) -> None:
        """Stream the artifact bytes to ``dest``.

        Atomic against partial writes — bytes go to ``<dest>.part``
        and are renamed on success. The signed URL is fetched fresh
        per call so resumes after a long delay still work.
        """
        url = await self.signed_url(handle)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout * 10)) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise ArtifactError(
                        f"R2 download failed ({resp.status_code}): {body[:200]!r}"
                    )
                with tmp.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(_CHUNK):
                        fh.write(chunk)
        os.replace(tmp, dest)

    async def list_for_project(
        self,
        project_id: str,
        *,
        kind: ArtifactKind | None = None,
        limit: int = 100,
    ) -> list[ArtifactHandle]:
        params: dict[str, str] = {"limit": str(limit)}
        if kind:
            params["kind"] = kind
        async with httpx.AsyncClient(base_url=self._base, timeout=self._timeout) as client:
            r = await client.get(
                f"/v1/projects/{project_id}/artifacts",
                params=params,
                headers=self._auth_headers(),
            )
            _raise_for_artifact_error(r)
            items = r.json().get("artifacts", [])
            return [ArtifactHandle.from_json(it) for it in items]

    async def delete(self, handle: ArtifactHandle | str) -> None:
        artifact_id = handle.id if isinstance(handle, ArtifactHandle) else handle
        async with httpx.AsyncClient(base_url=self._base, timeout=self._timeout) as client:
            r = await client.delete(
                f"/v1/artifacts/{artifact_id}",
                headers=self._auth_headers(),
            )
            _raise_for_artifact_error(r)

    async def pin(self, handle: ArtifactHandle | str) -> None:
        """Pin an artifact so the backend retention engine never
        auto-expires it (clears its expires_at)."""
        artifact_id = handle.id if isinstance(handle, ArtifactHandle) else handle
        async with httpx.AsyncClient(base_url=self._base, timeout=self._timeout) as client:
            r = await client.post(
                f"/v1/artifacts/{artifact_id}/pin",
                headers=self._auth_headers(),
            )
            _raise_for_artifact_error(r)

    async def unpin(self, handle: ArtifactHandle | str) -> None:
        """Unpin an artifact, re-arming the per-kind expiry clock."""
        artifact_id = handle.id if isinstance(handle, ArtifactHandle) else handle
        async with httpx.AsyncClient(base_url=self._base, timeout=self._timeout) as client:
            r = await client.post(
                f"/v1/artifacts/{artifact_id}/unpin",
                headers=self._auth_headers(),
            )
            _raise_for_artifact_error(r)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


async def _iter_file_async(path: Path, chunk: int) -> AsyncIterator[bytes]:
    """Yield ``chunk``-sized blocks from ``path`` without loading the
    whole file into memory. Stays sync under the hood (file I/O is fast
    relative to network), but is exposed as async so httpx's
    AsyncClient is happy."""
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            yield buf


def _raise_for_artifact_error(r: httpx.Response) -> None:
    if 200 <= r.status_code < 300:
        return
    if r.status_code == 503:
        raise ArtifactNotConfigured("artifact store not configured on backend")
    try:
        body = r.json()
        msg = body.get("error", {}).get("message", r.text[:200])
    except Exception:
        msg = r.text[:200]
    raise ArtifactError(f"{r.status_code}: {msg}")
