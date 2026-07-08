"""ArtifactStore round-trip against an in-process mock backend.

Validates that ``BackendArtifactStore`` builds the wire bodies the
backend expects (per ``backend/api/openapi.yaml`` schemas). The bytes
themselves are exchanged via two layers of presigned URLs, so the
test sets up a synthetic R2 layer as well — both signed URLs point
at the same MockTransport.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from lqh.artifacts import (
    ArtifactError,
    ArtifactHandle,
    ArtifactNotConfigured,
    BackendArtifactStore,
)


# ----------------------------------------------------------------------
# Mock backend ↔ R2
# ----------------------------------------------------------------------


class _FakeBackend:
    """In-memory stand-in for api.lqh.ai + R2.

    The backend issues "signed" URLs that point back at itself with a
    distinct path prefix; that way one MockTransport handles both
    layers (the real R2 hostname is never resolved). All state is
    keyed by R2 key.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.artifacts: dict[str, dict] = {}  # id -> registered row
        self._counter = 0

    # mock-only helpers
    def _new_id(self) -> str:
        self._counter += 1
        return f"art-{self._counter:04d}"

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        path = url.path

        # ----- R2 layer (presigned PUT / GET) -----
        if path.startswith("/r2/"):
            key = path[len("/r2/"):]
            if request.method == "PUT":
                self.objects[key] = request.content
                return httpx.Response(200)
            if request.method == "GET":
                if key not in self.objects:
                    return httpx.Response(404)
                return httpx.Response(200, content=self.objects[key])
            return httpx.Response(405)

        # ----- backend layer -----
        if path == "/v1/artifacts/upload-url" and request.method == "POST":
            body = _json(request)
            assert body["project_id"], "project_id missing"
            assert body["kind"], "kind missing"
            key = f"u/{body['project_id']}/{body['kind']}/{self._new_id()}"
            return httpx.Response(
                200,
                json={
                    "r2_key": key,
                    "upload_url": str(url.copy_with(path=f"/r2/{key}")),
                    "expires_in_seconds": 60,
                },
            )

        if path == "/v1/artifacts/register" and request.method == "POST":
            body = _json(request)
            key = body["r2_key"]
            if key not in self.objects:
                return httpx.Response(
                    400, json={"error": {"message": "no object at r2_key"}}
                )
            stored = self.objects[key]
            if body.get("size_bytes") and body["size_bytes"] != len(stored):
                return httpx.Response(
                    400, json={"error": {"message": "size mismatch"}}
                )
            art_id = self._new_id()
            self.artifacts[art_id] = {
                "id": art_id,
                "project_id": body["project_id"],
                "kind": body["kind"],
                "r2_key": key,
                "size_bytes": len(stored),
                "sha256": body.get("sha256"),
                "job_id": body.get("job_id"),
                "hf_repo": body.get("hf_repo"),
            }
            return httpx.Response(201, json={"id": art_id, "r2_key": key})

        if path.startswith("/v1/artifacts/") and path.endswith("/url") and request.method == "GET":
            art_id = path.split("/")[3]
            if art_id not in self.artifacts:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            row = self.artifacts[art_id]
            return httpx.Response(
                200,
                json={
                    "url": str(url.copy_with(path=f"/r2/{row['r2_key']}")),
                    "expires_in_seconds": 60,
                },
            )

        if path.startswith("/v1/projects/") and path.endswith("/artifacts") and request.method == "GET":
            pid = path.split("/")[3]
            kind = url.params.get("kind")
            rows = [
                r for r in self.artifacts.values()
                if r["project_id"] == pid and (not kind or r["kind"] == kind)
            ]
            return httpx.Response(200, json={"artifacts": rows})

        if path.startswith("/v1/artifacts/") and request.method == "DELETE":
            art_id = path.split("/")[3]
            if art_id not in self.artifacts:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            del self.artifacts[art_id]
            return httpx.Response(204)

        return httpx.Response(404, json={"error": {"message": "not mocked"}})


def _json(req: httpx.Request) -> dict:
    import json as _json_mod

    return _json_mod.loads(req.content.decode("utf-8"))


# ----------------------------------------------------------------------
# Store wrapper that injects MockTransport
# ----------------------------------------------------------------------


class _MockStore(BackendArtifactStore):
    """BackendArtifactStore variant that points all httpx clients at a
    shared MockTransport instead of the real network."""

    def __init__(self, backend: _FakeBackend) -> None:
        super().__init__(api_base="https://mock.lqh.test", token="test-token")
        self._backend = backend

    # Override the AsyncClient factory by patching the two callers.
    # In production, BackendArtifactStore opens fresh clients per call,
    # which makes mocking tricky — so we monkeypatch the module's
    # httpx.AsyncClient at test time instead. See ``patch_httpx``.


@pytest.fixture
def fake_backend(monkeypatch):
    be = _FakeBackend()
    transport = httpx.MockTransport(be.handler)

    real_async_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("lqh.artifacts.httpx.AsyncClient", _patched)
    return be


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_upload_register_download_roundtrip(tmp_path, fake_backend):
    src = tmp_path / "predictions.parquet"
    payload = b"hello, world\n" * 1000
    src.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()

    store = BackendArtifactStore(api_base="https://mock.lqh.test", token="t")

    async def go():
        h = await store.upload_file(
            src, project_id="demo", kind="predictions"
        )
        assert isinstance(h, ArtifactHandle)
        assert h.kind == "predictions"
        assert h.size_bytes == len(payload)

        url = await store.signed_url(h)
        assert url.startswith("https://mock.lqh.test/r2/")

        dest = tmp_path / "out.parquet"
        await store.download(h, dest)
        assert dest.read_bytes() == payload

        # Listing returns the artifact we just made.
        items = await store.list_for_project("demo", kind="predictions")
        assert len(items) == 1
        assert items[0].id == h.id

        # Sha computed locally and sent matches the backend record.
        backend_row = fake_backend.artifacts[h.id]
        assert backend_row["sha256"] == expected_sha

        # Cleanup.
        await store.delete(h)
        items2 = await store.list_for_project("demo")
        assert items2 == []

    asyncio.run(go())


def test_invalid_kind_rejected(tmp_path, fake_backend):
    store = BackendArtifactStore(api_base="https://mock.lqh.test", token="t")
    src = tmp_path / "x.bin"
    src.write_bytes(b"abc")

    async def go():
        with pytest.raises(ValueError):
            await store.upload_file(src, project_id="demo", kind="not-a-kind")

    asyncio.run(go())


def test_503_maps_to_not_configured(tmp_path, monkeypatch):
    """If the backend returns 503 (R2 unconfigured), surface a clear
    typed exception so callers can fall back gracefully."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "artifact store not configured"}})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("lqh.artifacts.httpx.AsyncClient", _patched)

    store = BackendArtifactStore(api_base="https://mock.lqh.test", token="t")
    src = tmp_path / "x.bin"
    src.write_bytes(b"abc")

    async def go():
        with pytest.raises(ArtifactNotConfigured):
            await store.upload_file(src, project_id="demo", kind="metrics")

    asyncio.run(go())


def test_register_size_mismatch_rejected(tmp_path, monkeypatch):
    """If the local-computed size doesn't match what's actually in R2,
    the backend's 400 should surface as an ArtifactError — protects
    against accidental truncation on the PUT side."""

    be = _FakeBackend()
    # Simulate a partial upload: trim the stored payload one byte
    # short of the size the client reports.
    original_handler = be.handler

    def trimming_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/r2/") and request.method == "PUT":
            be.objects[request.url.path[len("/r2/"):]] = request.content[:-1]
            return httpx.Response(200)
        return original_handler(request)

    transport = httpx.MockTransport(trimming_handler)
    real_async_client = httpx.AsyncClient

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("lqh.artifacts.httpx.AsyncClient", _patched)

    store = BackendArtifactStore(api_base="https://mock.lqh.test", token="t")
    src = tmp_path / "blob"
    src.write_bytes(b"x" * 256)

    async def go():
        with pytest.raises(ArtifactError):
            await store.upload_file(src, project_id="demo", kind="other")

    asyncio.run(go())
