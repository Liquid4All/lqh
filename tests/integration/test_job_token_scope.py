"""Integration test for the job-token scope enforcement added in
Step 2 of the training-infra build-out + the cross-project /
cross-job tightening from the review pass.

Why this test needs DB access: the plaintext of a job_tokens row is
only ever returned to the sandbox via cloud-secret injection — the
backend never exposes it through the API. To replay a token against
the routes, we either submit a real cloud job and scrape the token
out of the DB, or insert a synthetic row directly. We go with the
synthetic-insert path so the test is fast (no GPU cost, no wait).

The API base resolves via the standard lqh config path
(``default_api_base_url`` — env or ``~/.config/lqh/credentials``).
The job-token plaintext can't be obtained via the API (the backend
never returns it), so this test additionally needs direct DB access
via ``LQH_TEST_DATABASE_URL`` to insert a synthetic row.

Skipped unless:
  - the lqh client knows where the backend is (default_api_base_url
    resolves to a non-empty URL), AND
  - ``LQH_TEST_DATABASE_URL`` is set, AND
  - asyncpg is installed.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import unittest
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from lqh.auth import api_root
from lqh.config import default_api_base_url


def _enabled() -> tuple[bool, str]:
    if not default_api_base_url():
        return False, "no API base URL (set LQH_BASE_URL or run /login)"
    if not os.environ.get("LQH_TEST_DATABASE_URL"):
        return False, "LQH_TEST_DATABASE_URL not set (direct asyncpg required)"
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        return False, "asyncpg not installed; pip install asyncpg"
    return True, ""


def _hash_token(plaintext: str) -> str:
    """Mirror auth.HashToken — hex sha256."""
    return hashlib.sha256(plaintext.encode()).hexdigest()


async def _mint_test_job_token(
    dsn: str,
    *,
    project_id: str,
    scopes: list[str],
) -> tuple[str, str, str]:
    """Insert a cloud_jobs + job_tokens row pair and return
    (plaintext_token, job_id, project_id). The cloud_jobs row is in
    'running' status so LookupJobToken's join passes.

    Uses the bootstrap admin's (user_id, organization_id) since
    those are the only IDs we can be sure exist on a freshly-
    deployed stack. Cleanup is the caller's responsibility (we
    return the job_id).
    """
    import asyncpg

    plaintext = secrets.token_urlsafe(32)
    token_hash = _hash_token(plaintext)
    job_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    conn = await asyncpg.connect(dsn)
    try:
        # Pick any super_admin user + their org. The bootstrap
        # admin is super_admin by construction; without one the
        # test can't run.
        row = await conn.fetchrow(
            "SELECT id, organization_id FROM users "
            "WHERE role = 'super_admin' LIMIT 1"
        )
        if row is None:
            raise RuntimeError(
                "no super_admin user in LQH_TEST_DATABASE_URL — "
                "did bootstrap.Run get a chance to insert one?"
            )
        user_id = row["id"]
        org_id = row["organization_id"]

        # Insert a cloud_jobs row in 'running' status. Use a kind
        # the CHECK accepts; resource_spec defaults to '{}'.
        await conn.execute(
            "INSERT INTO cloud_jobs (id, user_id, organization_id, "
            "project_id, kind, status, provider, est_cost_micros) "
            "VALUES ($1, $2, $3, $4, 'train_sft', 'running', 'test', 0)",
            uuid.UUID(job_id), user_id, org_id, project_id,
        )
        await conn.execute(
            "INSERT INTO job_tokens "
            "(job_id, user_id, organization_id, project_id, "
            " token_hash, scopes, expires_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
            uuid.UUID(job_id), user_id, org_id, project_id,
            token_hash, scopes, expires_at,
        )
        return plaintext, job_id, project_id
    finally:
        await conn.close()


async def _cleanup_test_job(dsn: str, job_id: str) -> None:
    import asyncpg
    conn = await asyncpg.connect(dsn)
    try:
        # ON DELETE CASCADE on job_tokens.job_id takes care of the token row.
        await conn.execute("DELETE FROM cloud_jobs WHERE id = $1", uuid.UUID(job_id))
    finally:
        await conn.close()


@unittest.skipUnless(_enabled()[0], _enabled()[1])
class TestJobTokenScope(unittest.IsolatedAsyncioTestCase):
    """Replay a synthetic job token against the various routes that
    accept (or refuse) job-token bearers."""

    async def asyncSetUp(self) -> None:
        # api_root() strips trailing /v1 so /v1/projects, /v1/cloud,
        # /v1/account paths land single-prefixed.
        self._base = api_root().rstrip("/")
        self._dsn = os.environ["LQH_TEST_DATABASE_URL"]
        self._project_id = f"test_jobtoken_{int(__import__('time').time())}"
        self._token, self._job_id, _ = await _mint_test_job_token(
            self._dsn,
            project_id=self._project_id,
            scopes=["chat.score", "artifacts.write", "projects.read"],
        )
        self._hdr = {"Authorization": f"Bearer {self._token}"}

    async def asyncTearDown(self) -> None:
        await _cleanup_test_job(self._dsn, self._job_id)

    async def test_chat_score_only_allows_judges(self):
        """`chat.score` scope: judge model OK, non-judge → 403."""
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            # Non-judge model → 403.
            resp = await client.post(
                "/v1/chat/completions",
                headers=self._hdr,
                json={
                    "model": "small",  # pool model, not a judge
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(resp.status_code, 403, resp.text)
            # Judge model passes scope; may 5xx or 400 downstream but
            # the gate doesn't trip. Just assert it's NOT 403.
            resp = await client.post(
                "/v1/chat/completions",
                headers=self._hdr,
                json={
                    "model": "judge:small",
                    "messages": [{"role": "user", "content": "Rate 5"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "schema": {
                                "type": "object",
                                "properties": {"score": {"type": "integer"}},
                                "required": ["score"],
                            },
                        },
                    },
                },
            )
            self.assertNotEqual(resp.status_code, 403,
                                f"judge model rejected by scope check: {resp.text}")

    async def test_cross_project_denied(self):
        """A job token bound to project X must 403 on Y."""
        other_pid = self._project_id + "_other"
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            resp = await client.get(
                f"/v1/projects/{other_pid}",
                headers=self._hdr,
            )
            # Endpoint also 404s for non-existent project. We want
            # the 403 path, which fires BEFORE the DB lookup.
            self.assertEqual(resp.status_code, 403,
                             f"cross-project access not gated: {resp.status_code} {resp.text}")
            resp = await client.get(
                f"/v1/projects/{other_pid}/artifacts",
                headers=self._hdr,
            )
            self.assertEqual(resp.status_code, 403, resp.text)
            resp = await client.get(
                f"/v1/projects/{other_pid}/lineage",
                headers=self._hdr,
            )
            self.assertEqual(resp.status_code, 403, resp.text)

    async def test_cloud_jobs_denied_entirely(self):
        """DenyJobToken on /v1/cloud/jobs/*: sandbox shouldn't be
        able to submit new jobs, list, cancel, or stream them."""
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            # Submit attempt (multipart not necessary; the deny gate
            # fires before parsing).
            resp = await client.post("/v1/cloud/jobs", headers=self._hdr)
            self.assertEqual(resp.status_code, 403, resp.text)
            # Snapshot on the token's own job — still denied,
            # because the deny gate is by AuthMethod, not by JobID.
            resp = await client.get(
                f"/v1/cloud/jobs/{self._job_id}",
                headers=self._hdr,
            )
            self.assertEqual(resp.status_code, 403, resp.text)

    async def test_hf_token_endpoints_denied(self):
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            resp = await client.get("/v1/account/hf_token", headers=self._hdr)
            self.assertEqual(resp.status_code, 403, resp.text)
            resp = await client.post(
                "/v1/account/hf_token", headers=self._hdr,
                json={"token": "hf_x"},
            )
            self.assertEqual(resp.status_code, 403, resp.text)

    async def test_projects_list_denied(self):
        """Listing all projects via /v1/projects is denied for job
        tokens — they should hit /v1/projects/{bound-pid} directly."""
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            resp = await client.get("/v1/projects", headers=self._hdr)
            self.assertEqual(resp.status_code, 403, resp.text)

    async def test_artifact_delete_denied(self):
        """Artifact delete is no longer in artifacts.write — it's
        DenyJobToken on the route."""
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            # Use a random uuid; the deny gate runs before the DB
            # lookup, so a 403 here is the gate firing (a 404 would
            # mean the gate passed and we hit the not-found branch).
            random_id = str(uuid.uuid4())
            resp = await client.delete(
                f"/v1/artifacts/{random_id}",
                headers=self._hdr,
            )
            self.assertEqual(resp.status_code, 403, resp.text)

    async def test_revoked_token_rejected(self):
        """Revoke + retry: token should stop authenticating."""
        import asyncpg
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(
                "UPDATE job_tokens SET revoked_at = now() "
                "WHERE job_id = $1",
                uuid.UUID(self._job_id),
            )
        finally:
            await conn.close()
        async with httpx.AsyncClient(base_url=self._base, timeout=30.0) as client:
            resp = await client.get(
                f"/v1/projects/{self._project_id}",
                headers=self._hdr,
            )
            # Without an authenticated principal the artifacts
            # handler's requirePrincipal helper returns 401.
            self.assertEqual(resp.status_code, 401, resp.text)


if __name__ == "__main__":
    unittest.main()
