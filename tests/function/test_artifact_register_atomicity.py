"""Integration test for the artifact register atomicity fix from
the review pass (High-2).

Before the fix, /v1/artifacts/register inserted the artifact row
first and then validated/inserted the lineage. A malformed lineage
payload returned 400 AFTER the artifact row was created, so a
retry hit the r2_key UNIQUE constraint and failed unpredictably.

After the fix, lineage is validated before any DB write and both
inserts run in one transaction. A 400 must leave no artifact row.

Verification strategy: count artifacts under the test project_id
via /v1/projects/{pid}/artifacts before, between, and after the
attempts. The pid is unique per run so we don't collide with
prior tests.

Resolves API base + token the same way the lqh CLI does:
``lqh.config.default_api_base_url()`` and ``lqh.auth.get_token()``
read from env (``LQH_BASE_URL``, ``LQH_API_TOKEN``) and then from
``~/.config/lqh/credentials``. The test only skips when nothing
resolves at all — no extra env-var ceremony required for an
already-logged-in dev. The token must belong to a user with a real
DB principal (artifact endpoints reject the debug-key alias).
"""

from __future__ import annotations

import time
import unittest

import httpx

from lqh.auth import api_root, get_token
from lqh.config import default_api_base_url


def _enabled() -> tuple[bool, str]:
    if not default_api_base_url():
        return False, "no API base URL (set LQH_BASE_URL or run /login)"
    if not get_token():
        return False, "no lqh auth token (run /login or set LQH_API_TOKEN)"
    return True, ""


@unittest.skipUnless(_enabled()[0], _enabled()[1])
class TestArtifactRegisterAtomicity(unittest.TestCase):
    def setUp(self) -> None:
        # api_root() strips trailing /v1 so we can post to "/v1/..."
        # paths cleanly; default_api_base_url() leaves /v1 attached
        # (it's the OpenAI SDK base) and would produce /v1/v1/... .
        self._base = api_root().rstrip("/")
        self._hdr = {
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json",
        }
        self._project_id = f"test_atomicity_{int(time.time())}"

    def _list_artifacts(self) -> list[dict]:
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            resp = client.get(
                f"/v1/projects/{self._project_id}/artifacts",
                headers={"Authorization": self._hdr["Authorization"]},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json().get("artifacts", [])

    def _request_upload_url(self) -> tuple[str, str]:
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            resp = client.post(
                "/v1/artifacts/upload-url",
                headers=self._hdr,
                json={
                    "project_id": self._project_id,
                    "kind": "metrics",
                    "filename": "atomicity.json",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        return body["r2_key"], body["upload_url"]

    @staticmethod
    def _put_bytes(upload_url: str, payload: bytes) -> None:
        # Presigned R2 PUT; no auth header.
        with httpx.Client(timeout=60.0) as client:
            put = client.put(
                upload_url,
                content=payload,
                headers={"Content-Length": str(len(payload))},
            )
        if put.status_code not in (200, 201):
            raise RuntimeError(f"R2 PUT failed: {put.status_code} {put.text[:200]}")

    def test_malformed_lineage_leaves_no_orphan(self):
        # Project starts empty.
        before = self._list_artifacts()
        self.assertEqual(len(before), 0,
                         f"project {self._project_id} not empty at start; check uniqueness")

        # Stage bytes once. We reuse the r2_key across both register
        # attempts — that's what a retrying client would do.
        r2_key, upload_url = self._request_upload_url()
        payload = b'{"smoke": true}'
        self._put_bytes(upload_url, payload)

        # Attempt 1: malformed lineage (invalid artifact_kind). The
        # backend's buildLineageParams rejects this with 400. Before
        # the fix, an artifact row was already in the DB.
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            resp_bad = client.post(
                "/v1/artifacts/register",
                headers=self._hdr,
                json={
                    "project_id": self._project_id,
                    "kind": "metrics",
                    "r2_key": r2_key,
                    "size_bytes": len(payload),
                    "lineage": {
                        "artifact_kind": "not_a_valid_lineage_kind",
                    },
                },
            )
        self.assertEqual(resp_bad.status_code, 400,
                         f"expected 400 on malformed lineage; got {resp_bad.status_code}: {resp_bad.text}")

        # No artifact row should exist yet.
        mid = self._list_artifacts()
        self.assertEqual(
            len(mid), 0,
            f"orphan artifact row found after malformed-lineage 400: {mid}",
        )

        # Attempt 2: valid lineage on the SAME r2_key — must succeed.
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            resp_ok = client.post(
                "/v1/artifacts/register",
                headers=self._hdr,
                json={
                    "project_id": self._project_id,
                    "kind": "metrics",
                    "r2_key": r2_key,
                    "size_bytes": len(payload),
                    "lineage": {
                        "artifact_kind": "other",
                        "hyperparams": {"smoke": True},
                    },
                },
            )
        self.assertEqual(resp_ok.status_code, 201,
                         f"valid-lineage register failed: {resp_ok.status_code}: {resp_ok.text}")
        new_id = resp_ok.json()["id"]

        after = self._list_artifacts()
        self.assertEqual(len(after), 1,
                         f"expected exactly 1 artifact row, got {len(after)}: {after}")
        self.assertEqual(after[0]["id"], new_id)


if __name__ == "__main__":
    unittest.main()
