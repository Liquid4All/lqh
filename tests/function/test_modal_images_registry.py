"""Integration test for the image registry endpoints
(Step 6). Exercises register + promote + list + rollback round-trip.

The endpoint path on the backend is ``/v1/admin/gpu_images``
(legacy table name — the GPU provider is backend-implemented and
not a user-facing detail).

Resolves API base + token via the standard lqh config path
(``default_api_base_url`` + ``get_token``). The currently-logged-in
user must be a super_admin — the registry routes are gated by
``RequireSuperAdmin`` and the debug-key alias does NOT work (see
the docstring on backend/internal/router/router.go).

Cleanup: registered rows are NOT deleted (the registry deliberately
does not expose a delete endpoint — rows are provenance for
artifact_lineage). The test uses a unique `notes` tag per run so
operator inspection can spot the test entries and prune them by
hand if needed.
"""

from __future__ import annotations

import os
import time
import unittest

import httpx

from lqh.auth import api_root, get_token
from lqh.config import default_api_base_url


def _enabled() -> tuple[bool, str]:
    if not default_api_base_url():
        return False, "no API base URL (set LQH_BASE_URL or run /login)"
    if not get_token():
        return False, "no lqh auth token (run /login as a super-admin)"
    return True, ""


@unittest.skipUnless(_enabled()[0], _enabled()[1])
class TestImageRegistry(unittest.TestCase):
    """Verify register → promote → list → rollback contract."""

    def setUp(self) -> None:
        # api_root() strips trailing /v1 so /v1/admin/... is single-prefixed.
        self._base = api_root().rstrip("/")
        self._hdr = {
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json",
        }
        self._tag = f"integration-test-{int(time.time())}"
        # Snapshot the currently-active sft row BEFORE we register +
        # promote any test rows. tearDown re-promotes it so the
        # registry isn't left pointing at a fake im-test-... id that
        # would break every subsequent SFT cloud job (we did exactly
        # that and it took a debug round to notice).
        rows = self._get(
            "/v1/admin/gpu_images", {"purpose": "sft"}
        ).json().get("images", [])
        active = [r for r in rows if r.get("is_active")]
        self._restore_id = active[0]["id"] if active else None

    def tearDown(self) -> None:
        # Best-effort: re-promote the pre-test active row so any
        # follow-up cloud job sees a real image. If there was no
        # active row at setUp time, leave whatever the test promoted
        # alone — operator can clean it up.
        if self._restore_id is None:
            return
        try:
            self._post(f"/v1/admin/gpu_images/{self._restore_id}/promote")
        except Exception:
            # Don't fail the test on cleanup; operator can manually
            # promote via the build script's --promote flag.
            pass

    def _post(self, path: str, body: dict | None = None) -> httpx.Response:
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            return client.post(path, headers=self._hdr, json=body or {})

    def _get(self, path: str, params: dict | None = None) -> httpx.Response:
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            return client.get(path, headers=self._hdr, params=params or {})

    def test_register_promote_rollback(self):
        # 1. Register two rows for purpose=sft. We use the test_
        # prefix on image_id so they're obvious in a manual scan.
        resp_a = self._post("/v1/admin/gpu_images", {
            "purpose": "sft",
            "image_id": f"im-test-{self._tag}-A",
            "git_sha": "deadbeefcafe",
            "deps_hash": "0123456789abcdef",
            "notes": f"integration test {self._tag} row A",
        })
        self.assertEqual(resp_a.status_code, 201, resp_a.text)
        id_a = resp_a.json()["id"]
        self.assertFalse(resp_a.json()["is_active"], "new rows must start inactive")

        resp_b = self._post("/v1/admin/gpu_images", {
            "purpose": "sft",
            "image_id": f"im-test-{self._tag}-B",
            "git_sha": "feedfacefeed",
            "deps_hash": "fedcba9876543210",
            "notes": f"integration test {self._tag} row B",
        })
        self.assertEqual(resp_b.status_code, 201, resp_b.text)
        id_b = resp_b.json()["id"]

        # 2. List should include both, newest first.
        resp_list = self._get("/v1/admin/gpu_images", {"purpose": "sft"})
        self.assertEqual(resp_list.status_code, 200, resp_list.text)
        rows = resp_list.json()["images"]
        ids = [r["id"] for r in rows]
        self.assertIn(id_a, ids)
        self.assertIn(id_b, ids)
        # Either of our two should be ahead of any pre-existing
        # rows by created_at. Don't assert exact ordering since
        # other test/operator activity could interleave.

        # 3. Promote A → A.is_active=true.
        resp_promote = self._post(f"/v1/admin/gpu_images/{id_a}/promote")
        self.assertEqual(resp_promote.status_code, 200, resp_promote.text)
        self.assertTrue(resp_promote.json()["is_active"])
        self.assertIsNotNone(resp_promote.json().get("promoted_at"))

        # 4. List again; A must be the only active row for sft.
        rows = self._get("/v1/admin/gpu_images", {"purpose": "sft"}).json()["images"]
        actives = [r for r in rows if r["is_active"]]
        self.assertEqual(len(actives), 1, f"expected exactly 1 active row, got {len(actives)}")
        self.assertEqual(actives[0]["id"], id_a)

        # 5. Promote B (the rollback / forward-move case).
        resp_promote = self._post(f"/v1/admin/gpu_images/{id_b}/promote")
        self.assertEqual(resp_promote.status_code, 200)
        rows = self._get("/v1/admin/gpu_images", {"purpose": "sft"}).json()["images"]
        actives = [r for r in rows if r["is_active"]]
        self.assertEqual(len(actives), 1, "promote did not demote previous active row")
        self.assertEqual(actives[0]["id"], id_b)

        # 6. Idempotency: promote B again, expect no error and B
        # stays active. promoted_at SHOULD have bumped.
        first_promoted_at = actives[0]["promoted_at"]
        time.sleep(1.1)  # ensure now() ticks past the previous value
        resp_promote = self._post(f"/v1/admin/gpu_images/{id_b}/promote")
        self.assertEqual(resp_promote.status_code, 200)
        rows = self._get("/v1/admin/gpu_images", {"purpose": "sft"}).json()["images"]
        b_row = next(r for r in rows if r["id"] == id_b)
        self.assertTrue(b_row["is_active"])
        self.assertNotEqual(b_row["promoted_at"], first_promoted_at,
                            "re-promote did not bump promoted_at")

        # 7. Rollback: promote A back to active.
        self._post(f"/v1/admin/gpu_images/{id_a}/promote")
        rows = self._get("/v1/admin/gpu_images", {"purpose": "sft"}).json()["images"]
        actives = [r for r in rows if r["is_active"]]
        self.assertEqual(len(actives), 1)
        self.assertEqual(actives[0]["id"], id_a, "rollback did not flip active back to A")

    def test_purpose_validation(self):
        # Unknown purpose on register → 400.
        resp = self._post("/v1/admin/gpu_images", {
            "purpose": "not_a_real_purpose",
            "image_id": "im-x",
            "git_sha": "x",
            "deps_hash": "y",
        })
        self.assertEqual(resp.status_code, 400, resp.text)
        # Unknown purpose on list → 400.
        resp = self._get("/v1/admin/gpu_images", {"purpose": "also_not_real"})
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_promote_unknown_id(self):
        # Random uuid that doesn't exist → 404.
        import uuid as _uuid
        resp = self._post(f"/v1/admin/gpu_images/{_uuid.uuid4()}/promote")
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_non_super_admin_rejected(self):
        """A non-super-admin token must be rejected.

        Skipped silently when only one token is available — this
        check requires a second LQH_API_TOKEN_NONADMIN that
        authenticates to a regular user. CI is expected to provide it.
        """
        non_admin = os.environ.get("LQH_API_TOKEN_NONADMIN")
        if not non_admin:
            self.skipTest("no LQH_API_TOKEN_NONADMIN; can't verify role gating")
        hdr = {"Authorization": f"Bearer {non_admin}"}
        with httpx.Client(base_url=self._base, timeout=30.0) as client:
            resp = client.get(
                "/v1/admin/gpu_images",
                params={"purpose": "sft"},
                headers=hdr,
            )
        # 401 if the token is invalid; 403 if it authenticates but
        # role check fails. We want the 403 (auth ok, role no).
        self.assertEqual(resp.status_code, 403, resp.text)


if __name__ == "__main__":
    unittest.main()
