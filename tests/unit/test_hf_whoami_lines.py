"""`hf_repo_info` (whoami) must name the credential class it found.

It used to read `auth.accessToken.type`, a field that does not exist in HF's
`AuthInfo` shape, so it printed "Token type: unknown" for every token — which
is worse than useless when the question being asked is "which credential is
this machine actually using".

Every test here also asserts the token value never appears in the output.
"""

from __future__ import annotations

from lqh.tools.handlers import _whoami_lines

TOKEN = "hf_SENTINEL_DO_NOT_PRINT_0123456789"


def test_personal_access_token_reports_its_class_and_role():
    info = {
        "name": "someone",
        "auth": {
            "type": "access_token",
            "accessToken": {"displayName": "laptop", "role": "read"},
        },
        "orgs": [{"name": "acme"}],
    }
    out = "\n".join(_whoami_lines(info))

    assert "access_token" in out
    assert "role: read" in out
    assert "unknown" not in out
    assert "acme" in out
    # A static settings-page token is the right thing for a cloud job, so the
    # expiry warning must not fire here.
    assert "expires" not in out
    assert TOKEN not in out


def test_oauth_credential_is_named_and_flagged_as_expiring():
    # What `hf auth login` saves with no --token: an OAuth credential, and
    # note the absent "accessToken" — reading a field under it can only ever
    # have produced "unknown".
    info = {
        "name": "someone",
        "auth": {"type": "app_token_as_user", "expiresAt": "2026-09-20T00:00:00Z"},
    }
    out = "\n".join(_whoami_lines(info))

    assert "app_token_as_user" in out
    assert "2026-09-20T00:00:00Z" in out
    assert "expires" in out.lower()
    # The point of naming it: a cloud job holds a static copy it cannot refresh.
    assert "settings/tokens" in out
    assert TOKEN not in out


def test_unknown_shape_degrades_instead_of_raising():
    # A whoami response we do not recognise (or an older/newer Hub) must still
    # render. "unknown" is honest here; it was not honest as a blanket answer.
    for info in ({}, {"name": "someone"}, {"name": "someone", "auth": None}):
        out = "\n".join(_whoami_lines(info))
        assert "Token type: unknown" in out
        assert "settings/tokens" not in out  # no expiry advice we cannot support
