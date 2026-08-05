"""Tests for lqh.hf_token — env/env-file/hub-cache token resolution."""

from __future__ import annotations

import os
from contextvars import ContextVar

import pytest

from lqh.hf_token import (
    CLEARED,
    KIND_ENV,
    KIND_ENVFILE,
    KIND_HF_CLI,
    donatable_hf_token,
    hf_disclosure_line,
    hf_token_origin,
    local_hf_token,
    parse_env_file_value,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with no HF vars and no opt-out set."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LQH_ENV_FILE", "LQH_HF_DONATE"):
        monkeypatch.delenv(var, raising=False)
    # Neutralize a real ~/.cache/huggingface/token on the dev machine so tests
    # don't depend on whether the runner ever ran `huggingface-cli login`.
    monkeypatch.setattr("lqh.hf_token._hub_cached_token", lambda: None)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("HF_TOKEN=hf_plain", "hf_plain"),
        ("export HF_TOKEN=hf_exported", "hf_exported"),
        ('HF_TOKEN="hf_double"', "hf_double"),
        ("HF_TOKEN='hf_single'", "hf_single"),
        ("HF_TOKEN = hf_spaced", "hf_spaced"),
        ("HF_TOKEN=hf_trailing   ", "hf_trailing"),
        ("HF_TOKEN=hf_inline # a comment", "hf_inline"),
        ("# HF_TOKEN=hf_commented", None),
        ("HF_TOKENX=hf_other", None),
        ("XHF_TOKEN=hf_other", None),
        ("HF_TOKEN=", CLEARED),  # assigned nothing != never mentioned
        ("", None),
        ("no equals sign here", None),
        ("HF_TOKEN=hf_with=equals", "hf_with=equals"),
        ("OTHER=1\r\nHF_TOKEN=hf_crlf\r\n", "hf_crlf"),
        ('HF_TOKEN="hf_esc\\"quote"', 'hf_esc"quote'),
    ],
)
def test_parser_grammar(text, expected):
    assert parse_env_file_value(text, "HF_TOKEN") == expected


def test_parser_last_assignment_wins(tmp_path):
    """Matches env_secrets.append_env_secret, which appends duplicates.

    First-wins would make the CLI warn about "the stale one" while silently
    using it.
    """
    text = "HF_TOKEN=hf_old\n# rotated\nHF_TOKEN=hf_new\n"
    assert parse_env_file_value(text, "HF_TOKEN") == "hf_new"


def test_parser_empty_last_assignment_clears(tmp_path):
    """The file is append-only, so `HF_TOKEN=` at the end is how a user
    clears a stale token. It must be distinguishable from "never
    mentioned", or the value falls through to a lower-precedence source."""
    assert parse_env_file_value("HF_TOKEN=hf_old\nHF_TOKEN=\n", "HF_TOKEN") is CLEARED
    assert parse_env_file_value('HF_TOKEN=hf_old\nHF_TOKEN=""\n', "HF_TOKEN") is CLEARED
    assert parse_env_file_value("OTHER=1\n", "HF_TOKEN") is None


def test_env_local_can_revoke_a_token_from_env(tmp_path):
    """The whole point of an override file: clearing in the higher-
    precedence file must beat a stale value in the lower one."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_old_should_be_revoked\n")
    (tmp_path / ".env.local").write_text("HF_TOKEN=\n")
    assert donatable_hf_token(tmp_path) == (None, None)
    assert hf_token_origin(tmp_path) is None


def test_custom_env_file_can_revoke(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_old\n")
    (tmp_path / "custom.env").write_text("HF_TOKEN=\n")
    monkeypatch.setenv("LQH_ENV_FILE", "custom.env")
    assert donatable_hf_token(tmp_path) == (None, None)


def test_empty_env_var_revokes_a_file_token(tmp_path, monkeypatch):
    """`export HF_TOKEN=` is the highest-precedence revocation there is."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_old\n")
    monkeypatch.setenv("HF_TOKEN", "")
    assert donatable_hf_token(tmp_path) == (None, None)


def test_unreadable_file_does_not_revoke(tmp_path):
    """Only a revocation we actually read counts — an I/O error must not
    masquerade as one and hide a legitimate lower-precedence token."""
    (tmp_path / ".env.local").mkdir()  # unreadable as a file
    (tmp_path / ".env").write_text("HF_TOKEN=hf_still_valid\n")
    assert donatable_hf_token(tmp_path)[0] == "hf_still_valid"


def test_cleared_env_file_yields_no_donation(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_revoked\nHF_TOKEN=\n")
    assert donatable_hf_token(tmp_path) == (None, None)
    assert hf_token_origin(tmp_path) is None


def test_parser_ignores_other_keys(tmp_path):
    text = "OPENAI_API_KEY=sk_secret\nHF_TOKEN=hf_wanted\nDB_URL=postgres://x\n"
    assert parse_env_file_value(text, "HF_TOKEN") == "hf_wanted"


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------


def test_env_var_wins_over_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    token, origin = donatable_hf_token(tmp_path)
    assert token == "hf_from_env"
    assert origin.kind == KIND_ENV


def test_hf_token_wins_over_hub_token_var(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    token, _ = donatable_hf_token(tmp_path)
    assert token == "hf_primary"


def test_legacy_env_var_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    token, origin = donatable_hf_token(tmp_path)
    assert token == "hf_legacy"
    assert origin.kind == KIND_ENV


def test_env_local_wins_over_env(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_base\n")
    (tmp_path / ".env.local").write_text("HF_TOKEN=hf_local\n")
    token, origin = donatable_hf_token(tmp_path)
    assert token == "hf_local"
    assert origin.kind == KIND_ENVFILE
    assert origin.path.endswith(".env.local")


def test_lqh_env_file_override_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_base\n")
    (tmp_path / "custom.env").write_text("HF_TOKEN=hf_custom\n")
    monkeypatch.setenv("LQH_ENV_FILE", "custom.env")
    token, origin = donatable_hf_token(tmp_path)
    assert token == "hf_custom"
    assert origin.kind == KIND_ENVFILE


def test_lqh_env_file_accepts_absolute_path(tmp_path, monkeypatch):
    external = tmp_path / "elsewhere.env"
    external.write_text("HF_TOKEN=hf_abs\n")
    monkeypatch.setenv("LQH_ENV_FILE", str(external))
    token, _ = donatable_hf_token(tmp_path / "project")
    assert token == "hf_abs"


def test_env_file_wins_over_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("lqh.hf_token._hub_cached_token", lambda: "hf_from_hub")
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    token, origin = donatable_hf_token(tmp_path)
    assert token == "hf_from_file"
    assert origin.kind == KIND_ENVFILE


def test_hub_cache_is_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr("lqh.hf_token._hub_cached_token", lambda: "hf_from_hub")
    token, origin = donatable_hf_token(tmp_path)
    assert token == "hf_from_hub"
    assert origin.kind == KIND_HF_CLI
    assert origin.is_hub_cache


def test_no_token_anywhere(tmp_path):
    assert donatable_hf_token(tmp_path) == (None, None)
    assert hf_token_origin(tmp_path) is None
    assert local_hf_token(tmp_path) is None


def test_none_project_dir_still_reads_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_only")
    token, _ = donatable_hf_token(None)
    assert token == "hf_env_only"


# ---------------------------------------------------------------------------
# safety properties
# ---------------------------------------------------------------------------


def test_os_environ_is_not_mutated(tmp_path):
    """A .env read must not leak into the process env.

    subprocess_manager inherits os.environ into every local training run, and
    ssh_direct copies HF_TOKEN onto a remote host's disk.
    """
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\nOTHER_SECRET=nope\n")
    token, _ = donatable_hf_token(tmp_path)
    assert token == "hf_from_file"
    assert "HF_TOKEN" not in os.environ
    assert "OTHER_SECRET" not in os.environ


def test_origin_never_carries_the_value(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_SENTINEL\n")
    origin = hf_token_origin(tmp_path)
    assert "hf_SENTINEL" not in repr(origin)
    assert "hf_SENTINEL" not in hf_disclosure_line(tmp_path)


def test_parent_directory_env_is_not_searched(tmp_path):
    """No parent walk — a ~/.env must not be donated to every project."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_parent\n")
    project = tmp_path / "project"
    project.mkdir()
    assert donatable_hf_token(project) == (None, None)


def test_missing_and_unreadable_files_are_silent(tmp_path):
    (tmp_path / ".env").mkdir()  # a directory where a file is expected
    assert donatable_hf_token(tmp_path) == (None, None)


def test_oversized_env_file_ignored(tmp_path):
    big = tmp_path / ".env"
    big.write_text("HF_TOKEN=hf_real\n" + "# padding\n" * 200_000)
    assert big.stat().st_size > 1024 * 1024
    assert donatable_hf_token(tmp_path) == (None, None)


def test_broken_huggingface_hub_degrades(tmp_path, monkeypatch):
    """A broken hub install must read as "no token", never break a submit."""
    import lqh.hf_token as mod

    monkeypatch.undo()  # drop the autouse stub so the real helper runs

    def _boom(*_a, **_k):
        raise RuntimeError("hub is broken")

    monkeypatch.setattr("huggingface_hub.get_token", _boom)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LQH_ENV_FILE"):
        monkeypatch.delenv(var, raising=False)

    assert mod._hub_cached_token() is None
    assert donatable_hf_token(tmp_path) == (None, None)


# ---------------------------------------------------------------------------
# LQH_HF_DONATE opt-out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
def test_donation_opt_out(tmp_path, monkeypatch, value):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("LQH_HF_DONATE", value)
    assert donatable_hf_token(tmp_path) == (None, None)


def test_opt_out_still_reports_origin(tmp_path, monkeypatch):
    """The UI must be able to say "found, but disabled" rather than "not found"."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    origin = hf_token_origin(tmp_path)
    assert origin is not None
    assert origin.donation_enabled is False
    assert hf_disclosure_line(tmp_path) == ""


def test_opt_out_does_not_affect_local_use(tmp_path, monkeypatch):
    """LQH_HF_DONATE is about sending the token to us, not using it locally."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    assert local_hf_token(tmp_path) == "hf_from_file"


# ---------------------------------------------------------------------------
# disclosure copy
# ---------------------------------------------------------------------------


def test_disclosure_names_the_env_file(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    line = hf_disclosure_line(tmp_path)
    assert ".env" in line
    assert "LQH_HF_DONATE=0" in line


def test_disclosure_does_not_overstate_retention(tmp_path):
    """The backend holds the token (encrypted, per-job) so a replacement
    worker survives a restart. Copy claiming it is "not stored" would be
    false; it must scope the claim to the account instead."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    line = hf_disclosure_line(tmp_path).lower()
    assert "not stored" not in line
    assert "not added to your lqh account" in line
    assert "deleted" in line


def test_disclosure_does_not_claim_the_token_is_already_sent(tmp_path):
    """Donation is consented separately and can be declined, so this line
    can only promise a question."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    line = hf_disclosure_line(tmp_path)
    assert "you'll be asked" in line


def test_disclosure_distinguishes_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("lqh.hf_token._hub_cached_token", lambda: "hf_from_hub")
    line = hf_disclosure_line(tmp_path)
    assert "huggingface-cli login" in line


def test_disclosure_empty_without_token(tmp_path):
    assert hf_disclosure_line(tmp_path) == ""


def test_ambient_env_files_outside_the_test_tree_are_not_read():
    """The suite-wide isolation fixture has to cover env FILES, not just
    env vars.

    Deleting HF_TOKEN from the environment says nothing about a .env
    sitting in whatever directory a test happens to pass as project_dir —
    including the repository root. A developer with a real token there
    would have seen tests resolve it, which is both a wrong-for-the-wrong-
    reason pass and a token read that no test asked for.

    Uses a directory outside pytest's temp tree on purpose: tmp_path is
    exactly the case that must keep working.
    """
    import shutil
    import tempfile
    from pathlib import Path

    outside = Path(tempfile.mkdtemp(prefix="lqh-not-a-pytest-tmpdir-"))
    try:
        (outside / ".env").write_text("HF_TOKEN=hf_ambient_developer_token\n")
        assert hf_token_origin(outside) is None
        assert donatable_hf_token(outside) == (None, None)
    finally:
        shutil.rmtree(outside, ignore_errors=True)


# ---------------------------------------------------------------------------
# dotenv grammar: annotated secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        # The reported break: quoting a secret AND saying what it's for.
        ('HF_TOKEN="hf_quoted_and_noted"  # private read token', "hf_quoted_and_noted"),
        ("HF_TOKEN='hf_single_noted'  # read-only", "hf_single_noted"),
        ('HF_TOKEN="hf_quoted_plain"', "hf_quoted_plain"),
        ("HF_TOKEN=hf_bare  # a note", "hf_bare"),
        ("HF_TOKEN=hf_bare", "hf_bare"),
        # A '#' inside quotes is part of the value, not a comment.
        ('HF_TOKEN="hf_with#hash"', "hf_with#hash"),
        # Escaped quote inside a double-quoted value.
        ('HF_TOKEN="hf_with\\"quote"', 'hf_with"quote'),
        # Single quotes have no escape in POSIX; first quote closes.
        ("HF_TOKEN='hf_plain_single'", "hf_plain_single"),
    ],
)
def test_quoted_values_survive_an_inline_comment(tmp_path, line, expected):
    """Anchoring the closing quote on the last character meant a trailing
    comment defeated the match, the value fell through to the unquoted
    branch, and the token came back WITH its quotes — invalid everywhere,
    with nothing to indicate why."""
    (tmp_path / ".env").write_text(line + "\n")
    token, origin = donatable_hf_token(tmp_path)
    assert token == expected
    assert origin is not None


def test_an_unterminated_quote_is_taken_literally(tmp_path):
    """Malformed, but it must not crash or silently truncate to empty
    (which would read as a revocation and drop a working token)."""
    (tmp_path / ".env").write_text('HF_TOKEN="hf_unterminated\n')
    token, _origin = donatable_hf_token(tmp_path)
    assert token == '"hf_unterminated'


# ---------------------------------------------------------------------------
# provenance callers must not receive the plaintext
# ---------------------------------------------------------------------------


def test_provenance_resolution_drops_the_value(tmp_path):
    """`want_value=False` is what makes "the value is not in scope" a
    property of the code rather than a comment: hf_token_origin used to be
    handed the plaintext and simply not use it."""
    from lqh.hf_token import _resolve

    (tmp_path / ".env").write_text("HF_TOKEN=hf_should_not_escape\n")

    value, origin = _resolve(tmp_path, want_value=True)
    assert value == "hf_should_not_escape"
    assert origin is not None

    dropped, origin2 = _resolve(tmp_path, want_value=False)
    assert dropped is None
    assert origin2 == origin


# ---------------------------------------------------------------------------
# redact must not fail open on values the API accepts
# ---------------------------------------------------------------------------


def test_short_secrets_drop_the_body_rather_than_pass_it_through(tmp_path):
    """The submit API accepts any non-empty token up to 512 bytes, so
    "too short to be real" is not this layer's call. Returning the text
    unchanged below the redaction floor left the value in a tool result."""
    from lqh.hf_token import redact

    body = 'error: invalid token "abc"'
    scrubbed = redact(body, "abc")
    assert "abc" not in scrubbed
    assert "withheld" in scrubbed


def test_short_secret_not_present_leaves_the_body_intact(tmp_path):
    """Dropping every message that merely coexists with a short secret
    would throw away the diagnostics the user needs."""
    from lqh.hf_token import redact

    body = "error: repository not found"
    assert redact(body, "xyz") == body


# ---------------------------------------------------------------------------
# the project root the source helpers resolve against
# ---------------------------------------------------------------------------


def test_source_helpers_follow_the_selected_project_not_the_cwd(tmp_path, monkeypatch):
    """`lqh run --project DIR` selects a project WITHOUT chdir'ing into it,
    but lqh.sources derived everything from Path.cwd(). Local validation
    then authenticated with the caller's directory while the cloud submit
    donated the selected project's token — the same pipeline running under
    two different credentials, with nothing to indicate it."""
    from lqh import sources

    selected = tmp_path / "selected"
    elsewhere = tmp_path / "elsewhere"
    selected.mkdir()
    elsewhere.mkdir()
    (selected / ".env").write_text("HF_TOKEN=hf_selected_project\n")
    (elsewhere / ".env").write_text("HF_TOKEN=hf_some_other_project\n")

    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sources, "_PROJECT_ROOT", ContextVar("t", default=None))

    # Default: cwd, which is what the TUI and the sandbox want.
    assert sources.project_root() == elsewhere.resolve()
    assert sources._hf_dataset_token() == "hf_some_other_project"

    # After the run driver announces the selected project.
    sources.set_project_root(selected)
    assert sources.project_root() == selected.resolve()
    assert sources._hf_dataset_token() == "hf_selected_project"
    # ...which is the token the cloud submit would donate for that project.
    donated, _origin = donatable_hf_token(selected)
    assert donated == "hf_selected_project"


def test_run_cmd_scopes_the_selected_project(tmp_path, monkeypatch):
    """The wiring, not just the mechanism: --project has to reach
    lqh.sources during the run, and must not still be there after it.

    The scoping half is not pedantry — `lqh run` is a one-shot process in
    production but is also called in-process, and the first version of
    this leaked the selected project into every later caller, which broke
    unrelated path resolution in a way that only showed up as a
    FileNotFoundError somewhere else entirely."""
    from types import SimpleNamespace

    from lqh import sources
    from lqh.cli_cmds import run_cmd

    selected = tmp_path / "selected"
    selected.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    seen: list[Path] = []
    monkeypatch.setattr(
        run_cmd, "_cmd_run_guarded",
        lambda ns, fd: (seen.append(sources.project_root()), 0)[1],
    )

    rc = run_cmd.cmd_run(SimpleNamespace(project=str(selected)))

    assert rc == 0
    assert seen == [selected.resolve()]
    # ...and the process is back to normal afterwards.
    assert sources.project_root() == outside.resolve()
