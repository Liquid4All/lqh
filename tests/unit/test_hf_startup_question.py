"""The HF-donation question is asked once, at startup, before any work.

It used to arrive at the first cloud submit — mid-pipeline, phrased as
though that job needed the credential, when in fact nothing did: the offer
exists only because a token happens to sit on this machine. These tests
pin where it fires, what each answer records, and that answering it once
silences the per-job gate for good.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Captured before conftest's isolation fixture replaces it — the tests
# that write an installation-wide answer need the real config lookup,
# pointed at a redirected home.
from lqh.hf_token import _global_hf_donate as _REAL_GLOBAL_LOOKUP
from lqh.tools.handlers import _resolve_hf_donation
from lqh.tools.permissions import PermissionContext, load_permissions


class _StubApp:
    """Just enough of LqhApp to drive _settle_hf_donation."""

    def __init__(self, project_dir: Path, answer: str | None):
        from lqh.tui.app import LqhApp

        self.project_dir = project_dir
        self.auto_mode = False
        self._answer = answer
        self._shutdown_requested = False
        self.emitted: list[str] = []
        self.asked: list[list[str]] = []
        self.refreshed = 0
        self._HF_DONATE_ANSWERS = LqhApp._HF_DONATE_ANSWERS
        self._settle_hf_donation = LqhApp._settle_hf_donation.__get__(self)

    async def _emit(self, text: str) -> None:
        self.emitted.append(text)

    async def _wait_for_user_response(self, *, options=None, **_kw) -> str:
        self.asked.append(options)
        if self._answer is None:
            raise AssertionError("prompt fired but the test expected silence")
        return self._answer

    async def _refresh_hf_status(self) -> None:
        self.refreshed += 1

    @property
    def output(self) -> str:
        return "\n".join(self.emitted)


def _answers() -> list[str]:
    from lqh.tui.app import LqhApp

    return [label for label, _key in LqhApp._HF_DONATE_ANSWERS]


PROJECT_YES, GLOBAL_YES, PROJECT_NO, GLOBAL_NO = _answers()


@pytest.fixture
def cloud_project(tmp_path, monkeypatch) -> Path:
    (tmp_path / ".lqh").mkdir()
    (tmp_path / ".env").write_text("HF_TOKEN=hf_startup_question_fixture\n")
    monkeypatch.setattr("lqh.remote.compute.resolve_compute", lambda _d: "cloud")
    return tmp_path


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "home" / ".lqh"
    root.mkdir(parents=True)
    monkeypatch.setattr("lqh.config.config_dir", lambda: root)
    monkeypatch.setattr("lqh.config.config_path", lambda: root / "config.json")
    monkeypatch.setattr("lqh.hf_token._global_hf_donate", _REAL_GLOBAL_LOOKUP)
    return root


# ---------------------------------------------------------------------------
# when it fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asks_when_a_token_is_discoverable(cloud_project):
    app = _StubApp(cloud_project, PROJECT_YES)
    await app._settle_hf_donation()

    assert app.asked == [_answers()]
    # The copy has to answer "what would this be used for" — that was the
    # user's actual complaint, not just the timing.
    assert "gated or private models and datasets" in app.output
    assert "not added to your LQH account" in app.output
    assert "jobs still run" in app.output


@pytest.mark.asyncio
async def test_silent_without_a_token(tmp_path, monkeypatch):
    (tmp_path / ".lqh").mkdir()
    monkeypatch.setattr("lqh.remote.compute.resolve_compute", lambda _d: "cloud")
    app = _StubApp(tmp_path, None)
    await app._settle_hf_donation()

    assert app.asked == []
    assert app.emitted == []


@pytest.mark.asyncio
async def test_asks_even_when_training_runs_on_byo_compute(cloud_project, monkeypatch):
    """Only training follows the project's compute target. Data-gen, eval,
    GGUF conversion and HF transfers go to our sandbox regardless — so an
    ssh-pinned project must still be asked up front, or it gets the
    mid-pipeline prompt this change exists to remove."""
    monkeypatch.setattr("lqh.remote.compute.resolve_compute", lambda _d: "ssh:lab")
    app = _StubApp(cloud_project, PROJECT_NO)
    await app._settle_hf_donation()

    assert app.asked == [_answers()]
    for label in ("data-gen", "eval", "gguf", "transfer"):
        _donate, prompt = _resolve_hf_donation(
            cloud_project, PermissionContext(), None, label,
        )
        assert prompt is None, label


@pytest.mark.asyncio
async def test_silent_when_donation_is_switched_off(cloud_project, monkeypatch):
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    app = _StubApp(cloud_project, None)
    await app._settle_hf_donation()

    assert app.asked == []


@pytest.mark.asyncio
async def test_asked_only_once(cloud_project):
    await _StubApp(cloud_project, PROJECT_YES)._settle_hf_donation()

    again = _StubApp(cloud_project, None)
    await again._settle_hf_donation()
    assert again.asked == []


@pytest.mark.asyncio
async def test_a_hub_cli_token_says_where_it_came_from(tmp_path, monkeypatch):
    (tmp_path / ".lqh").mkdir()
    monkeypatch.setattr("lqh.remote.compute.resolve_compute", lambda _d: "cloud")
    monkeypatch.setattr("lqh.hf_token._hub_cached_token", lambda: "hf_from_hub_cli")
    app = _StubApp(tmp_path, PROJECT_NO)
    await app._settle_hf_donation()

    assert "huggingface-cli login" in app.output
    assert "didn't set it up for LQH" in app.output


# ---------------------------------------------------------------------------
# what each answer records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_yes(cloud_project):
    app = _StubApp(cloud_project, PROJECT_YES)
    await app._settle_hf_donation()

    perms = load_permissions(cloud_project)
    assert perms.hf_donate_allow_all is True
    assert perms.hf_donate_declined is False
    assert app.refreshed == 1


@pytest.mark.asyncio
async def test_project_no(cloud_project):
    app = _StubApp(cloud_project, PROJECT_NO)
    await app._settle_hf_donation()

    perms = load_permissions(cloud_project)
    assert perms.hf_donate_declined is True
    assert perms.hf_donate_allow_all is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,expected", [(GLOBAL_YES, "always"), (GLOBAL_NO, "never")],
)
async def test_installation_wide_answers_are_written_to_the_config(
    cloud_project, home, answer, expected,
):
    from lqh.config import load_config

    await _StubApp(cloud_project, answer)._settle_hf_donation()

    assert load_config().hf_donate == expected
    # ...and the project record agrees, so the answer holds even if the
    # config file is later reset.
    perms = load_permissions(cloud_project)
    assert perms.hf_donate_allow_all is (expected == "always")
    assert perms.hf_donate_declined is (expected == "never")


@pytest.mark.asyncio
async def test_quitting_at_the_prompt_records_nothing(cloud_project):
    """Ctrl+C resolves the prompt with the shutdown sentinel. Recording a
    credential decision nobody made is the one unacceptable outcome; being
    asked again next start is fine."""
    app = _StubApp(cloud_project, "anything")

    async def _quit(*, options=None, **_kw):
        app.asked.append(options)
        app._shutdown_requested = True
        return "\x00shutdown"

    app._wait_for_user_response = _quit
    await app._settle_hf_donation()

    perms = load_permissions(cloud_project)
    assert perms.hf_donate_allow_all is False
    assert perms.hf_donate_declined is False


@pytest.mark.asyncio
async def test_an_unrecognised_reply_is_not_consent(cloud_project):
    app = _StubApp(cloud_project, "sure, whatever")
    await app._settle_hf_donation()

    perms = load_permissions(cloud_project)
    assert perms.hf_donate_allow_all is False
    assert perms.hf_donate_declined is False
    assert "No answer recorded" in app.output


@pytest.mark.asyncio
async def test_a_failed_write_is_reported_not_swallowed(cloud_project, monkeypatch):
    monkeypatch.setattr(
        "lqh.tools.permissions.grant_hf_donate_permission",
        lambda _d: (_ for _ in ()).throw(OSError("read-only fs")),
    )
    app = _StubApp(cloud_project, PROJECT_YES)
    await app._settle_hf_donation()

    assert "Could not record" in app.output
    assert "asked again" in app.output


@pytest.mark.asyncio
async def test_a_half_written_answer_does_not_promise_another_question(
    cloud_project, home, monkeypatch,
):
    """A machine-wide answer is two writes. If the second fails, the first
    still stands and silences the question — telling the user they'll be
    asked again would strand them with a decision they cannot reach."""
    from lqh.config import update_config as _real_update

    monkeypatch.setattr(
        "lqh.config.update_config",
        lambda _m: (_ for _ in ()).throw(OSError("read-only home")),
    )
    app = _StubApp(cloud_project, GLOBAL_YES)
    await app._settle_hf_donation()

    assert "Could not record" in app.output
    assert "asked again next time" not in app.output
    assert "recorded only in part" in app.output
    assert "~/.lqh/config.json" in app.output
    assert _real_update is not None  # the real one is restored by monkeypatch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,names_config",
    [(PROJECT_YES, False), (PROJECT_NO, False),
     (GLOBAL_YES, True), (GLOBAL_NO, True)],
)
async def test_the_confirmation_says_where_to_undo_it(
    cloud_project, home, answer, names_config,
):
    """Revocation is the half of a consent design people reach for after
    changing their mind. Pointing a machine-wide answer at the project
    file only would have them edit a file that does not decide."""
    app = _StubApp(cloud_project, answer)
    await app._settle_hf_donation()

    assert ".lqh/permissions.json" in app.output
    assert ("~/.lqh/config.json" in app.output) is names_config


# ---------------------------------------------------------------------------
# the point of all of it: no mid-pipeline prompt afterwards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,donates", [(PROJECT_YES, True), (PROJECT_NO, False)],
)
async def test_answering_up_front_silences_every_job_prompt(
    cloud_project, answer, donates,
):
    await _StubApp(cloud_project, answer)._settle_hf_donation()

    for label in ("data-gen", "training", "eval", "gguf", "transfer"):
        donate, prompt = _resolve_hf_donation(
            cloud_project, PermissionContext(), None, label,
        )
        assert prompt is None, label
        assert donate is donates, label
