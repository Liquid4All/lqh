"""The 🤗 indicator must distinguish the two token sources.

Before donation existed, a cloud project's indicator reflected only the
backend-stored token — so a user with a token in their .env saw "✗" while
their next submit would in fact carry one.
"""

from __future__ import annotations

import pytest

from lqh.tui.status_bar import StatusBar


def _hf_segment(bar: StatusBar) -> str:
    """The rendered 🤗 fragment, whatever style it carries."""
    parts = bar.get_formatted_text()
    return next(text for _style, text in parts if "🤗" in text)


@pytest.fixture
def bar(tmp_path):
    return StatusBar(project_dir=tmp_path)


@pytest.mark.parametrize(
    "stored,local,expected",
    [
        (True, "your project .env", "🤗 ✓both"),
        (True, None, "🤗 ✓acct"),
        (False, "your project .env", "🤗 ✓env"),
        (None, "your project .env", "🤗 ✓env"),
        (False, None, "🤗 HF ✗"),
        (None, None, "🤗 HF ?"),
    ],
)
def test_cloud_indicator_names_the_source(bar, stored, local, expected):
    bar.compute_is_cloud = True
    bar.hf_cloud_configured = stored
    bar.hf_local_source = local
    bar.hf_local_donatable = local is not None
    assert _hf_segment(bar) == expected


@pytest.mark.parametrize(
    "stored,expected",
    [
        (True, "🤗 ✓acct"),
        (False, "🤗 HF ✗"),
        (None, "🤗 HF ?"),
    ],
)
def test_cloud_ignores_a_token_it_cannot_send(bar, stored, expected):
    """LQH_HF_DONATE=0 keeps a discoverable token off the wire. On cloud
    the question is "will a token reach the sandbox", so a suppressed one
    must not count — otherwise the bar promises HF access the job won't
    have."""
    bar.compute_is_cloud = True
    bar.hf_cloud_configured = stored
    bar.hf_local_source = "your project .env"
    bar.hf_local_donatable = False
    assert _hf_segment(bar) == expected


@pytest.mark.parametrize(
    "local,expected",
    [
        ("your HF_TOKEN environment variable", "🤗 HF ✓"),
        ("your project .env", "🤗 HF ✓"),
        ("your huggingface-cli login token", "🤗 HF ✓"),
        (None, "🤗 HF ✗"),
    ],
)
@pytest.mark.parametrize("donatable", [True, False])
def test_local_compute_uses_only_the_local_source(bar, local, expected, donatable):
    """On ssh/local a stored backend token is irrelevant, and the local
    source now covers .env and huggingface-cli login, not just the env var.

    Donatability is a cloud-only concern and must not enter here:
    LQH_HF_DONATE=0 stops the token being SENT anywhere, it does not stop
    a local `push` from authenticating with it. Reading the same field on
    both branches made the bar show HF ✗ to a local user whose tooling
    worked fine."""
    bar.compute_is_cloud = False
    bar.hf_cloud_configured = True  # must be ignored on this branch
    bar.hf_local_source = local
    bar.hf_local_donatable = donatable
    assert _hf_segment(bar) == expected


def test_indicator_never_renders_a_token_value(bar):
    """hf_local_source holds a source label; nothing should let a value in."""
    bar.compute_is_cloud = True
    bar.hf_local_source = "your project .env"
    rendered = "".join(text for _s, text in bar.get_formatted_text())
    assert "hf_" not in rendered


def test_defaults_are_unknown_not_false(bar):
    """A fresh bar must not claim "no token" before anything was resolved."""
    assert bar.hf_cloud_configured is None
    assert bar.hf_local_source is None
    assert _hf_segment(bar) == "🤗 HF ?"


# ---------------------------------------------------------------------------
# what _refresh_hf_status feeds the bar
# ---------------------------------------------------------------------------


class _StubApp:
    """Just enough of LqhApp to drive _refresh_hf_status."""

    def __init__(self, project_dir, agent=None):
        from lqh.tui.app import LqhApp

        self.project_dir = project_dir
        self._agent = agent
        self._status_bar = StatusBar(project_dir=project_dir)
        self._refresh_hf_status = LqhApp._refresh_hf_status.__get__(self)

    def _invalidate(self):
        pass


class _StubAgent:
    def __init__(self, *, auto, allow_donate=False):
        from lqh.agent_policy import AgentPolicy

        self.policy = AgentPolicy(
            auto_grant_permissions=auto, allow_hf_donate=allow_donate,
        )


@pytest.fixture
def cloud_project(tmp_path, monkeypatch):
    (tmp_path / ".lqh").mkdir()
    (tmp_path / ".env").write_text("HF_TOKEN=hf_status_bar_fixture_token\n")
    monkeypatch.setattr("lqh.remote.compute.resolve_compute", lambda _d: "cloud")
    monkeypatch.setattr("lqh.auth.get_token", lambda: None)
    return tmp_path


@pytest.mark.asyncio
async def test_auto_mode_does_not_promise_a_token_it_will_decline(cloud_project):
    """In auto mode the donation gate is answered "no" unless a durable
    grant exists, so a discoverable token does NOT reach the sandbox. The
    bar showed ✓env anyway — a green state for something that would never
    happen."""
    app = _StubApp(cloud_project, agent=_StubAgent(auto=True))
    await app._refresh_hf_status()

    assert app._status_bar.hf_local_source is not None  # found...
    assert app._status_bar.hf_local_donatable is False  # ...but not sendable
    assert _hf_segment(app._status_bar) == "🤗 HF ?"


@pytest.mark.asyncio
async def test_auto_mode_with_a_durable_grant_is_green(cloud_project):
    from lqh.tools.permissions import grant_hf_donate_permission

    grant_hf_donate_permission(cloud_project)
    app = _StubApp(cloud_project, agent=_StubAgent(auto=True))
    await app._refresh_hf_status()

    assert app._status_bar.hf_local_donatable is True


@pytest.mark.asyncio
async def test_interactive_mode_is_green_because_it_will_ask(cloud_project):
    """Interactive sessions get a prompt, so the token genuinely may be
    sent — the indicator reports availability, not a prediction."""
    app = _StubApp(cloud_project, agent=_StubAgent(auto=False))
    await app._refresh_hf_status()

    assert app._status_bar.hf_local_donatable is True


@pytest.mark.asyncio
async def test_stale_cloud_status_is_cleared(cloud_project, monkeypatch):
    """hf_cloud_configured had no path back to unknown: a True from an
    earlier cloud project kept rendering ✓acct after a logout or a switch
    to local compute, describing a backend nobody was asking."""
    app = _StubApp(cloud_project, agent=_StubAgent(auto=False))
    app._status_bar.hf_cloud_configured = True

    monkeypatch.setattr("lqh.remote.compute.resolve_compute", lambda _d: "local")
    await app._refresh_hf_status()

    assert app._status_bar.hf_cloud_configured is None
