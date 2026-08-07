"""The four answers to the HF-donation prompt, across every tool that asks.

hf_donate is the only permission domain where declining does not cancel
the action — the job still runs, without the token. That asymmetry lives
in the agent loop, so it is worth pinning per answer rather than only
through the data-gen handler.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lqh.tools.handlers import HF_DONATE_OPTIONS, ToolResult, _resolve_hf_donation
from lqh.tools.permissions import (
    PermissionContext,
    check_hf_donate_permission,
    grant_hf_donate_permission,
    load_permissions,
)

YES, YES_ALWAYS, NO, NO_ALWAYS = HF_DONATE_OPTIONS


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".lqh").mkdir()
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_dotenv_for_consent\n")
    return tmp_path


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["training", "eval", "data-gen", "gguf", "transfer"])
def test_every_job_kind_prompts_with_the_same_options(project, label):
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, label)

    assert donate is False
    assert prompt.content == "PERMISSION_REQUIRED"
    assert prompt.permission_key == f"hf_donate:{label}"
    assert prompt.options == HF_DONATE_OPTIONS
    assert label in prompt.question


def test_grant_skips_the_prompt(project):
    donate, prompt = _resolve_hf_donation(
        project, PermissionContext.granting("hf_donate"), None, "training",
    )
    assert donate is True
    assert prompt is None


def test_durable_grant_skips_the_prompt(project):
    grant_hf_donate_permission(project)
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "eval")
    assert donate is True
    assert prompt is None


def test_decline_donates_nothing_and_does_not_reprompt(project):
    """The asymmetry: an absent grant would look like "not asked yet" and
    re-prompt forever, so decline arrives as an explicit False."""
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), False, "training")
    assert donate is False
    assert prompt is None


def test_decline_is_not_persisted(project):
    """A remembered "no" would silently break a later run that needs the
    token, so only the explicit "and don't ask again" answer persists."""
    _resolve_hf_donation(project, PermissionContext(), False, "training")
    assert check_hf_donate_permission(project) is False
    assert load_permissions(project).hf_donate_declined is False


def test_no_prompt_without_a_discoverable_token(tmp_path):
    (tmp_path / ".lqh").mkdir()
    donate, prompt = _resolve_hf_donation(tmp_path, PermissionContext(), None, "training")
    assert donate is False
    assert prompt is None


def test_opt_out_suppresses_the_prompt(project, monkeypatch):
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "training")
    assert donate is False
    assert prompt is None


# ---------------------------------------------------------------------------
# the agent loop's handling of each answer
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Just enough of the agent to drive _handle_permission_response."""

    def __init__(self, project_dir: Path):
        from lqh.agent import Agent, AgentCallbacks

        self.project_dir = project_dir
        self.reinvoked: list[dict] = []
        self.hf_refreshes = 0
        self._handle_permission_response = Agent._handle_permission_response.__get__(self)
        self._hf_donation_recorded = Agent._hf_donation_recorded.__get__(self)
        self._chain_grants = Agent._chain_grants.__get__(self)
        self._DONATE_CHAIN_DOMAINS = Agent._DONATE_CHAIN_DOMAINS

        async def _recorded() -> None:
            self.hf_refreshes += 1

        self.callbacks = AgentCallbacks(on_hf_donation_recorded=_recorded)

        class _Policy:
            auto_grant_permissions = False
            no_user = False

        self.policy = _Policy()

    async def _reinvoke_tool(self, tool_name, tool_args, *, internal_kwargs=None):
        self.reinvoked.append({"tool": tool_name, "internal": internal_kwargs or {}})
        return ToolResult(content="reinvoked")


async def _answer(project: Path, tool: str, response: str) -> dict:
    agent = _FakeAgent(project)
    await agent._handle_permission_response(
        response, tool, {}, permission_key="hf_donate:training",
    )
    assert len(agent.reinvoked) == 1
    return agent.reinvoked[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool", ["start_training", "eval_hf_model", "run_data_gen_pipeline", "push", "gguf_convert"],
)
async def test_yes_grants_for_this_invocation_only(project, tool):
    call = await _answer(project, tool, YES)

    assert "hf_donate" in call["internal"]["_permissions"].grants
    assert call["internal"].get("_hf_donate") is None
    # "this time" must not persist — the prompt returns next submit.
    assert check_hf_donate_permission(project) is False


@pytest.mark.asyncio
async def test_yes_always_persists(project):
    call = await _answer(project, "start_training", YES_ALWAYS)

    assert "hf_donate" in call["internal"]["_permissions"].grants
    assert check_hf_donate_permission(project) is True
    assert load_permissions(project).hf_donate_allow_all is True


@pytest.mark.asyncio
async def test_no_reinvokes_with_an_explicit_decline(project):
    """Decline must re-invoke — the job still runs — carrying a value the
    handler can tell apart from "never asked"."""
    call = await _answer(project, "start_training", NO)

    assert call["internal"]["_hf_donate"] is False
    assert check_hf_donate_permission(project) is False
    assert load_permissions(project).hf_donate_declined is False


@pytest.mark.asyncio
async def test_no_always_persists_the_decline(project):
    """The reported bug: without a durable no, the offer came back at
    every cloud submit and the only way out was leaving the session to
    export LQH_HF_DONATE=0."""
    call = await _answer(project, "start_training", NO_ALWAYS)

    assert call["internal"]["_hf_donate"] is False
    assert load_permissions(project).hf_donate_declined is True
    assert check_hf_donate_permission(project) is False


@pytest.mark.asyncio
async def test_no_always_silences_every_later_submit(project):
    """What the answer is actually for: the next stage of the same
    pipeline must not ask again."""
    await _answer(project, "run_data_gen_pipeline", NO_ALWAYS)

    for label in ("training", "eval", "gguf"):
        donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, label)
        assert donate is False
        assert prompt is None


@pytest.mark.asyncio
async def test_no_always_is_offered_by_upload_jobs_too(project):
    """The required-token wording changes only the plain decline; the
    durable one has to be reachable from a transfer or GGUF push as
    well, or those sites keep re-asking."""
    from lqh.tools.handlers import HF_DONATE_REQUIRED_OPTIONS

    assert HF_DONATE_REQUIRED_OPTIONS[-1] == NO_ALWAYS
    await _answer(project, "gguf_convert", HF_DONATE_REQUIRED_OPTIONS[-1])
    assert load_permissions(project).hf_donate_declined is True


@pytest.mark.asyncio
async def test_the_upload_variants_plain_decline_stays_one_time(project):
    """Closest near-miss to the durable answer: it starts with "no" and
    contains "don't", so only the full phrase may separate them."""
    from lqh.tools.handlers import HF_DONATE_REQUIRED_OPTIONS

    call = await _answer(project, "gguf_convert", HF_DONATE_REQUIRED_OPTIONS[2])

    assert call["internal"]["_hf_donate"] is False
    assert load_permissions(project).hf_donate_declined is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    ["stop asking me about this, don't ask again", "cancel", ""],
)
async def test_free_text_declines_without_recording_anything(project, answer):
    """"Other" is free text, so an answer that merely mentions not being
    asked again is not the offered option. Pinned deliberately: the job
    is still declined, but nothing durable is inferred from prose about
    a credential — the offered answer is one keystroke away."""
    call = await _answer(project, "start_training", answer)

    assert call["internal"]["_hf_donate"] is False
    perms = load_permissions(project)
    assert perms.hf_donate_declined is False
    assert perms.hf_donate_allow_all is False


@pytest.mark.asyncio
async def test_no_always_undoes_a_standing_yes(project):
    """Changing your mind has to actually move the answer — leaving both
    flags set would put a contradiction on disk for the resolver."""
    grant_hf_donate_permission(project)
    await _answer(project, "start_training", NO_ALWAYS)

    perms = load_permissions(project)
    assert perms.hf_donate_allow_all is False
    assert perms.hf_donate_declined is True


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["auto_grant_permissions", "no_user"])
@pytest.mark.parametrize("answer_kind", ["yes", "no"])
async def test_a_headless_surface_never_persists_a_standing_answer(
    project, surface, answer_kind,
):
    """Neither headless surface reaches this handler for a donation key
    today, but if one starts synthesizing answers again it must not
    leave a standing decision behind about a credential nobody was
    asked about — in either direction."""
    agent = _FakeAgent(project)
    setattr(agent.policy, surface, True)

    await agent._handle_permission_response(
        YES_ALWAYS if answer_kind == "yes" else NO_ALWAYS,
        "start_training", {}, permission_key="hf_donate:training",
    )
    perms = load_permissions(project)
    assert perms.hf_donate_declined is False
    assert perms.hf_donate_allow_all is False
    assert agent.hf_refreshes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "no"])
async def test_a_standing_answer_refreshes_the_surfaces_indicator(project, answer):
    """The 🤗 indicator is derived from the standing answer and is
    otherwise only recomputed at startup — a mid-workflow answer that
    didn't notify would leave it describing the old decision all
    session."""
    agent = _FakeAgent(project)

    await agent._handle_permission_response(
        YES_ALWAYS if answer == "yes" else NO_ALWAYS,
        "start_training", {}, permission_key="hf_donate:training",
    )
    assert agent.hf_refreshes == 1


@pytest.mark.asyncio
async def test_a_one_time_answer_does_not_notify(project):
    """Nothing changed on disk, so there is nothing to re-render."""
    agent = _FakeAgent(project)

    for response in (YES, NO):
        await agent._handle_permission_response(
            response, "start_training", {}, permission_key="hf_donate:training",
        )
    assert agent.hf_refreshes == 0


@pytest.mark.asyncio
async def test_the_tui_actually_wires_the_refresh(project, monkeypatch):
    """The half of the fix the user sees. Without this the callback can
    be dropped from _create_agent and the suite stays green while the 🤗
    indicator goes stale for the rest of the session."""
    import asyncio

    from lqh.session import Session
    from lqh.tui.app import LqhApp

    monkeypatch.setenv("HOME", str(project))
    monkeypatch.setattr("lqh.tui.app.get_token", lambda: None)
    app = LqhApp(project)
    app._session = Session.create(project)
    app._invalidate = lambda: None

    refreshed = []

    async def _refresh():
        refreshed.append(1)

    app._refresh_hf_status = _refresh

    agent = app._create_agent()
    assert agent.callbacks.on_hf_donation_recorded is not None

    await agent.callbacks.on_hf_donation_recorded()
    # Scheduled, not awaited — the refresh can take a 15s round trip and
    # must not stall the agent between the answer and the re-invocation.
    await asyncio.sleep(0)
    assert refreshed == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,expected",
    [
        ("eval_hf_model", "cloud_eval_hf"),
        ("run_data_gen_pipeline", "cloud_data_gen"),
        ("push", "hf_push"),
        ("gguf_convert", "hf_push"),
    ],
)
async def test_answering_preserves_the_tools_own_consent(project, tool, expected):
    """Reaching the donation prompt proves the tool's own gate already
    passed, but invocation grants don't survive a re-invocation — so they
    have to be restated or the two prompts ping-pong forever."""
    for answer in (YES, NO):
        call = await _answer(project, tool, answer)
        assert expected in call["internal"]["_permissions"].grants, answer


# ---------------------------------------------------------------------------
# consent must behave the same way whichever surface you come from
# ---------------------------------------------------------------------------


def test_full_consent_does_not_imply_donation(project):
    """`lqh tool call` treats invocation as consent to RUN a tool. Handing
    over one of the user's credentials is a separate decision, so it needs
    --allow-hf-donate rather than riding along."""
    donate, prompt = _resolve_hf_donation(
        project, PermissionContext(full_consent=True), None, "training",
    )
    assert donate is False
    assert prompt is not None


def test_tool_call_opt_in_grants_donation(project):
    from lqh.cli_cmds.registry import full_consent_kwargs

    without = full_consent_kwargs({})["_permissions"]
    assert "hf_donate" not in without.grants
    assert without.allows_hf_donate(project) is False

    with_flag = full_consent_kwargs({}, allow_hf_donate=True)["_permissions"]
    assert with_flag.allows_hf_donate(project) is True


def test_headless_run_opt_in_grants_donation():
    from lqh.agent_policy import subagent_policy

    assert "hf_donate" not in subagent_policy().granted_domains
    assert "hf_donate" in subagent_policy(allow_hf_donate=True).granted_domains
    # And it is not smuggled in by --allow-publish, which is a different question.
    assert "hf_donate" not in subagent_policy(allow_publish=True).granted_domains


@pytest.mark.asyncio
async def test_a_synthesized_answer_never_donates(project):
    """Belt-and-braces on the auto-mode fix above. Auto mode no longer
    reaches this handler for a donation key, but if a future surface
    starts synthesizing answers again, the string it synthesizes must not
    read as consent."""
    call = await _answer(
        project, "start_training", "Execute and don't ask again for this project",
    )

    assert call["internal"]["_hf_donate"] is False
    assert check_hf_donate_permission(project) is False
    # Nor does it read as the durable *decline*: an answer synthesized for
    # some other domain must not settle the donation question either way.
    assert load_permissions(project).hf_donate_declined is False


@pytest.mark.asyncio
async def test_headless_declines_instead_of_halting(project):
    """A discoverable token must never turn a working headless run into
    needs_permission. Declining still runs the job, so on a surface with
    nobody to ask, the answer is "no" — not "stop"."""
    from lqh.agent import Agent

    agent = _FakeAgent(project)
    agent.policy.no_user = True

    handled = await Agent._headless_donation_decline.__get__(agent)(
        "hf_donate:training", "start_training", {},
    )
    assert handled is not None
    assert agent.reinvoked[0]["internal"]["_hf_donate"] is False


@pytest.mark.asyncio
async def test_headless_still_halts_for_other_domains(project):
    """Only donation degrades this way; a genuine publishing gate must
    still stop the run rather than silently proceeding."""
    from lqh.agent import Agent

    agent = _FakeAgent(project)
    agent.policy.no_user = True

    assert await Agent._headless_donation_decline.__get__(agent)(
        "hf_push:me/model", "push", {},
    ) is None
    assert not agent.reinvoked


# ---------------------------------------------------------------------------
# ordering: the donation answer must be decided before any blanket
# auto-approval gets to answer it
# ---------------------------------------------------------------------------


def _real_agent(project: Path, **kwargs):
    """A real Agent, so the test exercises the actual dispatch order in
    _handle_tool_call rather than a hand-rolled approximation of it."""
    from lqh.agent import Agent
    from lqh.session import Session

    return Agent(project, Session.create(project), **kwargs)




@pytest.fixture
def donation_gate(project, monkeypatch):
    """Route tool calls through the REAL donation gate.

    The point is to test the loop's dispatch order against the resolver
    that actually decides, so a grant (or the absence of one) reaches the
    loop the same way it would in production. Returns the list of calls;
    each entry's ``extra`` is the internal-kwargs channel.
    """
    calls: list[dict] = []

    async def fake_execute_tool(tool_name, arguments, project_dir, **extra):
        calls.append({"tool": tool_name, "extra": extra})
        donate, prompt = _resolve_hf_donation(
            project_dir, extra.get("_permissions"), extra.get("_hf_donate"), "training",
        )
        calls[-1]["donated"] = donate
        if prompt is not None:
            return prompt
        return ToolResult(content="submitted")

    monkeypatch.setattr("lqh.agent.execute_tool", fake_execute_tool)
    return calls


@pytest.mark.asyncio
async def test_auto_mode_declines_donation_without_asking(project, donation_gate):
    """TUI --auto auto-approves every PERMISSION_REQUIRED sentinel. That is
    consent to spend compute without being interrupted; it is NOT consent
    to put an HF token on the wire. The donation answer therefore has to
    be settled before the auto-grant branch runs — when it was settled
    after, `lqh --auto` donated a discoverable token silently, with no
    prompt and no way to opt out short of LQH_HF_DONATE=0."""
    agent = _real_agent(project, auto_mode=True)
    assert agent.policy.auto_grant_permissions is True
    assert agent.policy.allow_hf_donate is False

    await agent._handle_tool_call("start_training", {"type": "sft"})

    assert len(donation_gate) == 2, "expected exactly one re-invocation"
    assert donation_gate[-1]["donated"] is False
    assert donation_gate[-1]["extra"].get("_hf_donate") is False
    # And nothing durable was written from an answer nobody gave.
    assert check_hf_donate_permission(project) is False


@pytest.mark.asyncio
async def test_auto_mode_donates_under_a_durable_grant(project, donation_gate):
    """The way in for unattended runs is the durable grant a human writes
    by choosing "don't ask again". With it the gate never fires at all, so
    the decline above must not become a dead end for someone who already
    said yes."""
    grant_hf_donate_permission(project)
    agent = _real_agent(project, auto_mode=True)

    await agent._handle_tool_call("start_training", {"type": "sft"})

    assert len(donation_gate) == 1, "a granted donation must not prompt"
    assert donation_gate[0]["donated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "Do not send it",
        "cancel",
        "Other (please specify)",
        "Execute and don't ask again for this project",
        "send it later",
    ],
)
async def test_unrecognized_answers_decline(project, answer):
    """The TUI appends a free-text "Other" option, so the answer is not
    drawn from a fixed set. Anything that isn't an explicit yes has to
    decline: matching "not a no" instead sent the token for every string
    here, including an empty one. Declining costs a re-run; the inverse
    mistake cannot be taken back."""
    call = await _answer(project, "start_training", answer)

    assert call["internal"]["_hf_donate"] is False
    assert check_hf_donate_permission(project) is False


# ---------------------------------------------------------------------------
# only ask when the sandbox can actually use the token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifact_format,target_repo,should_offer",
    [
        # No push target and a full checkpoint: the sandbox neither
        # downloads a base nor uploads anything. Nothing to authenticate.
        ("full", None, False),
        # A push always needs the token.
        ("full", "me/model-gguf", True),
        # A LoRA merge downloads a possibly-gated base.
        ("lora", None, True),
        # Format unknown here — resolved server-side from lineage. Stay
        # permissive: skipping the token would hand the user a paid 401.
        (None, None, True),
    ],
)
async def test_gguf_only_offers_when_the_token_has_a_purpose(
    project, monkeypatch, artifact_format, target_repo, should_offer,
):
    """A donation that cannot be used is pure exposure — and after a
    "don't ask again" grant it happens silently, so the gate has to match
    the backend's own conditions rather than always offering."""
    from lqh.tools import handlers

    asked: list[str] = []
    real_resolve = handlers._resolve_hf_donation

    def spy(project_dir, permissions, hf_donate, label, **kw):
        asked.append(label)
        return real_resolve(project_dir, permissions, hf_donate, label, **kw)

    monkeypatch.setattr(handlers, "_resolve_hf_donation", spy)

    submitted: dict = {}

    async def fake_submit_gguf(**kwargs):
        submitted.update(kwargs)
        return "job-1"

    monkeypatch.setattr("lqh.remote.gguf_convert.submit_gguf", fake_submit_gguf)

    await handlers.handle_gguf_convert(
        project_dir=project,
        artifact_id="a1",
        quant_types=["Q4_K"],
        target_hf_repo=target_repo,
        artifact_format=artifact_format,
        # A push target raises the hf_push gate first; pre-grant it so
        # this test only observes the donation decision.
        _permissions=PermissionContext.granting("hf_push"),
        _hf_donate=False,
    )

    assert bool(asked) is should_offer
    assert submitted.get("donate_hf_token") is False


# ---------------------------------------------------------------------------
# the prompt must not promise what the workflow cannot deliver
# ---------------------------------------------------------------------------


def test_upload_jobs_do_not_claim_declining_still_runs_them(project):
    """A transfer, and a GGUF conversion with a push target, UPLOAD to
    Hugging Face — the backend rejects them outright when no token exists
    anywhere. The shared prompt used to tell those users "declining still
    runs the job", which then handed them a 400."""
    _donate, prompt = _resolve_hf_donation(
        project, PermissionContext(), None, "transfer", token_required=True,
    )

    assert "Declining still runs the job" not in prompt.question
    assert "UPLOADS to Hugging Face" in prompt.question
    assert "/hf_login" in prompt.question
    # ...and the plain decline OPTION cannot claim it either.
    assert prompt.options[2] == "No — don't send it (needs a token stored via /hf_login)"


def test_ordinary_jobs_keep_the_reassuring_wording(project):
    """Training, eval and data-gen genuinely do run without the token —
    over-warning there would push people into donating unnecessarily."""
    _donate, prompt = _resolve_hf_donation(
        project, PermissionContext(), None, "training",
    )

    assert "Declining still runs the job" in prompt.question
    assert "UPLOADS" not in prompt.question
    assert prompt.options[2] == "No — run this job without it"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_repo,required",
    [("me/model-gguf", True), (None, False)],
)
async def test_gguf_only_warns_when_it_actually_pushes(
    project, monkeypatch, target_repo, required,
):
    """A GGUF conversion needs a token unconditionally only when it
    pushes. Merging onto a PUBLIC base works without one, so the harsher
    copy would be wrong there."""
    from lqh.tools import handlers

    seen: dict = {}
    real = handlers._resolve_hf_donation

    def spy(project_dir, permissions, hf_donate, label, **kw):
        seen.update(kw)
        return real(project_dir, permissions, hf_donate, label, **kw)

    monkeypatch.setattr(handlers, "_resolve_hf_donation", spy)

    async def fake_submit_gguf(**kwargs):
        return "job-1"

    monkeypatch.setattr("lqh.remote.gguf_convert.submit_gguf", fake_submit_gguf)

    await handlers.handle_gguf_convert(
        project_dir=project,
        artifact_id="a1",
        quant_types=["Q4_K"],
        target_hf_repo=target_repo,
        artifact_format="lora",
        _permissions=PermissionContext.granting("hf_push"),
        _hf_donate=False,
    )

    assert seen.get("token_required") is required


# ---------------------------------------------------------------------------
# standing answers — the up-front question is what normally settles this,
# so the per-job gate must read it and stay quiet
# ---------------------------------------------------------------------------

# Captured before conftest's isolation fixture swaps it out, so the tests
# below can exercise the real ~/.lqh/config.json read against a redirected
# home rather than the developer's own.
from lqh.hf_token import _global_hf_donate as _REAL_GLOBAL_LOOKUP  # noqa: E402
from lqh.hf_token import (  # noqa: E402
    DONATE_ALWAYS,
    DONATE_NEVER,
    hf_donate_question_due,
    resolve_hf_donate_decision,
)
from lqh.tools.permissions import deny_hf_donate_permission  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Redirect ~/.lqh so installation-wide answers are test-local."""
    root = tmp_path / "home" / ".lqh"
    root.mkdir(parents=True)
    monkeypatch.setattr("lqh.config.config_dir", lambda: root)
    monkeypatch.setattr("lqh.config.config_path", lambda: root / "config.json")
    monkeypatch.setattr("lqh.hf_token._global_hf_donate", _REAL_GLOBAL_LOOKUP)
    return root


def _set_global(value):
    from lqh.config import update_config

    update_config(lambda c: setattr(c, "hf_donate", value))


def test_project_decline_skips_the_prompt(project):
    """The startup question's "no" is durable — unlike a mid-job decline."""
    deny_hf_donate_permission(project)
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "data-gen")
    assert donate is False
    assert prompt is None


def test_installation_yes_donates_without_asking(project, home):
    _set_global("always")
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "data-gen")
    assert donate is True
    assert prompt is None


def test_installation_no_skips_the_prompt(project, home):
    _set_global("never")
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "training")
    assert donate is False
    assert prompt is None


def test_project_answer_beats_the_installation_default(project, home):
    _set_global("never")
    grant_hf_donate_permission(project)
    assert resolve_hf_donate_decision(project) == DONATE_ALWAYS

    _set_global("always")
    deny_hf_donate_permission(project)
    assert resolve_hf_donate_decision(project) == DONATE_NEVER


def test_env_opt_out_beats_every_stored_answer(project, home, monkeypatch):
    _set_global("always")
    grant_hf_donate_permission(project)
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    assert resolve_hf_donate_decision(project) == DONATE_NEVER
    donate, prompt = _resolve_hf_donation(project, PermissionContext(), None, "eval")
    assert donate is False
    assert prompt is None


def test_invocation_opt_in_beats_a_stored_no(project):
    """`--allow-hf-donate` is the user saying yes about THIS run; a "no"
    they recorded in some earlier session must not veto it."""
    deny_hf_donate_permission(project)
    donate, prompt = _resolve_hf_donation(
        project, PermissionContext.granting("hf_donate"), None, "training",
    )
    assert donate is True
    assert prompt is None


def test_env_opt_out_still_beats_the_invocation_opt_in(project, monkeypatch):
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    donate, prompt = _resolve_hf_donation(
        project, PermissionContext.granting("hf_donate"), None, "training",
    )
    assert donate is False
    assert prompt is None


def test_granting_after_declining_clears_the_decline(project):
    deny_hf_donate_permission(project)
    grant_hf_donate_permission(project)
    perms = load_permissions(project)
    assert perms.hf_donate_allow_all is True
    assert perms.hf_donate_declined is False
    assert resolve_hf_donate_decision(project) == DONATE_ALWAYS


def test_a_garbage_config_value_is_not_a_yes(project, home):
    """A hand-edited config must not decide that a credential leaves the
    machine — anything but the two known answers reads as unanswered."""
    (home / "config.json").write_text('{"hf_donate": "yes please"}')
    assert resolve_hf_donate_decision(project) is None


# hf_donate_question_due — what the startup prompt asks itself


def test_question_is_due_when_a_token_is_found_and_unanswered(project):
    origin = hf_donate_question_due(project)
    assert origin is not None
    assert ".env" in (origin.path or "")


@pytest.mark.parametrize("record", [grant_hf_donate_permission, deny_hf_donate_permission])
def test_question_is_not_due_once_answered(project, record):
    record(project)
    assert hf_donate_question_due(project) is None


def test_question_is_not_due_without_a_token(tmp_path):
    (tmp_path / ".lqh").mkdir()
    assert hf_donate_question_due(tmp_path) is None


def test_question_is_not_due_when_donation_is_switched_off(project, monkeypatch):
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    assert hf_donate_question_due(project) is None


# the disclosure line in the surrounding consent prompts must agree


@pytest.mark.parametrize(
    "record,expected",
    [(grant_hf_donate_permission, "is sent with this job"),
     (deny_hf_donate_permission, "is NOT sent")],
)
def test_disclosure_reflects_a_standing_answer(project, record, expected):
    """Once answered, promising "you'll be asked" would describe a prompt
    that never comes."""
    from lqh.hf_token import hf_disclosure_line

    record(project)
    line = hf_disclosure_line(project)
    assert expected in line
    assert "you'll be asked" not in line


def test_the_context_helper_cannot_drift_from_the_gate(project, home, monkeypatch):
    """PermissionContext.allows_hf_donate reads the same resolver the gate
    does. Reading only hf_donate_allow_all left it blind to the
    installation-wide answer and to LQH_HF_DONATE=0."""
    ctx = PermissionContext()
    assert ctx.allows_hf_donate(project) is False

    _set_global("always")
    assert ctx.allows_hf_donate(project) is True

    deny_hf_donate_permission(project)
    assert ctx.allows_hf_donate(project) is False
    # ...and an explicit invocation grant still overrides that stored no,
    # exactly as it does in _resolve_hf_donation.
    assert PermissionContext.granting("hf_donate").allows_hf_donate(project) is True

    # The env opt-out outranks even the grant, on both surfaces.
    monkeypatch.setenv("LQH_HF_DONATE", "0")
    assert PermissionContext.granting("hf_donate").allows_hf_donate(project) is False
