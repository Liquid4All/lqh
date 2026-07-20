"""Agent behavior policies (CLI_PLAN §4.2).

Historically ``Agent(auto_mode: bool)`` bundled several independent
policies (skill injection, no-user interception, permission auto-grant,
compute defaulting, secret delivery, terminal tools). ``AgentPolicy``
makes each explicit; the boolean is now a preset selector so TUI
behavior is unchanged. The ``SUBAGENT`` preset backs headless
``lqh run`` (delegated tasks from third-party harnesses).
"""

from __future__ import annotations

from dataclasses import dataclass

# Outward-facing publishing tools: gated by policy on the sub-agent
# surface (CLI_PLAN §3.3 — "code + compute auto; publishing gated").
# hf_push additionally has a durable-store permission domain; the other
# two have no handler-side gate, so this policy gate is their only one.
PUBLISH_TOOLS = frozenset({
    "hf_push",
    "push_to_production",
    "create_inference_key",
})


@dataclass(frozen=True)
class AgentPolicy:
    # No human attached: intercept ask_user/show_file, nudge instead of
    # yielding on tool-less turns, decline overwrite confirmations, park
    # on training_status.
    no_user: bool = False
    # Skill injected as a sticky system message ("auto" | "subagent" | None).
    sticky_skill: str | None = None
    # PERMISSION_REQUIRED sentinels auto-grant project-wide (TUI --auto).
    auto_grant_permissions: bool = False
    # Invocation-scoped consent domains passed to every tool call
    # (PermissionContext.granting(*granted_domains)); never persisted.
    granted_domains: frozenset[str] = frozenset()
    # False: PUBLISH_TOOLS terminate the run with needs_permission.
    allow_publish: bool = True
    # COMPUTE_PICK_REQUIRED resolution when no_user: persist this target
    # ("cloud"); None terminates the run with needs_configuration.
    compute_default: str | None = None
    # One-time secret delivery: "prompt" (interactive), "env" (append to
    # .env), "result" (carried out-of-band to the run result payload).
    secret_delivery: str = "prompt"
    # Expose set_auto_stage / exit_auto_mode and honor exit_auto_mode.
    terminal_tools: bool = False
    # Refuse to LAUNCH compute (start_training / start_local_eval) unless a
    # compute target is explicitly configured (project or global) — the
    # implicit product default is billable cloud, which a delegated run
    # must never pick silently (needs_configuration instead).
    require_compute_config: bool = False


TUI_INTERACTIVE = AgentPolicy()

# ≡ the historical auto_mode=True behavior, verbatim.
TUI_AUTO = AgentPolicy(
    no_user=True,
    sticky_skill="auto",
    auto_grant_permissions=True,
    compute_default="cloud",
    secret_delivery="env",
    terminal_tools=True,
)


def subagent_policy(*, allow_publish: bool = False) -> AgentPolicy:
    """Policy for `lqh run` (CLI_PLAN §3.3, §4.2).

    Task-implied work (scripts, cloud data-gen, training) is auto-granted
    for the invocation; publishing is gated behind ``allow_publish``.
    Secrets always ride the result payload; the run driver additionally
    persists them to .env when the caller passed --save-secret.
    """
    domains = {"script", "cloud_data_gen", "training"}
    if allow_publish:
        domains.add("hf_push")
    return AgentPolicy(
        no_user=True,
        sticky_skill="subagent",
        auto_grant_permissions=False,
        granted_domains=frozenset(domains),
        allow_publish=allow_publish,
        compute_default=None,
        secret_delivery="result",
        terminal_tools=True,
        require_compute_config=True,
    )
