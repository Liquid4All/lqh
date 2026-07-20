"""Tool definition/handler parity and x-lqh metadata (CLI_PLAN §5.2).

The metadata drives the headless `lqh tool` surface: exposure is opt-in
(`cli=False` default), and the flags must stay aligned with both the
handler registry and the PermissionContext consent domains.
"""

from __future__ import annotations

from lqh.tools.definitions import METADATA_KEY, get_all_tools
from lqh.tools.handlers import TOOL_HANDLERS
from lqh.tools.permissions import PERMISSION_DOMAINS

# The frozen exposed set (CLI_PLAN §5.2). Changing exposure must be a
# conscious diff to this literal, not a side effect.
EXPOSED_TOOLS = {
    "summary", "list_user_data", "list_models", "list_skills",
    "get_eval_failures", "run_data_gen_pipeline", "run_data_filter",
    "run_scoring", "start_training", "training_status", "stop_training",
    "start_local_eval", "eval_hf_model", "hf_push", "hf_pull",
    "hf_repo_info", "pull", "push", "artifacts", "gguf_convert",
    "push_to_production", "list_deployments", "get_deployment",
    "stop_deployment", "restart_deployment", "create_inference_key",
    "list_inference_keys", "revoke_inference_key", "remote_list",
    "remote_add", "remote_bind", "remote_remove", "remote_remove_machine",
    "remote_setup", "remote_status", "compute_set",
}

META_KEYS = {"cli", "mutating", "needs_auth", "permission_domain", "needs_loop"}


def test_handlers_match_definitions() -> None:
    names = {
        t["function"]["name"]
        for t in get_all_tools(auto_mode=True, include_meta=True)
    }
    assert names == set(TOOL_HANDLERS)


def test_metadata_stripped_by_default() -> None:
    for auto_mode in (False, True):
        for tool in get_all_tools(auto_mode=auto_mode):
            assert METADATA_KEY not in tool
            assert set(tool) == {"type", "function"}


def test_metadata_present_and_complete_with_include_meta() -> None:
    for tool in get_all_tools(auto_mode=True, include_meta=True):
        meta = tool[METADATA_KEY]
        assert set(meta) == META_KEYS, tool["function"]["name"]
        assert isinstance(meta["cli"], bool)
        assert isinstance(meta["mutating"], bool)
        assert isinstance(meta["needs_auth"], bool)
        assert isinstance(meta["needs_loop"], bool)
        assert isinstance(meta["permission_domain"], list)


def test_cli_exposed_set_is_exactly_the_plan_set() -> None:
    exposed = {
        t["function"]["name"]
        for t in get_all_tools(auto_mode=True, include_meta=True)
        if t[METADATA_KEY]["cli"]
    }
    assert exposed == EXPOSED_TOOLS


def test_permission_domains_match_context() -> None:
    for tool in get_all_tools(auto_mode=True, include_meta=True):
        for domain in tool[METADATA_KEY]["permission_domain"]:
            assert domain in PERMISSION_DOMAINS, tool["function"]["name"]


def test_auto_only_tools_not_exposed() -> None:
    by_name = {
        t["function"]["name"]: t
        for t in get_all_tools(auto_mode=True, include_meta=True)
    }
    for name in ("set_auto_stage", "exit_auto_mode", "ask_user", "load_skill"):
        assert by_name[name][METADATA_KEY]["cli"] is False
