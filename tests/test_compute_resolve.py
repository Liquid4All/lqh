"""Default-compute precedence + the first-run picker plumbing.

Two layers of test:

  * Pure ``lqh.remote.compute`` resolution — explicit > project > global.
  * ``handle_start_training`` integration — verifies the picker sentinel
    is returned when nothing is set, and that a saved global default
    routes the run without prompting.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

import pytest

import lqh.config as lqh_config
import lqh.remote.compute as compute
import lqh.remote.config as remote_config


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HOME + the global remotes/compute paths at a tmp dir so
    tests can't leak state into the developer's real ~/.lqh."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # lqh.config.config_dir() reads Path.home() lazily, so setting HOME
    # is enough — but remote.config caches GLOBAL_CONFIG_DIR at import
    # time, so patch it too.
    monkeypatch.setattr(remote_config, "GLOBAL_CONFIG_DIR", home / ".lqh")
    yield home


# ---------------------------------------------------------------------
# Pure resolver
# ---------------------------------------------------------------------


def test_resolve_defaults_to_cloud_when_nothing_set(tmp_path, isolated_home):
    """LQH Cloud is the product default. Resolver never returns None —
    the first-run picker has been removed."""
    project = tmp_path / "proj"
    project.mkdir()
    assert compute.resolve_compute(project) == "cloud"


def test_global_default_resolves(tmp_path, isolated_home):
    project = tmp_path / "proj"
    project.mkdir()
    compute.save_global_default("cloud")
    assert compute.resolve_compute(project) == "cloud"


def test_project_overrides_global(tmp_path, isolated_home):
    project = tmp_path / "proj"
    project.mkdir()
    compute.save_global_default("cloud")
    compute.save_project_default(project, "ssh:lab")
    assert compute.resolve_compute(project) == "ssh:lab"


def test_explicit_beats_everything(tmp_path, isolated_home):
    project = tmp_path / "proj"
    project.mkdir()
    compute.save_global_default("cloud")
    compute.save_project_default(project, "ssh:lab")
    assert compute.resolve_compute(project, explicit="ssh:other") == "ssh:other"


def test_clearing_project_default_removes_file(tmp_path, isolated_home):
    project = tmp_path / "proj"
    project.mkdir()
    compute.save_project_default(project, "cloud")
    assert compute.compute_file_path(project).exists()
    compute.save_project_default(project, None)
    assert not compute.compute_file_path(project).exists()


def test_ssh_remote_name_normalizes_both_forms():
    assert compute.ssh_remote_name("ssh:lab-gpu") == "lab-gpu"
    assert compute.ssh_remote_name("lab-gpu") == "lab-gpu"  # legacy callers
    assert compute.ssh_remote_name("cloud") is None
    assert compute.ssh_remote_name(None) is None
    assert compute.is_cloud("cloud")
    assert not compute.is_cloud("ssh:lab-gpu")


# ---------------------------------------------------------------------
# compute_set tool
# ---------------------------------------------------------------------


def test_compute_set_global_persists(tmp_path, isolated_home):
    from lqh.tools.handlers import handle_compute_set

    project = tmp_path / "proj"
    project.mkdir()
    res = asyncio.run(handle_compute_set(project, value="cloud", scope="global"))
    assert "✅" in res.content
    assert compute.load_global_default() == "cloud"


def test_compute_set_project_persists(tmp_path, isolated_home):
    from lqh.tools.handlers import handle_compute_set

    project = tmp_path / "proj"
    project.mkdir()
    res = asyncio.run(handle_compute_set(project, value="ssh:lab", scope="project"))
    assert "✅" in res.content
    assert compute.load_project_default(project) == "ssh:lab"
    assert compute.load_global_default() is None  # global untouched


def test_compute_set_validates_value(tmp_path, isolated_home):
    from lqh.tools.handlers import handle_compute_set

    project = tmp_path / "proj"
    project.mkdir()
    res = asyncio.run(handle_compute_set(project, value="modal", scope="global"))
    assert res.content.lower().startswith("error")
    assert compute.load_global_default() is None


def test_compute_set_empty_clears(tmp_path, isolated_home):
    from lqh.tools.handlers import handle_compute_set

    project = tmp_path / "proj"
    project.mkdir()
    compute.save_global_default("cloud")
    asyncio.run(handle_compute_set(project, value="", scope="global"))
    assert compute.load_global_default() is None


# ---------------------------------------------------------------------
# handle_start_training picker behavior
# ---------------------------------------------------------------------


def _make_dataset(project: Path, name: str = "ds") -> str:
    """Create a minimal parquet dataset so handle_start_training's
    validation passes. Tiny on-disk file is enough — we never run the
    actual training subprocess in these tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = project / "datasets" / name
    ds_dir.mkdir(parents=True)
    table = pa.table({"prompt": ["hi"], "completion": ["hello"]})
    pq.write_table(table, ds_dir / "data.parquet")
    return f"datasets/{name}"


def test_first_run_routes_to_cloud_silently(tmp_path, isolated_home, monkeypatch):
    """Cloud-only project (no bring-your-own-compute remote bound, no
    local GPU), no default, no explicit remote → start_training routes
    straight to LQH Cloud without asking. The picker only fires when
    there's a real choice — a BYOC remote or a local GPU (see
    test_picker_fires_with_byoc_remote)."""
    import lqh.tools.handlers as handlers

    project = tmp_path / "proj"
    project.mkdir()
    ds = _make_dataset(project)

    from lqh.tools.permissions import grant_permission
    grant_permission(project, project_wide=True)

    # No BYOC remote and no local GPU → no choice to make → silent cloud.
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: False)

    recorded: dict = {}

    async def fake_remote(project_dir, run_dir, config, run_name, remote_name, api_key, **kw):
        from lqh.tools.handlers import ToolResult
        recorded["remote_name"] = remote_name
        return ToolResult(content=f"stub: ran on {remote_name}")

    monkeypatch.setattr(handlers, "_execute_start_training_remote", fake_remote)

    res = asyncio.run(handlers.handle_start_training(
        project,
        type="sft",
        base_model="LiquidAI/LFM2-1.2B",
        dataset=ds,
        eval_dataset=ds,
        scorer=None,
    ))

    assert recorded.get("remote_name") == "cloud"
    assert "cloud" in res.content
    # No user prompt should have been emitted.
    assert not res.requires_user_input


def test_cloud_training_config_uses_project_relative_dataset_paths(
    tmp_path, isolated_home, monkeypatch,
):
    """Cloud submit configs must not contain laptop-local absolute paths."""
    import lqh.tools.handlers as handlers

    project = tmp_path / "proj"
    project.mkdir()
    ds = _make_dataset(project)

    from lqh.tools.permissions import grant_permission
    grant_permission(project, project_wide=True)

    # No remote, no local GPU → silent cloud default (no picker).
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: False)

    recorded: dict = {}

    async def fake_remote(project_dir, run_dir, config, run_name, remote_name, api_key, **kw):
        from lqh.tools.handlers import ToolResult

        recorded["config"] = config
        recorded["remote_name"] = remote_name
        return ToolResult(content="stub: cloud")

    monkeypatch.setattr(handlers, "_execute_start_training_remote", fake_remote)

    asyncio.run(handlers.handle_start_training(
        project,
        type="sft",
        base_model="LiquidAI/LFM2-1.2B",
        dataset=ds,
        eval_dataset=ds,
        scorer=None,
    ))

    assert recorded["remote_name"] == "cloud"
    base_config = recorded["config"]["base_config"]
    assert base_config["dataset"] == "datasets/ds/data.parquet"
    assert base_config["eval_dataset"] == "datasets/ds/data.parquet"
    assert not Path(base_config["dataset"]).is_absolute()


def test_cloud_bundle_contains_relative_config_and_dataset(tmp_path):
    """The tarball should match the relative paths the cloud trainer reads."""
    from lqh.remote.bundle import build_bundle

    project = tmp_path / "proj"
    project.mkdir()
    _make_dataset(project)
    config = {
        "type": "sweep",
        "base_config": {
            "type": "sft",
            "base_model": "LiquidAI/LFM2-1.2B",
            "dataset": "datasets/ds/data.parquet",
            "manifest": ["dataset"],
        },
    }

    bundle = build_bundle(config, project)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        names = set(tar.getnames())
        cfg = json.loads(tar.extractfile("config.json").read().decode("utf-8"))

    assert cfg["base_config"]["dataset"] == "datasets/ds/data.parquet"
    assert "datasets/ds/data.parquet" in names


def test_picker_skipped_when_global_default_set(tmp_path, isolated_home, monkeypatch):
    """If the user previously picked cloud, start_training should route
    straight to the cloud branch (which we mock so no real network call)."""
    import lqh.tools.handlers as handlers

    project = tmp_path / "proj"
    project.mkdir()
    ds = _make_dataset(project)
    compute.save_global_default("cloud")

    # Replace _execute_start_training_remote with a stub that records
    # the remote_name it was called with. The picker should NOT fire;
    # control should flow straight through to this stub.
    # Grant project-wide training permission so the handler doesn't
    # stop at the permission check.
    from lqh.tools.permissions import grant_permission
    grant_permission(project, project_wide=True)

    recorded: dict = {}

    async def fake_remote(project_dir, run_dir, config, run_name, remote_name, api_key, **kw):
        from lqh.tools.handlers import ToolResult
        recorded["remote_name"] = remote_name
        return ToolResult(content=f"stub: ran on {remote_name}")

    monkeypatch.setattr(handlers, "_execute_start_training_remote", fake_remote)

    res = asyncio.run(handlers.handle_start_training(
        project,
        type="sft",
        base_model="LiquidAI/LFM2-1.2B",
        dataset=ds,
        eval_dataset=ds,
        scorer=None,
    ))

    assert recorded.get("remote_name") == "cloud"
    assert "cloud" in res.content


def test_picker_fires_with_byoc_remote(tmp_path, isolated_home, monkeypatch):
    """A project with a bound BYOC remote and no chosen target gets the
    one-time picker sentinel instead of routing immediately."""
    import lqh.tools.handlers as handlers
    from lqh.remote.backend import ProjectBinding, RemoteMachine
    from lqh.remote.config import add_binding, add_machine

    project = tmp_path / "proj"
    project.mkdir()
    ds = _make_dataset(project)
    add_machine(RemoteMachine(name="lab-gpu", type="ssh_direct", hostname="lab-gpu-01"))
    add_binding(project, ProjectBinding(name="lab-gpu", remote_root="/home/u/lab"))

    # Pin local GPU off so the option list is just Cloud + remote + else.
    monkeypatch.setattr(handlers, "_local_gpu_available", lambda: False)

    from lqh.tools.permissions import grant_permission
    grant_permission(project, project_wide=True)

    res = asyncio.run(handlers.handle_start_training(
        project,
        type="sft",
        base_model="LiquidAI/LFM2-1.2B",
        dataset=ds,
        eval_dataset=ds,
        scorer=None,
    ))

    assert res.requires_user_input
    assert res.content == "COMPUTE_PICK_REQUIRED"
    assert res.options[0] == "LQH Cloud (recommended)"
    assert any("lab-gpu" in o for o in res.options)
    assert res.options[-1].startswith("Something else")
