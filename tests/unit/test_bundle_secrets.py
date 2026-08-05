"""The submit bundle must never carry credential files to object storage.

The manifest names config keys rather than paths, so nobody sets out to
bundle a .env — but a key pointing at a *directory* is recursed with
rglob, which (unlike the glob module) includes dotfiles. A project with a
.env sitting next to its seed data would ship it to R2.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from lqh.remote.bundle import _write_bundle, build_bundle, is_secret_like


def _members(config: dict, project_dir: Path) -> set[str]:
    raw = build_bundle(config, project_dir)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        return {m.name for m in tar.getmembers()}


@pytest.mark.parametrize(
    "name,secret",
    [
        (".env", True),
        (".env.local", True),
        (".env.production", True),
        (".ENV", True),  # case-insensitive
        (".netrc", True),
        (".pgpass", True),
        (".git-credentials", True),
        ("credentials.json", True),
        ("id_rsa", True),
        ("server.pem", True),
        ("private.key", True),
        ("cert.p12", True),
        (".env.example", False),  # placeholders, and pipelines read them
        (".env.sample", False),
        (".env.template", False),
        ("data.parquet", False),
        ("seeds.txt", False),
        ("environment.yml", False),  # not an env file despite the name
    ],
)
def test_is_secret_like(name, secret):
    assert is_secret_like(name) is secret


def test_directory_recursion_excludes_dotenv(tmp_path: Path):
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "data.parquet").write_text("rows")
    (seeds / ".env").write_text("HF_TOKEN=hf_SHOULD_NOT_SHIP\n")
    (seeds / ".env.local").write_text("OPENAI_API_KEY=sk_nope\n")
    (seeds / ".env.example").write_text("HF_TOKEN=\n")
    (seeds / "key.pem").write_text("-----BEGIN PRIVATE KEY-----")

    config = {"manifest": ["source_paths"], "source_paths": ["seeds"]}
    names = _members(config, tmp_path)

    assert "seeds/data.parquet" in names
    assert "seeds/.env.example" in names
    assert "seeds/.env" not in names
    assert "seeds/.env.local" not in names
    assert "seeds/key.pem" not in names


def test_secret_bytes_are_absent_from_the_tarball(tmp_path: Path):
    """Belt and braces: grep the actual bytes, not just the member list."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "data.parquet").write_text("rows")
    (seeds / ".env").write_text("HF_TOKEN=hf_SENTINEL_MUST_NOT_SHIP\n")

    config = {"manifest": ["source_paths"], "source_paths": ["seeds"]}
    raw = build_bundle(config, tmp_path)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        blob = b"".join(
            tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        )
    assert b"hf_SENTINEL_MUST_NOT_SHIP" not in blob


def test_git_and_lqh_subtrees_are_skipped(tmp_path: Path):
    root = tmp_path / "inputs"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text("[remote]\n  url = https://user:pw@host/x\n")
    (root / ".lqh").mkdir()
    (root / ".lqh" / "permissions.json").write_text("{}")
    (root / "data.parquet").write_text("rows")

    config = {"manifest": ["source_paths"], "source_paths": ["inputs"]}
    names = _members(config, tmp_path)

    assert "inputs/data.parquet" in names
    assert not any(".git" in n for n in names)
    assert not any(".lqh" in n for n in names)


def test_custom_env_file_is_excluded(tmp_path: Path, monkeypatch):
    """LQH_ENV_FILE can name anything, so the filter has to ask for the
    configured name rather than pattern-match a static list — otherwise
    the file we read the token OUT of gets uploaded."""
    monkeypatch.setenv("LQH_ENV_FILE", "seeds/creds.txt")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "creds.txt").write_text("HF_TOKEN=hf_CUSTOM_FILE_LEAK\n")
    (seeds / "data.parquet").write_text("rows")

    config = {"manifest": ["source_paths"], "source_paths": ["seeds"]}
    raw = build_bundle(config, tmp_path)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = {m.name for m in tar.getmembers()}
        blob = b"".join(
            tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        )

    assert "seeds/data.parquet" in names
    assert "seeds/creds.txt" not in names
    assert b"hf_CUSTOM_FILE_LEAK" not in blob


@pytest.mark.parametrize("name", ["secrets.env", "prod.env", "local.ENV"])
def test_dot_env_suffix_is_excluded(tmp_path: Path, name):
    """The other half of the .env convention: <thing>.env, which an
    exact-name list misses entirely."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / name).write_text("HF_TOKEN=hf_suffix_leak\n")
    (seeds / "data.parquet").write_text("rows")

    names = _members({"manifest": ["source_paths"], "source_paths": ["seeds"]}, tmp_path)
    assert "seeds/data.parquet" in names
    assert f"seeds/{name}" not in names


def test_direct_secret_path_is_excluded(tmp_path: Path):
    """A config key naming a .env outright is a bug or an attack."""
    (tmp_path / ".env").write_text("HF_TOKEN=hf_nope\n")
    config = {"manifest": ["dataset"], "dataset": ".env"}
    assert _members(config, tmp_path) == {"config.json"}


def test_out_of_project_path_is_skipped_not_rerooted(tmp_path: Path):
    """Absolute manifest values must not silently upload whatever they
    point at — the old behaviour re-rooted them under extern/."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_ish = outside / "notes.txt"
    secret_ish.write_text("private")

    project = tmp_path / "project"
    project.mkdir()
    config = {"manifest": ["dataset"], "dataset": str(secret_ish)}
    names = _members(config, project)

    assert names == {"config.json"}
    assert not any(n.startswith("extern/") for n in names)


# --- referenced vs swept ---------------------------------------------
#
# A file the config points at directly is load-bearing: dropping it while
# leaving the reference in config.json means the sandbox looks for
# something that isn't there. That has to stop the submit. A file merely
# found inside a bundled directory is the ordinary ".env next to seed
# data" case, and a warning is the right weight.


def test_referenced_and_swept_skips_are_reported_separately(tmp_path: Path):
    import io as _io

    from lqh.remote.bundle import _write_bundle

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "data.parquet").write_text("rows")
    (seeds / ".env").write_text("HF_TOKEN=hf_swept\n")
    (tmp_path / "prod.env").write_text("HF_TOKEN=hf_referenced\n")

    config = {
        "manifest": ["source_paths", "dataset"],
        "source_paths": ["seeds"],
        "dataset": "prod.env",
    }
    referenced, swept = _write_bundle(_io.BytesIO(), config, tmp_path)

    assert referenced == ["prod.env"]
    assert swept == ["seeds/.env"]


@pytest.mark.asyncio
async def test_submit_refuses_when_a_referenced_file_was_excluded(tmp_path: Path):
    """Fail before the POST, not after a billable launch."""
    from lqh.remote.backend import RemoteConfig
    from lqh.remote.cloud import CloudBackend, CloudError

    project = tmp_path / "proj"
    (project / "runs" / "r1").mkdir(parents=True)
    (project / "creds.pem").write_text("-----BEGIN PRIVATE KEY-----")

    backend = CloudBackend(
        RemoteConfig(name="cloud", type="cloud", hostname="h", remote_root="cloud:lqh"),
        project,
        api_base="https://mock.lqh.test",
        token="t",
    )
    config = {"manifest": ["dataset"], "dataset": "creds.pem"}

    with pytest.raises(CloudError) as excinfo:
        await backend.submit_run(str(project / "runs" / "r1"), config)

    msg = str(excinfo.value)
    assert "refusing to submit" in msg
    assert "creds.pem" in msg


# ---------------------------------------------------------------------------
# bypasses: the filter has to hold for the paths that skip the walk
# ---------------------------------------------------------------------------


def test_configured_env_file_outranks_the_template_allowlist(tmp_path, monkeypatch):
    """`.env.example` is allowlisted because templates carry placeholders.
    But LQH_ENV_FILE naming it means the resolver reads a live token out of
    that exact file, and the allowlist ran first — so pointing LQH_ENV_FILE
    at any allowlisted name uploaded the token file itself."""
    monkeypatch.setenv("LQH_ENV_FILE", ".env.example")
    assert is_secret_like(".env.example") is True
    # The others stay allowlisted; only the configured one is reclassified.
    assert is_secret_like(".env.sample") is False


def test_manifest_named_protected_paths_are_excluded(tmp_path):
    """_SKIP_DIRS was enforced only while recursing into a directory. A
    manifest naming a file inside one reached tar.add untouched — and
    .git/config holds credentials in remote URLs, which is why the walk
    excludes it in the first place."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(
        "[remote]\n\turl = https://user:ghp_secret@github.com/x/y\n"
    )
    (tmp_path / ".lqh").mkdir()
    (tmp_path / ".lqh" / "permissions.json").write_text("{}")
    (tmp_path / "data.jsonl").write_text('{"a":1}\n')

    config = {
        "manifest": ["git", "lqh", "data"],
        "git": ".git/config",
        "lqh": ".lqh/permissions.json",
        "data": "data.jsonl",
    }
    buf = io.BytesIO()
    referenced, _swept = _write_bundle(buf, config, tmp_path)

    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        names = set(tar.getnames())
    assert names == {"config.json", "data.jsonl"}
    assert ".git/config" in referenced
    assert ".lqh/permissions.json" in referenced


def test_protected_dirs_are_excluded_when_named_as_a_directory(tmp_path):
    """Same rule for a manifest naming the directory itself."""
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "pyvenv.cfg").write_text("home = /usr\n")

    config = {"manifest": ["venv"], "venv": ".venv"}
    buf = io.BytesIO()
    referenced, _swept = _write_bundle(buf, config, tmp_path)

    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        assert tar.getnames() == ["config.json"]
    assert ".venv" in referenced


# ---------------------------------------------------------------------------
# the HuggingFace cache: a credential the name-only filter cannot see
# ---------------------------------------------------------------------------


def test_project_local_hf_cache_token_is_excluded(tmp_path):
    """`huggingface_hub.get_token()` reads a file named plainly `token`.
    That name is far too common to blocklist outright, and the cache
    normally lives under $HOME — but HF_HOME can point into the project,
    and then a manifest directory sweeps the credential straight into R2.
    Worse, it happens regardless of the donation answer: a user who
    DECLINED still uploads the token."""
    inputs = tmp_path / "input"
    cache = inputs / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("hf_the_actual_credential\n")
    (cache / "stored_tokens").write_text("{}\n")
    (inputs / "seed.jsonl").write_text('{"a": 1}\n')

    buf = io.BytesIO()
    _referenced, swept = _write_bundle(buf, {"manifest": ["p"], "p": "input"}, tmp_path)

    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        names = set(tar.getnames())
    assert names == {"config.json", "input/seed.jsonl"}
    assert "input/.cache/huggingface/token" in swept
    assert "input/.cache/huggingface/stored_tokens" in swept


def test_a_plain_token_file_outside_an_hf_cache_is_kept(tmp_path):
    """The rule is path-shaped on purpose. `token` is a legitimate data
    filename; blocklisting the bare name would silently drop real inputs
    and produce a baffling paid failure in the sandbox."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "token").write_text("not a credential, just data\n")

    buf = io.BytesIO()
    _referenced, swept = _write_bundle(buf, {"manifest": ["p"], "p": "input"}, tmp_path)

    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        assert "input/token" in tar.getnames()
    assert swept == []


def test_hf_token_path_names_its_own_credential_file(tmp_path, monkeypatch):
    """Same reasoning as LQH_ENV_FILE: the user told us where the token
    lives, so its name stops mattering."""
    from lqh.remote.bundle import is_secret_like

    monkeypatch.setenv("HF_TOKEN_PATH", "/somewhere/my-hf-creds.txt")
    assert is_secret_like("my-hf-creds.txt") is True
    assert is_secret_like("unrelated.txt") is False
