"""Build the submit bundle that ``CloudBackend`` ships to ``api.lqh.ai``.

A bundle is a gzipped tarball containing:

  config.json           # the run config, written verbatim
  <relative paths>      # every file the manifest points at, preserving
                        # its position under the project root

The tar's directory layout mirrors the project — so when the cloud
runner extracts it into ``/workspace/runs/<job_id>/inputs/`` and the
trainer's config refers to ``datasets/train.parquet``, the file is at
exactly that relative path. No path rewriting required.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

from lqh.sync import resolve_manifest

__all__ = ["build_bundle", "build_bundle_to_file", "is_protected_path", "is_secret_like"]

logger = logging.getLogger(__name__)

# Files that must never be uploaded, whatever the manifest says.
#
# The manifest names config keys, not paths, so nobody sets out to bundle
# a .env — but a key pointing at a *directory* (data_gen seed folders,
# multi-file datasets) is recursed, and Path.rglob("*") includes dotfiles
# unlike the glob module. A project with a .env next to its seed data
# would ship the credential to object storage, where it would sit for as
# long as the bundle is retained.
_SECRET_EXACT = {
    ".env", ".envrc", ".netrc", ".pgpass", ".npmrc", ".pypirc",
    ".git-credentials", "credentials", "credentials.json",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
}
_SECRET_PREFIXES = (".env.",)          # .env.local, .env.production
# ".env" as a *suffix* catches the other half of the convention —
# secrets.env, prod.env — which a static exact-name list misses.
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".env")
# Template files carry placeholders, not secrets, and pipelines sometimes
# read them. Checked before the .env. prefix rule.
_SECRET_ALLOW = {".env.example", ".env.sample", ".env.template", ".env.dist"}
# Directories skipped wholesale during recursion. .git/config can hold
# credentials in remote URLs; .lqh/ is local bookkeeping the sandbox has
# no use for.
_SKIP_DIRS = {".git", ".lqh", "__pycache__", ".venv", "node_modules"}

# Hugging Face keeps its credential in a file named plainly `token` (and
# newer versions `stored_tokens`) inside its cache dir — the very file
# `huggingface_hub.get_token()` reads, and therefore the third source
# this whole feature resolves from. A name-only filter can't see it:
# "token" is far too common a filename to blocklist outright, and the
# cache is normally under $HOME rather than the project. But HF_HOME can
# point anywhere, including into the project, at which point a manifest
# directory sweeps up the credential and ships it to R2 — even on a run
# where the user DECLINED donation, which is the worst version of it.
#
# So the rule is path-shaped, not name-shaped: these names count as
# secrets only under a `huggingface` directory.
_HF_CACHE_DIR_NAMES = {"huggingface"}
_HF_CACHE_SECRET_NAMES = {"token", "stored_tokens"}


def _configured_token_file_names() -> set[str]:
    """Basenames of credential files the HF env vars point at.

    Same reasoning as LQH_ENV_FILE: the user told us where their token
    lives, so we know it is a credential regardless of what it is called.
    HF_TOKEN_PATH names the file directly; HF_HOME relocates the cache
    whose `token` file get_token() reads.
    """
    names: set[str] = set()
    direct = (os.environ.get("HF_TOKEN_PATH") or "").strip()
    if direct:
        names.add(Path(direct).name.lower())
    return names


def _is_hf_cache_secret(rel: Path) -> bool:
    """Whether *rel* is the credential file inside a HuggingFace cache."""
    if rel.name.lower() not in _HF_CACHE_SECRET_NAMES:
        return False
    return any(part.lower() in _HF_CACHE_DIR_NAMES for part in rel.parts[:-1])


def is_protected_path(rel: Path) -> bool:
    """Whether *rel* must never be bundled, judged on the whole path.

    Complements :func:`is_secret_like`, which only sees a basename and so
    cannot express "this file is a secret because of where it sits".
    """
    if any(part in _SKIP_DIRS for part in rel.parts):
        return True
    if _is_hf_cache_secret(rel):
        return True
    return is_secret_like(rel.name)


def _configured_env_file_name() -> str | None:
    """Basename of ``$LQH_ENV_FILE``, if the user pointed us at one.

    ``lqh.hf_token`` reads a token out of whatever file this names, so it
    may be called anything at all — ``secrets.env``, ``creds.txt``. A
    static name list can't know, and a user who configured a custom env
    file is exactly the user with a token in it, so it has to be asked
    for by name rather than pattern-matched.
    """
    raw = (os.environ.get("LQH_ENV_FILE") or "").strip()
    return Path(raw).name.lower() if raw else None


def is_secret_like(name: str) -> bool:
    """Whether a file name looks like a credential store."""
    lowered = name.lower()
    # The user's own LQH_ENV_FILE outranks every rule below, including
    # the template allowlist. A name can be a convention in general and
    # a real credential store on this machine, and only the user knows
    # which: `LQH_ENV_FILE=.env.example` pointed the resolver at that
    # file, so it holds a live token no matter what it is called, and
    # allowlisting it first uploaded exactly the file the user told us
    # their token was in.
    if lowered == _configured_env_file_name():
        return True
    if lowered in _configured_token_file_names():
        return True
    if lowered in _SECRET_ALLOW:
        return False
    if lowered in _SECRET_EXACT:
        return True
    if lowered.startswith(_SECRET_PREFIXES):
        return True
    return lowered.endswith(_SECRET_SUFFIXES)


def build_bundle(
    config: dict[str, Any],
    project_dir: Path,
) -> bytes:
    """Build the in-memory tarball.

    Right for the common case — training configs plus modest datasets —
    where the bundle is uploaded via httpx multipart in one go. Bundles
    that may be large (data_gen bring-your-own image folders) should use
    :func:`build_bundle_to_file` so the bytes never sit in RAM; the
    cloud client then ships them via the presigned-PUT staging path.
    """
    buf = io.BytesIO()
    _write_bundle(buf, config, project_dir)
    return buf.getvalue()


def build_bundle_to_file(
    config: dict[str, Any],
    project_dir: Path,
    dest: Path,
) -> tuple[int, list[str], list[str]]:
    """Stream the tarball to *dest*.

    Returns ``(size_bytes, referenced_skips, swept_skips)``.

    Used for potentially-large bundles (image folders and other seed
    data riding a data_gen submit) so memory stays flat regardless of
    input size.

    The two skip lists are separated because they mean different things
    to the caller. ``referenced_skips`` are files a config key points at
    *directly*: the config still names them, so the sandbox will look for
    a file that isn't there — the submit must be refused before it costs
    anything. ``swept_skips`` were merely found inside a bundled
    directory, which is the ordinary case of a ``.env`` sitting next to
    seed data; a warning is enough.
    """
    with dest.open("wb") as fh:
        referenced, swept = _write_bundle(fh, config, project_dir)
    if referenced or swept:
        logger.info(
            "bundle excluded %d file(s): %s",
            len(referenced) + len(swept), ", ".join(referenced + swept),
        )
    return dest.stat().st_size, referenced, swept


def _write_bundle(
    fileobj: BinaryIO, config: dict[str, Any], project_dir: Path
) -> tuple[list[str], list[str]]:
    """Write the tarball; returns (referenced_skips, swept_skips).

    Skipped names are returned rather than dropped silently: a pipeline
    that genuinely wanted one of these files would otherwise fail inside
    a paid sandbox with no clue why. See build_bundle_to_file for what
    the caller does with each list.
    """
    paths = resolve_manifest(config, project_dir)
    seen: set[str] = set()
    referenced_skips: list[str] = []
    swept_skips: list[str] = []

    with tarfile.open(fileobj=fileobj, mode="w:gz") as tar:
        # 1. config.json at the tar root.
        cfg_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
        info = tarfile.TarInfo(name="config.json")
        info.size = len(cfg_bytes)
        tar.addfile(info, io.BytesIO(cfg_bytes))

        # 2. manifest files, anchored under the project dir.
        project_resolved = project_dir.resolve()
        for p in paths:
            p_resolved = p.resolve()
            try:
                rel = p_resolved.relative_to(project_resolved)
            except ValueError:
                # Manifest pointed outside the project. Skip rather than
                # re-root under extern/: a manifest value is influenced by
                # config the model can write, and silently uploading
                # whatever an absolute path points at is not a safety net.
                logger.warning("skipping out-of-project bundle path: %s", p_resolved)
                referenced_skips.append(p_resolved.name)
                continue
            arc = str(rel)
            if arc in seen:
                continue
            # _SKIP_DIRS applies to a directly-named path too, not just to
            # what recursion sweeps up. Naming `.git/config` in a manifest
            # reached tar.add untouched — the same file the directory walk
            # is careful to exclude, because it holds credentials in remote
            # URLs. The walk's filter was doing all the work, and a
            # manifest is the easier way to point at a file.
            # Whole-path check, not just the basename: a manifest may
            # name a file that is a credential because of WHERE it sits
            # (.git/config, a huggingface cache token) rather than what
            # it is called.
            if is_protected_path(rel):
                logger.info("excluding protected path from bundle: %s", arc)
                referenced_skips.append(arc)
                continue
            seen.add(arc)

            if p_resolved.is_dir():
                # Recurse for directory entries — typical for dataset
                # parquet shards. rglob includes dotfiles, hence the
                # filtering here rather than trust in the manifest.
                for f in sorted(p_resolved.rglob("*")):
                    if not f.is_file():
                        continue
                    try:
                        sub = f.relative_to(project_resolved)
                    except ValueError:
                        logger.warning("skipping out-of-project bundle path: %s", f)
                        swept_skips.append(f.name)
                        continue
                    if str(sub) in seen:
                        continue
                    if is_protected_path(sub):
                        logger.info("excluding protected path from bundle: %s", sub)
                        swept_skips.append(str(sub))
                        continue
                    seen.add(str(sub))
                    tar.add(f, arcname=str(sub))
            else:
                tar.add(p_resolved, arcname=arc)

    return referenced_skips, swept_skips
