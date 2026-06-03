"""R2 → Hugging Face transfer.

Two sides live here:

* ``submit_transfer(...)`` — the CLI client. Asks api.lqh.ai to launch a
  CPU-only transfer sandbox (POST /v1/cloud/transfers) and returns the
  job id. Used by the ``push lqh:<id> -> hf:owner/repo`` tool path.

* ``main()`` / ``python -m lqh.remote.transfer`` — the in-sandbox
  launcher. Runs inside the CPU sandbox the backend started: downloads
  the source artifact from R2 via a presigned URL, then uploads it to
  the user's HF repo with the injected HF_TOKEN. Bytes go R2 → sandbox
  → HF directly; nothing round-trips through the laptop or the backend.
  The backend records the artifact's hf_repo when the job completes.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
from pathlib import Path

# ----------------------------------------------------------------------
# CLI client side
# ----------------------------------------------------------------------


async def submit_transfer(
    *,
    project_id: str,
    source_artifact_id: str,
    target_hf_repo: str,
    private: bool = True,
) -> str:
    """Ask the backend to start a CPU transfer job. Returns the job id."""
    import httpx

    from lqh.auth import require_token
    from lqh.config import load_config

    config = load_config()
    token = require_token()
    base = config.api_base_url.rstrip("/")
    # api_base_url already ends in /v1 in normal config; tolerate both.
    url = base + ("/cloud/transfers" if base.endswith("/v1") else "/v1/cloud/transfers")

    body = {
        "source_artifact_id": source_artifact_id,
        "target_hf_repo": target_hf_repo,
        "private": private,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            url, json=body, headers={"Authorization": f"Bearer {token}"}
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"transfer submit failed ({r.status_code}): {r.text[:300]}")
        return r.json()["job_id"]


# ----------------------------------------------------------------------
# In-sandbox launcher side
# ----------------------------------------------------------------------


def _download(url: str, dest: Path, *, chunk: int = 1 << 20) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=httpx.Timeout(600.0)) as resp:
        if resp.status_code != 200:
            body = resp.read()
            raise RuntimeError(f"R2 download failed ({resp.status_code}): {body[:200]!r}")
        with dest.open("wb") as fh:
            for piece in resp.iter_bytes(chunk):
                fh.write(piece)


def _safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract a tar.gz, rejecting unsafe members (matches the launcher's
    bundle-extraction hardening)."""
    dest_abs = os.path.realpath(dest)
    with tarfile.open(tar_path, "r:gz") as t:
        members = []
        for m in t.getmembers():
            n = m.name
            if m.issym() or m.islnk() or m.isdev() or m.ischr() or m.isblk() or m.isfifo():
                raise RuntimeError(f"reject unsafe tar member {n!r}")
            if n.startswith("/") or n.startswith("\\") or ".." in n.replace("\\", "/").split("/"):
                raise RuntimeError(f"reject traversal in tar member {n!r}")
            target = os.path.realpath(os.path.join(dest_abs, n))
            if target != dest_abs and not target.startswith(dest_abs + os.sep):
                raise RuntimeError(f"tar member escapes dest: {n!r}")
            members.append(m)
        t.extractall(dest_abs, members=members)


def run_transfer(
    *, source_url: str, hf_repo: str, private: bool, filename: str, hf_token: str | None,
) -> None:
    """Download the source artifact and upload it to HF."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    # Idempotent: create the repo if it doesn't exist.
    api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        local = tmp / (filename or "artifact.bin")
        print(f"transfer: downloading {filename or 'artifact'} from R2 ...", flush=True)
        _download(source_url, local)

        is_tar = local.name.endswith(".tar.gz") or local.name.endswith(".tgz")
        if is_tar:
            extract_dir = tmp / "extracted"
            extract_dir.mkdir()
            print("transfer: extracting checkpoint ...", flush=True)
            _safe_extract(local, extract_dir)
            # If the tar contained a single top-level dir, upload its
            # contents at the repo root (the usual checkpoint shape).
            entries = [p for p in extract_dir.iterdir()]
            root = entries[0] if len(entries) == 1 and entries[0].is_dir() else extract_dir
            print(f"transfer: uploading folder to hf:{hf_repo} ...", flush=True)
            api.upload_folder(repo_id=hf_repo, folder_path=str(root), repo_type="model")
        else:
            print(f"transfer: uploading file to hf:{hf_repo} ...", flush=True)
            api.upload_file(
                repo_id=hf_repo,
                path_or_fileobj=str(local),
                path_in_repo=local.name,
                repo_type="model",
            )
    print(f"transfer: done -> hf:{hf_repo}", flush=True)


def main(argv: list[str] | None = None) -> int:
    source_url = os.environ.get("LQH_TRANSFER_SOURCE_URL", "")
    hf_repo = os.environ.get("LQH_TRANSFER_HF_REPO", "")
    private = os.environ.get("LQH_TRANSFER_PRIVATE", "1") != "0"
    filename = os.environ.get("LQH_TRANSFER_FILENAME", "artifact.bin")
    hf_token = os.environ.get("HF_TOKEN") or None

    if not source_url or not hf_repo:
        print("transfer: LQH_TRANSFER_SOURCE_URL and LQH_TRANSFER_HF_REPO are required",
              file=sys.stderr)
        return 2
    try:
        run_transfer(
            source_url=source_url,
            hf_repo=hf_repo,
            private=private,
            filename=filename,
            hf_token=hf_token,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a non-zero exit for the launcher
        print(f"transfer: failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
