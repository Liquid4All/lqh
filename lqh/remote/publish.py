"""End-of-run artifact publisher for remote backends.

Invoked as ``python -m lqh.remote.publish <run_dir>`` on the remote
machine after a successful train/infer run. Walks the run dir,
classifies each output, uploads to R2 via ``api.lqh.ai``, and writes
an ``artifacts.json`` manifest in the run dir that the client picks
up on the next rsync.

This replaces the previous "rsync everything including multi-GB
checkpoints back over SSH" path. Only the manifest + small files
(metrics, status, logs) flow over SSH after this; heavy artifacts
go remote → R2 → client directly via presigned URLs when needed.

Designed to be self-contained: stdlib + httpx, no lqh.train deps.
Auth and API base come from environment variables sourced by the
launcher (lqh/remote/ssh_direct.py writes them into
``.lqh-env/.env``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Self-contained imports: pull only the artifacts module from lqh.
# Avoid importing lqh.train so the remote venv doesn't need torch
# in its publish-only path.
from lqh.progress import emit_sentinel
from lqh.artifacts import (
    ArtifactError,
    ArtifactHandle,
    ArtifactKind,
    BackendArtifactStore,
)

logger = logging.getLogger(__name__)

__all__ = ["main", "publish_run", "PublishResult"]


@dataclass
class _Candidate:
    """One thing to publish: either a single file or a directory to be
    tar'd. ``kind`` is the registered artifact kind; ``relpath`` is the
    path relative to the run dir for human-readable manifest entries.

    ``lineage`` is an optional dict matching the ArtifactLineageInput
    schema; when set, it's passed to the register call so the backend
    writes an `artifact_lineage` row alongside the artifact row.
    """

    path: Path
    kind: ArtifactKind
    relpath: str
    is_dir: bool = False
    lineage: dict | None = None
    # "final" for a run's published model (model/ or model-lora/),
    # "intermediate" for per-step / per-iteration / per-sweep-config
    # checkpoints. Drives the backend retention engine's protection:
    # final checkpoints are kept indefinitely, intermediate ones expire
    # unless they're the project's best or have descendants. None for
    # non-checkpoint artifacts.
    checkpoint_role: str | None = None


@dataclass
class PublishResult:
    """Returned by ``publish_run`` and written to ``artifacts.json``."""

    artifacts: list[ArtifactHandle]
    failed: list[tuple[str, str]]  # (relpath, error_message)

    def to_json(self) -> dict:
        return {
            "artifacts": [
                {
                    "id": h.id,
                    "kind": h.kind,
                    "size_bytes": h.size_bytes,
                    "r2_key": h.r2_key,
                    "job_id": h.job_id,
                    "sha256": h.sha256,
                }
                for h in self.artifacts
            ],
            "failed": [
                {"relpath": rp, "error": err} for rp, err in self.failed
            ],
        }


def _load_lineage_sidecar(target: Path) -> dict | None:
    """Look for ``<target>.lineage.json`` (file) or
    ``<target>/lineage.json`` (dir). Returns the parsed dict or None.

    The trainer drops these next to each artifact it intends to
    publish: a SFT checkpoint dir gets ``checkpoints/4500/lineage.json``,
    a rollouts parquet gets ``iterations/2/predictions.parquet.lineage.json``.
    The publisher reads them on its sweep so the lineage row is
    written in the same call as the artifact registration.
    """
    if target.is_dir():
        cand = target / "lineage.json"
    else:
        cand = target.with_suffix(target.suffix + ".lineage.json")
    if cand.exists() and cand.is_file():
        try:
            return json.loads(cand.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read %s: %s", cand, exc)
    return None


def _adapter_base_model(target: Path) -> str | None:
    """Return PEFT adapter base model if ``target`` is an adapter dir."""
    cfg_path = target / "adapter_config.json"
    if not cfg_path.exists() or not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read %s: %s", cfg_path, exc)
        return None
    base = cfg.get("base_model_name_or_path")
    return base if isinstance(base, str) and base.strip() else None


def _infer_checkpoint_lineage(c: _Candidate) -> dict[str, Any] | None:
    """Infer lineage for adapter checkpoints when no sidecar was written.

    This is a defense-in-depth path for older trainers or manual artifact
    layouts that saved a LoRA adapter under ``model/`` instead of
    ``model-lora/``. The adapter manifest is the authoritative local signal.
    """
    if c.kind != "checkpoint" or not c.path.is_dir():
        return None
    base_model = _adapter_base_model(c.path)
    if base_model is None:
        return None
    return {
        "artifact_kind": "checkpoint",
        "training_method": "lora",
        "base_model": base_model,
        "hyperparams": {"lora_base": base_model},
        "parent_ids": [],
    }


def _dir_has_weights(d: Path) -> bool:
    """True when ``d`` looks like a real saved checkpoint (i.e. carries
    model weights), False when it's a sidecar dir holding only eval
    outputs.

    SFT writes two distinct shapes under ``checkpoints/``:
      * ``checkpoints/step_N/`` — eval-only dir produced by the
        per-checkpoint eval callback; contains ``predictions.parquet``
        and ``eval_request.json``, no weights.
      * ``checkpoints/checkpoint-N/`` (HF Trainer default) — real
        checkpoint with ``*.safetensors`` or ``adapter_model.*``.

    Tarring an eval-only dir as ``kind=checkpoint`` mislabels it in
    the artifact list and wastes R2 storage. Gating on the presence
    of weight files distinguishes them without coupling to a specific
    naming convention.
    """
    if not d.is_dir():
        return False
    for entry in d.iterdir():
        name = entry.name
        if name.endswith(".safetensors") or name == "pytorch_model.bin":
            return True
        if name.startswith("adapter_model.") or name.startswith("model.safetensors"):
            return True
    return False


def _run_reported_completed(run_dir: Path) -> bool:
    """Whether the workload wrote status.json with status=completed.

    lqh.remote.data_gen writes it after the pipeline finishes (completed
    only when at least one sample succeeded). Missing or unreadable
    status.json means the process died before finishing — never treat
    that as success.
    """
    try:
        status = json.loads((run_dir / "status.json").read_text())
    except Exception:
        return False
    return status.get("status") == "completed"


def _resolve_candidates(run_dir: Path) -> list[_Candidate]:
    """Enumerate publishable outputs in a run dir.

    Order matters only for stable logging — the upload itself happens
    in this order, which means small files land first and the user
    sees progress before a multi-GB checkpoint tar starts.
    """
    out: list[_Candidate] = []

    # --- run-root small files ---
    smalls: dict[str, ArtifactKind] = {
        "progress.jsonl": "metrics",
        "status.json": "metrics",
        "eval_request.json": "eval_result",
        "eval_result.json": "eval_result",
        "predictions.parquet": "predictions",
        "stdout.log": "logs",
        "stderr.log": "logs",
        "config.json": "other",
        "eval_history.json": "metrics",
        # Scoring-failure diagnostic (eval_hf / watcher scoring) — small
        # and load-bearing for debugging failed evals off-volume.
        "eval_error.json": "metrics",
        # data_gen sandbox output (lqh.remote.data_gen). Train run dirs
        # never carry a root-level data.parquet, so no collision.
        # data.partial.jsonl is deliberately NOT published — it's the
        # engine's resume scratch and stays on the volume. Same for the
        # infer loop's predictions.partial.jsonl (allowlist: neither is
        # ever matched).
        "data.parquet": "dataset",
    }
    for name, kind in smalls.items():
        p = run_dir / name
        if p.exists() and p.is_file():
            # A dataset artifact's mere EXISTENCE is treated as proof of
            # success downstream (the backend's orphan reconciler and the
            # TUI's restart-recovery both resolve a vanished data_gen job
            # as completed when one is registered). The engine writes
            # data.parquet even for a run with zero successful samples —
            # and an OOM mid-write can leave a truncated file — so
            # publish it only when the run itself reported success.
            if kind == "dataset" and not _run_reported_completed(run_dir):
                logger.info(
                    "skipping %s: run did not report completed status", name
                )
                continue
            out.append(_Candidate(path=p, kind=kind, relpath=name))

    # --- logs/ — the cloud launcher tees stdout/stderr into
    # run_dir/logs/ (see the cloud runner's buildLauncherScript). The
    # SSH-direct backend writes them at run_dir top-level, so we
    # support both layouts.
    logs_dir = run_dir / "logs"
    if logs_dir.exists() and logs_dir.is_dir():
        for name in ("stdout.log", "stderr.log"):
            p = logs_dir / name
            if p.exists() and p.is_file():
                out.append(_Candidate(path=p, kind="logs", relpath=f"logs/{name}"))

    # --- model/ or model-lora/ — the final dir written by the SFT
    # trainer (run_dir/model/ when merged, run_dir/model-lora/ when
    # saving the LoRA adapter alone). Tar-and-upload as a single
    # checkpoint artifact. The dir name carries forward to the R2
    # tar filename (model.tar.gz vs model-lora.tar.gz) so adapter
    # checkpoints are visually distinguishable in the artifact list
    # without needing a separate ArtifactKind. Without this branch
    # a successful fine-tune leaves no model state in R2.
    for sub in ("model", "model-lora"):
        p = run_dir / sub
        if _dir_has_weights(p):
            out.append(
                _Candidate(
                    path=p,
                    kind="checkpoint",
                    relpath=sub,
                    is_dir=True,
                    checkpoint_role="final",
                )
            )

    # --- checkpoints/<step>/ — directory per checkpoint, tar each ---
    # plus surface the eval-side files inside as their own typed
    # artifacts. This mirrors the iterations/ branch below: the whole
    # dir lands as one checkpoint tar (for restore), AND the small
    # eval-related files inside (predictions.parquet from
    # _run_checkpoint_eval, eval_request.json from write_eval_request)
    # are uploaded separately so consumers querying for ``kind=
    # predictions`` or ``kind=eval_result`` get sensible results
    # without having to download and untar the whole checkpoint.
    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists() and ckpt_root.is_dir():
        for sub in sorted(ckpt_root.iterdir()):
            if sub.is_dir():
                # Only tar the dir as a checkpoint artifact if it
                # actually carries weights — see _dir_has_weights.
                # Eval-only step_N dirs still surface their inner
                # predictions / eval_result files individually below.
                if _dir_has_weights(sub):
                    out.append(
                        _Candidate(
                            path=sub,
                            kind="checkpoint",
                            relpath=f"checkpoints/{sub.name}",
                            is_dir=True,
                            checkpoint_role="intermediate",
                        )
                    )
                for fname, kind in (
                    ("predictions.parquet", "predictions"),
                    ("eval_request.json", "eval_result"),
                    ("eval_result.json", "eval_result"),
                ):
                    p = sub / fname
                    if p.exists() and p.is_file():
                        out.append(
                            _Candidate(
                                path=p,
                                kind=kind,
                                relpath=f"checkpoints/{sub.name}/{fname}",
                            )
                        )
            elif sub.is_file():
                # eg checkpoints/best.json — small file alongside dirs
                # Skip lineage sidecars themselves; they're metadata
                # *about* artifacts, not artifacts in their own right.
                if sub.name.endswith(".lineage.json"):
                    continue
                out.append(
                    _Candidate(
                        path=sub,
                        kind="metrics",
                        relpath=f"checkpoints/{sub.name}",
                    )
                )

    # --- iterations/<i>/ — DPO iter dirs ---
    iter_root = run_dir / "iterations"
    if iter_root.exists() and iter_root.is_dir():
        for sub in sorted(iter_root.iterdir()):
            for fname, kind in (
                ("predictions.parquet", "predictions"),
                ("eval_predictions.parquet", "eval_result"),
                ("iter_request.json", "other"),
            ):
                p = sub / fname
                if p.exists() and p.is_file():
                    out.append(
                        _Candidate(
                            path=p,
                            kind=kind,
                            relpath=f"iterations/{sub.name}/{fname}",
                        )
                    )
            # Per-iteration checkpoint dir (lqh/train/dpo.py writes
            # iter_dir / "checkpoint" via save_pretrained). Gating on
            # _dir_has_weights skips the dir on incomplete iters or if
            # the saver was disabled.
            ckpt_sub = sub / "checkpoint"
            if _dir_has_weights(ckpt_sub):
                out.append(
                    _Candidate(
                        path=ckpt_sub,
                        kind="checkpoint",
                        relpath=f"iterations/{sub.name}/checkpoint",
                        is_dir=True,
                        checkpoint_role="intermediate",
                    )
                )

    # --- sweep_<id>/ — hyperparameter-sweep per-config sub-dirs ---
    # Each contains the child trainer's stdout.log, stderr.log,
    # progress.jsonl, eval_history.json, model{,-lora}/, and
    # checkpoints/. Without this branch all that's published from a
    # sweep run is the OUTER sweep's logs — the child's actual
    # training output is invisible, which is the failure mode that
    # made the before/after test impossible to debug.
    for sub in sorted(run_dir.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("sweep_")):
            continue
        # Small files at the sub-dir root.
        sub_smalls: dict[str, ArtifactKind] = {
            "config.json":       "other",
            "stdout.log":        "logs",
            "stderr.log":        "logs",
            "progress.jsonl":    "metrics",
            "eval_history.json": "metrics",
            "chosen_ce_summary.json": "metrics",
            "status.json":       "metrics",
        }
        for fname, kind in sub_smalls.items():
            p = sub / fname
            if p.exists() and p.is_file():
                out.append(
                    _Candidate(
                        path=p,
                        kind=kind,
                        relpath=f"{sub.name}/{fname}",
                    )
                )
        # Per-config model/checkpoints — same shape as the top-level
        # paths, but nested. Saves the trained adapter so a sweep
        # can be inspected after the fact.
        for mdir in ("model", "model-lora"):
            p = sub / mdir
            if _dir_has_weights(p):
                # Per-sweep-config model: "intermediate" so only the
                # winning config (best checkpoint) is protected from
                # auto-expiry; the losing configs age out.
                out.append(
                    _Candidate(
                        path=p,
                        kind="checkpoint",
                        relpath=f"{sub.name}/{mdir}",
                        is_dir=True,
                        checkpoint_role="intermediate",
                    )
                )

    # Single sweep to attach any lineage sidecar files. Cheap (just a
    # stat per candidate) and keeps the per-block logic above focused
    # on "what to publish" rather than "metadata to attach".
    for c in out:
        c.lineage = _load_lineage_sidecar(c.path)
        if c.lineage is None:
            c.lineage = _infer_checkpoint_lineage(c)

    return out


def _tar_directory(src: Path, dest: Path) -> None:
    """Create a gzipped tar at ``dest`` containing ``src`` rooted as its
    basename. Streaming, deterministic order, no compression-level
    knob (gzip default keeps a checkpoint at ~95% of original size for
    safetensors which barely compress; the win is one file instead of
    thousands)."""
    with tarfile.open(dest, "w:gz") as tar:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                arcname = f.relative_to(src.parent)
                tar.add(f, arcname=str(arcname))


async def publish_run(
    run_dir: Path,
    *,
    project_id: str,
    job_id: str | None = None,
    api_base: str | None = None,
    token: str | None = None,
) -> PublishResult:
    """Walk ``run_dir`` and upload artifacts. Returns a ``PublishResult``.

    Never raises on a per-file failure — that file ends up in
    ``failed`` instead. Raises only on a programmer error (bad args)
    or an unauthenticated backend.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    store = BackendArtifactStore(api_base=api_base, token=token)
    candidates = _resolve_candidates(run_dir)
    # Load-bearing artifacts first: a later failure (flaky network on a
    # log upload) must not be able to precede — and thereby doom — the
    # dataset a multi-hour data_gen run exists to produce.
    candidates.sort(key=lambda c: 0 if c.kind == "dataset" else 1)

    successes: list[ArtifactHandle] = []
    failures: list[tuple[str, str]] = []

    async def _upload_with_retries(path: Path, cand: "_Candidate") -> ArtifactHandle:
        # Catch-all per attempt: besides ArtifactError/OSError, uploads
        # can surface raw httpx transport errors, decode errors, etc. —
        # none of which should discard the rest of an otherwise
        # successful run. Two retries with a short pause ride out blips.
        last: Exception | None = None
        for attempt in range(3):
            try:
                return await store.upload_file(
                    path,
                    project_id=project_id,
                    kind=cand.kind,
                    job_id=job_id,
                    lineage=cand.lineage,
                    checkpoint_role=cand.checkpoint_role,
                )
            except Exception as exc:  # noqa: BLE001 — see comment above
                last = exc
                if attempt < 2:
                    logger.warning(
                        "publish attempt %d for %s failed (%s); retrying",
                        attempt + 1, cand.relpath, exc,
                    )
                    await asyncio.sleep(2.0 * (attempt + 1))
        assert last is not None
        raise last

    for cand in candidates:
        try:
            if cand.is_dir:
                # Tar into a tmp file alongside the run dir to avoid
                # filling /tmp on the GPU host (which is often small).
                with tempfile.TemporaryDirectory(dir=str(run_dir)) as td:
                    tar_path = Path(td) / f"{cand.path.name}.tar.gz"
                    _tar_directory(cand.path, tar_path)
                    handle = await _upload_with_retries(tar_path, cand)
            else:
                handle = await _upload_with_retries(cand.path, cand)
            logger.info("published %s -> %s", cand.relpath, handle.id)
            successes.append(handle)
            # In a cloud sandbox (LQH_JOB_ID set) this streams through the
            # event pipeline into the client's run_dir/artifacts.json, so
            # the TUI can resolve e.g. the dataset artifact without listing
            # the whole project. No-op outside sandboxes.
            emit_sentinel("artifact", {
                "artifact_id": handle.id,
                "kind": handle.kind,
                "relpath": cand.relpath,
            })
        except Exception as exc:  # noqa: BLE001 — per-candidate isolation
            logger.warning("publish failed for %s: %s", cand.relpath, exc)
            failures.append((cand.relpath, str(exc)))

    result = PublishResult(artifacts=successes, failed=failures)
    # Write manifest into run dir; the client picks it up via the
    # progress-sync rsync include list.
    manifest_path = run_dir / "artifacts.json"
    manifest_path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lqh.remote.publish",
        description="Publish a finished run's artifacts to api.lqh.ai (R2).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--project-id",
        required=True,
        help="Project identifier (typically the lqh project directory name).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Optional job UUID to associate artifacts with (cloud runs).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("LQH_BASE_URL"),
        help="API base URL (defaults to LQH_BASE_URL or https://api.lqh.ai/v1).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LQH_API_TOKEN") or os.environ.get("LQH_DEBUG_API_KEY"),
        help="Bearer token (defaults to $LQH_API_TOKEN or $LQH_DEBUG_API_KEY).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.token is None:
        print(
            "publish: no auth token available — set LQH_API_TOKEN in the remote env",
            file=sys.stderr,
        )
        return 2

    result = asyncio.run(
        publish_run(
            args.run_dir,
            project_id=args.project_id,
            job_id=args.job_id,
            api_base=args.api_base,
            token=args.token,
        )
    )
    print(
        f"published {len(result.artifacts)} artifacts, "
        f"{len(result.failed)} failed",
        file=sys.stderr,
    )
    # data_gen jobs (the launcher fails them on publish error): only the
    # dataset artifact is load-bearing — a failed log/metrics upload must
    # not flip an otherwise successful generation to failed.
    if os.environ.get("LQH_KIND") == "data_gen":
        if any(h.kind == "dataset" for h in result.artifacts):
            return 0
        print("publish: no dataset artifact was published", file=sys.stderr)
        return 1
    # eval_hf jobs (the launcher likewise fails them on publish error):
    # only eval_result.json is load-bearing — a failed log upload must
    # not flip a scored eval to failed, but a missing/failed
    # eval_result.json upload must. NOTE kind alone is not enough:
    # eval_request.json shares kind "eval_result" (smalls map above) and
    # is published even when scoring failed, so match the r2_key
    # basename suffix ("<hex>-<name>", minted by the backend).
    if os.environ.get("LQH_KIND") == "eval_hf":
        if any(
            h.kind == "eval_result" and h.r2_key.endswith("eval_result.json")
            for h in result.artifacts
        ):
            return 0
        print("publish: no eval_result artifact was published", file=sys.stderr)
        return 1
    # Surface a non-zero exit if anything failed so the launcher's
    # caller can decide whether to mark the run partially successful.
    return 0 if not result.failed else 1


if __name__ == "__main__":
    sys.exit(main())
