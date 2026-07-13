"""Standalone HuggingFace eval — runs any HF checkpoint on a GPU sandbox.

Sandbox entrypoint mapped from the backend's ``eval_hf`` cloud-job
kind (see backend/internal/handler/cloud_jobs.go module whitelist).
Invoked as ``python -m lqh.infer.eval_hf <config.json>``.

Two use cases this serves:

  1. **External eval** — "evaluate someone else's LoRA / full model
     (public or private HF repo) on our eval set + judge".
  2. **Baseline benchmarking** — "take an off-the-shelf LFM2 fine-tune
     from the hub and compare against our trained checkpoint".

Config shape::

    {
      "hf_repo":         "Qwen/Qwen3.5-3B-Instruct",
      "revision":        "main",                # optional, defaults to main
      "training_method": "lora" | "full",
      "base_model":      "Qwen/Qwen3.5-3B-Instruct",   # required iff lora
      "eval_dataset":    "evals/translation.parquet",   # path inside bundle
      "scorer":          "scorers/translation.md",      # path inside bundle
      "system_prompt":   "...",                         # optional
      "judge_size":      "small" | "medium" | "large",  # optional, default small
      "max_new_tokens":  4096                            # optional
    }

Flow:

  1. Download the HF repo with ``huggingface_hub.snapshot_download``
     (HF_TOKEN forwarded by the backend for private repos).
  2. Build a normal ``lqh.infer`` config pointing at the downloaded
     dir; for LoRA, also set ``base_override`` so the adapter merges
     onto the caller-specified base.
  3. Delegate to ``lqh.infer.__main__._run_inference`` for the
     prediction loop — same generation, decoding, tool-call parsing,
     and JSON-schema constrained decoding as a normal infer run.
  4. Drop a ``predictions.parquet.lineage.json`` sidecar so the
     publish step (lqh/remote/publish.py) registers the
     ``artifact_lineage`` row with HF repo / revision / training
     method / judge / hyperparams. The HF repo is recorded as
     ``base_model``; ``parent_ids`` is empty because the parent is
     external (not an LQH artifact UUID).
  5. Inline-score the predictions via ``lqh.train.cloud_score``
     (no-op outside a cloud sandbox) so the published
     ``eval_result.json`` artifact carries the judge summary.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["main"]


_VALID_TRAINING_METHODS = {"lora", "full"}


def _validate(config: dict[str, Any]) -> None:
    if not isinstance(config.get("hf_repo"), str) or not config["hf_repo"].strip():
        raise ValueError("eval_hf config.hf_repo is required (string)")
    method = config.get("training_method")
    if method not in _VALID_TRAINING_METHODS:
        raise ValueError(
            f"eval_hf config.training_method must be one of "
            f"{sorted(_VALID_TRAINING_METHODS)}; got {method!r}"
        )
    if method == "lora" and not config.get("base_model"):
        raise ValueError(
            "eval_hf config.base_model is required when training_method='lora'"
        )
    if not config.get("eval_dataset"):
        raise ValueError("eval_hf config.eval_dataset is required")


def _download_checkpoint(repo: str, revision: str | None, dest_root: Path) -> Path:
    """Snapshot-download ``repo@revision`` into a subdirectory of
    ``dest_root``. Uses HF_TOKEN from the env when present (private
    repos). Returns the absolute local path the model can be loaded
    from.

    Defers the huggingface_hub import so the module load-time cost
    is paid only when this entrypoint actually runs (saves test
    import time, keeps non-eval_hf paths unaffected).
    """
    from huggingface_hub import snapshot_download

    # Use the repo slug as the on-disk dir name; revision is appended
    # so a single sandbox can host multiple revisions side by side
    # (useful for ablations).
    slug = repo.replace("/", "__")
    rev = revision or "main"
    target = dest_root / f"{slug}@{rev}"
    target.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    logger.info("snapshot_download %s@%s → %s", repo, rev, target)
    snapshot_download(
        repo_id=repo,
        revision=rev,
        local_dir=str(target),
        token=token,
    )
    return target


def _write_lineage_sidecar(
    run_dir: Path,
    config: dict[str, Any],
    *,
    judge: str,
) -> None:
    """Drop ``predictions.parquet.lineage.json`` so the publisher
    registers an ``artifact_lineage`` row. The parent is external
    (HF), recorded via ``base_model``.
    """
    pred_path = run_dir / "predictions.parquet"
    if not pred_path.exists():
        return
    hf_repo = config["hf_repo"]
    revision = config.get("revision") or "main"
    base_model_id = f"{hf_repo}@{revision}"
    lineage: dict[str, Any] = {
        "artifact_kind": "rollout",
        "training_method": config["training_method"],
        "base_model": base_model_id,
        "reward_model": judge,
        "hyperparams": {
            "max_new_tokens": int(config.get("max_new_tokens", 4096)),
            "system_prompt_present": bool(config.get("system_prompt")),
        },
        # parent_ids stays empty: the parent is an external HF repo
        # rather than an LQH artifact UUID. The repo + revision live
        # in base_model above so it's still discoverable.
        "parent_ids": [],
    }
    if config["training_method"] == "lora":
        lineage["hyperparams"]["lora_base"] = config["base_model"]
    sidecar = pred_path.with_suffix(pred_path.suffix + ".lineage.json")
    sidecar.write_text(json.dumps(lineage, indent=2) + "\n")
    logger.info("wrote lineage sidecar at %s", sidecar)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.infer.eval_hf <config.json>", file=sys.stderr)
        sys.exit(1)
    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    _validate(config)
    run_dir = config_path.parent
    (run_dir / "pid").write_text(str(os.getpid()))
    from lqh.train.progress import begin_run_attempt, write_status
    begin_run_attempt(run_dir)

    try:
        from lqh.progress import ProgressReporter

        ProgressReporter(
            task_kind="evaluation", label="HF model evaluation",
            run_dir=run_dir,
        ).update(
            phase="setup", phase_label="downloading checkpoint",
            overall_fraction=0, force=True,
        )
        # 1. Download HF checkpoint to a sandbox-local dir.
        dl_root = run_dir / "hf_checkpoints"
        local_dir = _download_checkpoint(
            config["hf_repo"], config.get("revision"), dl_root,
        )

        # 2. Build a lqh.infer-shaped config pointing at the download.
        infer_config: dict[str, Any] = {
            "type": "infer",
            "base_model": str(local_dir),
            "dataset": config["eval_dataset"],
            "max_new_tokens": int(config.get("max_new_tokens", 4096)),
            "progress_label": "HF model evaluation",
        }
        if config["training_method"] == "lora":
            # _run_inference forwards `base_override` to
            # load_for_inference, which uses it instead of the
            # adapter_config.json's recorded base. Required when
            # adapter_config.json points at a repo that's since been
            # renamed / privated; harmless otherwise.
            infer_config["base_override"] = config["base_model"]
        if sp := config.get("system_prompt"):
            infer_config["system_prompt"] = sp
        if rf := config.get("response_format"):
            infer_config["response_format"] = rf
        # Carry the scorer through so cloud_score finds it via the
        # same resolution rule the DPO / sweep paths use.
        if scorer := config.get("scorer"):
            infer_config["scorer"] = scorer

        # 3. Delegate to the regular infer loop. It writes
        # predictions.parquet + eval_request.json + status sentinels
        # exactly the way a normal infer run does.
        from lqh.infer.__main__ import _run_inference

        _run_inference(run_dir, infer_config)

        # 4. Lineage sidecar (publisher picks it up automatically).
        judge_size = (config.get("judge_size") or "small").lower()
        judge = f"judge:{judge_size}"
        _write_lineage_sidecar(run_dir, config, judge=judge)

        # 5. Inline scoring — same scoped LQH_API_TOKEN path as the
        # DPO + SFT-sweep eval-of-best uses. Writes eval_result.json
        # next to predictions so the publisher classifies it
        # accordingly.
        #
        # We deliberately use print() rather than logger.* here so the
        # diagnostic shows up in the published stdout.log without
        # needing a logging.basicConfig — the eval_hf sandbox runs
        # without any logging handler so logger.warning is invisible.
        print("eval_hf: starting inline scoring step", flush=True)
        try:
            from lqh.train.cloud_score import is_cloud_mode, score_run_eval_inline

            if not is_cloud_mode():
                print(
                    "eval_hf: SKIPPED inline scoring — is_cloud_mode() is "
                    f"False (LQH_API_TOKEN set={'LQH_API_TOKEN' in os.environ}, "
                    f"LQH_BASE_URL set={'LQH_BASE_URL' in os.environ}).",
                    flush=True,
                )
            else:
                summary = score_run_eval_inline(run_dir, infer_config)
                if summary is None:
                    print(
                        "eval_hf: WARNING — inline scoring returned None. "
                        "Likely cause: scorer path didn't resolve under "
                        f"cwd={Path.cwd()} (config.scorer="
                        f"{infer_config.get('scorer')!r}) "
                        "or predictions.parquet missing.",
                        flush=True,
                    )
                else:
                    print(
                        f"eval_hf: inline scoring summary keys="
                        f"{sorted(summary.keys()) if isinstance(summary, dict) else type(summary).__name__}",
                        flush=True,
                    )
                    # Surface the judge summary on the rollout lineage
                    # as real_metric — this is the row the backend's
                    # BestCheckpointForProject query sorts by. The
                    # lqh.scoring.run_scoring contract puts the stats
                    # under summary["scores"] = {mean, median, std,
                    # min, max} (see lqh/scoring.py:563), NOT at the
                    # top level — early versions of this code looked
                    # for `summary["mean"]` and silently no-op'd.
                    scores_block = (
                        summary.get("scores")
                        if isinstance(summary, dict) else None
                    )
                    mean = (
                        scores_block.get("mean")
                        if isinstance(scores_block, dict) else None
                    )
                    n_scored = (
                        summary.get("num_scored")
                        if isinstance(summary, dict) else None
                    )
                    pred_lineage = run_dir / "predictions.parquet.lineage.json"
                    if pred_lineage.exists():
                        payload = json.loads(pred_lineage.read_text())
                        if mean is not None:
                            payload["real_metric"] = {
                                "name": "judge_score_mean",
                                "value": float(mean),
                                "n": int(n_scored or 0),
                            }
                            print(
                                f"eval_hf: real_metric stamped "
                                f"(mean={float(mean):.3f}, n={int(n_scored or 0)})",
                                flush=True,
                            )
                        else:
                            print(
                                "eval_hf: WARNING — summary has no scores.mean; "
                                f"lineage real_metric NOT stamped. summary={summary!r}",
                                flush=True,
                            )
                        pred_lineage.write_text(
                            json.dumps(payload, indent=2) + "\n"
                        )
                    else:
                        print(
                            "eval_hf: WARNING — no lineage sidecar at "
                            f"{pred_lineage}; can't stamp real_metric.",
                            flush=True,
                        )
        except Exception as exc:  # noqa: BLE001
            # Surface the exception type + message + traceback to
            # stdout so the published log captures what went wrong.
            # The publisher uploads stdout.log → operator can grep.
            import traceback
            print(
                f"eval_hf: ERROR in inline scoring: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
    except Exception as exc:
        write_status(run_dir, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
