"""Hyperparameter sweep orchestrator (spawned as its own subprocess).

The sweep parent reads a single ``sweep_config.json`` that wraps a base
``lqh.train`` config plus a grid specification. It then runs the child
training subprocess (``python -m lqh.train``) sequentially for each grid
point, reads a cheap proxy metric per run, and picks the winner.

Why sweep is the default, and which proxy we use for which mode
================================================================
Hyperparameter sweeping is on by default because the cost structure for
fine-tuning is severely asymmetric:

  - Data generation is expensive (rollout chosen + rollout rejected +
    judge both ≈ several hours for DPO; for SFT a few hours of pipeline
    generation + judge filtering).
  - Training on a fixed dataset is cheap (~5–10 min per config on toka).
  - Judge eval on held-out is expensive again (~10 min per config).

We sweep training cheaply, pick a winner with an in-training proxy that
costs essentially nothing, then optionally pay for one judge eval on
the winner. The handler can be told ``enable_sweep=False`` if the user
explicitly says "just train one config".

Picking the proxy
-----------------
Both proxies were validated empirically on ar_to_de (results live in
``results/proxy_validation/`` from 2026-05-11 / 12).

SFT  →  ``eval_loss`` from HF Trainer.
        Pearson r = −0.90 with judge_mean. Top-1 picked correctly.
        SAFE because SFT cross-entropy directly measures the absolute
        probability the policy assigns to the gold response — there
        is no hackable ratio.

DPO  →  ``eval_ce_chosen_mean`` (written by ``_ChosenCECallback`` in
        ``lqh/train/dpo.py`` to ``iter_000/chosen_ce_summary.json``).
        Spearman ρ = −1.000 with judge_mean. Top-1 picked correctly,
        top-3 3/3.

        DPO ``eval_loss`` is intentionally NOT used for selection.
        It correlates with judge in the WRONG direction (r = +0.92).
        Reason: DPO loss = −log σ(β · (log P(chosen) − log P(rejected)));
        the policy can drive that loss to zero by dragging BOTH
        log-probs down with rejected falling fastest, even when the
        absolute probability of generating chosen has collapsed. This
        is classic DPO reward-hacking (cf. Pal et al. *Smaug / DPO-Positive*).
        ``eval_rewards/margins`` has the same failure mode.

        Chosen-CE is safe because the reference is FROZEN: only a true
        increase in P_policy(chosen) lowers it.

The selection function is hard-wired in this module. The agent never
sees DPO eval_loss in ``training_status`` (filtered out by
``_format_status`` in ``lqh/tools/handlers.py``).

Subprocess contract
-------------------
- Spawned exactly like ``lqh.train``: ``python -m lqh.train.sweep <cfg>``.
- Writes ``pid``, ``progress.jsonl``, ``stdout.log``, ``stderr.log`` in
  the same shape so ``SubprocessManager`` treats it identically.
- After each child run, appends a row to ``runs.jsonl`` and rewrites
  ``sweep_summary.json`` so a status read mid-sweep already sees the
  partial leaderboard.
- On completion, the winner's ``model/`` directory is exposed at the
  top level as a symlink (or copy) so downstream tools (``training_status``,
  ``start_local_eval``) can find it without sweep-awareness.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lqh.train.progress import write_progress, write_status
from lqh.train.resume import is_continuation


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """One point in the hyperparameter grid."""

    id: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ChildProgressContext:
    """Scoped parent-run metadata for forwarding child progress rows."""

    parent_run_dir: Path
    config_id: str
    config_index: int
    n_configs: int
    offset: int = 0
    last_step: int | None = None
    emitted_eval_keys: set[tuple[int, str]] = field(default_factory=set)


_CHILD_PROGRESS_CONTEXT: ContextVar[_ChildProgressContext | None] = ContextVar(
    "_CHILD_PROGRESS_CONTEXT",
    default=None,
)


def sft_grid_small() -> list[SweepPoint]:
    """SFT grid: lr ∈ {2e-5, 5e-5, 1e-4} × epochs ∈ {2, 3} = 6 configs.

    Pushes above lr=5e-5/epochs=3 (the config that won the validation
    grid at its top edge — optimum was likely higher than the tested
    range).
    """
    points: list[SweepPoint] = []
    for lr in (2e-5, 5e-5, 1e-4):
        for epochs in (2, 3):
            points.append(
                SweepPoint(
                    id=f"sft_lr{lr:g}_e{epochs}",
                    overrides={
                        "training": {"learning_rate": lr, "num_epochs": epochs},
                    },
                )
            )
    return points


def dpo_grid_small() -> list[SweepPoint]:
    """DPO grid: lr ∈ {3e-7, 1e-6, 3e-6} × β ∈ {0.05, 0.10} = 6 configs.

    Brackets the calm/collapse boundary observed on ar_to_de
    (calm at lr=1e-6, full collapse at lr=5e-6). 3e-6 may collapse but
    the chosen-CE proxy detects it and excludes from the winner pool.
    """
    points: list[SweepPoint] = []
    for lr in (3e-7, 1e-6, 3e-6):
        for beta in (0.05, 0.10):
            points.append(
                SweepPoint(
                    id=f"dpo_lr{lr:g}_b{beta:g}",
                    overrides={
                        "training": {"learning_rate": lr},
                        "dpo_beta": beta,
                    },
                )
            )
    return points


def sft_grid_tiny() -> list[SweepPoint]:
    """3-config SFT smoke grid: pick the lower-epoch row at each lr."""
    return [p for p in sft_grid_small() if p.id.endswith("_e2")]


def dpo_grid_tiny() -> list[SweepPoint]:
    """3-config DPO smoke grid: pick β=0.10 at each lr."""
    return [p for p in dpo_grid_small() if p.id.endswith("_b0.1")]


def resolve_grid(run_type: str, size: str) -> list[SweepPoint]:
    if run_type == "sft":
        return {"tiny": sft_grid_tiny, "small": sft_grid_small}.get(
            size, sft_grid_small
        )()
    if run_type in ("dpo", "on_policy_dpo"):
        return {"tiny": dpo_grid_tiny, "small": dpo_grid_small}.get(
            size, dpo_grid_small
        )()
    raise ValueError(f"unknown run_type for grid: {run_type!r}")


# ---------------------------------------------------------------------------
# Proxy reading
# ---------------------------------------------------------------------------

# (proxy_key, file_to_read_from, direction)
# direction is "min" for both — lower is better.
PROXY_SPEC: dict[str, tuple[str, str, str]] = {
    "sft": ("eval_loss", "eval_history.json", "min"),
    "dpo": ("eval_ce_chosen_mean", "iterations/iter_000/chosen_ce_summary.json", "min"),
}


# Collapse threshold: ce_chosen_delta_ref > 0.5 nats means the policy is
# noticeably worse than the reference at generating chosen — the boundary
# we saw between calm (Δref ≈ −0.01) and the first collapsed config
# (Δref ≈ +1.14) in the validation data.
COLLAPSE_DELTA_REF_THRESHOLD = 0.5


def _proxy_for(run_type: str) -> tuple[str, str, str]:
    key = "sft" if run_type == "sft" else "dpo"
    return PROXY_SPEC[key]


def _read_sft_proxy(sub_run_dir: Path) -> dict[str, Any]:
    _, fname, _ = PROXY_SPEC["sft"]
    path = sub_run_dir / fname
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text())
    except Exception:
        return {}
    # The last entry containing eval_loss is the post-load_best eval.
    for entry in reversed(entries):
        if "eval_loss" in entry:
            return {
                "primary": entry["eval_loss"],
                "eval_runtime": entry.get("eval_runtime"),
                "epoch": entry.get("epoch"),
            }
    return {}


def _read_dpo_proxy(sub_run_dir: Path) -> dict[str, Any]:
    _, fname, _ = PROXY_SPEC["dpo"]
    path = sub_run_dir / fname
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    out: dict[str, Any] = {}
    primary = payload.get("eval_ce_chosen_mean")
    if primary is not None:
        out["primary"] = primary
    for k in (
        "eval_ce_chosen_p90",
        "eval_ce_chosen_p95",
        "eval_ce_chosen_max",
        "eval_ce_chosen_delta_ref",
        "eval_ce_rejected_mean",
        "ref_ce_chosen_mean",
    ):
        if k in payload:
            out[k] = payload[k]
    return out


def _is_collapsed(proxy: dict[str, Any], run_type: str) -> bool:
    """DPO-only collapse detector. SFT never collapses in this sense."""
    if run_type == "sft":
        return False
    dref = proxy.get("eval_ce_chosen_delta_ref")
    return dref is not None and dref > COLLAPSE_DELTA_REF_THRESHOLD


def _pick_winner(
    rows: list[dict[str, Any]], run_type: str
) -> dict[str, Any] | None:
    """Pick the row with the lowest proxy value that is NOT collapsed."""
    candidates = [
        r for r in rows
        if r.get("primary") is not None and not r.get("collapsed", False)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["primary"])


# ---------------------------------------------------------------------------
# Child subprocess driver
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — overrides win, scalars in base are replaced."""
    out = copy.deepcopy(base)

    def _merge(a: dict[str, Any], b: dict[str, Any]) -> None:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                _merge(a[k], v)
            else:
                a[k] = v

    _merge(out, overrides)
    return out


def _run_child(sub_run_dir: Path, sub_config: dict[str, Any]) -> int:
    """Launch one lqh.train subprocess and block until it finishes."""
    sub_run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = sub_run_dir / "config.json"
    cfg_path.write_text(json.dumps(sub_config, indent=2) + "\n")

    stderr_path = sub_run_dir / "stderr.log"
    with (sub_run_dir / "stdout.log").open("w") as stdout_f, \
         stderr_path.open("w") as stderr_f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "lqh.train", str(cfg_path)],
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
        )
        while True:
            _forward_child_progress(sub_run_dir)
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(2.0)
        _forward_child_progress(sub_run_dir)
    # When a child fails, echo the tail of its stderr to OUR stderr
    # so the failure is visible in the sweep's published logs. The
    # child's sub_run_dir/stderr.log isn't picked up by publish.py
    # (which only walks the run_dir's standard layout), so without
    # this echo the only thing R2 sees is `rc=1 eval_loss=n/a`,
    # which is useless for debugging.
    if rc != 0:
        try:
            stderr_tail = stderr_path.read_text(errors="replace").splitlines()[-60:]
        except OSError:
            stderr_tail = ["(could not re-read child stderr)"]
        print(
            f"\nsweep: child {sub_run_dir.name} failed with rc="
            f"{rc}; last 60 stderr lines:",
            file=sys.stderr,
            flush=True,
        )
        for line in stderr_tail:
            print(f"  {line}", file=sys.stderr, flush=True)
    return rc


def _forward_child_progress(sub_run_dir: Path) -> None:
    """Forward new child progress rows into the sweep parent's progress log.

    The cloud runner only sees the sweep parent's stdout. Child SFT/DPO
    progress rows are written under ``sweep_<config>/progress.jsonl`` and
    would otherwise be invisible until the config completes.
    """
    ctx = _CHILD_PROGRESS_CONTEXT.get()
    if ctx is None:
        return
    progress_path = sub_run_dir / "progress.jsonl"
    if not progress_path.exists():
        return
    try:
        with progress_path.open("rb") as fh:
            fh.seek(ctx.offset)
            chunk = fh.read()
    except OSError:
        return
    if not chunk:
        return

    parts = chunk.split(b"\n")
    if chunk.endswith(b"\n"):
        raw_lines = parts[:-1]
        ctx.offset += len(chunk)
    else:
        # The child writes JSONL with open/write/close per row, but avoid
        # consuming a partially visible final line if a poll races the write.
        raw_lines = parts[:-1]
        ctx.offset += len(chunk) - len(parts[-1])

    for raw_bytes in raw_lines:
        raw = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        _forward_child_progress_row(ctx, row)


def _forward_child_progress_row(
    ctx: _ChildProgressContext,
    row: dict[str, Any],
) -> None:
    step = row.get("step")
    if not isinstance(step, int):
        return

    eval_loss = row.get("eval_loss")
    eval_key: tuple[int, str] | None = None
    if eval_loss is not None:
        eval_key = (step, str(eval_loss))

    should_emit = step != ctx.last_step
    if eval_key is not None and eval_key not in ctx.emitted_eval_keys:
        should_emit = True
    if not should_emit:
        return

    ctx.last_step = step
    if eval_key is not None:
        ctx.emitted_eval_keys.add(eval_key)

    extra: dict[str, Any] = {
        "phase": "sweep_config_progress",
        "config_id": ctx.config_id,
        "config_index": ctx.config_index,
        "n_configs": ctx.n_configs,
        "child_step": step,
    }
    if row.get("loss") is not None:
        extra["child_loss"] = row["loss"]
    if row.get("lr") is not None:
        extra["child_lr"] = row["lr"]
    if row.get("epoch") is not None:
        extra["child_epoch"] = row["epoch"]
    if eval_loss is not None:
        extra["child_eval_loss"] = eval_loss
    max_steps = row.get("max_steps")
    if isinstance(max_steps, int) and max_steps > 0:
        extra["child_max_steps"] = max_steps

    # Parent sweep rows normally use top-level `step` as the config index.
    # Forwarded child rows intentionally use the child's trainer step so
    # existing progress readers still show motion; phase/child_* identify
    # the row as within-config progress.
    write_progress(
        ctx.parent_run_dir,
        step=step,
        loss=row.get("loss"),
        lr=row.get("lr"),
        epoch=row.get("epoch"),
        extra=extra,
    )


def _winner_model_dir(run_dir: Path) -> Path | None:
    """Return the materialized sweep winner dir, preserving adapter names."""
    for name in ("model", "model-lora"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _build_eval_of_best_config(base: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    """Compose the infer config the eval-of-best step runs against.

    Returns None when there's nothing to evaluate against — either no
    ``eval_dataset`` was supplied in the sweep's base config, or no
    winner model has been materialized at the run root yet.
    """
    eval_dataset = base.get("eval_dataset")
    if not eval_dataset:
        return None
    model_path = _winner_model_dir(run_dir)
    if model_path is None:
        return None
    # Build a minimal lqh.infer config that matches what the rest of
    # the stack expects (matches the shape produced by
    # handle_start_local_eval).
    cfg: dict[str, Any] = {
        "type": "infer",
        "base_model": str(model_path.resolve()),
        "dataset": eval_dataset,
        "max_new_tokens": int(base.get("max_new_tokens", 4096)),
        "manifest": ["base_model", "dataset"],
    }
    if scorer := base.get("scorer"):
        cfg["scorer"] = scorer
        cfg["manifest"].append("scorer")
    if sp := base.get("system_prompt"):
        cfg["system_prompt"] = sp
    if rf := base.get("response_format"):
        cfg["response_format"] = rf
    return cfg


def _run_eval_of_best(run_dir: Path, base: dict[str, Any]) -> dict[str, Any]:
    """Run ``lqh.infer`` against the sweep's winner model and surface
    the predictions at the run-dir level so the watcher scores them.

    Self-contained: builds the infer config, runs it as a subprocess
    in a ``eval_of_best/`` subdir (so its progress.jsonl / pid don't
    collide with the sweep's), then symlinks the resulting
    ``predictions.parquet`` + ``eval_request.json`` up to ``run_dir``
    so the existing scoring loop picks them up without changes.

    Returns a small summary dict for the sweep's final status payload
    or ``{"skipped": "<reason>"}`` if eval was a no-op (no eval_dataset
    in base config, no winner model, or infer subprocess crashed).
    """
    cfg = _build_eval_of_best_config(base, run_dir)
    if cfg is None:
        return {"skipped": "no eval_dataset or no winner model"}

    eval_dir = run_dir / "eval_of_best"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = eval_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"sweep: starting eval-of-best on winner ({cfg['base_model']})", flush=True)
    write_progress(
        run_dir,
        step=0,
        extra={"phase": "eval_of_best_start", "dataset": cfg.get("dataset")},
    )

    log_path = eval_dir / "infer.log"
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "lqh.infer", str(cfg_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rc = proc.returncode
        except FileNotFoundError as exc:
            # Python missing — practically impossible since we're
            # running under it, but defend against PATH oddities in
            # restricted sandboxes anyway.
            return {"skipped": f"failed to spawn infer subprocess: {exc}"}

    if rc != 0:
        print(f"sweep: eval-of-best failed (rc={rc}); see eval_of_best/infer.log", flush=True)
        return {"skipped": f"infer subprocess exit code {rc}"}

    # Surface predictions + eval_request at the run-dir level so the
    # remote watcher's existing path detection scores them. Use
    # symlinks (cheap, no duplicated disk) and fall back to copies
    # on filesystems that reject symlinks.
    for fname in ("predictions.parquet", "eval_request.json"):
        src = eval_dir / fname
        if not src.exists():
            continue
        dst = run_dir / fname
        if dst.is_symlink() or dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            import shutil
            shutil.copy2(src, dst)

    summary: dict[str, Any] = {
        "ok": True,
        "dataset": cfg.get("dataset"),
        "model": cfg["base_model"],
    }

    # Cloud-mode: score the eval predictions inline so we emit a
    # complete real-metric (judge_score) artifact without depending
    # on the laptop watcher. score_run_eval_inline is a no-op for
    # SSH backends — the laptop watcher continues to pick up the
    # symlinked predictions + score them as before.
    try:
        from lqh.train.cloud_score import score_run_eval_inline

        score_summary = score_run_eval_inline(run_dir, base)
        if score_summary is not None:
            summary["score_summary"] = score_summary
    except Exception as exc:  # noqa: BLE001
        print(f"sweep: inline eval-of-best scoring failed: {exc}", flush=True)

    return summary


def _materialize_best_model(winner_dir: Path, run_dir: Path) -> None:
    """Expose the winner's model directly at the sweep run root.

    Looks for ``winner_dir/model`` first (full / merged-LoRA case),
    falls back to ``winner_dir/model-lora`` (adapter-only LoRA case;
    sft.py:_save_final_model writes there when ``lora.merge`` is
    false). ``load_for_inference`` happily resolves an adapter dir
    (PeftModel + merge_and_unload), so eval-of-best works against
    either layout. The destination preserves the source directory name
    so publish sees adapter checkpoints as ``model-lora.tar.gz``.

    Tries a symlink first (instant, no disk duplication). Falls back
    to ``shutil.copytree`` on filesystems that don't support symlinks.
    """
    src: Path | None = None
    dst_name: str | None = None
    for candidate in ("model", "model-lora"):
        candidate_path = winner_dir / candidate
        if candidate_path.exists():
            src = candidate_path
            dst_name = candidate
            break
    if src is None or dst_name is None:
        return
    dst = run_dir / dst_name
    # Clear any prior winner artifact (symlink or copy).
    for stale_name in ("model", "model-lora"):
        stale = run_dir / stale_name
        if not (stale.is_symlink() or stale.exists()):
            continue
        try:
            if stale.is_symlink() or not stale.is_dir():
                stale.unlink()
            else:
                import shutil
                shutil.rmtree(stale)
        except OSError:
            pass
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:
        import shutil
        shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------


def _completed_sweep_rows(runs_jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Read successful per-config rows from a previous sweep attempt."""
    if not runs_jsonl_path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    try:
        lines = runs_jsonl_path.read_text().splitlines()
    except OSError:
        return done
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        config_id = row.get("config_id")
        if isinstance(config_id, str) and row.get("rc") == 0:
            done[config_id] = row
    return done


def _write_sweep_summary(
    path: Path,
    *,
    run_type: str,
    grid_size: str,
    n_configs: int,
    proxy_key: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_elapsed_s = sum(
        float(r["elapsed_s"])
        for r in rows
        if isinstance(r.get("elapsed_s"), (int, float))
    )
    summary = {
        "mode": run_type,
        "grid_size": grid_size,
        "n_configs": n_configs,
        "n_completed": len(rows),
        "total_elapsed_s": total_elapsed_s,
        "proxy_key": proxy_key,
        "collapse_threshold_delta_ref": COLLAPSE_DELTA_REF_THRESHOLD,
        "rows": rows,
        "winner": _pick_winner(rows, run_type),
    }
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _dpo_no_proxy_message(run_dir: Path, rows: list[dict[str, Any]]) -> str | None:
    """Explain the common DPO no-winner case: completed configs with no
    held-out preference pairs, so the chosen-CE proxy was never written."""
    if not rows:
        return None

    completed_without_proxy = [
        r for r in rows
        if r.get("rc") == 0 and r.get("primary") is None and not r.get("collapsed")
    ]
    if len(completed_without_proxy) != len(rows):
        return None

    prefs: list[int] = []
    eval_pairs: list[int] = []
    n_iters = 0
    for row in completed_without_proxy:
        sub_dir = row.get("sub_dir")
        if not isinstance(sub_dir, str):
            continue
        iter_root = run_dir / sub_dir / "iterations"
        if not iter_root.exists():
            continue
        for result_path in sorted(iter_root.glob("iter_*/dpo_result.json")):
            try:
                payload = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            n_iters += 1
            if isinstance(payload.get("num_preferences"), int):
                prefs.append(payload["num_preferences"])
            if isinstance(payload.get("eval_pairs"), int):
                eval_pairs.append(payload["eval_pairs"])

    if not n_iters or not eval_pairs or any(n > 0 for n in eval_pairs):
        return None

    pref_range = (
        f"{min(prefs)}-{max(prefs)}" if prefs and min(prefs) != max(prefs)
        else str(prefs[0]) if prefs
        else "too few"
    )
    return (
        "DPO generated too few usable preference pairs to compute the sweep "
        "proxy. All completed configs had 0 held-out preference pairs, so "
        "iterations/iter_000/chosen_ce_summary.json was not written and "
        "eval_ce_chosen_mean is unavailable. This usually means the policy "
        "rollouts are already too close to the chosen/gold answers for this "
        f"DPO prompt count: observed only {pref_range} preference pairs per "
        "iteration. DPO cannot be selected/applied reliably at this size; "
        "increase the DPO prompt count substantially, use prompts with more "
        "headroom, or skip DPO for this benchmark point."
    )


def _no_winner_message(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    """Explain *why* no winner was selected, broken down by failure mode.

    The old message ("all N configs failed or collapsed") conflated three
    very different outcomes — children that crashed (rc != 0), children that
    collapsed (proxy past the threshold), and children that completed but
    produced no usable proxy metric. They want different fixes, so spell out
    which configs landed where and point crashed ones at their stderr.log.
    """
    if not rows:
        return "sweep produced no configs to evaluate; no model selected"

    crashed = [r for r in rows if r.get("rc") not in (0, None)]
    collapsed = [r for r in rows if r.get("rc") in (0, None) and r.get("collapsed")]
    no_proxy = [
        r for r in rows
        if r.get("rc") in (0, None)
        and not r.get("collapsed")
        and r.get("primary") is None
    ]

    def _ids(items: list[dict[str, Any]]) -> str:
        return ", ".join(str(r.get("config_id", "?")) for r in items)

    parts: list[str] = [
        f"no model selected: none of {len(rows)} sweep config(s) "
        f"produced a usable winner."
    ]
    if crashed:
        # Surface the actual log path for the first crashed child so the
        # cause (e.g. a bad base_model path) is one click away.
        first = crashed[0]
        log_hint = ""
        sub_dir = first.get("sub_dir")
        if isinstance(sub_dir, str):
            log_hint = f" (see {run_dir / sub_dir / 'stderr.log'})"
        rcs = sorted({r.get("rc") for r in crashed})
        parts.append(
            f"{len(crashed)} crashed before completing "
            f"[rc={','.join(str(c) for c in rcs)}]: {_ids(crashed)}{log_hint}"
        )
    if collapsed:
        parts.append(
            f"{len(collapsed)} collapsed "
            f"(proxy past the {COLLAPSE_DELTA_REF_THRESHOLD} threshold): "
            f"{_ids(collapsed)}"
        )
    if no_proxy:
        parts.append(
            f"{len(no_proxy)} completed but wrote no proxy metric: {_ids(no_proxy)}"
        )
    return " ".join(parts)


def sweep_loop(run_dir: Path, sweep_config: dict[str, Any]) -> None:
    """Sweep entry point invoked by ``python -m lqh.train.sweep <cfg>``."""
    base = sweep_config["base_config"]
    run_type = base.get("type", "sft")
    grid_size = sweep_config.get("grid_size", "small")
    proxy_key, _, _ = _proxy_for(run_type)

    if sweep_config.get("grid_override"):
        # External override (e.g. from the test harness). Each entry must
        # be {"id": str, "overrides": {...}}.
        grid = [SweepPoint(id=p["id"], overrides=p["overrides"])
                for p in sweep_config["grid_override"]]
    else:
        grid = resolve_grid(run_type, grid_size)

    print(
        f"sweep: type={run_type} grid_size={grid_size} n_configs={len(grid)} "
        f"proxy={proxy_key}",
        flush=True,
    )
    write_progress(
        run_dir, step=0,
        extra={
            "phase": "sweep_start",
            "n_configs": len(grid),
            "grid_size": grid_size,
            "run_type": run_type,
            "proxy_key": proxy_key,
        },
    )

    rows: list[dict[str, Any]] = []
    sweep_summary_path = run_dir / "sweep_summary.json"
    runs_jsonl_path = run_dir / "runs.jsonl"
    completed_rows = _completed_sweep_rows(runs_jsonl_path) if is_continuation() else {}
    if completed_rows:
        print(
            f"sweep: continuation detected; {len(completed_rows)} completed config(s) "
            "will be skipped",
            flush=True,
        )

    for i, point in enumerate(grid):
        sub_run_dir = run_dir / f"sweep_{point.id}"
        sub_config = _deep_merge(base, point.overrides)
        # Child runs MUST not recurse into another sweep.
        sub_config.pop("enable_sweep", None)

        if point.id in completed_rows:
            row = dict(completed_rows[point.id])
            rows.append(row)
            print(
                f"\n[{i+1}/{len(grid)}] {point.id} already completed; skipping",
                flush=True,
            )
            write_progress(
                run_dir, step=i + 1,
                extra={
                    "phase": "sweep_config_done",
                    "config_id": point.id,
                    "rc": row.get("rc"),
                    "primary": row.get("primary"),
                    "collapsed": row.get("collapsed", False),
                    "config_index": i,
                    "n_configs": len(grid),
                    "resumed": True,
                },
            )
            running_summary = _write_sweep_summary(
                sweep_summary_path,
                run_type=run_type,
                grid_size=grid_size,
                n_configs=len(grid),
                proxy_key=proxy_key,
                rows=rows,
            )
            running_winner = running_summary["winner"]
            if (
                running_winner is not None
                and running_winner["config_id"] == point.id
            ):
                _materialize_best_model(sub_run_dir, run_dir)
            continue

        write_progress(
            run_dir, step=i,
            extra={
                "phase": "sweep_config_start",
                "config_id": point.id,
                "config_index": i,
                "n_configs": len(grid),
            },
        )
        print(f"\n[{i+1}/{len(grid)}] {point.id}", flush=True)

        t0 = datetime.now(timezone.utc).timestamp()
        progress_ctx = _ChildProgressContext(
            parent_run_dir=run_dir,
            config_id=point.id,
            config_index=i,
            n_configs=len(grid),
        )
        token = _CHILD_PROGRESS_CONTEXT.set(progress_ctx)
        try:
            rc = _run_child(sub_run_dir, sub_config)
        finally:
            _CHILD_PROGRESS_CONTEXT.reset(token)
        elapsed = datetime.now(timezone.utc).timestamp() - t0

        if run_type == "sft":
            proxy = _read_sft_proxy(sub_run_dir)
        else:
            proxy = _read_dpo_proxy(sub_run_dir)
        collapsed = _is_collapsed(proxy, run_type)

        row: dict[str, Any] = {
            "config_id": point.id,
            "overrides": point.overrides,
            "rc": rc,
            "primary": proxy.get("primary"),
            "collapsed": collapsed,
            "elapsed_s": elapsed,
            "sub_dir": sub_run_dir.name,
        }
        for k, v in proxy.items():
            if k != "primary":
                row[k] = v
        rows.append(row)

        try:
            with runs_jsonl_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass

        primary_str = (
            f"{row['primary']:.4f}" if row["primary"] is not None else "n/a"
        )
        flag = " [COLLAPSED]" if collapsed else ""
        print(
            f"  → rc={rc} {proxy_key}={primary_str}{flag} elapsed={elapsed:.0f}s",
            flush=True,
        )

        write_progress(
            run_dir, step=i + 1,
            extra={
                "phase": "sweep_config_done",
                "config_id": point.id,
                "rc": rc,
                "primary": row["primary"],
                "collapsed": collapsed,
                "config_index": i,
                "n_configs": len(grid),
            },
        )

        sweep_summary = _write_sweep_summary(
            sweep_summary_path,
            run_type=run_type,
            grid_size=grid_size,
            n_configs=len(grid),
            proxy_key=proxy_key,
            rows=rows,
        )
        running_winner = sweep_summary["winner"]

        if (running_winner is not None
                and running_winner["config_id"] == point.id):
            _materialize_best_model(sub_run_dir, run_dir)

    winner = _pick_winner(rows, run_type)
    if winner is None:
        dpo_msg = _dpo_no_proxy_message(run_dir, rows) if run_type != "sft" else None
        msg = dpo_msg or _no_winner_message(run_dir, rows)
        print(msg, flush=True)
        write_status(run_dir, "failed", error=msg)
        raise SystemExit(1)

    print(
        f"\nsweep complete. winner={winner['config_id']} "
        f"{proxy_key}={winner['primary']:.4f}",
        flush=True,
    )

    # Eval-of-best: if the base config has an eval_dataset, run the
    # winner against it now while the GPU is still warm. The result
    # (predictions.parquet + eval_request.json at the run-dir level)
    # gets scored by the host-side watcher / cloud SSE consumer
    # exactly the same way a standalone start_local_eval would be.
    #
    # Gated by sweep_config["eval_best"] (default true so cloud runs
    # produce a final score without a second submit; callers that
    # only want the sweep can pass eval_best=false explicitly).
    eval_summary: dict[str, Any] = {}
    if sweep_config.get("eval_best", True):
        eval_summary = _run_eval_of_best(run_dir, base)
        print(f"sweep: eval-of-best summary: {eval_summary}", flush=True)

    write_status(
        run_dir, "completed",
        extra={
            "winner": winner["config_id"],
            "winner_primary": winner["primary"],
            "n_configs": len(rows),
            "eval_of_best": eval_summary,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.train.sweep <sweep_config.json>", file=sys.stderr)
        sys.exit(1)
    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.exists():
        print(f"Sweep config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = json.loads(cfg_path.read_text())
    run_dir = cfg_path.parent
    (run_dir / "pid").write_text(str(os.getpid()))
    try:
        sweep_loop(run_dir, cfg)
    except Exception as exc:
        write_status(run_dir, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
