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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lqh.train.progress import write_progress, write_status


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """One point in the hyperparameter grid."""

    id: str
    overrides: dict[str, Any] = field(default_factory=dict)


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

    with (sub_run_dir / "stdout.log").open("w") as stdout_f, \
         (sub_run_dir / "stderr.log").open("w") as stderr_f:
        proc = subprocess.run(
            [sys.executable, "-m", "lqh.train", str(cfg_path)],
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
        )
    return proc.returncode


def _materialize_best_model(winner_dir: Path, run_dir: Path) -> None:
    """Expose the winner's model directly as ``run_dir/model``.

    Tries a symlink first (instant, no disk duplication). Falls back to
    ``shutil.copytree`` on filesystems that don't support symlinks.
    """
    src = winner_dir / "model"
    if not src.exists():
        return
    dst = run_dir / "model"
    # Clear any prior winner artifact (symlink or copy).
    if dst.is_symlink() or dst.exists():
        try:
            if dst.is_symlink() or not dst.is_dir():
                dst.unlink()
            else:
                import shutil
                shutil.rmtree(dst)
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

    for i, point in enumerate(grid):
        sub_run_dir = run_dir / f"sweep_{point.id}"
        sub_config = _deep_merge(base, point.overrides)
        # Child runs MUST not recurse into another sweep.
        sub_config.pop("enable_sweep", None)

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
        rc = _run_child(sub_run_dir, sub_config)
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

        running_winner = _pick_winner(rows, run_type)
        sweep_summary = {
            "mode": run_type,
            "grid_size": grid_size,
            "n_configs": len(grid),
            "n_completed": len(rows),
            "proxy_key": proxy_key,
            "collapse_threshold_delta_ref": COLLAPSE_DELTA_REF_THRESHOLD,
            "rows": rows,
            "winner": running_winner,
        }
        sweep_summary_path.write_text(
            json.dumps(sweep_summary, indent=2, default=str) + "\n"
        )

        if (running_winner is not None
                and running_winner["config_id"] == point.id):
            _materialize_best_model(sub_run_dir, run_dir)

    winner = _pick_winner(rows, run_type)
    if winner is None:
        msg = (
            f"all {len(rows)} sweep configs failed or collapsed; "
            f"no model selected"
        )
        print(msg, flush=True)
        write_status(run_dir, "failed", error=msg)
        return

    print(
        f"\nsweep complete. winner={winner['config_id']} "
        f"{proxy_key}={winner['primary']:.4f}",
        flush=True,
    )
    write_status(
        run_dir, "completed",
        extra={
            "winner": winner["config_id"],
            "winner_primary": winner["primary"],
            "n_configs": len(rows),
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
