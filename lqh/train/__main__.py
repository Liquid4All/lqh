"""Entry point for ``python -m lqh.train <config.json>``.

Reads the run config and dispatches to the appropriate training loop.
All torch/transformers imports happen inside the dispatched functions,
keeping import-time lightweight so error messages are immediate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.train <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    run_dir = config_path.parent

    # Write PID file so the main process can track us.
    (run_dir / "pid").write_text(str(__import__("os").getpid()))
    from lqh.train.progress import begin_run_attempt, write_status
    begin_run_attempt(run_dir)

    run_type = config.get("type", "sft")

    try:
        if run_type == "sft":
            from lqh.train.sft import sft_loop

            sft_loop(run_dir, config)
        elif run_type in ("on_policy_dpo", "dpo"):
            from lqh.train.dpo import dpo_loop

            dpo_loop(run_dir, config)
        else:
            print(f"Unknown training type: {run_type!r}", file=sys.stderr)
            sys.exit(1)
    except TimeoutError:
        # DPO timeout waiting for preferences — handled inside dpo_loop
        # which writes "interrupted" status. This is a safety net.
        write_status(run_dir, "interrupted", error="Timeout waiting for preferences")
    except Exception as exc:
        # Write failure to progress so the watcher can detect it. A CUDA
        # OOM is flagged explicitly (oom=True) so the backend classifies
        # the lease as `oom` rather than `preempted` and batch auto-tuning
        # can self-heal — see lqh.train.progress.write_status.
        is_oom = _looks_like_oom(exc)
        if is_oom:
            # Self-heal: write back a smaller batch profile so the next
            # run uses it (GPU_TYPE.md §6). Best-effort, never raises.
            from lqh.train.calibrate import report_oom_downgrade

            report_oom_downgrade(config)
        write_status(run_dir, "failed", error=str(exc), oom=is_oom)
        raise


def _looks_like_oom(exc: BaseException) -> bool:
    """Best-effort CUDA out-of-memory detection without importing torch.

    torch.cuda.OutOfMemoryError subclasses RuntimeError; we match on the
    type name and message so this stays a zero-dependency check at the
    dispatch layer (torch is imported only inside the training loops).
    """
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "outofmemory" in msg


if __name__ == "__main__":
    main()
