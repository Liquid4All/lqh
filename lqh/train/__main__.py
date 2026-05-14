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
        from lqh.train.progress import write_status

        write_status(run_dir, "interrupted", error="Timeout waiting for preferences")
    except Exception as exc:
        # Write failure to progress so the watcher can detect it.
        from lqh.train.progress import write_status

        write_status(run_dir, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
