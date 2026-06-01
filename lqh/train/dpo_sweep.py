"""DPO hyperparameter sweep — thin alias over ``lqh.train.sweep``.

The sweep orchestrator in ``lqh.train.sweep`` is already polymorphic:
when the base config carries ``type=on_policy_dpo`` (or ``type=dpo``)
it picks the DPO grid, the DPO proxy (eval_ce_chosen_mean), and the
DPO collapse detector. This module exists for two reasons:

1. The cloud backend's module whitelist
   (``backend/internal/handler/cloud_jobs.go``) maps ``train_dpo_sweep``
   to ``lqh.train.dpo_sweep`` so the kind→entrypoint dispatch is
   uniform with ``train_sft_sweep → lqh.train.sweep``.

2. We can defensively normalise the inbound sweep config: if a
   client submits without ``base_config.type`` set, we force
   ``on_policy_dpo`` so the sweep doesn't fall through to the SFT
   grid. (SFT-shaped configs sent under this entrypoint are a
   client bug and are rejected explicitly.)

No business logic lives here — once the config is normalised, we
hand off to ``sweep_loop`` unchanged. New DPO-sweep behavior should
go in ``sweep.py`` so it's available to both entrypoints.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from lqh.train.progress import write_status
from lqh.train.sweep import sweep_loop


_DPO_TYPES = {"dpo", "on_policy_dpo"}


def _normalise(cfg: dict) -> dict:
    """Ensure base_config.type is a DPO type. Mutates and returns
    ``cfg``; raises ValueError on a non-DPO base."""
    base = cfg.get("base_config")
    if not isinstance(base, dict):
        raise ValueError("sweep_config.base_config is required (dict)")
    t = base.get("type")
    if t is None:
        base["type"] = "on_policy_dpo"
    elif t not in _DPO_TYPES:
        raise ValueError(
            f"lqh.train.dpo_sweep requires base_config.type in {_DPO_TYPES}, "
            f"got {t!r}"
        )
    return cfg


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lqh.train.dpo_sweep <sweep_config.json>",
              file=sys.stderr)
        sys.exit(1)
    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.exists():
        print(f"Sweep config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = json.loads(cfg_path.read_text())
    run_dir = cfg_path.parent
    (run_dir / "pid").write_text(str(os.getpid()))
    try:
        cfg = _normalise(cfg)
        sweep_loop(run_dir, cfg)
    except Exception as exc:
        write_status(run_dir, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
