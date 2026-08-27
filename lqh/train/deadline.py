"""Stop training before the sandbox's wall-clock cap.

A cloud job runs inside a sandbox with a hard lifetime cap: the
wall-clock limit the job was submitted with (``timeout_minutes``, which
SFT and DPO let the client choose). When the provider kills the sandbox
AT that cap, the launcher never reaches its publish step: checkpoints
already written to the volume are never uploaded, no logs come back, and
the reserved GPU time bought nothing. The user's only evidence is a job
that failed on timeout.

The backend passes the cap as an absolute unix time in
``LQH_DEADLINE_EPOCH``. This callback ends training a reserve before it,
so HF Trainer performs its normal final save, the trainer writes its
usual artifacts, and the launcher publishes as it would after any
ordinary run. A capped run then yields a usable checkpoint instead of
nothing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # transformers only exists in the training image
    # HF's CallbackHandler duck-types every callback (it getattr's the
    # event name), so the base class is documentation, not a contract.
    # The stand-in keeps this module — and its unit test — importable in
    # the CLI environment, which has no torch stack.
    class TrainerCallback:  # type: ignore[no-redef]
        pass


# Covers HF's final save plus the tar + upload the launcher does after
# the trainer returns. Generous because publish moves multi-GB artifacts
# and has been seen to go minutes without output.
#
# ponytail: one constant, not a config knob. Make it per-kind (or read it
# from the config) only when a real run overruns it — stopped_early.json
# records the step and the deadline, which is the evidence needed.
PUBLISH_RESERVE_SECONDS = 20 * 60


def deadline_epoch() -> float | None:
    """Absolute unix time the sandbox dies at, or None when unset."""
    raw = os.environ.get("LQH_DEADLINE_EPOCH", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class DeadlineStopCallback(TrainerCallback):
    """Ends training in time for the run to save and publish."""

    def __init__(
        self,
        run_dir: Path,
        *,
        label: str,
        reserve_seconds: float = PUBLISH_RESERVE_SECONDS,
        deadline: float | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.label = label
        self.reserve_seconds = max(0.0, float(reserve_seconds))
        if deadline is None:
            deadline = deadline_epoch()
        self.deadline = deadline
        self.stop_at = None if deadline is None else deadline - self.reserve_seconds
        self.triggered = False
        self.stopped_at_step: int | None = None

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        if self.triggered or self.stop_at is None:
            return
        if time.time() < self.stop_at:
            return
        self.triggered = True
        self.stopped_at_step = int(getattr(state, "global_step", 0) or 0)
        # should_save as well as should_training_stop: the loop may exit
        # between HF's own save points, and the whole purpose here is to
        # leave a checkpoint behind.
        control.should_training_stop = True
        control.should_save = True
        max_steps = getattr(state, "max_steps", 0)
        print(
            f"{self.label}: wall-clock deadline reached at step "
            f"{self.stopped_at_step}"
            + (f"/{max_steps}" if isinstance(max_steps, int) and max_steps > 0 else "")
            + f"; stopping now so the run can save and publish "
            f"(reserve {int(self.reserve_seconds / 60)} min). "
            f"Resubmit with a larger timeout_minutes to train longer.",
            flush=True,
        )
        self._write_marker(state)

    def _write_marker(self, state: Any) -> None:
        """Record the early stop where publish and a reader can see it."""
        payload = {
            "reason": "wall_clock_deadline",
            "label": self.label,
            "stopped_at_step": self.stopped_at_step,
            "max_steps": (
                int(state.max_steps)
                if isinstance(getattr(state, "max_steps", None), int)
                else None
            ),
            "epoch": (
                float(state.epoch)
                if isinstance(getattr(state, "epoch", None), (int, float))
                else None
            ),
            "deadline_epoch": self.deadline,
            "reserve_seconds": self.reserve_seconds,
        }
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "stopped_early.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
        except OSError as exc:
            # A run must never die because the marker could not be written.
            print(f"{self.label}: could not write stopped_early.json: {exc}")
