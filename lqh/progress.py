"""Shared progress reporting for foreground, local, SSH, and cloud work.

The wire representation is deliberately plain JSON.  Background producers
append it to ``progress.jsonl`` and cloud sandboxes mirror the same object via
the existing ``LQH_EVENT_JSON`` sentinel.  Foreground producers hand the
``ProgressEvent`` directly to the TUI.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = 1
_SENTINEL_PREFIX = "LQH_EVENT_JSON:"
_CLOUD_ENV_MARKER = "LQH_JOB_ID"
PROGRESS_FILE = "progress.jsonl"
OBSERVER_PROGRESS_FILE = "observer_progress.jsonl"
RUN_ATTEMPT_ENV = "LQH_RUN_ATTEMPT_ID"

TRAINING_END = 0.90
FINAL_INFERENCE_END = 0.95
DPO_ROLLOUT_SHARE = 0.30
DPO_PREFERENCE_SHARE = 0.30
DPO_OPTIMIZER_SHARE = 0.40
DPO_HELD_OUT_SHARE = 0.10
DEFAULT_DPO_ITERATIONS = 5
DPO_JUDGING_SHARE = DPO_PREFERENCE_SHARE * 2 / 3


@dataclass(frozen=True)
class FinalScoringContext:
    progress_dir: Path
    start: float
    task_kind: str


def final_scoring_context(
    output_dir: Path,
    config: dict[str, Any],
) -> FinalScoringContext | None:
    """Return the headline scoring plan, or None for internal checkpoints."""
    is_checkpoint = output_dir.parent.name == "checkpoints"
    is_final_checkpoint = is_checkpoint and output_dir.name == "final"
    if is_checkpoint and not is_final_checkpoint:
        return None
    is_training = (
        is_final_checkpoint
        or config.get("type") not in {"infer", "eval_hf"}
    )
    progress_dir = output_dir.parent.parent if is_final_checkpoint else output_dir
    configured_type = str(config.get("type", "training"))
    default_kind = (
        "dpo" if configured_type in {"dpo", "on_policy_dpo"}
        else configured_type if is_training else "evaluation"
    )
    task_kind = str(config.get(
        "progress_task_kind",
        config.get("_progress_task_kind", default_kind),
    ))
    return FinalScoringContext(
        progress_dir=progress_dir,
        start=FINAL_INFERENCE_END if is_training else 0.50,
        task_kind=task_kind,
    )


def nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def write_error_marker(path: Path, message: str) -> None:
    try:
        path.write_text(json.dumps({"error": message}) + "\n")
    except OSError:
        pass


def has_final_scoring(config: dict[str, Any]) -> bool:
    """Whether this workflow promises a final judged result."""
    return bool(
        config.get("eval_on_checkpoints")
        and config.get("eval_dataset")
        and config.get("scorer")
    )


def has_final_inference(config: dict[str, Any]) -> bool:
    """Whether training includes a final inference/evaluation pass."""
    return bool(
        config.get("eval_on_checkpoints") and config.get("eval_dataset")
    )


def training_end_for(config: dict[str, Any]) -> float:
    if has_final_scoring(config):
        return TRAINING_END
    return FINAL_INFERENCE_END if has_final_inference(config) else 1.0


def final_result_dir(run_dir: Path, config: dict[str, Any]) -> Path | None:
    """Return the one artifact directory that gates whole-run completion."""
    run_type = config.get("type")
    effective = config.get("base_config", {}) if run_type == "sweep" else config
    if not effective.get("scorer"):
        return None
    if run_type == "sft":
        if not has_final_inference(effective):
            return None
        return run_dir / "checkpoints" / "final"
    if run_type in {"infer", "eval_hf", "sweep", "dpo", "on_policy_dpo"}:
        return run_dir
    return None


def has_pending_final_result(run_dir: Path, config: dict[str, Any]) -> bool:
    """Whether the promised headline result has been requested but not resolved."""
    target = final_result_dir(run_dir, config)
    if target is None or not (target / "eval_request.json").exists():
        return False
    return not (
        (target / "eval_result.json").exists()
        or (target / "eval_error.json").exists()
    )


def dpo_overall_fraction(
    iteration: int,
    num_iterations: int,
    within_iteration: float,
    training_end: float = 1.0,
) -> float:
    """Map a normalized DPO-iteration fraction into whole-job training."""
    count = max(1, num_iterations)
    within = min(1.0, max(0.0, _finite_number(within_iteration, 0.0)))
    end = min(1.0, max(0.0, training_end))
    return min(end, end * (max(0, iteration) + within) / count)


def dpo_optimizer_share(has_held_out_eval: bool) -> float:
    """Reserve the final tenth of an iteration for held-out evaluation."""
    return (
        DPO_OPTIMIZER_SHARE - DPO_HELD_OUT_SHARE
        if has_held_out_eval
        else DPO_OPTIMIZER_SHARE
    )


def dpo_judging_fraction(
    iteration: int,
    num_iterations: int,
    completed: int,
    total: int,
    training_end: float = 1.0,
) -> float:
    return dpo_overall_fraction(
        iteration,
        num_iterations,
        DPO_ROLLOUT_SHARE
        + DPO_JUDGING_SHARE * completed / max(total, 1),
        training_end,
    )


def dpo_preferences_ready_fraction(
    iteration: int,
    num_iterations: int,
    training_end: float = 1.0,
) -> float:
    return dpo_overall_fraction(
        iteration,
        num_iterations,
        DPO_ROLLOUT_SHARE + DPO_PREFERENCE_SHARE,
        training_end,
    )


@dataclass(frozen=True)
class ProgressEvent:
    """One user-facing progress observation.

    ``overall_fraction`` is the authoritative whole-job value.  Phase counts
    are retained because ``84/200 samples`` is more useful than a bare percent.
    An incomplete event is intentionally capped below 1.0 by ``as_payload``;
    only ``result_ready=True`` may render as 100%.
    """

    task_kind: str
    label: str
    phase: str
    phase_label: str
    completed: float = 0
    total: float | None = None
    unit: str = "items"
    overall_fraction: float = 0
    detail: str | None = None
    concurrency: int | None = None
    step: int | None = None
    loss: float | None = None
    lr: float | None = None
    epoch: float | None = None
    attempt_id: str | None = None
    result_ready: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        fraction = _finite_number(self.overall_fraction, 0.0)
        payload["overall_fraction"] = (
            1.0 if self.result_ready else min(0.999999, max(0.0, fraction))
        )
        payload["completed"] = max(0.0, _finite_number(self.completed, 0.0))
        if self.total is not None:
            payload["total"] = max(0.0, _finite_number(self.total, 0.0))
        for key in ("loss", "lr", "epoch"):
            value = payload.get(key)
            if value is not None:
                parsed = _finite_number(value, math.nan)
                payload[key] = parsed if math.isfinite(parsed) else None
        payload["timestamp"] = self.timestamp
        return {key: value for key, value in payload.items() if value is not None}


ProgressCallback = Callable[[ProgressEvent], Any]


class ProgressReporter:
    """Emit progress through a callback and/or the durable run protocol."""

    def __init__(
        self,
        *,
        task_kind: str,
        label: str,
        callback: ProgressCallback | None = None,
        run_dir: Path | None = None,
        file_name: str = PROGRESS_FILE,
        min_interval: float = 0.25,
        legacy_callback: bool = False,
        attempt_id: str | None = None,
    ) -> None:
        self.task_kind = task_kind
        self.label = label
        self.callback = callback
        self.run_dir = run_dir
        self.file_name = file_name
        self.min_interval = min_interval
        if os.environ.get(_CLOUD_ENV_MARKER):
            # Cloud sentinels become durable backend DB rows. One headline
            # update per second is responsive while bounding long-job volume.
            self.min_interval = max(self.min_interval, 1.0)
        self._legacy_callback = legacy_callback
        self._attempt_id = attempt_id or os.environ.get(RUN_ATTEMPT_ENV)
        self._last_emit = 0.0
        self._last_fraction = 0.0
        self._last_legacy_counts: tuple[int, int] | None = None

    def update(
        self,
        *,
        phase: str,
        phase_label: str,
        completed: float = 0,
        total: float | None = None,
        unit: str = "items",
        overall_fraction: float = 0,
        detail: str | None = None,
        concurrency: int | None = None,
        step: int | None = None,
        loss: float | None = None,
        lr: float | None = None,
        epoch: float | None = None,
        result_ready: bool = False,
        force: bool = False,
    ) -> ProgressEvent:
        finite_fraction = _finite_number(overall_fraction, self._last_fraction)
        fraction = max(self._last_fraction, min(1.0, finite_fraction))
        event = ProgressEvent(
            task_kind=self.task_kind,
            label=self.label,
            phase=phase,
            phase_label=phase_label,
            completed=completed,
            total=total,
            unit=unit,
            overall_fraction=fraction,
            detail=detail,
            concurrency=concurrency,
            step=step,
            loss=loss,
            lr=lr,
            epoch=epoch,
            attempt_id=self._attempt_id,
            result_ready=result_ready,
        )
        now = time.monotonic()
        terminal = result_ready or (
            total is not None and total > 0 and completed >= total
        )
        if self.callback is not None and self._legacy_callback and total is not None:
            progress_counts = (int(completed), int(total))
            if progress_counts != self._last_legacy_counts:
                try:
                    self.callback(  # type: ignore[call-arg]
                        *progress_counts, int(concurrency or 0),
                    )
                    self._last_legacy_counts = progress_counts
                except Exception:
                    pass
        if (
            force
            or terminal
            or now - self._last_emit >= self.min_interval
        ):
            payload = event.as_payload()
            event = ProgressEvent(**payload)
            if self.callback is not None:
                try:
                    if not self._legacy_callback:
                        self.callback(event)
                except Exception:
                    # Display telemetry must not abort the underlying work.
                    pass
            if self.run_dir is not None:
                write_progress_event(
                    self.run_dir, event, file_name=self.file_name,
                )
            self._last_emit = now
            self._last_fraction = payload["overall_fraction"]
        return event


def write_progress_event(
    run_dir: Path,
    event: ProgressEvent,
    *,
    file_name: str = PROGRESS_FILE,
) -> None:
    """Persist and, in a cloud sandbox, stream one common progress event."""
    payload = event.as_payload()
    if not payload.get("attempt_id"):
        attempt_id = os.environ.get(RUN_ATTEMPT_ENV)
        if attempt_id:
            payload["attempt_id"] = attempt_id
    _emit_sentinel("progress", payload)
    _append_jsonl(run_dir / file_name, payload)


def read_progress_events(run_dir: Path, last_n: int = 256) -> list[dict[str, Any]]:
    """Merge producer and observer tails for whole-job progress display.

    This is deliberately *not* a chronological JSONL merge: finite-fraction
    v1 rows sort after legacy/status rows and by fraction because clocks can
    differ across machines. Callers needing status or trainer metrics should
    use the content-specific readers in ``lqh.train.progress`` instead.
    """
    tagged: list[tuple[float, int, dict[str, Any]]] = []
    order = 0
    for name in (PROGRESS_FILE, OBSERVER_PROGRESS_FILE):
        for row in read_jsonl_tail(run_dir / name, last_n=last_n):
            ts = _timestamp_seconds(row.get("timestamp")) or 0.0
            tagged.append((ts, order, row))
            order += 1
    def order_key(item: tuple[float, int, dict[str, Any]]) -> tuple[float, float, int]:
        ts, sequence, row = item
        fraction = row.get("overall_fraction")
        if isinstance(fraction, (int, float)) and math.isfinite(float(fraction)):
            # Fractions are monotonic across machines; timestamps are not.
            return (1.0, float(fraction), sequence)
        return (0.0, ts, sequence)

    tagged.sort(key=order_key)
    return [item[2] for item in tagged[-last_n:]]


def select_display_event(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the headline row, including terminal and scoring-retry rules."""
    candidates = list(rows)
    if not candidates:
        return None
    ready = [row for row in candidates if row.get("result_ready")]
    if ready:
        return ready[-1]

    def attempt_number(row: dict[str, Any]) -> int | None:
        phase = str(row.get("phase", ""))
        if not phase.startswith("scoring_attempt_"):
            return None
        try:
            return int(phase.rsplit("_", 1)[-1])
        except ValueError:
            return None

    attempted = [
        (attempt, row)
        for row in candidates
        if (attempt := attempt_number(row)) is not None
    ]
    if attempted:
        latest_attempt = max(attempt for attempt, _ in attempted)
        candidates = [row for attempt, row in attempted if attempt == latest_attempt]
    return max(
        enumerate(candidates),
        key=lambda item: (float(item[1].get("overall_fraction", 0)), item[0]),
    )[1]


def read_jsonl_tail(path: Path, *, last_n: int = 256) -> list[dict[str, Any]]:
    """Read a bounded JSONL tail without scanning a multi-hour run file."""
    if not path.exists() or last_n <= 0:
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            block = 8192
            chunks: list[bytes] = []
            newline_count = 0
            pos = end
            while pos > 0 and newline_count <= last_n:
                take = min(block, pos)
                pos -= take
                fh.seek(pos)
                chunk = fh.read(take)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
            data = b"".join(reversed(chunks))
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for raw in data.splitlines()[-last_n:]:
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def percent_for(event: ProgressEvent | dict[str, Any]) -> int | None:
    data = event.as_payload() if isinstance(event, ProgressEvent) else event
    fraction = data.get("overall_fraction")
    if not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)):
        return None
    if data.get("result_ready") or float(fraction) >= 1.0:
        return 100
    return min(99, max(0, math.floor(float(fraction) * 100)))


def estimate_eta_seconds(
    events: Iterable[ProgressEvent | dict[str, Any]],
    *,
    now: float | None = None,
) -> float | None:
    """Return a conservative ETA from stable, recent whole-job movement.

    A phase change starts a fresh sample window.  This avoids carrying an
    optimizer rate into model inference or judge scoring.
    """
    parsed: list[tuple[float, float, str]] = []
    for event in events:
        data = event.as_payload() if isinstance(event, ProgressEvent) else event
        fraction = data.get("overall_fraction")
        ts = _timestamp_seconds(data.get("timestamp"))
        phase = data.get("phase")
        if (
            isinstance(fraction, (int, float))
            and math.isfinite(float(fraction))
            and ts is not None
            and isinstance(phase, str)
        ):
            parsed.append((ts, float(fraction), phase))
    if not parsed:
        return None

    # Keep the contiguous current phase, then distinct advances.
    current_phase = parsed[-1][2]
    phase_rows: list[tuple[float, float, str]] = []
    for row in reversed(parsed[-1024:]):
        if row[2] != current_phase:
            break
        phase_rows.append(row)
    phase_rows.reverse()
    advances: list[tuple[float, float]] = []
    for ts, fraction, _ in phase_rows:
        if not advances or fraction > advances[-1][1]:
            advances.append((ts, fraction))
    if len(advances) < 5:
        return None
    elapsed = advances[-1][0] - advances[0][0]
    if elapsed < 15.0 or advances[-1][1] <= advances[0][1]:
        return None

    rates: list[float] = []
    intervals: list[float] = []
    for (t0, f0), (t1, f1) in zip(advances, advances[1:]):
        dt = t1 - t0
        df = f1 - f0
        if dt > 0 and df > 0:
            rates.append(df / dt)
            intervals.append(dt)
    if len(rates) < 4:
        return None
    median_rate = statistics.median(rates)
    deviations = [abs(rate - median_rate) for rate in rates]
    if median_rate <= 0 or statistics.median(deviations) / median_rate > 0.35:
        return None

    wall_now = now if now is not None else time.time()
    stale_after = max(30.0, 3 * statistics.median(intervals))
    if wall_now - advances[-1][0] > stale_after:
        return None
    return max(0.0, (1.0 - advances[-1][1]) / median_rate)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def format_event_oneline(
    event: ProgressEvent | dict[str, Any],
    *,
    history: Iterable[ProgressEvent | dict[str, Any]] = (),
    observed_at: float | None = None,
) -> tuple[str, int | None]:
    data = event.as_payload() if isinstance(event, ProgressEvent) else event
    pct = percent_for(data)
    phase = str(data.get("phase_label") or data.get("phase") or "working")
    parts = [phase]
    completed = data.get("completed")
    total = data.get("total")
    unit = str(data.get("unit") or "items")
    if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
        parts.append(f"{_count(completed)}/{_count(total)} {unit}")
    if pct is not None:
        parts.append(f"{pct}%")
    eta_now = None
    if observed_at is not None:
        last_ts = _timestamp_seconds(data.get("timestamp"))
        if last_ts is not None:
            # Rates use producer timestamps; staleness uses the local time at
            # which the latest advance was observed, avoiding machine skew.
            eta_now = last_ts + max(0.0, time.time() - observed_at)
    eta = estimate_eta_seconds(history, now=eta_now)
    if eta is not None and pct not in (None, 100):
        parts.append(f"ETA {format_duration(eta)} at current rate")
    detail = data.get("detail")
    if isinstance(detail, str) and detail:
        parts.append(detail)
    concurrency = data.get("concurrency")
    if isinstance(concurrency, int) and concurrency > 1:
        parts.append(f"up to {concurrency} concurrent")
    return (" · ".join(parts), pct)


def _count(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _finite_number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, default=_json_default) + "\n").encode()
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
    except OSError:
        # Telemetry must never take down training or scoring.
        return


def _emit_sentinel(kind: str, payload: dict[str, Any]) -> None:
    if not os.environ.get(_CLOUD_ENV_MARKER):
        return
    try:
        print(
            _SENTINEL_PREFIX + " " + json.dumps(
                {"kind": kind, "payload": payload}, default=_json_default,
            ),
            flush=True,
        )
    except Exception:
        pass


def relay_cloud_sentinel(line: str) -> bool:
    """Relay a child sandbox sentinel through its parent process stdout."""
    if not line.startswith(_SENTINEL_PREFIX):
        return False
    if not os.environ.get(_CLOUD_ENV_MARKER):
        return False
    try:
        print(line.rstrip("\n"), flush=True)
    except Exception:
        pass
    return True


def _json_default(obj: Any) -> Any:
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(obj)


# Shared low-level protocol hooks used by the legacy training facade.
append_jsonl = _append_jsonl
emit_sentinel = _emit_sentinel
json_default = _json_default
