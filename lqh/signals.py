"""Startup attention signals — the "unknown unknowns" briefing.

An agent with read tools can pull anything it suspects exists; what it
cannot do is suspect the right things after days away. This module
computes a short list of facts the agent would not know to look for
(see PERSISTENCY_PLAN.md, "push signals, not dossiers"):

* jobs still running, and jobs that reached a terminal state since the
  last session (diffed against ``.lqh/job_seen.json``);
* cloud submissions with unknown fate (an idempotency marker without an
  accepted job — billing-relevant);
* SPEC.md drift relative to the last cloud-submitted spec hash;
* cloud snapshot staleness/offline state.

Everything is computed from the filesystem and the already-fetched
snapshot — no network access here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lqh.fsio import atomic_write_json, file_lock

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_SEEN_SCHEMA_VERSION = 1


@dataclass
class Signal:
    kind: str
    text: str


def _seen_path(project_dir: Path) -> Path:
    return project_dir / ".lqh" / "job_seen.json"


def load_seen_states(project_dir: Path) -> dict[str, str]:
    """Return the last recorded per-run states ``{run_name: state}``."""
    try:
        data = json.loads(_seen_path(project_dir).read_text(encoding="utf-8"))
        runs = data.get("runs", {})
        return {
            name: str(info.get("state", ""))
            for name, info in runs.items()
            if isinstance(info, dict)
        }
    except (OSError, ValueError, AttributeError):
        return {}


def record_seen_states(project_dir: Path, states: dict[str, str]) -> None:
    """Merge ``states`` into ``.lqh/job_seen.json`` (best-effort, atomic).

    The read–merge–write runs under a cross-process lock so two CLIs
    observing different runs cannot drop each other's updates.
    """
    if not states:
        return
    try:
        with file_lock(project_dir / ".lqh" / "job_seen.lock"):
            current = load_seen_states(project_dir)
            current.update(states)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            atomic_write_json(
                _seen_path(project_dir),
                {
                    "schema_version": _SEEN_SCHEMA_VERSION,
                    "runs": {
                        name: {"state": state, "at": now}
                        for name, state in current.items()
                    },
                },
            )
    except OSError:
        logger.warning("could not record seen job states", exc_info=True)


def progress_terminal_state(run_dir: Path) -> str | None:
    """Terminal state recorded in progress.jsonl, or None.

    Reads only the terminal ``status`` rows the trainers/eval runners
    append (and the cloud event replayer mirrors). Used for remote runs
    where PID-liveness is meaningless locally.
    """
    path = run_dir / "progress.jsonl"
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row.get("status")
        if status in ("completed", "failed", "cancelled", "interrupted"):
            return "failed" if status == "interrupted" else status
    return None


def observe_run_states(project_dir: Path) -> dict[str, str]:
    """Best-effort current state per run, from run-directory files only.

    Mirrors the sources the TUI job scanner uses: ``cloud_state.json`` /
    synced ``progress.jsonl`` for remote (cloud AND ssh) runs, PID +
    progress.jsonl for local ones. An orphaned submit intent is reported
    as the pseudo-state ``"submit_fate_unknown"``.

    NOTE: for remote runs this reads whatever the last sync left on
    disk — callers that need "finished while LQH was closed" must
    refresh remote state first (the TUI runs one scan before building
    signals).
    """
    from lqh.subprocess_manager import SubprocessManager

    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return {}
    manager = SubprocessManager()
    states: dict[str, str] = {}
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        has_config = (run_dir / "config.json").exists()
        has_remote = (run_dir / "remote_job.json").exists()
        has_intent = (run_dir / "submit_intent.json").exists()
        # A response-loss orphan is exactly a directory with ONLY
        # submit_intent.json — cloud submission writes the intent before
        # the POST and config.json only after acceptance. Requiring
        # config.json here would hide the very state this exists to find.
        if not (has_config or has_remote or has_intent):
            continue
        # A marker owned by a DIFFERENT project identity (run dir copied
        # in by hand) must not be observed as this project's job.
        try:
            from lqh.project_identity import marker_is_foreign

            foreign = False
            for marker_name in ("remote_job.json", "submit_intent.json"):
                marker_file = run_dir / marker_name
                if not marker_file.exists():
                    continue
                marker = json.loads(marker_file.read_text(encoding="utf-8"))
                if marker_is_foreign(project_dir, marker):
                    logger.warning(
                        "runs/%s: %s belongs to another project identity; skipping",
                        name, marker_name,
                    )
                    foreign = True
                    break
            if foreign:
                continue
        except (OSError, ValueError):
            pass
        try:
            if has_remote:
                status = None
                try:
                    cloud_state = json.loads(
                        (run_dir / "cloud_state.json").read_text(encoding="utf-8")
                    )
                    status = cloud_state.get("status")
                except (OSError, ValueError):
                    pass
                if status in _TERMINAL_STATES:
                    states[name] = status
                else:
                    # SSH runs have no cloud_state.json; cloud runs may
                    # have a stale one. The synced progress log is the
                    # next-best terminal evidence for both.
                    states[name] = progress_terminal_state(run_dir) or "running"
            elif has_intent:
                states[name] = "submit_fate_unknown"
            else:
                states[name] = manager.get_status(run_dir).state
        except Exception:
            states[name] = "unknown"
    return states


def finished_while_away_signals(
    project_dir: Path, run_states: dict[str, str]
) -> list[Signal]:
    """One-shot diff signals: jobs that went terminal since last recorded.

    Terminal now, but last seen non-terminal → finished while LQH was
    closed. Runs never seen before stay silent (first startup after the
    feature ships must not announce ancient completed runs).

    These are DIFF signals: they consume the ``job_seen.json`` baseline,
    so the caller computes them once per CLI open (recording the new
    baseline via ``record_seen_states``) and re-injects the same list for
    /clear and /resume — recomputing after recording would silently drop
    them.
    """
    seen = load_seen_states(project_dir)
    signals: list[Signal] = []
    finished_away = sorted(
        n for n, s in run_states.items()
        if s in _TERMINAL_STATES and seen.get(n) not in (None, s)
        and seen.get(n) not in _TERMINAL_STATES
    )
    for name in finished_away[:5]:
        signals.append(Signal(
            "finished_while_away",
            f"job finished while LQH was closed: runs/{name} → {run_states[name]}",
        ))
    if len(finished_away) > 5:
        signals.append(Signal(
            "finished_while_away",
            f"…and {len(finished_away) - 5} more jobs finished while LQH was closed",
        ))
    return signals


def collect_signals(
    project_dir: Path,
    *,
    snapshot: dict | None,
    snapshot_fresh: bool,
    run_states: dict[str, str] | None = None,
    jobs_refreshed: bool = True,
) -> list[Signal]:
    """Compute the stateless attention signals for a session open.

    Stateless: derived from what is on disk right now, safe to recompute
    on every open. The one-shot finished-while-away diff lives in
    ``finished_while_away_signals`` (see its docstring for why).
    """
    signals: list[Signal] = []
    states = observe_run_states(project_dir) if run_states is None else run_states

    if not jobs_refreshed:
        signals.append(Signal(
            "refresh_failed",
            "the startup job-state refresh failed or timed out — run states "
            "below come from possibly-stale local files; verify with "
            "training_status/remote_status before trusting them",
        ))

    running = sorted(n for n, s in states.items() if s == "running")
    if running:
        shown = ", ".join(f"runs/{n}" for n in running[:5])
        more = f" (+{len(running) - 5} more)" if len(running) > 5 else ""
        signals.append(Signal(
            "running",
            f"{len(running)} job(s) still running: {shown}{more}",
        ))

    unknown = sorted(n for n, s in states.items() if s == "unknown")
    if unknown:
        shown = ", ".join(f"runs/{n}" for n in unknown[:5])
        more = f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else ""
        signals.append(Signal(
            "state_unknown",
            f"{len(unknown)} job(s) whose state could not be determined: "
            f"{shown}{more} — inspect them before assuming anything",
        ))

    orphans = sorted(n for n, s in states.items() if s == "submit_fate_unknown")
    for name in orphans:
        signals.append(Signal(
            "submit_fate_unknown",
            f"cloud submission with unknown fate: runs/{name}/submit_intent.json "
            "(billing-relevant — check remote_status/artifacts before resubmitting)",
        ))

    # Spec drift vs the last cloud-submitted spec hash.
    from lqh.project_meta import compute_spec_sha256

    local_hash = compute_spec_sha256(project_dir)
    snap_payload = (snapshot or {}).get("snapshot") or {}
    cloud_spec_hash = snap_payload.get("spec_sha256")
    if cloud_spec_hash and local_hash and local_hash != cloud_spec_hash:
        signals.append(Signal(
            "spec_drift",
            "SPEC.md has changed since the last cloud submission "
            "(hash mismatch) — cloud artifacts may be based on the older spec",
        ))

    # Spec drift vs local artifact manifests (written at finalization by
    # Phase 4 tooling; read here whenever present).
    if local_hash:
        drifted = _manifest_spec_drift(project_dir, local_hash)
        if drifted:
            shown = ", ".join(drifted[:3])
            more = f" (+{len(drifted) - 3} more)" if len(drifted) > 3 else ""
            signals.append(Signal(
                "artifact_spec_drift",
                f"SPEC.md has changed since these artifacts were produced: "
                f"{shown}{more} — reuse, supplement, or regenerate deliberately",
            ))

    # A "fresh" snapshot can still carry stale sections (a partial
    # enrichment failure) — that is an unknown-unknown the agent must be
    # told about, since the summary is pull-only.
    stale_sections = (snapshot or {}).get("stale_sections") or []
    if snapshot_fresh and stale_sections:
        signals.append(Signal(
            "snapshot_partial",
            f"cloud snapshot refresh was partial — {', '.join(stale_sections)} "
            "carried from an older snapshot (or unavailable); verify with "
            "the artifacts/list_deployments tools",
        ))

    if not snapshot_fresh:
        if snapshot is not None:
            fetched = snapshot.get("fetched_at") or "an unknown time"
            signals.append(Signal(
                "snapshot_stale",
                f"cloud snapshot is stale (last fetched {fetched}; offline?) — "
                "cloud job/deployment facts may be outdated",
            ))
        else:
            signals.append(Signal(
                "snapshot_unavailable",
                "cloud state is unavailable (offline, not logged in, or auth "
                "failure) and no cached snapshot exists — cloud jobs, "
                "artifacts, and deployments are invisible right now",
            ))

    return signals


# Everywhere finalization manifests can appear (datasets, eval runs,
# training runs). Extend as Phase 4 adds producers.
_MANIFEST_GLOBS = (
    ("datasets", "datasets/*/manifest.json"),
    ("evals", "evals/runs/*/manifest.json"),
    ("runs", "runs/*/manifest.json"),
)


def _manifest_spec_drift(project_dir: Path, local_hash: str) -> list[str]:
    """Artifact names whose manifest records a different spec hash."""
    drifted: list[str] = []
    for prefix, pattern in _MANIFEST_GLOBS:
        for manifest_path in sorted(project_dir.glob(pattern)):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            recorded = manifest.get("spec_sha256") or manifest.get("spec_hash")
            if recorded and recorded != local_hash:
                drifted.append(f"{prefix}/{manifest_path.parent.name}")
    return drifted


def format_signal_block(signals: list[Signal]) -> str | None:
    """Render signals as the injectable context block, or None if empty."""
    if not signals:
        return None
    lines = ["⚡ Attention signals (changed while you were away):"]
    lines.extend(f"- {s.text}" for s in signals)
    lines.append(
        "Investigate with your tools (summary, training_status, "
        "remote_status, artifacts) before acting on these."
    )
    return "\n".join(lines)
