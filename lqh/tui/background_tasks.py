"""Registry of background tasks visible in the TUI status bar.

A "background task" is any long-running activity whose completion will
later notify the agent (e.g. a remote training/eval run watched by
``RunWatcher``). The registry is purely a *display-state* abstraction —
the actual lifecycle still lives with the producer (watchers, etc.).

Producers call ``register`` / ``update`` / ``unregister``; the status
bar reads ``snapshot()`` each repaint. ``on_change`` fires after every
mutation so the UI can ``invalidate()`` immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable


@dataclass(frozen=True)
class BackgroundTask:
    task_id: str
    kind: str            # "train" | "eval" | future producers
    label: str           # short human label (e.g. run_name)
    state: str           # "running" | "pending" | ...
    remote: str | None = None


class BackgroundTaskRegistry:
    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._on_change = on_change

    def register(self, task: BackgroundTask) -> None:
        self._tasks[task.task_id] = task
        self._notify()

    def update(self, task_id: str, **fields: object) -> None:
        existing = self._tasks.get(task_id)
        if existing is None:
            return
        self._tasks[task_id] = replace(existing, **fields)  # type: ignore[arg-type]
        self._notify()

    def unregister(self, task_id: str) -> None:
        if self._tasks.pop(task_id, None) is not None:
            self._notify()

    def snapshot(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def _notify(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass
