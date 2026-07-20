"""Back-compat re-export — the registry moved to ``lqh.background_tasks``
so the headless JobSupervisor doesn't import from the TUI package."""

from lqh.background_tasks import BackgroundTask, BackgroundTaskRegistry

__all__ = ["BackgroundTask", "BackgroundTaskRegistry"]
