"""Remote fine-tuning backends.

Provides ``RemoteBackend`` ABC and concrete implementations for running
training/inference on remote machines while keeping scoring and orchestration
local.
"""

from lqh.remote.backend import (
    JobStatus,
    ProjectBinding,
    RemoteBackend,
    RemoteConfig,
    RemoteMachine,
)
from lqh.remote.config import (
    add_binding,
    add_machine,
    add_remote,
    get_binding,
    get_machine,
    get_remote,
    load_bindings,
    load_machines,
    load_remotes,
    remove_binding,
    remove_machine,
    remove_remote,
    save_bindings,
    save_machines,
    save_remotes,
)

__all__ = [
    "JobStatus",
    "ProjectBinding",
    "RemoteBackend",
    "RemoteConfig",
    "RemoteMachine",
    "add_binding",
    "add_machine",
    "add_remote",
    "get_binding",
    "get_machine",
    "get_remote",
    "load_bindings",
    "load_machines",
    "load_remotes",
    "remove_binding",
    "remove_machine",
    "remove_remote",
    "save_bindings",
    "save_machines",
    "save_remotes",
]
