"""Best-effort local environment snapshot for feedback submissions.

Collected once per /feedback and attached as the `metadata` field
(FEEDBACK_2.md) so bug reports carry the OS/CPU/RAM/GPU/Python context they
were generated in. Every probe is individually fault-isolated: a failure
yields a missing/None field, never an exception — metadata must never block
a feedback submission. The whole collection is designed to finish in well
under a second (notably: torch is only consulted if it is already imported;
we never pay its multi-second import just for a version string).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata as _importlib_metadata

# Packages worth knowing the version of when debugging a report. Missing
# ones are simply omitted.
_PACKAGES_OF_INTEREST = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "httpx",
    "textual",
    "openai",
)


def _cpu_model() -> str | None:
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            return out.stdout.strip() or None
    except Exception:
        pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return round(total / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def _gpu_info() -> dict | None:
    # Prefer torch — but only if the process already imported it (the TUI
    # status bar does when torch is installed). Importing torch here would
    # add seconds to /feedback for no benefit.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                return {
                    "source": "torch",
                    "cuda": getattr(torch.version, "cuda", None),
                    "devices": [
                        {
                            "name": torch.cuda.get_device_name(i),
                            "vram_gb": round(
                                torch.cuda.get_device_properties(i).total_memory
                                / (1024**3), 1,
                            ),
                        }
                        for i in range(n)
                    ],
                }
            return {"source": "torch", "devices": []}
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            devices = []
            driver = None
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    devices.append({"name": parts[0], "vram": parts[1]})
                    driver = parts[2]
            return {"source": "nvidia-smi", "driver": driver, "devices": devices}
    except Exception:
        pass
    return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _PACKAGES_OF_INTEREST:
        try:
            versions[pkg] = _importlib_metadata.version(pkg)
        except Exception:
            continue
    return versions


def collect_environment() -> dict:
    """Return a JSON-safe snapshot of the local environment. Never raises."""
    info: dict = {}
    try:
        info["os"] = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        }
    except Exception:
        pass
    try:
        info["python"] = {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        }
    except Exception:
        pass
    try:
        info["cpu"] = {"count": os.cpu_count(), "model": _cpu_model()}
    except Exception:
        pass
    info["ram_gb"] = _ram_gb()
    info["gpu"] = _gpu_info()
    info["packages"] = _package_versions()
    return info
