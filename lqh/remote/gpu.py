"""GPU detection and status querying with vendor auto-discovery.

Supports NVIDIA (nvidia-smi) today, with the abstraction ready for
AMD (amd-smi / rocm-smi) in the future.  The probe auto-detects which
vendor tools are available on the remote host.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from lqh.remote.ssh_helpers import ssh_run

logger = logging.getLogger(__name__)

__all__ = [
    "GpuInfo",
    "GpuStatus",
    "detect_gpu_vendor",
    "query_gpu_info",
    "query_gpu_status",
]


@dataclass
class GpuInfo:
    """Static GPU information (doesn't change between calls)."""

    index: int
    name: str
    memory_total_mib: int
    vendor: str  # "nvidia" | "amd"


@dataclass
class GpuStatus:
    """Live GPU utilization snapshot."""

    index: int
    name: str
    vendor: str  # "nvidia" | "amd"
    gpu_utilization_pct: int  # 0-100
    memory_used_mib: int
    memory_total_mib: int
    temperature_c: int | None = None

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib

    @property
    def memory_utilization_pct(self) -> int:
        if self.memory_total_mib == 0:
            return 0
        return round(100 * self.memory_used_mib / self.memory_total_mib)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "memory_free_mib": self.memory_free_mib,
        }
        if self.temperature_c is not None:
            d["temperature_c"] = self.temperature_c
        return d


async def detect_gpu_vendor(hostname: str) -> str | None:
    """Auto-detect which GPU vendor tools are available on the remote.

    Returns ``"nvidia"``, ``"amd"``, or ``None`` if no GPU tools found.
    """
    # Check NVIDIA first (most common for training)
    _, _, rc = await ssh_run(hostname, "command -v nvidia-smi", timeout=10.0)
    if rc == 0:
        return "nvidia"

    # Check AMD ROCm (amd-smi is the newer tool, rocm-smi is legacy)
    _, _, rc = await ssh_run(hostname, "command -v amd-smi || command -v rocm-smi", timeout=10.0)
    if rc == 0:
        return "amd"

    return None


async def query_gpu_info(hostname: str) -> list[GpuInfo]:
    """Query static GPU information (names, memory).

    Auto-detects vendor.  Returns empty list if no GPUs found.
    """
    vendor = await detect_gpu_vendor(hostname)
    if vendor is None:
        return []

    if vendor == "nvidia":
        return await _nvidia_gpu_info(hostname)
    elif vendor == "amd":
        return await _amd_gpu_info(hostname)

    return []


async def query_gpu_status(hostname: str) -> list[GpuStatus]:
    """Query live GPU utilization (utilization %, memory, temperature).

    Auto-detects vendor.  Returns empty list if no GPUs found.
    """
    vendor = await detect_gpu_vendor(hostname)
    if vendor is None:
        return []

    if vendor == "nvidia":
        return await _nvidia_gpu_status(hostname)
    elif vendor == "amd":
        return await _amd_gpu_status(hostname)

    return []


# ---------------------------------------------------------------------------
# NVIDIA implementation
# ---------------------------------------------------------------------------


async def _nvidia_gpu_info(hostname: str) -> list[GpuInfo]:
    cmd = (
        "nvidia-smi --query-gpu=index,name,memory.total "
        "--format=csv,noheader,nounits"
    )
    stdout, _, rc = await ssh_run(hostname, cmd, timeout=15.0)
    if rc != 0 or not stdout:
        return []

    gpus: list[GpuInfo] = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mib=int(parts[2]),
                vendor="nvidia",
            ))
    return gpus


async def _nvidia_gpu_status(hostname: str) -> list[GpuStatus]:
    cmd = (
        "nvidia-smi "
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    stdout, _, rc = await ssh_run(hostname, cmd, timeout=15.0)
    if rc != 0 or not stdout:
        return []

    gpus: list[GpuStatus] = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            temp: int | None = None
            try:
                temp = int(parts[5])
            except (ValueError, IndexError):
                pass
            gpus.append(GpuStatus(
                index=int(parts[0]),
                name=parts[1],
                vendor="nvidia",
                gpu_utilization_pct=int(parts[2]),
                memory_used_mib=int(parts[3]),
                memory_total_mib=int(parts[4]),
                temperature_c=temp,
            ))
    return gpus


# ---------------------------------------------------------------------------
# AMD implementation (placeholder — ready for ROCm integration)
# ---------------------------------------------------------------------------


async def _amd_gpu_info(hostname: str) -> list[GpuInfo]:
    # TODO: implement when AMD ROCm support is needed.
    # amd-smi static --gpu --json  or  rocm-smi --showid --showmeminfo vram --json
    logger.warning("AMD GPU info query not yet implemented")
    return []


async def _amd_gpu_status(hostname: str) -> list[GpuStatus]:
    # TODO: implement when AMD ROCm support is needed.
    # amd-smi monitor --gpu-use --vram-use --temp --json
    logger.warning("AMD GPU status query not yet implemented")
    return []
