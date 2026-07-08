from __future__ import annotations

from pathlib import Path

from lqh.tools import handlers


async def test_get_deployment_handles_nullable_usage_fields(
    tmp_path: Path,
    monkeypatch,
):
    async def fake_backend_json(method, path, *, json_body=None, params=None):
        if path == "/v1/deployments/dep-1":
            return 200, {
                "id": "dep-1",
                "name": "lfm-smoke-v3",
                "status": "running",
                "desired_status": "running",
                "tier": "prod",
                "base_model": None,
                "gpu_type": "L40S",
                "gpu_count": 1,
                "min_containers": 0,
                "max_containers": 1,
                "replicas": None,
                "billed_per_hour_estimate": 7_800_000,
                "billed_cost_micros": 0,
                "gpu_seconds": None,
                "created_at": "2026-06-12T00:00:00Z",
            }
        if path == "/v1/deployments/dep-1/usage":
            return 200, {
                "totals": {
                    "requests": None,
                    "errors": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "avg_ttft_ms": None,
                    "avg_duration": None,
                },
                "billed_gpu_cost_micros": None,
                "gpu_seconds": None,
            }
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(handlers, "_backend_json", fake_backend_json)

    result = await handlers.handle_get_deployment(tmp_path, deployment_id="dep-1")

    assert "Deployment lfm-smoke-v3 (dep-1):" in result.content
    assert "0 GPU-seconds" in result.content
    assert "avg TTFT n/a ms, avg duration n/a s" in result.content
