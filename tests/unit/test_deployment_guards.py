from pathlib import Path

import pytest

from lqh.tools.handlers import handle_push_to_production


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tier": "prod"}, "Production deployments are temporarily unavailable"),
        ({"min_containers": 1}, "min_containers must be 0"),
    ],
)
async def test_push_rejects_non_development_scaling(kwargs, message) -> None:
    result = await handle_push_to_production(
        Path("."), artifact_id="artifact", name="model", **kwargs
    )

    assert result.ok is False
    assert result.error_kind == "validation"
    assert message in result.content
