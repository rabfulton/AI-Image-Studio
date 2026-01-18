import asyncio

import pytest

from ai_image_studio.core.execution import ExecutionContext


def test_execution_context_external_cancelled_trips_check():
    cancelled = False

    def external_cancelled() -> bool:
        return cancelled

    ctx = ExecutionContext(job_id=__import__("uuid").uuid4(), external_cancelled=external_cancelled)

    ctx.check_cancelled()  # ok

    cancelled = True
    with pytest.raises(asyncio.CancelledError):
        ctx.check_cancelled()

