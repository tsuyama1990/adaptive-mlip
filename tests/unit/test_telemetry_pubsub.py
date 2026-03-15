from typing import Any
import asyncio

import pytest

from pyacemaker.domain_models.telemetry import SimulationState, StateChangePayload, TelemetryFrame
from pyacemaker.logger import TelemetryBroker


@pytest.fixture
def clean_broker() -> TelemetryBroker:
    broker = TelemetryBroker()
    # Reset queue for each test
    broker.queue = asyncio.Queue(maxsize=10)
    broker._initialized = True
    broker.loop = None
    return broker


@pytest.mark.asyncio
async def test_telemetry_broker_drop_oldest(clean_broker: TelemetryBroker) -> None:
    # We must explicitly set the loop for the sync publisher to work
    loop = asyncio.get_running_loop()
    clean_broker.initialize_loop(loop)

    # Fill the queue up to maxsize (10)
    for i in range(10):
        frame = TelemetryFrame(
            workflow_id="test",
            step_number=i,
            current_state=SimulationState.RUNNING_MD,
            positions=[],
            forces=None,
            variances=None,
        )
        clean_broker.publish(frame)

    # Allow the synchronous thread-safe calls to process
    await asyncio.sleep(0.01)

    # Assert queue is full
    assert clean_broker.queue.full()

    # Push 11th item
    frame_11 = TelemetryFrame(
        workflow_id="test",
        step_number=10,
        current_state=SimulationState.RUNNING_MD,
        positions=[],
        forces=None,
        variances=None,
    )
    clean_broker.publish(frame_11)

    # Allow process
    await asyncio.sleep(0.01)

    # The oldest item (step_number=0) should be dropped. First item should now be step_number=1
    first_item = await clean_broker.queue.get()
    assert getattr(first_item, "step_number", None) == 1


@pytest.mark.asyncio
async def test_telemetry_broker_state_change(clean_broker: TelemetryBroker) -> None:
    loop = asyncio.get_running_loop()
    clean_broker.initialize_loop(loop)

    payload = StateChangePayload(workflow_id="test", new_state=SimulationState.RUNNING_DFT)
    clean_broker.publish(payload)

    await asyncio.sleep(0.01)

    retrieved = await clean_broker.queue.get()
    assert isinstance(retrieved, StateChangePayload)
    assert retrieved.new_state == SimulationState.RUNNING_DFT
