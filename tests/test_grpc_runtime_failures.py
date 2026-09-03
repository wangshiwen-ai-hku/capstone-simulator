"""Network runtime failures must be bounded and cannot reuse stale results."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import grpc
import pytest

from agent.service import MockAgentService, load_agent_configs
from interfaces.proto.mars.v1 import runtime_pb2, runtime_service_pb2_grpc
from mars.runtime import GrpcRuntimeAdapter
from tests.test_runtime_port import _command


@asynccontextmanager
async def _server(service_type=MockAgentService, *, delay_ms=1.0):
    config = replace(
        load_agent_configs("configs/mars/agents.local.json")[0],
        execution_delay_ms=delay_ms,
    )
    service = service_type(config)
    server = grpc.aio.server()
    runtime_service_pb2_grpc.add_AgentRuntimeServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield server, service, f"127.0.0.1:{port}"
    finally:
        await server.stop(0)
        active = tuple(service._active.values())
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


class _SlowAckAgent(MockAgentService):
    async def DispatchTask(self, request, context):
        response = await super().DispatchTask(request, context)
        await asyncio.sleep(0.05)
        return response


class _DuplicateCompletionAgent(MockAgentService):
    def __init__(self, config):
        super().__init__(config)
        self._history.append(
            runtime_pb2.AttemptCompletion(
                dispatch_id="old-dispatch",
                attempt_id="old-attempt",
                task_id="old-task",
                agent_id=config.agent_id,
                ok=True,
            )
        )

    async def _complete(self, dispatch_id, request):
        await super()._complete(dispatch_id, request)
        for queue in tuple(self._subscribers):
            queue.put_nowait(self._history[-1])


class _InvalidCompletionAgent(MockAgentService):
    async def _complete(self, dispatch_id, request):
        await asyncio.sleep(0.01)
        message = runtime_pb2.AttemptCompletion(
            dispatch_id=dispatch_id,
            attempt_id=request.attempt_id,
            task_id="wrong-task",
            agent_id=self.config.agent_id,
            ok=True,
        )
        self._history.append(message)
        for queue in tuple(self._subscribers):
            queue.put_nowait(message)
        self._active.pop(request.attempt_id, None)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("parameter", ["timeout_seconds", "completion_timeout_seconds"])
def test_timeouts_are_finite_and_positive(parameter, value):
    with pytest.raises(ValueError, match="finite and positive"):
        GrpcRuntimeAdapter({"robot_1": "127.0.0.1:50051"}, **{parameter: value})


def test_completion_timeout_cancels_work_and_does_not_hang():
    async def run():
        async with _server(delay_ms=10_000) as (_, service, endpoint):
            runtime = GrpcRuntimeAdapter(
                {"robot_1": endpoint}, completion_timeout_seconds=0.05
            )
            try:
                await runtime.start(0)
                ack = await runtime.dispatch(_command())
                with pytest.raises(
                    TimeoutError, match="execution outcome may be unknown"
                ) as failure:
                    await asyncio.wait_for(
                        runtime.receive_completion(ack.dispatch_id), 1
                    )
                assert isinstance(failure.value.__cause__, asyncio.TimeoutError)
                assert not service._active
                assert not runtime._pending
            finally:
                await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("receive_before_disconnect", [True, False])
def test_stream_disconnect_fails_current_and_later_receivers(receive_before_disconnect):
    async def run():
        async with _server(delay_ms=10_000) as (server, _, endpoint):
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint}, timeout_seconds=0.1)
            try:
                await runtime.start(0)
                ack = await runtime.dispatch(_command())
                receiver = None
                if receive_before_disconnect:
                    receiver = asyncio.create_task(
                        runtime.receive_completion(ack.dispatch_id)
                    )
                    await asyncio.sleep(0)
                await server.stop(0)
                if receiver is None:
                    receiver = asyncio.create_task(
                        runtime.receive_completion(ack.dispatch_id)
                    )
                with pytest.raises(RuntimeError, match="completion stream failed"):
                    await asyncio.wait_for(receiver, 1)
            finally:
                await runtime.close()

    asyncio.run(run())


def test_completion_can_arrive_before_dispatch_acknowledgement():
    async def run():
        async with _server(_SlowAckAgent, delay_ms=0) as (_, _, endpoint):
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint})
            try:
                await runtime.start(0)
                ack = await runtime.dispatch(_command())
                completion = await asyncio.wait_for(
                    runtime.receive_completion(ack.dispatch_id), 1
                )
                assert completion.ok
                assert completion.dispatch_id == ack.dispatch_id
                assert runtime._completed_by_agent == {"robot_1": 1}
            finally:
                await runtime.close()

    asyncio.run(run())


def test_lost_dispatch_ack_still_allows_cancellation():
    async def run():
        async with _server(_SlowAckAgent, delay_ms=10_000) as (_, service, endpoint):
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint}, timeout_seconds=0.02)
            try:
                await runtime.start(0)
                command = _command()
                with pytest.raises(grpc.aio.AioRpcError) as failure:
                    await runtime.dispatch(command)
                assert failure.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                assert command.attempt_id in service._active
                assert await runtime.cancel(command.attempt_id, "lost_ack", 0)
                assert not service._active
            finally:
                await runtime.close()

    asyncio.run(run())


def test_completion_deadline_starts_at_dispatch_not_receive():
    async def run():
        async with _server(delay_ms=60) as (_, _, endpoint):
            runtime = GrpcRuntimeAdapter(
                {"robot_1": endpoint}, completion_timeout_seconds=0.02
            )
            try:
                await runtime.start(0)
                ack = await runtime.dispatch(_command())
                await asyncio.sleep(0.1)
                with pytest.raises(TimeoutError, match="completion timeout") as failure:
                    await runtime.receive_completion(ack.dispatch_id)
                assert isinstance(failure.value.__cause__, asyncio.TimeoutError)
                assert runtime._completed_by_agent == {"robot_1": 0}
            finally:
                await runtime.close()

    asyncio.run(run())


def test_historical_and_duplicate_results_are_ignored():
    async def run():
        async with _server(_DuplicateCompletionAgent) as (_, _, endpoint):
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint})
            try:
                await runtime.start(0)
                ack = await runtime.dispatch(_command())
                completion = await runtime.receive_completion(ack.dispatch_id)
                assert completion.ok
                await asyncio.sleep(0.01)
                assert runtime._completed_by_agent == {"robot_1": 1}
                assert not runtime._buffered
                with pytest.raises(ValueError, match="unknown or consumed"):
                    await runtime.receive_completion(ack.dispatch_id)
                duplicate = await runtime.dispatch(_command())
                assert not duplicate.accepted
                assert duplicate.error_code == "duplicate_attempt"
            finally:
                await runtime.close()

    asyncio.run(run())


def test_corrupt_current_completion_fails_without_accepting_wrong_result():
    async def run():
        async with _server(_InvalidCompletionAgent) as (_, _, endpoint):
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint})
            try:
                await runtime.start(0)
                ack = await runtime.dispatch(_command())
                with pytest.raises(RuntimeError, match="does not match its command"):
                    await asyncio.wait_for(
                        runtime.receive_completion(ack.dispatch_id), 1
                    )
                assert runtime._completed_by_agent == {"robot_1": 0}
            finally:
                await runtime.close()

    asyncio.run(run())


def test_partial_registration_failure_closes_all_channels(monkeypatch):
    async def run():
        async with _server() as (_, _, endpoint):
            channels = []
            insecure_channel = grpc.aio.insecure_channel

            def tracked_channel(*args, **kwargs):
                channel = insecure_channel(*args, **kwargs)
                channels.append(channel)
                return channel

            monkeypatch.setattr(grpc.aio, "insecure_channel", tracked_channel)
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint, "wrong_id": endpoint})
            with pytest.raises(RuntimeError, match="rejected registration"):
                await runtime.start(0)
            assert len(channels) == 2
            assert all(
                channel.get_state() == grpc.ChannelConnectivity.SHUTDOWN
                for channel in channels
            )
            assert not runtime._channels
            assert not runtime._stream_tasks
            assert not runtime._started

    asyncio.run(run())


def test_close_cancels_work_and_unblocks_receivers():
    async def run():
        async with _server(delay_ms=10_000) as (_, service, endpoint):
            runtime = GrpcRuntimeAdapter({"robot_1": endpoint})
            await runtime.start(0)
            ack = await runtime.dispatch(_command())
            receiver = asyncio.create_task(runtime.receive_completion(ack.dispatch_id))
            await asyncio.sleep(0)
            await runtime.close()
            with pytest.raises(RuntimeError, match="runtime closed"):
                await asyncio.wait_for(receiver, 1)
            assert not service._active
            assert not runtime._channels
            assert not runtime._stream_tasks
            await runtime.close()

    asyncio.run(run())
