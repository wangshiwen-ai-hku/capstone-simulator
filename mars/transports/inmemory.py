"""Deterministic MARS transport for tests and the web simulator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..models import Assignment, NodeSnapshot, TaskCompletion, TaskInstance, WorkflowSpec
from .base import NodeRegistration, TransportCapabilities


class InMemoryTransport:
    capabilities = TransportCapabilities(
        discovery=False,
        reliable_control=True,
        best_effort_telemetry=False,
        feedback=True,
        cancellation=True,
        liveliness=False,
    )

    def __init__(self) -> None:
        self.registrations: dict[str, NodeRegistration] = {}
        self.node_states: dict[str, NodeSnapshot] = {}
        self.workflows: dict[str, WorkflowSpec] = {}
        self.dispatches: list[tuple[TaskInstance, Assignment]] = []
        self.cancelled: dict[str, str] = {}
        self._completions: asyncio.Queue[TaskCompletion] = asyncio.Queue()

    async def register(self, registration: NodeRegistration) -> bool:
        duplicate = registration.node_id in self.registrations
        self.registrations[registration.node_id] = registration
        return not duplicate

    async def publish_node_state(self, snapshot: NodeSnapshot) -> None:
        self.node_states[snapshot.node_id] = snapshot

    async def submit_workflow(self, workflow: WorkflowSpec) -> str:
        if workflow.workflow_id in self.workflows:
            return workflow.workflow_id
        self.workflows[workflow.workflow_id] = workflow
        return workflow.workflow_id

    async def dispatch(self, task: TaskInstance, assignment: Assignment) -> str:
        self.dispatches.append((task, assignment))
        return f"inmemory:{task.task_id}:{len(self.dispatches)}"

    async def cancel(self, task_id: str, reason: str) -> bool:
        self.cancelled[task_id] = reason
        return True

    async def report_completion(self, completion: TaskCompletion) -> None:
        await self._completions.put(completion)

    async def completions(self) -> AsyncIterator[TaskCompletion]:
        while True:
            yield await self._completions.get()
