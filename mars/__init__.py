"""MARS scheduling core used by the API, simulator, and transports."""

from .dag import DagValidationError, TaskManager, validate_workflow
from .models import (
    ArtifactRef,
    ExecutionMode,
    FailurePolicy,
    NodeKind,
    TaskClass,
    TaskInstance,
    TaskSpec,
    TaskState,
    WorkflowSpec,
    WorkflowState,
)

__all__ = [
    "ArtifactRef",
    "DagValidationError",
    "ExecutionMode",
    "FailurePolicy",
    "NodeKind",
    "TaskClass",
    "TaskInstance",
    "TaskManager",
    "TaskSpec",
    "TaskState",
    "WorkflowSpec",
    "WorkflowState",
    "validate_workflow",
]

__version__ = "0.2.0"
