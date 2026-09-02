"""Backward-compatible imports for the renamed Authoring Assistant service."""

from .authoring_assistant import (
    AuthoringAssistantService,
    ModellingSession,
)


MarsAgentService = AuthoringAssistantService

__all__ = [
    "AuthoringAssistantService",
    "MarsAgentService",
    "ModellingSession",
]
