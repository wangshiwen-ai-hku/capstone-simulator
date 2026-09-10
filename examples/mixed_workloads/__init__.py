"""Synthetic navigation with real CPU work and native CUDA map inflation."""

from .pipeline import GPU_TASK_TYPE, PORT_TYPES, execute

__all__ = ["GPU_TASK_TYPE", "PORT_TYPES", "execute"]
