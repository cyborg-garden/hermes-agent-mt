"""Intelligent routing plugin: opt-in cost routing by task type.

Main turn: when ``routing.intelligent: true`` (default OFF), ``task_router``
classifies each turn; cheap-safe work routes to ``routing.cheap_model`` and the
rest stays on the primary. Subagents: ``delegation.backend: task-routing``
routes each child the same way. Cheap routes escalate to the primary on error,
empty reply or refusal. See ``registration.py`` and the README.
"""
from .registration import register

__all__ = ["register"]
