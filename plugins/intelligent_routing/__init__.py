"""Intelligent routing plugin — opt-in pre-turn cost routing (Option A).

When enabled per-agent (``routing.intelligent: true`` in config.yaml, default
OFF), a cheap local heuristic classifies each incoming turn by task type, routes
mechanical/orchestration work to a cheap-metered tier (default
``deepseek/deepseek-v3.2`` via OpenRouter), and surfaces the decision in the
model-awareness prompt line. It fails open to the premium tier on any uncertainty.

The swap is ACTUATED through the routing-override interface extension
(``agent/routing_override.py`` + one call site in ``agent/turn_context.py``) —
the clean plugin form of upstream PR #43534, per teknium1's steer + his offer to
extend the plugin interface. Turn-scoped and fail-safe. See ``registration.py``
and the README.
"""
from .registration import register

__all__ = ["register"]
