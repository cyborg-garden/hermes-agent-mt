"""Delegation Provider ABC and registry (minimal form).

A delegation provider decides which route one ``delegate_task`` task takes: which model,
provider and endpoint the child uses. Providers register via
:meth:`PluginContext.register_delegation_provider`; ``delegation.backend`` in config.yaml
selects the active one. Unset (or ``default``) keeps the built-in behaviour: every child
takes the ``delegation.*`` route, else the parent's own model. Core keeps the whole child
lifecycle; a provider only answers "which route does this task take".

Route vocabulary (:meth:`DelegationProvider.route_task` returns a dict with any of):

* ``model``, ``provider``, ``base_url``, ``api_key``, ``api_mode`` - shaped like the
  ``delegation`` config section and resolved by the same credential resolver. An empty
  route (or one that fails to resolve) is the default route.
* ``fallback_providers`` - the child's fallback chain, replacing the parent's.
* ``reasoning_effort`` - the child's reasoning effort, replacing ``delegation.reasoning_effort``.
* ``label`` - short text shown with the child's progress.
* ``escalate`` - ``True`` marks a cheap route: if the child's reply is empty or a refusal, core
  moves it to the next ``fallback_providers`` entry at once instead of retrying the cheap model
  (put the stronger model first in that list). API errors already walk the chain.
* ``protected_watch`` - ``True`` arms the protected-content check on the child: before each
  call, a protected marker (default: the Papa Reo transcript footer, plus any
  ``protected_markers`` strings on the route) moves it to the first fallback entry, or ends it.
"""

from __future__ import annotations

import abc
import logging
import threading
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

#: Keys a route may carry; anything else in a provider's answer is ignored.
ROUTE_KEYS = frozenset({
    "model", "provider", "base_url", "api_key", "api_mode", "reasoning_effort",
    "fallback_providers", "label", "escalate", "protected_markers", "protected_watch",
})

#: ``delegation.backend`` values that mean the built-in behaviour.
DEFAULT_BACKENDS = frozenset({"", "default"})


class DelegationRouteError(ValueError):
    """Raised by :meth:`DelegationProvider.route_task` to refuse a task."""


class DelegationProvider(abc.ABC):
    """A pluggable delegation strategy. :attr:`name` is the ``delegation.backend`` value."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def summary(self) -> str:
        return ""

    @abc.abstractmethod
    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        """The route for one task (``goal``, ``context``, ``index``). ``config`` is the
        ``delegation`` config section (read-only)."""


_registry: dict[str, DelegationProvider] = {}
_lock = threading.Lock()


def register_provider(provider: DelegationProvider) -> None:
    with _lock:
        _registry[str(provider.name)] = provider


def get_provider(name: Optional[str]) -> Optional[DelegationProvider]:
    with _lock:
        return _registry.get(str(name or ""))


def clean_route(route: Any) -> dict[str, Any]:
    """Only the known route keys of a provider's answer ({} for anything not a dict)."""
    if not isinstance(route, Mapping):
        return {}
    return {k: v for k, v in route.items() if k in ROUTE_KEYS}
