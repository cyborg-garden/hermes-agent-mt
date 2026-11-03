"""``telegram.ext.ChatMemberHandler`` must be optional, not load-bearing.

The Telegram adapter imports the SDK once at module scope inside a single
``try: ... except ImportError:`` block. Everything named in the mandatory
``from telegram.ext import (...)`` tuple is therefore all-or-nothing: if ONE
name is missing, the whole block raises and the ``except ImportError`` branch
runs, setting ``TELEGRAM_AVAILABLE = False``, ``ParseMode = None``,
``ChatType = None`` and ``InlineKeyboardMarkup = Any``. The adapter still
imports, but every send path then dies on ``'NoneType' object has no attribute
'MARKDOWN_V2'`` / ``Any cannot be instantiated``.

``ChatMemberHandler`` only powers the bot-added-to-group HSM auto-approval gate
(``TELEGRAM_GROUP_AUTOAPPROVE``). A build of python-telegram-bot that lacks it
must cost us that one feature, nothing more. Note the blast radius is NOT
contained by the feature flag: the flag is read inside the handler, while the
import and handler registration happen unconditionally at import/connect time.

These tests import the adapter module FRESH against a stubbed ``telegram``
package so they exercise the real module-level import block rather than
whatever the first importer happened to cache.
"""

import importlib
import logging
import sys
import types
from types import SimpleNamespace

import pytest


# ── Minimal but honest fake of the SDK surface the adapter imports ──────


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


class _FakeNetworkError(Exception):
    pass


class _FakeBadRequest(_FakeNetworkError):
    pass


class _FakeTimedOut(_FakeNetworkError):
    pass


class _FakeChatMemberHandler:
    MY_CHAT_MEMBER = "my_chat_member"

    def __init__(self, callback, chat_member_types=None):
        self.callback = callback
        self.chat_member_types = chat_member_types


def _install_fake_telegram(monkeypatch, *, with_chat_member_handler: bool):
    """Put a fake ``telegram`` package tree in ``sys.modules``.

    ``with_chat_member_handler=False`` reproduces an SDK build (or a test
    double) that exports everything the adapter genuinely needs EXCEPT
    ``ChatMemberHandler``.
    """
    telegram = types.ModuleType("telegram")
    telegram.Update = object
    telegram.Bot = object
    telegram.Message = object
    telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
    telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
    telegram.InputMediaPhoto = object

    error = types.ModuleType("telegram.error")
    error.NetworkError = _FakeNetworkError
    error.BadRequest = _FakeBadRequest
    error.TimedOut = _FakeTimedOut
    telegram.error = error

    constants = types.ModuleType("telegram.constants")
    constants.ParseMode = SimpleNamespace(
        MARKDOWN_V2="MarkdownV2", MARKDOWN="Markdown", HTML="HTML",
    )
    constants.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private",
    )
    telegram.constants = constants

    ext = types.ModuleType("telegram.ext")
    ext.Application = object
    ext.CommandHandler = object
    ext.CallbackQueryHandler = object
    ext.InlineQueryHandler = object
    ext.MessageHandler = object
    ext.TypeHandler = object
    ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    ext.filters = object
    if with_chat_member_handler:
        ext.ChatMemberHandler = _FakeChatMemberHandler

    request = types.ModuleType("telegram.request")
    request.HTTPXRequest = object

    for name, module in (
        ("telegram", telegram),
        ("telegram.error", error),
        ("telegram.constants", constants),
        ("telegram.ext", ext),
        ("telegram.request", request),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _reimport_adapter(monkeypatch, *, with_chat_member_handler: bool):
    """Import ``plugins.platforms.telegram.adapter`` fresh against the fake."""
    _install_fake_telegram(monkeypatch, with_chat_member_handler=with_chat_member_handler)
    monkeypatch.delitem(
        sys.modules, "plugins.platforms.telegram.adapter", raising=False,
    )
    module = importlib.import_module("plugins.platforms.telegram.adapter")
    # Leave no cached copy behind: this module object was built against fakes.
    monkeypatch.delitem(
        sys.modules, "plugins.platforms.telegram.adapter", raising=False,
    )
    return module


# ── The regression this file exists for ────────────────────────────────


def test_adapter_stays_available_when_ext_lacks_chat_member_handler(monkeypatch):
    """A missing ChatMemberHandler must NOT collapse the whole SDK import.

    This is the failure that broke seven unrelated thread-routing tests: the
    adapter fell into its ``except ImportError`` branch, so ParseMode became
    None and every MarkdownV2 send raised.
    """
    adapter = _reimport_adapter(monkeypatch, with_chat_member_handler=False)

    assert adapter.TELEGRAM_AVAILABLE is True
    assert adapter.ChatMemberHandler is None

    # The rest of the SDK surface must be the REAL (faked) objects, not the
    # None/Any placeholders from the all-or-nothing fallback.
    assert adapter.ParseMode is not None
    assert adapter.ParseMode.MARKDOWN_V2 == "MarkdownV2"
    assert adapter.ChatType is not None
    assert adapter.ChatType.SUPERGROUP == "supergroup"
    assert adapter.filters is not None
    # ``Any cannot be instantiated`` was the second CI symptom.
    assert adapter.InlineKeyboardMarkup([[1]]).inline_keyboard == [[1]]


def test_adapter_binds_chat_member_handler_when_the_sdk_has_it(monkeypatch):
    """The feature is still wired up on an SDK that exports the symbol."""
    adapter = _reimport_adapter(monkeypatch, with_chat_member_handler=True)

    assert adapter.TELEGRAM_AVAILABLE is True
    assert adapter.ChatMemberHandler is _FakeChatMemberHandler
    assert adapter.ParseMode.MARKDOWN_V2 == "MarkdownV2"


# ── Handler registration degrades to a no-op, loudly when it matters ────


def _bare_adapter(adapter_module):
    """An adapter instance with just enough state to register handlers."""
    from gateway.config import Platform

    instance = object.__new__(adapter_module.TelegramAdapter)
    handlers = []
    instance._app = SimpleNamespace(handlers=handlers, add_handler=handlers.append)
    # ``name`` is a read-only property derived from the platform.
    instance._platform = Platform.TELEGRAM
    instance.platform = Platform.TELEGRAM
    return instance


def test_registration_skips_handler_when_symbol_is_missing(monkeypatch):
    adapter_module = _reimport_adapter(monkeypatch, with_chat_member_handler=False)
    monkeypatch.delenv("TELEGRAM_GROUP_AUTOAPPROVE", raising=False)
    instance = _bare_adapter(adapter_module)

    assert instance._register_my_chat_member_handler() is False
    assert instance._app.handlers == []


def test_registration_warns_when_autoapprove_is_on_but_symbol_is_missing(
    monkeypatch, caplog,
):
    """Silently dropping an ENABLED security gate would be the worse bug."""
    adapter_module = _reimport_adapter(monkeypatch, with_chat_member_handler=False)
    monkeypatch.setenv("TELEGRAM_GROUP_AUTOAPPROVE", "true")
    instance = _bare_adapter(adapter_module)

    with caplog.at_level(logging.WARNING, logger=adapter_module.logger.name):
        assert instance._register_my_chat_member_handler() is False

    assert instance._app.handlers == []
    assert any(
        "ChatMemberHandler" in record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


def test_registration_installs_handler_when_symbol_is_present(monkeypatch):
    adapter_module = _reimport_adapter(monkeypatch, with_chat_member_handler=True)
    instance = _bare_adapter(adapter_module)

    assert instance._register_my_chat_member_handler() is True
    assert len(instance._app.handlers) == 1
    handler = instance._app.handlers[0]
    assert isinstance(handler, _FakeChatMemberHandler)
    assert handler.chat_member_types == _FakeChatMemberHandler.MY_CHAT_MEMBER
    assert handler.callback == instance._handle_my_chat_member


@pytest.mark.parametrize("with_cmh", [True, False])
def test_lazy_sdk_reimport_also_treats_chat_member_handler_as_optional(
    monkeypatch, with_cmh,
):
    """``check_telegram_requirements`` has its own import tuple, same trap."""
    adapter_module = _reimport_adapter(monkeypatch, with_chat_member_handler=with_cmh)

    # Force the lazy path to actually run by pretending the SDK is absent.
    monkeypatch.setattr(adapter_module, "TELEGRAM_AVAILABLE", False)
    monkeypatch.setattr(adapter_module, "ParseMode", None)
    monkeypatch.setattr(adapter_module, "ChatMemberHandler", None)
    monkeypatch.setitem(
        sys.modules, "tools.lazy_deps", SimpleNamespace(ensure=lambda *a, **k: True),
    )

    assert adapter_module.check_telegram_requirements() is True
    assert adapter_module.TELEGRAM_AVAILABLE is True
    assert adapter_module.ParseMode.MARKDOWN_V2 == "MarkdownV2"
    if with_cmh:
        assert adapter_module.ChatMemberHandler is _FakeChatMemberHandler
    else:
        assert adapter_module.ChatMemberHandler is None
