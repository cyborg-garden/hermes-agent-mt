"""Classified raw HTTP and model-client call sites, and protected-session exceptions.

Read with ``agent/protected_session.py``. A protected session (one holding Papa Reo
transcript text) may only send text to the configured primary model.

``tests/agent/test_protected_egress_callsites.py`` scans the runtime packages for
raw HTTP and model-client call sites and fails when a file is missing here or its
count changes. Adding a call site means deciding which category it falls in:

``model_client``     OpenAI / Anthropic / Bedrock / Gemini clients. Every request goes
                     through the httpx or botocore hook, which refuses a protected
                     session's model request to anything but the configured primary.
``tool``             Runs inside a tool, plugin hook or memory-provider scope. In a
                     protected session the requests, urllib, aiohttp, httpx and
                     botocore hooks refuse outbound requests except to loopback, the
                     primary, or an allowed scope; ``BLOCKED_TOOLS`` refuses the tool
                     itself where it talks over a channel the hooks do not see
                     (browser CDP, websockets).
``side_task``        Background work skipped outright in a protected session.
``source``           Where protected text comes from (audio in, transcript out).
``user_channel``     Platform adapters delivering replies to the person; the
                     conversation's own channel, outside any scope.
``configured_brain`` A configured alternative to the AIAgent loop (Letta door mode);
                     it is that gateway's primary.
``no_content``       Auth, billing, catalogs, health and update checks, setup flows:
                     no conversation text.
``offline``          Batch and research tools that never run in a live session.
"""
from __future__ import annotations

CATEGORIES = frozenset({
    "model_client", "tool", "side_task", "source", "user_channel", "configured_brain",
    "no_content", "offline",
})

# Scopes (``tool:<name>``, ``plugin:<name>``, ``memory:<name>``) allowed to send in a
# protected session. The transcription tool is where the text comes from: it sends
# audio out and gets the transcript back. Extend per agent with
# ``routing.protected_egress_allow``.
ALLOWED_SCOPES = frozenset({
    "tool:transcribe_recording",
    "plugin:papareo_transcribe",
})

# Tools refused outright in a protected session: they send text out over channels
# the HTTP hooks cannot see (a browser over CDP, websockets) or exist to send it out.
# Subprocess tools (terminal, execute_code) and every MCP
# tool (stdio servers are subprocesses; streamable-HTTP servers post from a task
# opened at connect time) are refused too: the hooks cannot see their traffic.
BLOCKED_TOOLS = frozenset({
    "text_to_speech", "image_generate", "video_generate", "web_search", "web_extract",
    "x_search", "terminal", "execute_code",
})
BLOCKED_TOOL_PREFIXES = ("browser_", "mcp_")

# Plugins whose hooks export what they receive (traces, telemetry), usually from a
# long-lived sender thread started before any session: in a protected session
# their hook callbacks are not called at all. Matched against the plugin's module
# slug (``observability__langfuse`` for plugins/observability/langfuse).
EXPORTER_PLUGIN_PREFIXES = ("observability", "langfuse", "nemo_relay", "home_log_router",
                            "telemetry", "sentry", "otel", "opentelemetry")

CALL_SITES: dict = {
    "agent/account_usage.py": (4, "no_content"),
    "agent/agent_runtime_helpers.py": (1, "model_client"),
    "agent/anthropic_adapter.py": (5, "model_client"),
    "agent/auxiliary_client.py": (5, "model_client"),
    "agent/azure_identity_adapter.py": (4, "model_client"),
    "agent/bedrock_adapter.py": (2, "model_client"),
    "agent/gemini_native_adapter.py": (2, "model_client"),
    "agent/image_gen_provider.py": (1, "tool"),
    "agent/moa_loop.py": (1, "side_task"),
    "agent/model_metadata.py": (12, "no_content"),
    "agent/models_dev.py": (1, "no_content"),
    "agent/pet/generate/imagegen.py": (2, "tool"),
    "agent/pet/manifest.py": (1, "no_content"),
    "agent/pet/store.py": (3, "no_content"),
    "agent/redact.py": (1, "no_content"),
    "agent/secret_sources/bitwarden.py": (1, "no_content"),
    "agent/transports/anthropic.py": (1, "model_client"),
    "agent/usage_pricing.py": (2, "no_content"),
    "agent/video_gen_provider.py": (2, "tool"),
    "gateway/letta_brain.py": (1, "configured_brain"),
    "gateway/platforms/base.py": (3, "user_channel"),
    "gateway/platforms/bluebubbles.py": (1, "user_channel"),
    "gateway/platforms/qqbot/adapter.py": (2, "user_channel"),
    "gateway/platforms/qqbot/onboard.py": (2, "no_content"),
    "gateway/platforms/signal.py": (2, "user_channel"),
    "gateway/platforms/weixin.py": (4, "user_channel"),
    "gateway/platforms/whatsapp_cloud.py": (1, "user_channel"),
    "gateway/platforms/yuanbao.py": (4, "user_channel"),
    "gateway/platforms/yuanbao_media.py": (3, "user_channel"),
    "gateway/relay/__init__.py": (3, "user_channel"),
    "gateway/relay/ws_transport.py": (2, "user_channel"),
    "hermes_cli/auth.py": (18, "no_content"),
    "hermes_cli/auth_commands.py": (1, "no_content"),
    "hermes_cli/banner.py": (1, "no_content"),
    "hermes_cli/browser_connect.py": (1, "no_content"),
    "hermes_cli/codex_models.py": (1, "no_content"),
    "hermes_cli/config.py": (1, "no_content"),
    "hermes_cli/copilot_auth.py": (3, "no_content"),
    "hermes_cli/dashboard_register.py": (1, "no_content"),
    "hermes_cli/debug.py": (3, "no_content"),
    "hermes_cli/diagnostics_upload.py": (2, "no_content"),
    "hermes_cli/dingtalk_auth.py": (1, "no_content"),
    "hermes_cli/doctor.py": (6, "no_content"),
    "hermes_cli/gateway.py": (1, "no_content"),
    "hermes_cli/gateway_enroll.py": (1, "no_content"),
    "hermes_cli/github_app_token.py": (3, "no_content"),
    "hermes_cli/model_catalog.py": (1, "no_content"),
    "hermes_cli/models.py": (4, "no_content"),
    "hermes_cli/nous_account.py": (1, "no_content"),
    "hermes_cli/nous_billing.py": (1, "no_content"),
    "hermes_cli/proxy/server.py": (1, "model_client"),
    "hermes_cli/runtime_provider.py": (2, "no_content"),
    "hermes_cli/security_audit.py": (2, "no_content"),
    "hermes_cli/skills_hub.py": (6, "no_content"),
    "hermes_cli/status.py": (1, "no_content"),
    "hermes_cli/telegram_managed_bot.py": (2, "no_content"),
    "hermes_cli/urllib_security.py": (2, "no_content"),
    "hermes_cli/web_server.py": (15, "no_content"),
    "hermes_cli/webhook.py": (1, "no_content"),
    "hermes_constants.py": (2, "no_content"),
    "mini_swe_runner.py": (2, "offline"),
    "plugins/browser/browser_use/provider.py": (3, "tool"),
    "plugins/browser/browserbase/provider.py": (5, "tool"),
    "plugins/browser/firecrawl/provider.py": (3, "tool"),
    "plugins/cron_providers/chronos/_nas_client.py": (2, "tool"),
    "plugins/dashboard_auth/nous/__init__.py": (2, "no_content"),
    "plugins/dashboard_auth/self_hosted/__init__.py": (3, "no_content"),
    "plugins/image_gen/deepinfra/__init__.py": (1, "tool"),
    "plugins/image_gen/krea/__init__.py": (2, "tool"),
    "plugins/image_gen/openai-codex/__init__.py": (3, "tool"),
    "plugins/image_gen/openai/__init__.py": (2, "tool"),
    "plugins/image_gen/openrouter/__init__.py": (1, "tool"),
    "plugins/image_gen/xai/__init__.py": (1, "tool"),
    "plugins/memory/hindsight/__init__.py": (1, "tool"),
    "plugins/memory/honcho/oauth.py": (1, "no_content"),
    "plugins/memory/mem0/_backend.py": (1, "tool"),
    "plugins/memory/mem0/_setup.py": (3, "no_content"),
    "plugins/memory/retaindb/__init__.py": (3, "tool"),
    "plugins/memory/supermemory/__init__.py": (1, "tool"),
    "plugins/platforms/dingtalk/adapter.py": (2, "user_channel"),
    "plugins/platforms/discord/adapter.py": (6, "user_channel"),
    "plugins/platforms/feishu/adapter.py": (4, "user_channel"),
    "plugins/platforms/google_chat/oauth.py": (1, "no_content"),
    "plugins/platforms/homeassistant/adapter.py": (4, "user_channel"),
    "plugins/platforms/line/adapter.py": (5, "user_channel"),
    "plugins/platforms/matrix/adapter.py": (6, "user_channel"),
    "plugins/platforms/mattermost/adapter.py": (2, "user_channel"),
    "plugins/platforms/ntfy/adapter.py": (2, "user_channel"),
    "plugins/platforms/photon/adapter.py": (5, "user_channel"),
    "plugins/platforms/photon/auth.py": (11, "no_content"),
    "plugins/platforms/slack/adapter.py": (6, "user_channel"),
    "plugins/platforms/sms/adapter.py": (3, "user_channel"),
    "plugins/platforms/teams/adapter.py": (2, "user_channel"),
    "plugins/platforms/telegram/adapter.py": (1, "user_channel"),
    "plugins/platforms/telegram/telegram_network.py": (1, "user_channel"),
    "plugins/platforms/wecom/adapter.py": (5, "user_channel"),
    "plugins/platforms/wecom/callback_adapter.py": (1, "user_channel"),
    "plugins/platforms/whatsapp/adapter.py": (6, "user_channel"),
    "plugins/spotify/client.py": (1, "tool"),
    "plugins/swarm_map_policy/__init__.py": (3, "no_content"),
    "plugins/teams_pipeline/pipeline.py": (2, "tool"),
    "plugins/video_gen/xai/__init__.py": (1, "tool"),
    "plugins/web/brave_free/provider.py": (1, "tool"),
    "plugins/web/searxng/provider.py": (1, "tool"),
    "plugins/web/tavily/provider.py": (3, "tool"),
    "plugins/web/xai/provider.py": (1, "tool"),
    "run_agent.py": (1, "model_client"),
    "tools/browser_camofox.py": (6, "tool"),
    "tools/browser_cdp_tool.py": (1, "tool"),
    "tools/browser_supervisor.py": (1, "tool"),
    "tools/browser_tool.py": (1, "tool"),
    "tools/discord_tool.py": (1, "tool"),
    "tools/environments/managed_modal.py": (1, "tool"),
    "tools/homeassistant_tool.py": (4, "tool"),
    "tools/kanban_tools.py": (1, "tool"),
    "tools/mcp_oauth_manager.py": (1, "no_content"),
    "tools/mcp_tool.py": (1, "tool"),
    "tools/microsoft_graph_auth.py": (1, "no_content"),
    "tools/microsoft_graph_client.py": (2, "tool"),
    "tools/osv_check.py": (1, "no_content"),
    "tools/send_message_tool.py": (4, "tool"),
    "tools/skills_hub.py": (21, "tool"),
    "tools/tirith_security.py": (1, "no_content"),
    "tools/transcription_tools.py": (4, "source"),
    "tools/tts_tool.py": (5, "tool"),
    "tools/vision_tools.py": (2, "tool"),
    "tools/x_search_tool.py": (1, "tool"),
    "trajectory_compressor.py": (2, "offline"),
    "tui_gateway/server.py": (1, "no_content"),
}
