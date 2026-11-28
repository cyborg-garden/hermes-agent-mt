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

# The only tools that run in a protected session (default deny). Each reads local
# state or asks the person, and none persists text beyond the session or sends it
# anywhere. Every other tool, including any added later, is refused, also the
# agent-loop tools (memory, delegate_task) that bypass the registry.
# transcribe_recording runs under its own restrictions (no URL, gateway-downloaded
# recordings only, not from a child). tests/agent/test_protected_tool_allowlist.py
# fails when this lists a tool that is not registered.
PROTECTED_SESSION_TOOLS = frozenset({
    "read_file", "search_files", "read_document", "clarify", "todo_list", "skill_view",
    "skills_list", "session_search", "transcribe_recording",
})

# Tools refused outright in a protected session (kept for reference: the default-
# deny list above already refuses them): they send text out over channels
# the HTTP hooks cannot see (a browser over CDP, websockets) or exist to send it out.
# Subprocess tools (terminal, execute_code) and every MCP
# tool (stdio servers are subprocesses; streamable-HTTP servers post from a task
# opened at connect time) are refused too: the hooks cannot see their traffic.
BLOCKED_TOOLS = frozenset({
    "text_to_speech", "image_generate", "video_generate", "web_search", "web_extract",
    "x_search", "terminal", "execute_code", "send_message",
})
BLOCKED_TOOL_PREFIXES = ("browser_", "mcp_")

# Allowed tools that take a URL the model chooses and fetch it in a subprocess:
# in a protected session only a URL the person sent may be fetched. Tool name ->
# argument name. A local attachment path is unaffected.
URL_ARGUMENT_TOOLS = {"transcribe_recording": "url"}

# Tools whose results carry Papa Reo text with no footer in some modes (the
# transcribe tool's compliant mode returns ``engine: papareo`` plus ``text``).
TRANSCRIPT_TOOLS = frozenset({"transcribe_recording"})

# Scopes that may reach loopback in a protected session. Loopback is not safe by
# default: signal-cli, local bridges and VPN proxies on 127.0.0.1 relay outward.
LOOPBACK_SCOPES = frozenset({
    "tool:transcribe_recording", "plugin:papareo_transcribe",
})

# The chat platforms' own API hosts (and subdomains). Work a tool hands to the
# gateway loop may reach only these, loopback, and the primary. Self-hosted
# platforms (Matrix, Mattermost, ...) add theirs with routing.protected_delivery_hosts.
DELIVERY_HOSTS = (
    "discord.com", "discordapp.com", "discord.gg", "discordapp.net", "discord.media",
    "api.telegram.org", "slack.com", "slack-edge.com", "graph.facebook.com",
    "graph.microsoft.com", "botframework.com", "api.line.me", "api-data.line.me",
    "open.feishu.cn", "open.larksuite.com", "qyapi.weixin.qq.com", "api.weixin.qq.com",
    "oapi.dingtalk.com", "api.dingtalk.com", "chat.googleapis.com", "api.twilio.com",
)

# Plugins whose hooks export what they receive (traces, telemetry), usually from a
# long-lived sender thread started before any session: in a protected session
# their hook callbacks are not called at all. Matched against the plugin's module
# slug (``observability__langfuse`` for plugins/observability/langfuse).
EXPORTER_PLUGIN_PREFIXES = ("observability", "langfuse", "nemo_relay", "home_log_router",
                            "telemetry", "sentry", "otel", "opentelemetry")

CALL_SITES: dict = {
    "agent/account_usage.py": (3, "no_content"),
    "agent/agent_init.py": (1, "model_client"),
    "agent/agent_runtime_helpers.py": (2, "model_client"),
    "agent/anthropic_adapter.py": (3, "model_client"),
    "agent/anthropic_credentials.py": (1, "no_content"),
    "agent/anthropic_endpoints.py": (1, "model_client"),
    "agent/auxiliary_client.py": (4, "model_client"),
    "agent/azure_identity_adapter.py": (2, "model_client"),
    "agent/bedrock_adapter.py": (2, "model_client"),
    "agent/chat_completion_local.py": (1, "model_client"),
    "agent/error_classifier.py": (1, "no_content"),
    "agent/gemini_native_adapter.py": (2, "model_client"),
    "agent/model_metadata.py": (4, "no_content"),
    "agent/model_metadata_http.py": (1, "no_content"),
    "agent/models_dev.py": (1, "no_content"),
    "agent/pet/generate/imagegen.py": (2, "tool"),
    "agent/provider_media.py": (1, "tool"),
    "agent/proxy_sources/iron_proxy.py": (1, "no_content"),
    "agent/review_idle_queue.py": (1, "side_task"),
    "agent/transports/anthropic.py": (1, "model_client"),
    "agent/turn_api_request.py": (1, "model_client"),
    "agent/verify/runner.py": (1, "tool"),
    "agent/video_gen_provider.py": (1, "tool"),
    "cron/monitor.py": (1, "no_content"),
    "gateway/host_rendezvous.py": (1, "no_content"),
    "gateway/letta_brain.py": (1, "configured_brain"),
    "gateway/relay/__init__.py": (3, "user_channel"),
    "gateway/relay/link.py": (1, "user_channel"),
    "gateway/relay/media.py": (2, "user_channel"),
    "gateway/relay/ws_transport.py": (1, "user_channel"),
    "gateway/wake.py": (1, "user_channel"),
    "hermes_cli/auth.py": (2, "no_content"),
    "hermes_cli/auth_codex.py": (1, "no_content"),
    "hermes_cli/auth_commands.py": (1, "no_content"),
    "hermes_cli/auth_minimax.py": (2, "no_content"),
    "hermes_cli/auth_nous.py": (1, "no_content"),
    "hermes_cli/auth_oauth_pkce_plugin.py": (1, "no_content"),
    "hermes_cli/auth_openrouter.py": (1, "no_content"),
    "hermes_cli/auth_qwen.py": (1, "no_content"),
    "hermes_cli/auth_xai.py": (3, "no_content"),
    "hermes_cli/auth_zai_kimi.py": (1, "no_content"),
    "hermes_cli/browser_connect.py": (1, "no_content"),
    "hermes_cli/codex_models.py": (1, "no_content"),
    "hermes_cli/config_show.py": (1, "no_content"),
    "hermes_cli/dashboard_register.py": (1, "no_content"),
    "hermes_cli/debug.py": (2, "no_content"),
    "hermes_cli/diagnostics_upload.py": (1, "no_content"),
    "hermes_cli/dingtalk_auth.py": (1, "no_content"),
    "hermes_cli/doctor_connectivity.py": (8, "no_content"),
    "hermes_cli/doctor_live.py": (1, "no_content"),
    "hermes_cli/gateway_connections_oauth.py": (1, "no_content"),
    "hermes_cli/gateway_connections_reach.py": (1, "no_content"),
    "hermes_cli/gateway_connections_ssh.py": (1, "no_content"),
    "hermes_cli/gateway_enroll.py": (1, "no_content"),
    "hermes_cli/github_app_token.py": (2, "no_content"),
    "hermes_cli/local_runtime/catalog.py": (1, "no_content"),
    "hermes_cli/local_runtime/detect.py": (1, "no_content"),
    "hermes_cli/local_runtime/endpoint.py": (1, "no_content"),
    "hermes_cli/local_runtime/hf_browse.py": (1, "no_content"),
    "hermes_cli/local_runtime/load_progress.py": (1, "no_content"),
    "hermes_cli/local_runtime/supervisor.py": (2, "no_content"),
    "hermes_cli/model_catalog.py": (1, "no_content"),
    "hermes_cli/models_catalog_static.py": (1, "no_content"),
    "hermes_cli/models_local.py": (1, "no_content"),
    "hermes_cli/nous_account.py": (1, "no_content"),
    "hermes_cli/nous_billing.py": (2, "no_content"),
    "hermes_cli/observability/shared_metrics_sender.py": (1, "no_content"),
    "hermes_cli/plugin_catalog.py": (1, "no_content"),
    "hermes_cli/plugin_catalog_presence.py": (1, "no_content"),
    "hermes_cli/plugin_host_child.py": (1, "tool"),
    "hermes_cli/plugin_oauth.py": (1, "no_content"),
    "hermes_cli/plugin_packs.py": (1, "no_content"),
    "hermes_cli/plugin_provider_requests.py": (1, "no_content"),
    "hermes_cli/plugins_activation.py": (1, "no_content"),
    "hermes_cli/plugins_updates.py": (2, "no_content"),
    "hermes_cli/proxy/server.py": (1, "model_client"),
    "hermes_cli/runtime_provider.py": (2, "no_content"),
    "hermes_cli/runtime_provider_backends.py": (1, "no_content"),
    "hermes_cli/security_audit.py": (1, "no_content"),
    "hermes_cli/shared_session_attach.py": (1, "no_content"),
    "hermes_cli/source_check.py": (1, "no_content"),
    "hermes_cli/source_releases.py": (1, "no_content"),
    "hermes_cli/status.py": (1, "no_content"),
    "hermes_cli/telegram_managed_bot.py": (2, "no_content"),
    "hermes_cli/tools_config_post_setup.py": (1, "no_content"),
    "hermes_cli/urllib_security.py": (2, "no_content"),
    "hermes_cli/web_routers/audio.py": (1, "no_content"),
    "hermes_cli/web_routers/config_env.py": (2, "no_content"),
    "hermes_cli/web_routers/files.py": (1, "no_content"),
    "hermes_cli/web_routers/local_models.py": (1, "no_content"),
    "hermes_cli/web_routers/oauth.py": (2, "no_content"),
    "hermes_cli/web_routers/webapp_relay_proxy.py": (1, "no_content"),
    "hermes_cli/web_server.py": (1, "no_content"),
    "hermes_cli/web_server_cron.py": (1, "no_content"),
    "hermes_cli/web_server_gateway.py": (1, "no_content"),
    "hermes_cli/web_server_messaging.py": (1, "no_content"),
    "hermes_cli/web_server_oauth.py": (2, "no_content"),
    "hermes_cli/webapp_relay_target.py": (1, "no_content"),
    "hermes_cli/webhook.py": (1, "no_content"),
    "plugins/browser/_common.py": (1, "tool"),
    "plugins/cron_providers/chronos/_nas_client.py": (1, "tool"),
    "plugins/dashboard_auth/_shared.py": (1, "no_content"),
    "plugins/dashboard_auth/self_hosted/__init__.py": (2, "no_content"),
    "plugins/image_gen/_common.py": (1, "tool"),
    "plugins/image_gen/deepinfra/__init__.py": (1, "tool"),
    "plugins/image_gen/krea/__init__.py": (2, "tool"),
    "plugins/image_gen/meta-ai/__init__.py": (1, "tool"),
    "plugins/image_gen/openai-codex/__init__.py": (3, "tool"),
    "plugins/image_gen/openai/__init__.py": (1, "tool"),
    "plugins/image_gen/openrouter/__init__.py": (1, "tool"),
    "plugins/image_gen/xai/__init__.py": (1, "tool"),
    "plugins/memory/retaindb/__init__.py": (1, "tool"),
    "plugins/model-providers/opencode-zen/__init__.py": (1, "no_content"),
    "plugins/model-providers/solstice/auth.py": (1, "no_content"),
    "plugins/platforms/a2a/adapter.py": (1, "user_channel"),
    "plugins/platforms/a2a/tools.py": (1, "tool"),
    "plugins/platforms/bluebubbles/adapter.py": (1, "user_channel"),
    "plugins/platforms/buzz/adapter.py": (2, "user_channel"),
    "plugins/platforms/dingtalk/adapter.py": (2, "user_channel"),
    "plugins/platforms/discord/adapter.py": (4, "user_channel"),
    "plugins/platforms/discord/adapter_media.py": (1, "user_channel"),
    "plugins/platforms/feishu/adapter.py": (2, "user_channel"),
    "plugins/platforms/google_chat/oauth.py": (1, "no_content"),
    "plugins/platforms/line/adapter.py": (1, "user_channel"),
    "plugins/platforms/matrix/adapter.py": (5, "user_channel"),
    "plugins/platforms/mattermost/adapter.py": (2, "user_channel"),
    "plugins/platforms/ntfy/adapter.py": (2, "user_channel"),
    "plugins/platforms/photon/adapter.py": (5, "user_channel"),
    "plugins/platforms/photon/auth.py": (6, "no_content"),
    "plugins/platforms/qqbot/adapter.py": (1, "user_channel"),
    "plugins/platforms/qqbot/onboard.py": (1, "no_content"),
    "plugins/platforms/qqbot/send.py": (1, "user_channel"),
    "plugins/platforms/signal/adapter.py": (2, "user_channel"),
    "plugins/platforms/signal/setup.py": (1, "no_content"),
    "plugins/platforms/signal/standalone.py": (1, "user_channel"),
    "plugins/platforms/slack/adapter.py": (3, "user_channel"),
    "plugins/platforms/sms/adapter.py": (1, "user_channel"),
    "plugins/platforms/teams/adapter.py": (1, "user_channel"),
    "plugins/platforms/teams/summary_writer.py": (1, "user_channel"),
    "plugins/platforms/telegram/telegram_network.py": (1, "user_channel"),
    "plugins/platforms/wecom/adapter.py": (2, "user_channel"),
    "plugins/platforms/wecom/callback_adapter.py": (1, "user_channel"),
    "plugins/platforms/weixin/adapter.py": (1, "user_channel"),
    "plugins/platforms/whatsapp/adapter.py": (3, "user_channel"),
    "plugins/platforms/whatsapp_cloud/adapter.py": (1, "user_channel"),
    "plugins/swarm_map_policy/__init__.py": (3, "no_content"),
    "plugins/teams_pipeline/pipeline.py": (1, "tool"),
    "plugins/video_gen/openrouter/__init__.py": (2, "tool"),
    "plugins/video_gen/xai/__init__.py": (1, "tool"),
    "plugins/web/_common.py": (1, "tool"),
    "plugins/web/firecrawl/provider.py": (1, "tool"),
    "plugins/web/keenable/provider.py": (2, "tool"),
    "plugins/web/keyless_mcp.py": (1, "tool"),
    "plugins/web/perplexity/provider.py": (3, "tool"),
    "plugins/web/tavily/provider.py": (1, "tool"),
    "plugins/web/xai/provider.py": (1, "tool"),
    "run_agent.py": (1, "model_client"),
    "tools/browser_cdp_tool.py": (1, "tool"),
    "tools/browser_launcher.py": (1, "tool"),
    "tools/browser_session_cdp.py": (1, "tool"),
    "tools/browser_supervisor.py": (1, "tool"),
    "tools/cronjob_tools.py": (1, "tool"),
    "tools/discord_tool.py": (1, "tool"),
    "tools/environments/managed_modal.py": (1, "tool"),
    "tools/kanban_tools.py": (1, "tool"),
    "tools/mcp_oauth_device.py": (1, "no_content"),
    "tools/mcp_oauth_manager.py": (1, "no_content"),
    "tools/microsoft_graph_auth.py": (1, "no_content"),
    "tools/microsoft_graph_client.py": (1, "tool"),
    "tools/osv_check.py": (1, "no_content"),
    "tools/schema_sanitizer.py": (2, "no_content"),
    "tools/send_message_senders.py": (1, "tool"),
    "tools/skills_hub.py": (1, "tool"),
    "tools/skills_hub_github.py": (1, "tool"),
    "tools/transcription_cloud.py": (2, "source"),
    "tools/tts_elevenlabs_voices.py": (1, "tool"),
    "tools/tts_streaming.py": (3, "tool"),
    "tools/tts_tool_providers.py": (1, "tool"),
    "tools/url_safety.py": (2, "tool"),
    "tools/voice_live.py": (1, "tool"),
    "tools/x_search_tool.py": (1, "tool"),
    "trajectory_compressor.py": (2, "offline"),
    "tui_gateway/hosted_room_peer_deadline.py": (2, "no_content"),
    "tui_gateway/methods_browser.py": (1, "no_content"),
}
