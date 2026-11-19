# Deployment overlays

Optional files for running Hermes in Docker on a single host.
They sit beside the main compose file and change no upstream code.

## LiteLLM proxy sidecar

Sends model requests through AWS Bedrock (Claude) and Google Vertex AI (Gemini), so agents don't need provider API keys.

1. Set your Google Cloud project in `litellm-config.yaml` (`vertex_project`).
2. Put these in `~/.hermes/.env`:
   - `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for Bedrock
   - `GOOGLE_CLOUD_API_KEY` for Vertex AI
3. Start the stack with the overlay:

   ```bash
   docker compose -f docker-compose.yml -f docker/deploy/docker-compose.litellm.yml up -d
   ```

4. Point each agent's model at the proxy in `config.yaml`:

   ```yaml
   model:
     provider: custom
     base_url: http://litellm-proxy:4000/v1
   ```

## Signal voice memo paths

When Hermes runs in Docker and signal-cli runs on the host (through `host.docker.internal`), signal-cli can't open attachment paths inside the container, such as `/opt/data/cache/audio/tts.ogg`.

`signal-host-path.patch` rewrites container paths (`/opt/data/...`) to host paths (`~/.hermes/...`) before handing them to signal-cli.
Set this in the agent's environment:

```
SIGNAL_HOST_DATA_DIR=~/.hermes
```

The patch was written against `gateway/platforms/signal.py`.
The Signal adapter now lives in `plugins/platforms/signal/adapter.py`, so `git apply docker/deploy/signal-host-path.patch` will not apply cleanly. Port the change by hand.

## Telegram proxy

Upstream already reads `TELEGRAM_PROXY`. No patch is needed. Set it if your network requires a proxy:

```
TELEGRAM_PROXY=socks5://host:port
```
