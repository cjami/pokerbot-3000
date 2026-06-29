# Pokerbot 3000

PokerBot 3000 is a live multimodal poker experience powered by Gemma 4 31B via Cerebras, starring Reachy Mini and friendbot Eliza.

## Tech Stack

- Python 3.12+ with FastAPI, Jinja2, Pydantic, and Uvicorn.
- Frontend assets built with Tailwind CSS 4 and esbuild.
- Development tooling with uv, Ruff, ty, pytest, and pre-commit.

## Prerequisites

- `uv` for Python environment management: https://docs.astral.sh/uv/getting-started/installation/
- Node.js and npm for building browser assets: https://nodejs.org/

## Quick Start

```shell
uv sync
npm install
make assets
uv run pokerbot-3000
```

The local app defaults to `http://127.0.0.1:8000/`.

## Development

Create a local `.env` file for Cerebras access:

```shell
CEREBRAS_API_KEY=your_cerebras_api_key_here
CEREBRAS_MODEL=gemma-4-31b
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_ORCHESTRATOR_VOICE_ID=your_orchestrator_voice_id_here
ELEVENLABS_ELIZA_VOICE_ID=your_eliza_voice_id_here
ELEVENLABS_REACHY_VOICE_ID=your_reachy_voice_id_here
ELEVENLABS_MODEL=eleven_flash_v2_5
ELEVENLABS_ORCHESTRATOR_SPEED=0.82
ELEVENLABS_ELIZA_SPEED=0.92
ELEVENLABS_REACHY_SPEED=0.92
ELEVENLABS_STT_MODEL=scribe_v1
ELEVENLABS_STT_LANGUAGE=en
POKERBOT_VAD_RMS_THRESHOLD=0.012
POKERBOT_VAD_SILENCE_MS=650
POKERBOT_VAD_MIN_PHRASE_MS=220
POKERBOT_VAD_MAX_PHRASE_MS=8000
```

The dashboard uses the browser camera API for public-board frames, so OBS Virtual Camera and other browser-visible devices can be selected directly in the app.
Human voice input uses the browser microphone selector and streams 16 kHz mono PCM to the server-side energy VAD + ElevenLabs Scribe pipeline.
Eliza's thin client is available at `http://127.0.0.1:8000/clients/eliza`.
For another machine on the local network, start the app with `make app` and open `http://<host-ip>:8000/clients/eliza`.
Chrome and Edge block camera access on insecure LAN origins by default; on the Eliza machine, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<host-ip>:8000`, enable the flag, and relaunch the browser.
Reachy Mini can be connected with `make reachy-bridge`. By default this uses `http://reachy-mini.local:8000/`; override it with `make reachy-bridge REACHY_DAEMON_URL=http://10.0.0.39:8000/`.

Check model access without starting the web server:

```shell
uv run pokerbot-3000 --check-llm
```

```shell
make setup
make assets
make test
make lint
make format
make app
make reachy-bridge
```

Without Make:

```shell
uv run pytest
uv run ruff check .
uv run ty check
npm exec -- tailwindcss -i src/pokerbot_3000/web/static/input.css -o build/web/static/styles.css --minify
npm exec -- esbuild src/pokerbot_3000/web/static/app.js --bundle --minify --format=esm --outfile=build/web/static/app.js
```
