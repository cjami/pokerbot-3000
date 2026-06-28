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
ELEVENLABS_MODEL=eleven_flash_v2_5
ELEVENLABS_ORCHESTRATOR_SPEED=0.82
```

The dashboard uses the browser camera API for public-board frames, so OBS Virtual Camera and other browser-visible devices can be selected directly in the app.

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
```

Without Make:

```shell
uv run pytest
uv run ruff check .
uv run ty check
npm exec -- tailwindcss -i src/pokerbot_3000/web/static/input.css -o build/web/static/styles.css --minify
npm exec -- esbuild src/pokerbot_3000/web/static/app.js --bundle --minify --format=esm --outfile=build/web/static/app.js
```
