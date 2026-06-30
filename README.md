# PokerBot 3000

PokerBot 3000 is a live multimodal poker experience powered by Gemma 4 31B via Cerebras, starring Reachy Mini and friendbot Eliza.

## What It Does

- Runs a three-seat no-limit Hold'em table with a human, Reachy, and Eliza.
- Uses cameras to read the public board, private agent cards, and showdown reveals.
- Takes clear human poker actions and table talk from a microphone.
- Streams live game state, events, voice, and client status to the operator dashboard.

## Live Session Flow

1. Start the session from the dashboard.
2. Select camera and microphone devices in the browser.
3. Let the dashboard capture board cards when the game asks for them.
4. Let Eliza and Reachy submit private-card views from their thin clients or bridge.
5. Speak human actions such as `call`, `check`, `raise to 100`, or `all in`.
6. Watch agent decisions, speech, presentation events, showdown resolution, and the next hand.

## How It Works

- Gemma 4 acts as a VLM for live card recognition, reading public-board frames, private-card views, and showdown reveals.
- The poker engine advances until it needs outside input, then resumes when a camera, voice, agent, or presentation event arrives.
- Shared game state and event streams keep the dashboard, Eliza, and Reachy coordinated while private cards stay hidden from the other players.
- Gemma 4 also handles table talk and chooses agent actions; ElevenLabs transcribes the human and voices the agents.
- Presentation events carry emotion and gesture cues, so Eliza can show an emoji face and Reachy can move with the moment.

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

Create a local `.env` file for Cerebras and ElevenLabs access:

```shell
CEREBRAS_API_KEY=your_cerebras_api_key_here
CEREBRAS_MODEL=gemma-4-31b
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_ORCHESTRATOR_VOICE_ID=your_orchestrator_voice_id_here
ELEVENLABS_ELIZA_VOICE_ID=your_eliza_voice_id_here
ELEVENLABS_REACHY_VOICE_ID=your_reachy_voice_id_here
ELEVENLABS_MODEL=eleven_flash_v2_5
ELEVENLABS_OUTPUT_FORMAT=mp3_22050_32
ELEVENLABS_ORCHESTRATOR_SPEED=0.82
ELEVENLABS_ELIZA_SPEED=0.92
ELEVENLABS_REACHY_SPEED=0.92
ELEVENLABS_STT_MODEL=scribe_v2
ELEVENLABS_STT_LANGUAGE=en
ELEVENLABS_STT_KEYTERMS=fold,check,call,raise,raise to,bet,all in,one hundred,two hundred,chips,Reachy,Eliza
POKERBOT_VAD_RMS_THRESHOLD=0.012
POKERBOT_VAD_SILENCE_MS=650
POKERBOT_VAD_MIN_PHRASE_MS=220
POKERBOT_VAD_MAX_PHRASE_MS=8000
```

The dashboard uses browser-visible cameras for public-board frames, including OBS Virtual Camera.
Human voice input uses the browser microphone selector and streams 16 kHz mono PCM to the server-side VAD and ElevenLabs Scribe pipeline.
Eliza's thin client is available at `http://127.0.0.1:8000/clients/eliza`.
For another machine on the local network, start the app with `make app` and open `http://<host-ip>:8000/clients/eliza`.
Chrome and Edge block camera access on insecure LAN origins by default; on the Eliza machine, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<host-ip>:8000`, enable the flag, and relaunch the browser.
Reachy Mini can be connected with `make reachy-bridge`.

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
npm exec -- esbuild src/pokerbot_3000/web/static/app.js src/pokerbot_3000/web/static/eliza.js --bundle --minify --format=esm --outdir=build/web/static
```
