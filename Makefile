.PHONY: app assets format lint reachy-bridge setup test

REACHY_DAEMON_URL ?= http://reachy-mini.local:8000/
REACHY_BRIDGE_ARGS ?=

setup:
	uv sync
	npm install

assets:
	node -e "require('fs').mkdirSync('build/web/static',{recursive:true})"
	npm exec -- tailwindcss -i src/pokerbot_3000/web/static/input.css -o build/web/static/styles.css --minify
	npm exec -- esbuild src/pokerbot_3000/web/static/app.js src/pokerbot_3000/web/static/eliza.js --bundle --minify --format=esm --outdir=build/web/static

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ty check

format:
	uv run ruff check --fix .
	uv run ruff format .

app: assets
	uv run python -m pokerbot_3000 --server-name 0.0.0.0 --no-browser

reachy-bridge:
	uv run --group reachy pokerbot-reachy-bridge --reachy-daemon-url "$(REACHY_DAEMON_URL)" $(REACHY_BRIDGE_ARGS)
