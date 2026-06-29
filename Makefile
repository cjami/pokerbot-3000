.PHONY: app assets format lint setup test

setup:
	uv sync
	npm install

assets:
	node -e "require('fs').mkdirSync('build/web/static',{recursive:true})"
	npm exec -- tailwindcss -i src/pokerbot_3000/web/static/input.css -o build/web/static/styles.css --minify
	npm exec -- esbuild src/pokerbot_3000/web/static/app.js --bundle --minify --format=esm --outfile=build/web/static/app.js

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ty check

format:
	uv run ruff check --fix .
	uv run ruff format .

app: assets
	uv run python -m pokerbot_3000 --no-browser
