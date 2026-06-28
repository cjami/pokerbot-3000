# POKERBOT 3000

## Project Description

PokerBot 3000 is a live multimodal poker experience powered by Gemma 4 31B via Cerebras, starring Reachy Mini and friendbot Eliza.

## Project Structure

- `src/pokerbot_3000/app/` contains FastAPI application wiring.
- `src/pokerbot_3000/web/templates/` contains Jinja2 templates.
- `src/pokerbot_3000/web/static/` contains source frontend assets.
- `build/web/static/` contains generated frontend assets from `make assets`.
- `tests/` contains pytest tests.
- `pyproject.toml` defines dependencies, Ruff linting, pytest settings, package metadata, and console scripts.
- `.pre-commit-config.yaml` defines local hooks that run formatting, linting, type checking, and tests.
- `package.json` and `package-lock.json` define frontend build dependencies.
- `Makefile` provides common development commands.

## Development Workflow

- Always use modern Python practices for Python 3.12+.
- Use TDD where appropriate to keep a considered design and protect key behaviours.
- Do not test content, configurations or anything that is likely to change design.
- Tidy-up and refactor after changes - make sure to follow SOLID principles.
- Run `make lint` and `make test` after changes.
- Use `uv run` for Python commands.
- Do not commit changes unless instructed.
- Do not start the web server unless instructed.

## Comments

- Keep all comments concise, clear, and suitable for inclusion in final production.
- Only use comments when the intent cannot be explained through thoughtful naming or code structure.
