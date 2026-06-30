"""Command-line entry point for PokerBot 3000."""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from ipaddress import ip_address
from typing import Final

import uvicorn

from pokerbot_3000 import __version__
from pokerbot_3000.llm import CerebrasClientError, CerebrasConfig, CerebrasConfigurationError, CerebrasLlmClient

DEFAULT_SERVER_NAME: Final = "127.0.0.1"
DEFAULT_SERVER_PORT: Final = 8000
APP_IMPORT_STRING: Final = "pokerbot_3000.app.server:create_app"


def main(argv: list[str] | None = None) -> None:
    """Launch the local PokerBot 3000 web application."""
    args = _parse_args(argv)
    if args.check_llm:
        asyncio.run(_check_llm())
        return

    if not args.no_browser:
        webbrowser.open(f"http://{_display_host(args.server_name)}:{args.server_port}/")

    uvicorn.run(
        APP_IMPORT_STRING,
        factory=True,
        host=args.server_name,
        port=args.server_port,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pokerbot-3000", description="Launch the PokerBot 3000 local web app.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="Check Cerebras Gemma access without starting the app.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser.")
    parser.add_argument("--server-name", default=DEFAULT_SERVER_NAME, help="Host/interface for the local web server.")
    parser.add_argument("--server-port", default=DEFAULT_SERVER_PORT, type=int, help="Port for the local web server.")
    return parser.parse_args(argv)


def _display_host(server_name: str) -> str:
    try:
        return DEFAULT_SERVER_NAME if ip_address(server_name).is_unspecified else server_name
    except ValueError:
        return server_name


async def _check_llm() -> None:
    try:
        config = CerebrasConfig.from_env()
        client = CerebrasLlmClient(config)
        result = await client.check_access()
    except (CerebrasClientError, CerebrasConfigurationError) as exc:
        msg = f"Cerebras check failed: {exc}"
        raise SystemExit(msg) from None

    if result.ok:
        print(f"Cerebras access OK for {result.model}: {result.reply}")
        return

    msg = f"Cerebras model check failed for {result.model}. Listed={result.model_listed}; reply={result.reply!r}"
    raise SystemExit(msg)


if __name__ == "__main__":
    main(sys.argv[1:])
