"""Run PokerBot 3000 as a module."""

from __future__ import annotations

import sys

from pokerbot_3000.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
