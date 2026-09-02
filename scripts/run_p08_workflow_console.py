"""Compatibility wrapper for the maintained combined-console package entry point."""

from __future__ import annotations

from spatial_mapping_phase2.console_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
