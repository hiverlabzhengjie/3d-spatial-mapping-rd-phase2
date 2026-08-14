"""Run the localhost-only P03 capture console."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from p03_capture_cli import _environment, build_service, build_temporal_service

from spatial_mapping_phase2.p03_capture_web import create_p03_capture_app


def main() -> None:
    environment = _environment(Path(".env"))
    service = build_service(environment)
    try:
        uvicorn.run(
            create_p03_capture_app(
                service, temporal_factory=lambda: build_temporal_service(environment)
            ),
            host="127.0.0.1",
            port=8033,
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
