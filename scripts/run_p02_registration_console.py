"""Run the localhost P02 floor-plan and camera mounting-reference registration console."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from spatial_mapping_phase2.p02_registration_web import create_p02_registration_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Local mutable registration workspace, normally below the D: artifact root",
    )
    parser.add_argument(
        "--secret-file",
        type=Path,
        default=Path(".env"),
        help="Ignored local environment file used for RTSP endpoints",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("P02 registration console may bind only to localhost")
    app = create_p02_registration_app(arguments.workspace, arguments.secret_file)
    uvicorn.run(app, host=arguments.host, port=arguments.port, access_log=False)


if __name__ == "__main__":
    main()
