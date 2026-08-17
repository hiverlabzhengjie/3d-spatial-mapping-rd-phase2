"""Run the localhost P04 Camera 3 calibration correspondence console."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from spatial_mapping_phase2.p04_calibration_service import (
    P03PreviewCandidateCapturer,
    load_p04_camera3_endpoint,
)
from spatial_mapping_phase2.p04_calibration_web import create_p04_calibration_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Local P04 workspace, normally below the D: artifact root",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8044)
    parser.add_argument(
        "--secret-file",
        type=Path,
        default=Path(".env"),
        help="Ignored local environment file containing the Camera 3 RTSP endpoint",
    )
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("P04 calibration console may bind only to localhost")
    endpoint = load_p04_camera3_endpoint(arguments.secret_file)
    app = create_p04_calibration_app(
        arguments.workspace,
        candidate_capturer=P03PreviewCandidateCapturer(endpoint),
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port, access_log=False)


if __name__ == "__main__":
    main()
