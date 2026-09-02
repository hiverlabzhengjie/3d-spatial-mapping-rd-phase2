"""Source-checkout compatibility wrapper for the packaged XR02 worker."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_mapping_phase2.xr02_worker_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
