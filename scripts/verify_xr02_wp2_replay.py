"""Verify retained XR02 WP2 replay evidence without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_mapping_phase2.xr02_replay_verification import verify_wp2_replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = (
        json.dumps(verify_wp2_replay(args.manifest.resolve()), indent=2, sort_keys=True) + "\n"
    )
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError("verification output already exists")
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
