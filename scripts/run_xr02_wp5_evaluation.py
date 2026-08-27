"""Run the deterministic XR02 WP5 metadata scale evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spatial_mapping_phase2.xr02_scale_evaluation import (  # noqa: E402
    ScaleEvaluationConfig,
    run_scale_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XR02 WP5 metadata scale evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--cameras-per-scene", type=int, default=10)
    parser.add_argument("--people-per-scene", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=160)
    args = parser.parse_args()
    config = ScaleEvaluationConfig(
        scene_count=args.scenes,
        cameras_per_scene=args.cameras_per_scene,
        people_per_scene=args.people_per_scene,
        ticks=args.ticks,
    )
    result = run_scale_evaluation(config)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
