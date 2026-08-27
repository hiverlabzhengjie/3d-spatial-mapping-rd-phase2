from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_mapping_phase2.xr02_association import AssociationConfig, office_topology
from spatial_mapping_phase2.xr02_global_domain import GlobalTrackState, SignalProfile
from spatial_mapping_phase2.xr02_global_journal import (
    GlobalAssociationJournal,
    verify_global_journal,
)
from spatial_mapping_phase2.xr02_replay_eval import (
    ReplayPartition,
    ScenarioRun,
    aggregate_runs,
    load_replay_partition,
    run_scenario,
)
from spatial_mapping_phase2.xr02_wp2_input import load_wp2_journal_read_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic XR02 WP3 association replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--wp2-botsort-journal", type=Path, required=True)
    parser.add_argument("--wp2-deepocsort-journal", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError("output directory must not already exist")
    args.output.mkdir(parents=True)
    config = AssociationConfig()
    topology = office_topology()
    wp2_audits = [
        load_wp2_journal_read_only(args.wp2_botsort_journal, "botsort-fixed-v1"),
        load_wp2_journal_read_only(args.wp2_deepocsort_journal, "deepocsort-fixed-v1"),
    ]
    partitions = [
        load_replay_partition(args.tuning),
        load_replay_partition(args.heldout),
    ]
    if partitions[0].partition != "tuning" or partitions[1].partition != "heldout":
        raise RuntimeError("fixture roles changed")

    scorecard: dict[str, dict[str, dict[str, int | float | str]]] = {}
    all_runs: dict[tuple[str, SignalProfile], tuple[ScenarioRun, ...]] = {}
    journal_evidence: list[dict[str, object]] = []
    determinism: list[dict[str, object]] = []
    for partition in partitions:
        scorecard[partition.partition] = {}
        for profile in SignalProfile:
            runs = tuple(
                run_scenario(scenario, profile, topology, config)
                for scenario in partition.scenarios
            )
            second_pass = tuple(
                run_scenario(scenario, profile, topology, config)
                for scenario in partition.scenarios
            )
            first_signature = _run_signature(runs)
            second_signature = _run_signature(second_pass)
            if first_signature != second_signature:
                raise RuntimeError("association replay is not deterministic")
            all_runs[(partition.partition, profile)] = runs
            aggregate = aggregate_runs(runs)
            aggregate["fixture_sha256"] = partition.source_sha256
            scorecard[partition.partition][profile.value] = aggregate
            journal_path = args.output / f"{partition.partition}-{profile.value}.jsonl"
            journal = GlobalAssociationJournal(journal_path)
            for run in runs:
                for result in run.results:
                    journal.append(result)
            verified = verify_global_journal(journal_path)
            journal_evidence.append(
                {
                    "partition": partition.partition,
                    "profile": profile.value,
                    "path": str(journal_path.resolve()),
                    "bytes": journal_path.stat().st_size,
                    "sha256": _file_sha256(journal_path),
                    "records": verified.records,
                    "final_sha256": verified.final_sha256,
                    "decision_signature_sha256": verified.signature_sha256,
                }
            )
            determinism.append(
                {
                    "partition": partition.partition,
                    "profile": profile.value,
                    "first_signature_sha256": first_signature,
                    "second_signature_sha256": second_signature,
                    "matched": True,
                }
            )

    gate = _evaluate_gate(scorecard["heldout"])
    rerun_path = args.output / "wp3-heldout-floor-trajectories.rrd"
    _write_rerun(
        rerun_path,
        partitions[1],
        all_runs[("heldout", SignalProfile.COMBINED)],
    )
    scorecard_path = args.output / "scorecard.json"
    _write_json(scorecard_path, scorecard)
    manifest: dict[str, object] = {
        "schema": "xr02.wp3.replay_manifest.v1",
        "scope": "XR02 WP3 cached replay association only; no live RTSP or operator service",
        "command_argv": sys.argv,
        "association_config": asdict(config),
        "topology": {
            "overlap_edges": topology.overlap_edges,
            "transition_edges": topology.transition_edges,
        },
        "input_authority": [audit.as_dict() for audit in wp2_audits],
        "fixture_partitions": [
            {
                "partition": partition.partition,
                "path": partition.source_path,
                "bytes": Path(partition.source_path).stat().st_size,
                "sha256": partition.source_sha256,
                "scenario_count": len(partition.scenarios),
            }
            for partition in partitions
        ],
        "profiles": [profile.value for profile in SignalProfile],
        "determinism": determinism,
        "journals": journal_evidence,
        "scorecard": _identity(scorecard_path),
        "rerun": _identity(rerun_path),
        "gate": gate,
        "packages": _package_versions(),
        "claims": {
            "standard_idf1_or_hota": False,
            "production_or_live_performance": False,
            "survey_accuracy": False,
            "heldout_used_for_threshold_tuning": False,
            "labels_exposed_to_associator": False,
        },
    }
    manifest_path = args.output / "wp3-replay-manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(_identity(manifest_path), indent=2))
    return 0


def _run_signature(runs: tuple[ScenarioRun, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            [run.decision_signature_sha256 for run in runs],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _evaluate_gate(
    heldout: dict[str, dict[str, int | float | str]],
) -> dict[str, object]:
    combined = heldout[SignalProfile.COMBINED.value]
    baselines = [
        heldout[SignalProfile.SPATIAL_ONLY.value],
        heldout[SignalProfile.APPEARANCE_ONLY.value],
    ]
    strongest_f1 = max(float(item["pairwise_identity_f1"]) for item in baselines)
    lowest_false_merges = min(int(item["false_merge_pairs"]) for item in baselines)
    passed = (
        float(combined["pairwise_identity_f1"]) > strongest_f1
        and int(combined["false_merge_pairs"]) <= lowest_false_merges
        and int(combined["ambiguous_observations"]) > 0
    )
    return {
        "passed": passed,
        "metric": "macro pairwise identity F1 on bounded heldout fixtures",
        "combined": combined["pairwise_identity_f1"],
        "strongest_simple_baseline": strongest_f1,
        "combined_false_merge_pairs": combined["false_merge_pairs"],
        "lowest_simple_baseline_false_merge_pairs": lowest_false_merges,
        "combined_ambiguous_observations": combined["ambiguous_observations"],
    }


def _write_rerun(
    path: Path,
    partition: ReplayPartition,
    runs: tuple[ScenarioRun, ...],
) -> None:
    import rerun as rr

    recording: Any = rr.new_recording("xr02-wp3-heldout-trajectories", recording_id=path.stem)
    recording.save(str(path))
    recording.log(
        "xr02/wp3/status",
        rr.TextDocument(
            "Bounded labelled cached-replay evidence for the combined WP3 policy. "
            "It is not live, survey-grade, MOTChallenge IDF1/HOTA, or a production claim.",
            media_type="text/markdown",
        ),
        static=True,
    )
    recording.log(
        "xr02/wp3/world/floor_z0",
        rr.LineStrips3D(
            [
                [
                    [-2.0, -1.0, 0.0],
                    [10.0, -1.0, 0.0],
                    [10.0, 10.0, 0.0],
                    [-2.0, 10.0, 0.0],
                    [-2.0, -1.0, 0.0],
                ]
            ],
            colors=[[190, 190, 190]],
            radii=0.02,
        ),
        static=True,
    )
    for scenario_index, (scenario, run) in enumerate(zip(partition.scenarios, runs, strict=True)):
        paths: dict[str, list[list[float]]] = {}
        for tick, result in zip(scenario.ticks, run.results, strict=True):
            recording.set_time_sequence("replay_tick", scenario_index * 100 + tick.tick_index)
            assignments = {item.observation_id: item for item in result.assignments}
            ambiguous_points: list[list[float]] = []
            for labelled in tick.labelled_observations:
                observation = labelled.observation
                assignment = assignments[observation.observation_id]
                point = [*observation.world_xy_metres, 0.08]
                if assignment.state is GlobalTrackState.AMBIGUOUS:
                    ambiguous_points.append(point)
                elif assignment.global_track_id is not None:
                    paths.setdefault(assignment.global_track_id, []).append(point)
            if ambiguous_points:
                recording.log(
                    f"xr02/wp3/scenarios/{scenario.scenario_id}/ambiguous",
                    rr.Points3D(
                        ambiguous_points,
                        colors=[[255, 80, 80]] * len(ambiguous_points),
                        radii=0.12,
                    ),
                )
        for global_id, points in sorted(paths.items()):
            if len(points) >= 2:
                recording.log(
                    f"xr02/wp3/scenarios/{scenario.scenario_id}/trajectories/{global_id}",
                    rr.LineStrips3D([points], radii=0.05),
                    static=True,
                )
            recording.log(
                f"xr02/wp3/scenarios/{scenario.scenario_id}/points/{global_id}",
                rr.Points3D(points, labels=[global_id] * len(points), radii=0.09),
                static=True,
            )
    recording.flush(blocking=True)
    recording.disconnect()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "rerun-sdk", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
