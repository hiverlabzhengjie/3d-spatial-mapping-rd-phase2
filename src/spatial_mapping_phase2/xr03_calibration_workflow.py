"""Intrinsic-first, per-camera D034 calibration workflow for XR03.

The operator-facing result is a complete ``T_world_from_camera``. Translation remains the
reviewed fixed optical centre required by D034; only orientation is solved from linked points.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatial_mapping_phase2.p04_intrinsic_fleet import CameraIntrinsicEstimate
from spatial_mapping_phase2.p04_pose_domain import CameraIntrinsics, PoseSolveError
from spatial_mapping_phase2.p05_fixed_center_orientation import (
    D034OrientationSolution,
    D034SolverConfig,
    evaluate_d034_frozen_validation,
    solve_d034_orientation,
)
from spatial_mapping_phase2.p05_pose_candidates import D033_INTRINSICS
from spatial_mapping_phase2.p06_da3_evaluation import T_camera_from_world
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy
from spatial_mapping_phase2.xr03_intrinsic_policy import build_grouped_intrinsic_candidates


class CalibrationWorkflowError(ValueError):
    """Raised when the intrinsic/calibration workflow cannot advance safely."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "byte_count": path.stat().st_size,
    }


class SceneCalibrationRepository:
    """Append-only intrinsic batches, attempts and operator decisions for one scene."""

    def __init__(
        self,
        path: Path,
        project_id: str,
        scene_id: str,
        camera_ids: Sequence[str],
    ) -> None:
        self.path = path.resolve()
        self.project_id = project_id
        self.scene_id = scene_id
        self.camera_ids = tuple(camera_ids)
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise CalibrationWorkflowError("calibration repository requires a unique scene roster")
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS intrinsic_batches (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calibration_attempts (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    intrinsic_batch_sha256 TEXT NOT NULL,
                    source_revision INTEGER NOT NULL,
                    payload_sha256 TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS calibration_attempt_camera_sequence
                    ON calibration_attempts(camera_id, sequence DESC);
                CREATE TABLE IF NOT EXISTS calibration_decisions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    attempt_sha256 TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    UNIQUE(camera_id, attempt_sha256, decision)
                );
                """
            )
            expected = {
                "schema_version": "xr03-scene-calibration-store-v1",
                "project_id": self.project_id,
                "scene_id": self.scene_id,
                "camera_ids_json": _canonical_json({"camera_ids": list(self.camera_ids)}),
            }
            existing = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            if existing:
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise CalibrationWorkflowError(
                        "calibration store metadata differs from the configured scene roster"
                    )
            else:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
                )

    def append_intrinsic_batch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        digest = _payload_sha256(value)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO intrinsic_batches(
                    created_at_utc, payload_sha256, payload_json
                ) VALUES (?, ?, ?)
                """,
                (datetime.now(UTC).isoformat(), digest, _canonical_json(value)),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('active_batch_sha256', ?)",
                (digest,),
            )
        return {**value, "payload_sha256": digest}

    def active_intrinsic_batch(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json, payload_sha256
                FROM intrinsic_batches
                WHERE payload_sha256 = (
                    SELECT value FROM metadata WHERE key = 'active_batch_sha256'
                )
                """
            ).fetchone()
        if row is None:
            return None
        value = _json_object(str(row["payload_json"]))
        return {**value, "payload_sha256": str(row["payload_sha256"])}

    def append_attempt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        camera_id = _required_string(value, "camera_id")
        if camera_id not in self.camera_ids:
            raise CalibrationWorkflowError(
                "calibration attempt camera is outside the scene roster"
            )
        batch_sha256 = _required_sha256(value, "intrinsic_batch_sha256")
        source_revision = value.get("source_revision")
        if not isinstance(source_revision, int) or isinstance(source_revision, bool):
            raise CalibrationWorkflowError("calibration source revision must be an integer")
        digest = _payload_sha256(value)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO calibration_attempts(
                    created_at_utc, camera_id, intrinsic_batch_sha256, source_revision,
                    payload_sha256, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    camera_id,
                    batch_sha256,
                    source_revision,
                    digest,
                    _canonical_json(value),
                ),
            )
        return {**value, "payload_sha256": digest}

    def latest_attempt(self, camera_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json, payload_sha256
                FROM calibration_attempts
                WHERE camera_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (camera_id,),
            ).fetchone()
        if row is None:
            return None
        value = _json_object(str(row["payload_json"]))
        return {**value, "payload_sha256": str(row["payload_sha256"])}

    def decide(
        self,
        camera_id: str,
        attempt_sha256: str,
        decision: str,
        reason: str | None,
    ) -> dict[str, Any]:
        if decision not in {"strict-visual-review", "operator-override"}:
            raise CalibrationWorkflowError("unsupported calibration decision")
        latest = self.latest_attempt(camera_id)
        if latest is None or latest["payload_sha256"] != attempt_sha256:
            raise CalibrationWorkflowError("calibration decision must target the latest attempt")
        normalized_reason = None if reason is None else reason.strip()
        if decision == "operator-override" and not normalized_reason:
            raise CalibrationWorkflowError("operator override requires a non-blank reason")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO calibration_decisions(
                    created_at_utc, camera_id, attempt_sha256, decision, reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    camera_id,
                    attempt_sha256,
                    decision,
                    normalized_reason,
                ),
            )
        return {
            "camera_id": camera_id,
            "attempt_sha256": attempt_sha256,
            "decision": decision,
            "reason": normalized_reason,
        }

    def decision(self, camera_id: str, attempt_sha256: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT created_at_utc, decision, reason
                FROM calibration_decisions
                WHERE camera_id = ? AND attempt_sha256 = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (camera_id, attempt_sha256),
            ).fetchone()
        if row is None:
            return None
        return {
            "created_at_utc": str(row["created_at_utc"]),
            "decision": str(row["decision"]),
            "reason": None if row["reason"] is None else str(row["reason"]),
        }

    def history(self) -> dict[str, Any]:
        """Return a concise read-only timeline for Scene History & Storage."""

        with self._connect() as connection:
            batches = connection.execute(
                """
                SELECT created_at_utc, payload_sha256, payload_json
                FROM intrinsic_batches ORDER BY sequence DESC
                """
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT created_at_utc, camera_id, payload_sha256, payload_json
                FROM calibration_attempts ORDER BY sequence DESC
                """
            ).fetchall()
            decisions = connection.execute(
                """
                SELECT created_at_utc, camera_id, attempt_sha256, decision, reason
                FROM calibration_decisions ORDER BY sequence DESC
                """
            ).fetchall()
        return {
            "schema_version": "xr03-calibration-history-v1",
            "database_path": str(self.path),
            "intrinsic_batches": [
                {
                    "created_at_utc": str(row["created_at_utc"]),
                    "payload_sha256": str(row["payload_sha256"]),
                    "camera_policy_sha256": _json_object(str(row["payload_json"])).get(
                        "camera_policy_sha256"
                    ),
                }
                for row in batches
            ],
            "attempts": [
                {
                    "created_at_utc": str(row["created_at_utc"]),
                    "camera_id": str(row["camera_id"]),
                    "payload_sha256": str(row["payload_sha256"]),
                    "automated_status": _json_object(str(row["payload_json"])).get(
                        "automated_status"
                    ),
                    "selected_intrinsic_label": _json_object(str(row["payload_json"])).get(
                        "selected_intrinsic_label"
                    ),
                }
                for row in attempts
            ],
            "decisions": [
                {
                    "created_at_utc": str(row["created_at_utc"]),
                    "camera_id": str(row["camera_id"]),
                    "attempt_sha256": str(row["attempt_sha256"]),
                    "decision": str(row["decision"]),
                    "reason": None if row["reason"] is None else str(row["reason"]),
                }
                for row in decisions
            ],
        }


class IntegratedCalibrationWorkflowAdapter:
    """Use P04 linked points and the frozen D034 gates in the integrated console."""

    def __init__(
        self,
        repository: SceneCalibrationRepository,
        calibration_services: Mapping[str, Any],
        intrinsic_evidence_path: Path,
        facility_export_path: Path,
        output_root: Path,
    ) -> None:
        self.repository = repository
        self.calibration_services = dict(calibration_services)
        self.intrinsic_evidence_path = intrinsic_evidence_path.resolve()
        self.facility_export_path = facility_export_path.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def history(self) -> dict[str, Any]:
        return self.repository.history()

    def determine_intrinsics(self, policy: SceneCameraPolicy) -> dict[str, Any]:
        estimates, evidence = _load_intrinsic_estimates(self.intrinsic_evidence_path)
        grouped = build_grouped_intrinsic_candidates(estimates, policy)
        by_camera = {estimate.camera_id: estimate for estimate in estimates}
        group_by_camera = {
            camera_id: group for group in policy.intrinsic_groups for camera_id in group.camera_ids
        }
        target_by_camera = {
            str(target["camera_id"]): target
            for group in grouped["groups"]
            for target in group["targets"]
        }
        assignments: list[dict[str, Any]] = []
        for camera_id in policy.camera_ids:
            estimate = by_camera[camera_id]
            initial = _initial_intrinsic_candidate(estimate, evidence)
            candidates = [initial, _estimate_candidate(estimate)]
            for item in target_by_camera[camera_id]["candidates"]:
                if not isinstance(item, dict):
                    raise CalibrationWorkflowError("group intrinsic candidate is malformed")
                candidates.append(_fleet_candidate(item))
            candidates = _deduplicate_candidates(candidates)
            group = group_by_camera[camera_id]
            assignments.append(
                {
                    "camera_id": camera_id,
                    "group_id": group.group_id,
                    "lens_model": group.lens_model,
                    "initial_assignment_label": candidates[0]["label"],
                    "eligible_candidates": candidates,
                }
            )
        payload = {
            "schema_version": "xr03-scene-intrinsic-assignment-v1",
            "project_id": policy.project_id,
            "scene_id": policy.scene_id,
            "camera_ids": list(policy.camera_ids),
            "camera_policy_sha256": policy.sha256,
            "intrinsic_policy_sha256": policy.intrinsic_policy_sha256,
            "intrinsic_evidence": _identity(self.intrinsic_evidence_path),
            "assignment_policy": (
                "initial camera-specific/default assignment plus group-eligible D036 challengers"
            ),
            "group_candidate_evidence": grouped,
            "assignments": assignments,
        }
        return self.repository.append_intrinsic_batch(payload)

    def status(self, policy: SceneCameraPolicy) -> dict[str, Any]:
        batch = self.repository.active_intrinsic_batch()
        evidence_identity = (batch or {}).get("intrinsic_evidence")
        evidence_current = bool(
            isinstance(evidence_identity, dict)
            and self.intrinsic_evidence_path.is_file()
            and evidence_identity.get("sha256") == _file_sha256(self.intrinsic_evidence_path)
        )
        facility_sha256 = (
            _file_sha256(self.facility_export_path)
            if self.facility_export_path.is_file()
            else None
        )
        batch_ready = bool(
            batch
            and _intrinsic_batch_matches_policy(batch, policy)
            and tuple(batch.get("camera_ids", ())) == policy.camera_ids
            and evidence_current
        )
        assignments = {
            str(item["camera_id"]): item
            for item in (batch or {}).get("assignments", [])
            if isinstance(item, dict) and "camera_id" in item
        }
        cameras: list[dict[str, Any]] = []
        issues: list[str] = []
        warnings: list[str] = []
        batch_sha256 = batch.get("payload_sha256") if isinstance(batch, dict) else None
        for camera_id in policy.camera_ids:
            service = self.calibration_services.get(camera_id)
            input_status = (
                service.calibration_readiness()
                if service is not None
                else {
                    "camera_id": camera_id,
                    "current_export_ready": False,
                    "calibrate_ready": False,
                    "reason": "No linked-point workspace is configured",
                }
            )
            attempt = self.repository.latest_attempt(camera_id)
            attempt_facility = (attempt or {}).get("facility_export")
            current_attempt = bool(
                batch_ready
                and attempt
                and attempt.get("intrinsic_batch_sha256") == batch_sha256
                and attempt.get("source_revision") == input_status.get("source_revision")
                and attempt.get("correspondence_export_sha256")
                == input_status.get("current_export_sha256")
                and isinstance(attempt_facility, dict)
                and attempt_facility.get("sha256") == facility_sha256
            )
            decision = (
                self.repository.decision(camera_id, str(attempt["payload_sha256"]))
                if current_attempt and attempt is not None
                else None
            )
            strict_ready = bool(
                decision
                and decision["decision"] == "strict-visual-review"
                and attempt
                and attempt.get("automated_status") == "accepted"
            )
            override_ready = bool(
                decision
                and decision["decision"] == "operator-override"
                and attempt
                and attempt.get("can_override") is True
            )
            ready = strict_ready or override_ready
            record: dict[str, Any] = {
                "camera_id": camera_id,
                "ready": ready,
                "readiness": (
                    "strict-ready"
                    if strict_ready
                    else "operator-accepted-with-warning"
                    if override_ready
                    else "pending"
                ),
                "warning": (
                    "Automated calibration gates failed; operator override is active."
                    if override_ready
                    else None
                ),
                "input": input_status,
                "assignment": assignments.get(camera_id),
                "attempt": attempt if current_attempt else None,
                "decision": decision,
                "calibrate_enabled": batch_ready and bool(input_status.get("calibrate_ready")),
            }
            if current_attempt and attempt is not None and attempt.get("pose") is not None:
                record.update(_camera_summary_from_attempt(attempt, ready, override_ready))
            cameras.append(record)
            if not ready:
                issues.append(f"{camera_id} calibration is not ready")
            if override_ready:
                warnings.append(f"{camera_id} uses an operator-accepted calibration override")
        if not batch_ready:
            issues.insert(
                0,
                "Determine intrinsic profiles for the active lens-group policy and "
                "current evidence",
            )
        return {
            "schema_version": "xr03-calibration-workflow-status-v1",
            "intrinsics_ready": batch_ready,
            "intrinsic_batch": batch if batch_ready else None,
            "cameras": cameras,
            "all_cameras_ready": batch_ready and all(camera["ready"] for camera in cameras),
            "issues": issues,
            "warnings": warnings,
        }

    def calibrate_camera(self, camera_id: str, policy: SceneCameraPolicy) -> dict[str, Any]:
        status = self.status(policy)
        batch = status["intrinsic_batch"]
        if not isinstance(batch, dict):
            raise CalibrationWorkflowError(
                "determine and assign scene intrinsics before calibrating a camera"
            )
        camera_status = next(
            (item for item in status["cameras"] if item["camera_id"] == camera_id), None
        )
        if camera_status is None:
            raise CalibrationWorkflowError("camera is outside the active scene roster")
        input_status = camera_status["input"]
        if not input_status.get("calibrate_ready"):
            raise CalibrationWorkflowError(
                str(input_status.get("reason") or "linked points are not ready")
            )
        existing = camera_status.get("attempt")
        if isinstance(existing, dict):
            return existing
        service = self.calibration_services[camera_id]
        correspondence_path = Path(str(input_status["current_export_path"]))
        correspondence = _read_json(correspondence_path)
        _, validation_seal = service.export_d034_validation_seal()
        assignment = next(item for item in batch["assignments"] if item["camera_id"] == camera_id)
        fixed_center = _fixed_center(self.facility_export_path, camera_id)
        solve_items = [
            item for item in correspondence.get("landmarks", []) if item.get("role") == "solve"
        ]
        validation_items = validation_seal.get("validation_landmarks")
        if (
            len(solve_items) != 4
            or not isinstance(validation_items, list)
            or len(validation_items) != 2
        ):
            raise CalibrationWorkflowError(
                "camera calibration requires exactly four solve and two validation points"
            )
        config = D034SolverConfig()
        candidate_results: list[dict[str, Any]] = []
        solved: list[
            tuple[tuple[float, float, float, float], dict[str, Any], D034OrientationSolution]
        ] = []
        solve_ids, solve_world, solve_image = _landmark_values(solve_items)
        for candidate in assignment["eligible_candidates"]:
            intrinsics = _intrinsics_from_candidate(candidate)
            result: dict[str, Any] = {
                "label": candidate["label"],
                "reason": candidate["reason"],
                "intrinsics": intrinsics.to_dict(),
            }
            try:
                orientation = solve_d034_orientation(
                    intrinsics,
                    fixed_center,
                    solve_ids,
                    solve_world,
                    solve_image,
                    config=config,
                )
            except PoseSolveError as error:
                result.update({"status": "rejected-solve", "diagnostic": str(error)})
            else:
                sorted_errors = sorted(orientation.solve_reprojection_errors_pixels)[:3]
                trimmed_rmse = math.sqrt(sum(value * value for value in sorted_errors) / 3)
                score = (
                    -float(len(orientation.inlier_indices)),
                    trimmed_rmse,
                    orientation.maximum_perturbation_rotation_degrees,
                    orientation.subset_spread_degrees,
                )
                result.update(
                    {
                        "status": "passed-solve-gates",
                        "orientation": orientation.to_dict(),
                        "rank_metrics": {
                            "consensus_count": len(orientation.inlier_indices),
                            "trimmed_three_rmse_pixels": trimmed_rmse,
                            "maximum_perturbation_rotation_degrees": (
                                orientation.maximum_perturbation_rotation_degrees
                            ),
                            "subset_spread_degrees": orientation.subset_spread_degrees,
                        },
                    }
                )
                solved.append((score, result, orientation))
            candidate_results.append(result)
        selected: dict[str, Any] | None = None
        selected_orientation: D034OrientationSolution | None = None
        if solved:
            solved.sort(key=lambda item: item[0])
            best_score = solved[0][0]
            tied = [
                item
                for item in solved
                if item[0][0] == best_score[0]
                and item[0][1] <= best_score[1] + config.ambiguity_trimmed_rmse_pixels
            ]
            initial_label = assignment["initial_assignment_label"]
            chosen = next((item for item in tied if item[1]["label"] == initial_label), solved[0])
            selected, selected_orientation = chosen[1], chosen[2]
            selected["status"] = "selected-frozen-before-validation"
        payload = self._attempt_payload(
            camera_id,
            policy,
            batch,
            input_status,
            correspondence_path,
            assignment,
            candidate_results,
            selected,
            selected_orientation,
            validation_items,
            fixed_center,
            config,
        )
        return self.repository.append_attempt(payload)

    def _attempt_payload(
        self,
        camera_id: str,
        policy: SceneCameraPolicy,
        batch: dict[str, Any],
        input_status: dict[str, Any],
        correspondence_path: Path,
        assignment: dict[str, Any],
        candidate_results: list[dict[str, Any]],
        selected: dict[str, Any] | None,
        orientation: D034OrientationSolution | None,
        validation_items: list[dict[str, Any]],
        fixed_center: list[float],
        config: D034SolverConfig,
    ) -> dict[str, Any]:
        diagnostics: list[str] = []
        validation_dict: dict[str, Any] | None = None
        pose: dict[str, Any] | None = None
        evidence: dict[str, Any] | None = None
        automated_status = "rejected"
        can_override = False
        if selected is None or orientation is None:
            diagnostics.append(
                "No eligible intrinsic profile produced a pose that passed every D034 solve gate."
            )
            for result in candidate_results:
                if result.get("status") == "rejected-solve":
                    diagnostics.append(f"{result['label']}: {result['diagnostic']}")
        else:
            intrinsics = _intrinsics_from_dict(selected["intrinsics"])
            validation_ids, validation_world, validation_image = _landmark_values(validation_items)
            pose = orientation.to_dict()
            evidence = {
                "frame_id": input_status["approved_frame_id"],
                "image_width_pixels": int(selected["intrinsics"]["width_pixels"]),
                "image_height_pixels": int(selected["intrinsics"]["height_pixels"]),
                "solve_observed_pixels": [
                    item for item in _image_points_from_path(correspondence_path, "solve")
                ],
                "solve_projected_pixels": [
                    list(point) for point in orientation.solve_projected_pixels
                ],
                "validation_observed_pixels": validation_image,
                "validation_projected_pixels": [],
            }
            try:
                validation = evaluate_d034_frozen_validation(
                    intrinsics,
                    orientation.T_world_from_camera,
                    orientation.solve_landmark_ids,
                    validation_ids,
                    validation_world,
                    validation_image,
                    threshold_pixels=config.solve_inlier_threshold_pixels,
                )
            except PoseSolveError as error:
                diagnostics.append(
                    "The frozen pose cannot be operator-overridden because validation is "
                    f"physically invalid: {error}"
                )
            else:
                validation_dict = validation.to_dict()
                automated_status = validation.status
                can_override = validation.status == "rejected"
                if validation.status == "accepted":
                    diagnostics.append(
                        "All D034 solve gates passed and both validation points are within 30 px."
                    )
                else:
                    failed = [
                        f"{landmark_id} {error:.2f} px"
                        for landmark_id, error, passed in zip(
                            validation.validation_landmark_ids,
                            validation.individual_reprojection_errors_pixels,
                            validation.individual_pass,
                            strict=True,
                        )
                        if not passed
                    ]
                    diagnostics.append(
                        "The pose is finite, but validation failed: " + ", ".join(failed)
                    )
                evidence["validation_projected_pixels"] = [
                    list(point) for point in validation.projected_pixels
                ]
        return {
            "schema_version": "xr03-camera-calibration-attempt-v1",
            "camera_id": camera_id,
            "camera_policy_sha256": policy.sha256,
            "intrinsic_policy_sha256": policy.intrinsic_policy_sha256,
            "intrinsic_batch_sha256": batch["payload_sha256"],
            "source_revision": int(input_status["source_revision"]),
            "correspondence_export_path": str(correspondence_path.resolve()),
            "correspondence_export_sha256": _file_sha256(correspondence_path),
            "facility_export": _identity(self.facility_export_path),
            "assigned_initial_intrinsic_label": assignment["initial_assignment_label"],
            "selected_intrinsic_label": None if selected is None else selected["label"],
            "selected_intrinsics": None if selected is None else selected["intrinsics"],
            "fixed_center_world_metres": fixed_center,
            "solver": {"authority": "D034", "config": config.to_dict()},
            "candidate_results": candidate_results,
            "pose": pose,
            "validation": validation_dict,
            "automated_status": automated_status,
            "can_override": can_override,
            "diagnostics": diagnostics,
            "evidence_overlay": evidence,
            "authority_note": (
                "Automated acceptance still requires explicit physical-overlay review. A manual "
                "override never changes automated_status."
            ),
        }

    def review_camera(
        self, camera_id: str, attempt_sha256: str, policy: SceneCameraPolicy
    ) -> dict[str, Any]:
        camera = _camera_status(self.status(policy), camera_id)
        attempt = camera.get("attempt")
        if not isinstance(attempt, dict) or attempt.get("payload_sha256") != attempt_sha256:
            raise CalibrationWorkflowError("review must target the current calibration attempt")
        if attempt.get("automated_status") != "accepted":
            raise CalibrationWorkflowError(
                "strict visual review can approve only an automated numeric pass"
            )
        return self.repository.decide(
            camera_id, attempt_sha256, "strict-visual-review", "physical overlay reviewed"
        )

    def override_camera(
        self,
        camera_id: str,
        attempt_sha256: str,
        reason: str,
        acknowledged: bool,
        policy: SceneCameraPolicy,
    ) -> dict[str, Any]:
        if not acknowledged:
            raise CalibrationWorkflowError(
                "operator override requires explicit risk acknowledgement"
            )
        camera = _camera_status(self.status(policy), camera_id)
        attempt = camera.get("attempt")
        if not isinstance(attempt, dict) or attempt.get("payload_sha256") != attempt_sha256:
            raise CalibrationWorkflowError("override must target the current calibration attempt")
        if attempt.get("automated_status") == "accepted":
            raise CalibrationWorkflowError("use strict visual review for a passing calibration")
        if attempt.get("can_override") is not True:
            raise CalibrationWorkflowError("this failure produced no usable pose to override")
        return self.repository.decide(camera_id, attempt_sha256, "operator-override", reason)

    def prepare_reconstruction_inputs(
        self,
        policy: SceneCameraPolicy,
        baseline_directory: Path,
        output_directory: Path,
    ) -> Path:
        status = self.status(policy)
        if not status["all_cameras_ready"]:
            raise CalibrationWorkflowError(
                "every enabled camera must be reviewed or explicitly overridden"
            )
        if output_directory.exists():
            raise CalibrationWorkflowError("calibrated reconstruction input already exists")
        baseline_input = baseline_directory / "input-manifest.json"
        baseline_run = baseline_directory / "run-manifest.json"
        baseline = _read_json(baseline_input)
        if not baseline_run.is_file():
            raise CalibrationWorkflowError("baseline reconstruction run manifest is missing")
        records = baseline.get("cameras")
        if (
            not isinstance(records, list)
            or tuple(
                str(record.get("camera_id")) for record in records if isinstance(record, dict)
            )
            != policy.camera_ids
        ):
            raise CalibrationWorkflowError(
                "baseline reconstruction input differs from the enabled scene roster"
            )
        inputs_dir = output_directory / "inputs"
        inputs_dir.mkdir(parents=True)
        by_camera = {camera["camera_id"]: camera for camera in status["cameras"]}
        calibrated_records: list[dict[str, Any]] = []
        for source_record in records:
            if not isinstance(source_record, dict):
                raise CalibrationWorkflowError("baseline camera input is malformed")
            camera_id = str(source_record["camera_id"])
            camera = by_camera[camera_id]
            attempt = camera["attempt"]
            if not isinstance(attempt, dict):
                raise CalibrationWorkflowError(f"{camera_id} has no current calibration attempt")
            calibrated_records.append(
                _prepare_camera_input(source_record, attempt, inputs_dir, camera["readiness"])
            )
        manifest = {
            "schema_version": "xr03-calibrated-da3-input-manifest-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "camera_policy_sha256": policy.sha256,
            "intrinsic_policy_sha256": policy.intrinsic_policy_sha256,
            "intrinsic_batch_sha256": status["intrinsic_batch"]["payload_sha256"],
            "baseline_input": _identity(baseline_input),
            "cameras": calibrated_records,
            "authority": (
                "current D034 strict/operator-qualified calibration inputs for one immutable "
                "scene-joint DA3 candidate"
            ),
        }
        _write_json(output_directory / "input-manifest.json", manifest)
        _write_json(
            output_directory / "run-manifest.json",
            {
                "schema_version": "xr03-calibrated-da3-input-run-v1",
                "success": True,
                "input_manifest": _identity(output_directory / "input-manifest.json"),
                "baseline_run": _identity(baseline_run),
            },
        )
        return output_directory


def _load_intrinsic_estimates(
    path: Path,
) -> tuple[tuple[CameraIntrinsicEstimate, ...], dict[str, Any]]:
    value = _read_json(path)
    if value.get("schema_version") == "xr03-independent-intrinsic-estimates-v1":
        records = value.get("estimates")
        profile = None
        image_size = None
        authority = value.get("authority")
    elif value.get("schema_version") == "p04-intrinsic-fleet-study-v1":
        models = value.get("models")
        if not isinstance(models, list):
            raise CalibrationWorkflowError("D027 intrinsic models are missing")
        model = next(
            (item for item in models if item.get("camera_model") == "simple_radial"), None
        )
        if not isinstance(model, dict):
            raise CalibrationWorkflowError("D027 simple-radial evidence is missing")
        records = model.get("per_camera_estimates")
        profile = value.get("profile_version")
        image_size = value.get("image_size_pixels")
        authority = value.get("decision_authority")
    else:
        raise CalibrationWorkflowError("unsupported intrinsic evidence schema")
    if not isinstance(records, list):
        raise CalibrationWorkflowError("independent intrinsic estimates are missing")
    estimates: list[CameraIntrinsicEstimate] = []
    for record in records:
        if not isinstance(record, dict):
            raise CalibrationWorkflowError("intrinsic estimate is malformed")
        width = record.get("width_pixels")
        height = record.get("height_pixels")
        if image_size is not None:
            if not isinstance(image_size, list) or len(image_size) != 2:
                raise CalibrationWorkflowError("intrinsic evidence image size is malformed")
            width, height = image_size
        if (
            not isinstance(width, int | float)
            or isinstance(width, bool)
            or not isinstance(height, int | float)
            or isinstance(height, bool)
        ):
            raise CalibrationWorkflowError("intrinsic evidence image size is malformed")
        estimates.append(
            CameraIntrinsicEstimate(
                camera_id=_required_string(record, "camera_id"),
                profile_version=str(record.get("profile_version") or profile or ""),
                model=str(record.get("model") or "simple_radial"),
                width_pixels=int(width),
                height_pixels=int(height),
                fx_pixels=float(record["fx_pixels"]),
                fy_pixels=float(record["fy_pixels"]),
                cx_pixels=float(record["cx_pixels"]),
                cy_pixels=float(record["cy_pixels"]),
                distortion=tuple(float(item) for item in record["distortion"]),
                within_camera_focal_cv=float(record["within_camera_focal_cv"]),
            )
        )
    return tuple(estimates), {"authority": authority, "schema_version": value["schema_version"]}


def _initial_intrinsic_candidate(
    estimate: CameraIntrinsicEstimate, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        evidence.get("authority") == "D027"
        and estimate.model == "simple_radial"
        and (estimate.width_pixels, estimate.height_pixels) == (1920, 1080)
    ):
        return {
            "label": "d033-candidate-b",
            "reason": "D033 provisional starting default under D036",
            "intrinsics": D033_INTRINSICS.to_dict(),
            "source": "authorized-starting-default",
        }
    return _estimate_candidate(estimate)


def _estimate_candidate(estimate: CameraIntrinsicEstimate) -> dict[str, Any]:
    intrinsics = _intrinsics_from_estimate(estimate)
    return {
        "label": f"independent:{estimate.camera_id}",
        "reason": "camera-specific independent estimate retained as mandatory challenger",
        "intrinsics": intrinsics.to_dict(),
        "source": "independent-camera-estimate",
    }


def _fleet_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = CameraIntrinsics(
        str(value["model"]),
        int(value["width_pixels"]),
        int(value["height_pixels"]),
        float(value["fx_pixels"]),
        float(value["fy_pixels"]),
        float(value["cx_pixels"]),
        float(value["cy_pixels"]),
        tuple(float(item) for item in value["distortion"]),
    )
    return {
        "label": f"group:{value['method']}",
        "reason": "equal-camera leave-one-out lens-group challenger",
        "intrinsics": intrinsics.to_dict(),
        "source": "lens-group-profile",
        "included_camera_ids": list(value["included_camera_ids"]),
    }


def _deduplicate_candidates(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        digest = _payload_sha256(value["intrinsics"])
        if digest not in seen:
            seen.add(digest)
            output.append(value)
    return output


def _intrinsics_from_estimate(value: CameraIntrinsicEstimate) -> CameraIntrinsics:
    if value.model not in {"pinhole", "simple_radial", "simple_divisional", "radial"}:
        raise CalibrationWorkflowError(
            f"intrinsic model is not supported by the D034 solver: {value.model}"
        )
    return CameraIntrinsics(
        value.model,
        value.width_pixels,
        value.height_pixels,
        value.fx_pixels,
        value.fy_pixels,
        value.cx_pixels,
        value.cy_pixels,
        value.distortion,
    )


def _intrinsics_from_candidate(value: Mapping[str, Any]) -> CameraIntrinsics:
    intrinsics = value.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise CalibrationWorkflowError("intrinsic candidate is malformed")
    return _intrinsics_from_dict(intrinsics)


def _intrinsics_from_dict(value: Mapping[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        str(value["model"]),
        int(value["width_pixels"]),
        int(value["height_pixels"]),
        float(value["fx_pixels"]),
        float(value["fy_pixels"]),
        float(value["cx_pixels"]),
        float(value["cy_pixels"]),
        tuple(float(item) for item in value["distortion"]),
    )


def _fixed_center(path: Path, camera_id: str) -> list[float]:
    facility = _read_json(path)
    values = facility.get("camera_mounting_priors")
    if not isinstance(values, list):
        raise CalibrationWorkflowError("facility export has no camera mounting priors")
    camera = next((item for item in values if item.get("camera_id") == camera_id), None)
    if not isinstance(camera, dict):
        raise CalibrationWorkflowError(f"facility export has no fixed centre for {camera_id}")
    center = camera.get("C_world_mount_prior")
    if not isinstance(center, dict):
        raise CalibrationWorkflowError(f"{camera_id} fixed centre is malformed")
    result = [float(center[key]) for key in ("x_metres", "y_metres", "z_metres")]
    if not all(math.isfinite(value) for value in result):
        raise CalibrationWorkflowError(f"{camera_id} fixed centre is non-finite")
    return result


def _landmark_values(
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[list[float]], list[list[float]]]:
    return (
        [str(item["landmark_id"]) for item in items],
        [
            [
                float(item["world_point"]["x_metres"]),
                float(item["world_point"]["y_metres"]),
                float(item["world_point"]["z_metres"]),
            ]
            for item in items
        ],
        [[float(item["image_point"]["u"]), float(item["image_point"]["v"])] for item in items],
    )


def _image_points_from_path(path: Path, role: str) -> list[list[float]]:
    value = _read_json(path)
    return [
        [float(item["image_point"]["u"]), float(item["image_point"]["v"])]
        for item in value.get("landmarks", [])
        if item.get("role") == role
    ]


def _camera_status(status: Mapping[str, Any], camera_id: str) -> dict[str, Any]:
    value = next(
        (item for item in status.get("cameras", []) if item.get("camera_id") == camera_id),
        None,
    )
    if not isinstance(value, dict):
        raise CalibrationWorkflowError("camera is outside the active scene roster")
    return value


def _intrinsic_batch_matches_policy(
    batch: Mapping[str, Any], policy: SceneCameraPolicy
) -> bool:
    """Accept current identities and legacy batches with the same lens grouping."""

    identity = batch.get("intrinsic_policy_sha256")
    if isinstance(identity, str):
        return identity == policy.intrinsic_policy_sha256
    assignments = batch.get("assignments")
    if not isinstance(assignments, list):
        return False
    actual = {
        str(item.get("camera_id")): (
            str(item.get("group_id")),
            str(item.get("lens_model")),
        )
        for item in assignments
        if isinstance(item, Mapping)
    }
    expected = {
        camera_id: (group.group_id, group.lens_model)
        for group in policy.intrinsic_groups
        for camera_id in group.camera_ids
    }
    return actual == expected


def _camera_summary_from_attempt(
    attempt: Mapping[str, Any], ready: bool, override: bool
) -> dict[str, Any]:
    intrinsics = attempt.get("selected_intrinsics")
    pose = attempt.get("pose")
    if not isinstance(intrinsics, dict) or not isinstance(pose, dict):
        return {}
    transform = np.asarray(pose["T_world_from_camera"], dtype=np.float64)
    yaw, pitch, roll = _euler_zyx_degrees(transform[:3, :3])
    return {
        "status": "Ready with override" if override else "Ready" if ready else "Needs review",
        "intrinsics": {
            "model": str(intrinsics["model"]),
            "resolution": [int(intrinsics["width_pixels"]), int(intrinsics["height_pixels"])],
            "fx_pixels": float(intrinsics["fx_pixels"]),
            "fy_pixels": float(intrinsics["fy_pixels"]),
            "cx_pixels": float(intrinsics["cx_pixels"]),
            "cy_pixels": float(intrinsics["cy_pixels"]),
            "distortion": [float(item) for item in intrinsics["distortion"]],
        },
        "pose": {
            "frame": "world",
            "transform": "T_world_from_camera",
            "position_metres": [float(value) for value in transform[:3, 3]],
            "orientation_zyx_degrees": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "matrix": transform.tolist(),
            "translation_authority": "reviewed fixed optical centre; not optimized",
        },
    }


def _euler_zyx_degrees(rotation: Any) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) > 1e-9:
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    else:
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _prepare_camera_input(
    source_record: Mapping[str, Any],
    attempt: Mapping[str, Any],
    inputs_dir: Path,
    readiness: str,
) -> dict[str, Any]:
    camera_id = str(source_record["camera_id"])
    source = source_record.get("source")
    if not isinstance(source, dict):
        raise CalibrationWorkflowError(f"{camera_id} baseline source record is missing")
    source_path = Path(_required_string(source, "path"))
    if not source_path.is_file() or _file_sha256(source_path) != _required_sha256(
        source, "sha256"
    ):
        raise CalibrationWorkflowError(f"{camera_id} baseline source image identity changed")
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    intrinsics_value = attempt.get("selected_intrinsics")
    pose = attempt.get("pose")
    if image is None or not isinstance(intrinsics_value, dict) or not isinstance(pose, dict):
        raise CalibrationWorkflowError(f"{camera_id} calibrated input is incomplete")
    intrinsics = _intrinsics_from_dict(intrinsics_value)
    if (image.shape[1], image.shape[0]) != (
        intrinsics.width_pixels,
        intrinsics.height_pixels,
    ):
        raise CalibrationWorkflowError(f"{camera_id} source image size differs from calibration")
    K = np.asarray(
        [
            [intrinsics.fx_pixels, 0.0, intrinsics.cx_pixels],
            [0.0, intrinsics.fy_pixels, intrinsics.cy_pixels],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if intrinsics.model == "pinhole":
        coefficients = np.zeros(5, dtype=np.float64)
    elif intrinsics.model == "simple_radial":
        coefficients = np.asarray([intrinsics.distortion[0], 0, 0, 0, 0], dtype=np.float64)
    elif intrinsics.model == "radial":
        coefficients = np.asarray(
            [intrinsics.distortion[0], intrinsics.distortion[1], 0, 0, 0], dtype=np.float64
        )
    else:
        raise CalibrationWorkflowError(
            "simple-divisional reconstruction rectification is not implemented"
        )
    size = (intrinsics.width_pixels, intrinsics.height_pixels)
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, coefficients, np.eye(3, dtype=np.float64), K, size, cv2.CV_32FC1
    )
    derivative = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    valid_mask = (
        (map_x >= 0)
        & (map_x <= intrinsics.width_pixels - 1)
        & (map_y >= 0)
        & (map_y <= intrinsics.height_pixels - 1)
    )
    evaluation = source_record.get("evaluation_mask")
    rectangles = (
        evaluation.get("rectangles_xyxy_derivative_pixels", [])
        if isinstance(evaluation, dict)
        else []
    )
    for rectangle in rectangles:
        if isinstance(rectangle, list) and len(rectangle) == 4:
            x0, y0, x1, y1 = (int(value) for value in rectangle)
            valid_mask[y0:y1, x0:x1] = False
    derivative_path = inputs_dir / f"{camera_id}-pinhole.png"
    mask_path = inputs_dir / f"{camera_id}-evaluation-mask.png"
    if not cv2.imwrite(str(derivative_path), derivative) or not cv2.imwrite(
        str(mask_path), valid_mask.astype(np.uint8) * 255
    ):
        raise CalibrationWorkflowError(f"failed to write calibrated input for {camera_id}")
    T_world = np.asarray(pose["T_world_from_camera"], dtype=np.float64)
    T_camera = T_camera_from_world(T_world)
    record = copy.deepcopy(dict(source_record))
    record["pinhole_derivative"] = {
        **_identity(derivative_path),
        "operation": "OpenCV calibrated distortion removal at source dimensions",
        "source_intrinsic_label": attempt["selected_intrinsic_label"],
        "authority": "XR03 current calibration derivative",
    }
    record["evaluation_mask"] = {
        **_identity(mask_path),
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "excluded_pixel_count": int(valid_mask.size - np.count_nonzero(valid_mask)),
        "rectangles_xyxy_derivative_pixels": rectangles,
        "use": "evaluation only; model input pixels remain unmasked",
    }
    record["intrinsics"] = {
        **intrinsics.to_dict(),
        "K_pinhole": K.tolist(),
        "authority": "XR03 D086 current intrinsic assignment",
    }
    record["seed_transform"] = {
        "T_world_from_camera": T_world.tolist(),
        "T_camera_from_world_for_DA3": T_camera.tolist(),
        "status": readiness,
        "authority": "D034 strict result or explicit D086 operator-qualified override",
    }
    record["calibration_attempt_sha256"] = attempt["payload_sha256"]
    record["output_authority"] = (
        "warning-qualified operator override; not automated calibration acceptance"
        if readiness == "operator-accepted-with-warning"
        else "D034 strict reviewed calibration"
    )
    return record


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CalibrationWorkflowError(f"{key} must be a non-blank string")
    return result.strip()


def _required_sha256(value: Mapping[str, Any], key: str) -> str:
    result = _required_string(value, key)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise CalibrationWorkflowError(f"{key} must be a lowercase SHA-256")
    return result


def _json_object(value: str) -> dict[str, Any]:
    result = json.loads(value)
    if not isinstance(result, dict):
        raise CalibrationWorkflowError("stored calibration payload is malformed")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationWorkflowError(f"required JSON is unreadable: {path.name}") from error
    if not isinstance(result, dict):
        raise CalibrationWorkflowError(f"required JSON must be an object: {path.name}")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
