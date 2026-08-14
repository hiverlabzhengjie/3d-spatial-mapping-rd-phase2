"""Build the local-only P02 facility-reference evidence package.

Client plan images, annotations, digitized coordinates, and derived display products remain below
the artifact root. Git retains only this reusable builder, contracts, tests, hashes, and sanitized
stage summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from spatial_mapping_phase2.p01_observability import OwnerInputManifest
from spatial_mapping_phase2.p02_facility_reference import (
    P02_SCHEMA_VERSION,
    CeilingHeightCheck,
    ControlPoint,
    DisplayCorrespondence,
    EvidenceStatus,
    FacilityFrameDefinition,
    FrameReviewState,
    IndependentSpotCheck,
    P02ContractError,
    Point3Metres,
    RectangularPillarDimensions,
    SourceIdentity,
    SpotCheckCoverage,
    derive_mounting_point_prior,
    fit_plan_display_transform,
    require_d010_checks,
)

LOCAL_SCHEMA_VERSION = "p02-local-facility-input-v2"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P02ContractError(f"could not load local P02 JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise P02ContractError("local P02 input root must be an object")
    return payload


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise P02ContractError(f"{key} must be an object")
    return value


def _require_array(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise P02ContractError(f"{key} must be an array")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise P02ContractError(f"{key} must be a non-blank string")
    return value


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise P02ContractError(f"{key} must be numeric")
    return float(value)


def _optional_number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise P02ContractError(f"{key} must be numeric or null")
    return float(value)


def _parse_sources(payload: dict[str, Any]) -> tuple[SourceIdentity, ...]:
    sources: list[SourceIdentity] = []
    for item in _require_array(payload, "sources"):
        if not isinstance(item, dict):
            raise P02ContractError("each source must be an object")
        source_path = Path(_require_string(item, "path"))
        expected_hash = _require_string(item, "sha256")
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise P02ContractError(f"source hash mismatch for {source_path.name}")
        sources.append(
            SourceIdentity(
                source_id=_require_string(item, "source_id"),
                sha256=actual_hash,
                source_kind=_require_string(item, "source_kind"),
                authority_note=_require_string(item, "authority_note"),
            )
        )
    return tuple(sources)


def _parse_frame(
    payload: dict[str, Any], sources: tuple[SourceIdentity, ...]
) -> FacilityFrameDefinition:
    item = _require_object(payload, "frame")
    return FacilityFrameDefinition(
        schema_version=P02_SCHEMA_VERSION,
        frame_id=_require_string(item, "frame_id"),
        origin_feature_id=_require_string(item, "origin_feature_id"),
        origin_feature_meaning=_require_string(item, "origin_feature_meaning"),
        x_direction_meaning=_require_string(item, "x_direction_meaning"),
        y_direction_meaning=_require_string(item, "y_direction_meaning"),
        z_direction_meaning=_require_string(item, "z_direction_meaning"),
        units=_require_string(item, "units"),
        floor_reference=_require_string(item, "floor_reference"),
        x_axis=tuple(item["x_axis"]),
        y_axis=tuple(item["y_axis"]),
        z_axis=tuple(item["z_axis"]),
        review_state=FrameReviewState(_require_string(item, "review_state")),
        sources=sources,
    )


def _parse_controls(
    payload: dict[str, Any], sources: tuple[SourceIdentity, ...]
) -> tuple[ControlPoint, ...]:
    controls: list[ControlPoint] = []
    for item in _require_array(payload, "control_points"):
        if not isinstance(item, dict):
            raise P02ContractError("each control point must be an object")
        controls.append(
            ControlPoint(
                feature_id=_require_string(item, "feature_id"),
                meaning=_require_string(item, "meaning"),
                world=Point3Metres(
                    _number(item, "x_metres"),
                    _number(item, "y_metres"),
                    _number(item, "z_metres"),
                ),
                horizontal_uncertainty_metres=_optional_number(
                    item, "horizontal_uncertainty_metres"
                ),
                vertical_uncertainty_metres=_optional_number(item, "vertical_uncertainty_metres"),
                status=EvidenceStatus(_require_string(item, "status")),
                sources=sources,
            )
        )
    return tuple(controls)


def _require_rectangular_pillar_geometry(
    payload: dict[str, Any], sources: tuple[SourceIdentity, ...]
) -> RectangularPillarDimensions:
    item = _require_object(payload, "rectangular_pillar_geometry")
    source_by_id = {source.source_id: source for source in sources}
    source_id = _require_string(item, "source_id")
    if source_id not in source_by_id:
        raise P02ContractError(f"unknown rectangular-pillar source_id: {source_id}")
    dimensions = RectangularPillarDimensions(
        long_side_metres=_number(item, "long_side_metres"),
        long_side_uncertainty_metres=_number(item, "long_side_uncertainty_metres"),
        short_side_metres=_optional_number(item, "short_side_metres"),
        short_side_uncertainty_metres=_optional_number(item, "short_side_uncertainty_metres"),
        source=source_by_id[source_id],
    )
    dimensions.require_resolved()
    return dimensions


def _parse_correspondences(payload: dict[str, Any]) -> tuple[DisplayCorrespondence, ...]:
    correspondences: list[DisplayCorrespondence] = []
    for item in _require_array(payload, "display_correspondences"):
        if not isinstance(item, dict):
            raise P02ContractError("each display correspondence must be an object")
        correspondences.append(
            DisplayCorrespondence(
                feature_id=_require_string(item, "feature_id"),
                world_x_metres=_number(item, "world_x_metres"),
                world_y_metres=_number(item, "world_y_metres"),
                pixel_u=_number(item, "pixel_u"),
                pixel_v=_number(item, "pixel_v"),
            )
        )
    return tuple(correspondences)


def _parse_spot_checks(
    payload: dict[str, Any], sources: tuple[SourceIdentity, ...]
) -> tuple[IndependentSpotCheck, ...]:
    source_by_id = {source.source_id: source for source in sources}
    checks: list[IndependentSpotCheck] = []
    for item in _require_array(payload, "spot_checks"):
        if not isinstance(item, dict):
            raise P02ContractError("each spot check must be an object")
        source_id = _require_string(item, "source_id")
        if source_id not in source_by_id:
            raise P02ContractError(f"unknown spot-check source_id: {source_id}")
        checks.append(
            IndependentSpotCheck(
                check_id=_require_string(item, "check_id"),
                coverage=SpotCheckCoverage(_require_string(item, "coverage")),
                start_feature_id=_require_string(item, "start_feature_id"),
                end_feature_id=_require_string(item, "end_feature_id"),
                measured_distance_metres=_number(item, "measured_distance_metres"),
                measurement_uncertainty_metres=_optional_number(
                    item, "measurement_uncertainty_metres"
                ),
                plan_distance_metres=_number(item, "plan_distance_metres"),
                plan_uncertainty_metres=_optional_number(item, "plan_uncertainty_metres"),
                method=_require_string(item, "method"),
                source=source_by_id[source_id],
                status=EvidenceStatus(_require_string(item, "status")),
            )
        )
    return tuple(checks)


def _parse_ceiling(
    payload: dict[str, Any], sources: tuple[SourceIdentity, ...]
) -> CeilingHeightCheck | None:
    item = payload.get("ceiling_height_check")
    if item is None:
        return None
    if not isinstance(item, dict):
        raise P02ContractError("ceiling_height_check must be an object or null")
    source_by_id = {source.source_id: source for source in sources}
    source_id = _require_string(item, "source_id")
    return CeilingHeightCheck(
        height_metres=_number(item, "height_metres"),
        uncertainty_metres=_number(item, "uncertainty_metres"),
        location_feature_id=_require_string(item, "location_feature_id"),
        method=_require_string(item, "method"),
        source=source_by_id[source_id],
        status=EvidenceStatus(_require_string(item, "status")),
    )


def _preserve_inputs(payload: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    input_root = output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=False)
    preserved: list[dict[str, Any]] = []
    for item in _require_array(payload, "sources"):
        if not isinstance(item, dict):
            raise P02ContractError("each source must be an object")
        source_path = Path(_require_string(item, "path"))
        destination = input_root / source_path.name
        shutil.copy2(source_path, destination)
        preserved.append(
            {
                "source_id": _require_string(item, "source_id"),
                "relative_path": destination.relative_to(output_root).as_posix(),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return preserved


def _draw_display_plan(
    payload: dict[str, Any],
    display_image: Path,
    output_path: Path,
    controls: tuple[ControlPoint, ...],
    correspondences: tuple[DisplayCorrespondence, ...],
) -> None:
    image = cv2.imread(str(display_image), cv2.IMREAD_COLOR)
    if image is None:
        raise P02ContractError("canonical display raster could not be opened")
    transform = fit_plan_display_transform(correspondences)
    for correspondence in correspondences:
        observed = (round(correspondence.pixel_u), round(correspondence.pixel_v))
        predicted = tuple(
            map(
                round,
                transform.pixel_from_world(
                    correspondence.world_x_metres,
                    correspondence.world_y_metres,
                ),
            )
        )
        cv2.circle(image, observed, 3, (80, 80, 80), -1, cv2.LINE_AA)
        cv2.line(image, observed, predicted, (160, 160, 160), 1, cv2.LINE_AA)
    colours = {
        "office-cam-01-pillar-centre": (50, 50, 230),
        "office-cam-02-pillar-centre": (30, 180, 50),
        "office-cam-03-pillar-centre": (230, 100, 30),
        "office-cam-04-pillar-centre": (20, 180, 240),
    }
    for control in controls:
        u, v = transform.pixel_from_world(control.world.x_metres, control.world.y_metres)
        point = (round(u), round(v))
        colour = colours.get(control.feature_id, (180, 50, 180))
        cv2.circle(image, point, 10, colour, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            control.feature_id,
            (point[0] + 12, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    control_by_id = {control.feature_id: control for control in controls}
    camera_references = _require_object(payload, "camera_reference_feature_ids")
    annotations = _require_object(payload, "camera_annotations")
    camera_colours = {
        "office-cam-01": (50, 50, 230),
        "office-cam-02": (30, 180, 50),
        "office-cam-03": (230, 100, 30),
        "office-cam-04": (20, 180, 240),
    }
    for camera_id, raw_annotation in annotations.items():
        if not isinstance(raw_annotation, dict):
            raise P02ContractError("each camera annotation must be an object")
        reference_id = camera_references.get(camera_id)
        if not isinstance(reference_id, str) or reference_id not in control_by_id:
            raise P02ContractError(f"missing display reference for {camera_id}")
        vector = raw_annotation.get("rough_pan_vector_world_xy")
        if (
            not isinstance(vector, list)
            or len(vector) != 2
            or not all(isinstance(value, int | float) for value in vector)
        ):
            raise P02ContractError(f"invalid rough pan vector for {camera_id}")
        control = control_by_id[reference_id]
        start = transform.pixel_from_world(control.world.x_metres, control.world.y_metres)
        end = transform.pixel_from_world(
            control.world.x_metres + float(vector[0]),
            control.world.y_metres + float(vector[1]),
        )
        cv2.arrowedLine(
            image,
            tuple(map(round, start)),
            tuple(map(round, end)),
            camera_colours[camera_id],
            3,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    fitted_origin = transform.pixel_from_world(0.0, 0.0)
    fitted_x_end = transform.pixel_from_world(1.0, 0.0)
    fitted_y_end = transform.pixel_from_world(0.0, 1.0)
    origin_control = next(
        correspondence
        for correspondence in correspondences
        if correspondence.feature_id == "origin"
    )
    origin = (round(origin_control.pixel_u), round(origin_control.pixel_v))
    x_end = (
        round(origin[0] + fitted_x_end[0] - fitted_origin[0]),
        round(origin[1] + fitted_x_end[1] - fitted_origin[1]),
    )
    y_end = (
        round(origin[0] + fitted_y_end[0] - fitted_origin[0]),
        round(origin[1] + fitted_y_end[1] - fitted_origin[1]),
    )
    cv2.drawMarker(image, origin, (255, 0, 255), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
    cv2.arrowedLine(image, origin, x_end, (255, 0, 255), 3)
    cv2.arrowedLine(image, origin, y_end, (255, 128, 0), 3)
    cv2.putText(image, "+X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(image, "+Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    if not cv2.imwrite(str(output_path), image):
        raise P02ContractError("georeferenced display plan could not be written")


def build(local_input: Path, owner_input: Path, output_root: Path) -> None:
    payload = _load_json(local_input)
    if payload.get("schema_version") != LOCAL_SCHEMA_VERSION:
        raise P02ContractError("unsupported local P02 input schema_version")
    if output_root.exists():
        raise P02ContractError("output root already exists; choose a new immutable run directory")

    sources = _parse_sources(payload)
    frame = _parse_frame(payload, sources)
    _require_rectangular_pillar_geometry(payload, sources)
    output_root.mkdir(parents=True)
    controls = _parse_controls(payload, sources)
    correspondences = _parse_correspondences(payload)
    spot_checks = _parse_spot_checks(payload, sources)
    ceiling = _parse_ceiling(payload, sources)
    display_transform = fit_plan_display_transform(correspondences)
    preserved_inputs = _preserve_inputs(payload, output_root)

    owner_manifest = OwnerInputManifest.from_json_file(owner_input)
    control_by_id = {control.feature_id: control for control in controls}
    camera_references = _require_object(payload, "camera_reference_feature_ids")
    priors = []
    for camera in owner_manifest.cameras:
        reference_id = camera_references.get(camera.identity.camera_id)
        if not isinstance(reference_id, str) or reference_id not in control_by_id:
            raise P02ContractError(f"missing camera reference for {camera.identity.camera_id}")
        priors.append(
            derive_mounting_point_prior(
                camera,
                control_by_id[reference_id],
                plan_control_residual_metres=None,
                lens_to_reference_offset_metres=None,
                sources=sources,
            )
        )

    gate_reasons: list[str] = []
    if ceiling is None or ceiling.status != EvidenceStatus.ACCEPTED:
        gate_reasons.append("accepted finished-floor-to-ceiling height is missing")
    for coverage in SpotCheckCoverage:
        matching = tuple(check for check in spot_checks if check.coverage == coverage)
        if not any(
            check.status == EvidenceStatus.ACCEPTED and check.consistent_within_uncertainty is True
            for check in matching
        ):
            gate_reasons.append(
                f"{coverage.value} lacks an accepted uncertainty-complete spot check"
            )
    plan_uncertainty = _require_object(payload, "plan_uncertainty")
    if plan_uncertainty.get("printed_dimension_uncertainty_metres") is None:
        gate_reasons.append("printed dimension accuracy/uncertainty remains unquantified")
    try:
        require_d010_checks(ceiling, spot_checks)
    except P02ContractError as error:
        if not gate_reasons:
            gate_reasons.append(str(error))
    gate_status = "ready" if not gate_reasons else "blocked"
    gate_reason = None if not gate_reasons else "; ".join(gate_reasons)

    display_source = next(
        Path(item["path"])
        for item in _require_array(payload, "sources")
        if isinstance(item, dict) and item.get("source_id") == "canonical-display-raster"
    )
    display_output = output_root / "georeferenced_display_plan.png"
    _draw_display_plan(payload, display_source, display_output, controls, correspondences)

    residuals = []
    for point in correspondences:
        predicted_u, predicted_v = display_transform.pixel_from_world(
            point.world_x_metres, point.world_y_metres
        )
        residuals.append(
            {
                "feature_id": point.feature_id,
                "residual_u_pixels": predicted_u - point.pixel_u,
                "residual_v_pixels": predicted_v - point.pixel_v,
            }
        )

    facility_record = {
        "schema_version": P02_SCHEMA_VERSION,
        "frame": asdict(frame),
        "metric_authority": "printed dimensions plus verified permanent physical references",
        "display_transform_authority": "visualization and annotation support only",
        "display_transform": asdict(display_transform),
        "display_residuals": residuals,
        "control_points": [asdict(control) for control in controls],
        "printed_dimension_segments": _require_array(payload, "printed_dimension_segments"),
        "annotation_registration": _require_object(payload, "annotation_registration"),
        "camera_annotations": _require_object(payload, "camera_annotations"),
        "spot_checks": [asdict(check) for check in spot_checks],
        "ceiling_height_check": None if ceiling is None else asdict(ceiling),
        "landmark_database": _require_object(payload, "landmark_database"),
        "plan_uncertainty": plan_uncertainty,
        "mounting_point_priors": [asdict(prior) for prior in priors],
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "downstream_boundary": (
            "C_world_mount_prior is not an optical centre, orientation, pose, or "
            "T_world_from_camera"
        ),
    }
    facility_path = output_root / "facility_reference.json"
    facility_path.write_text(json.dumps(facility_record, indent=2) + "\n", encoding="utf-8")

    outputs = []
    for path in (facility_path, display_output):
        outputs.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "p02-artifact-manifest-v1",
        "preserved_inputs": preserved_inputs,
        "outputs": outputs,
        "gate_status": gate_status,
        "gate_reason": gate_reason,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-input", type=Path, required=True)
    parser.add_argument("--owner-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.local_input, arguments.owner_input, arguments.output_root)


if __name__ == "__main__":
    main()
