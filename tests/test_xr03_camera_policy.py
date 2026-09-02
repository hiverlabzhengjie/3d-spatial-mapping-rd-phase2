from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from spatial_mapping_phase2.xr03_camera_policy import (
    CameraPolicyError,
    CameraPolicyRepository,
    SceneCameraPolicy,
)

CAMERAS = ("camera-1", "camera-2", "camera-3")


def _policy(*, overlap_13: str = "no_overlap") -> SceneCameraPolicy:
    return SceneCameraPolicy.build(
        "project-a",
        "scene-a",
        CAMERAS,
        (
            {
                "group_id": "lens-b",
                "lens_model": "Model B",
                "camera_ids": ["camera-3"],
            },
            {
                "group_id": "lens-a",
                "lens_model": "Model A",
                "camera_ids": ["camera-2", "camera-1"],
            },
        ),
        (
            {
                "camera_id_a": "camera-1",
                "camera_id_b": "camera-2",
                "verdict": "overlap",
            },
            {
                "camera_id_a": "camera-1",
                "camera_id_b": "camera-3",
                "verdict": overlap_13,
            },
            {
                "camera_id_a": "camera-2",
                "camera_id_b": "camera-3",
                "verdict": "overlap",
            },
        ),
    )


def test_blank_policy_defaults_every_pair_to_no_overlap() -> None:
    policy = SceneCameraPolicy.blank("project-a", "scene-a", CAMERAS)

    assert policy.overlap_complete is True
    assert policy.overlap_edges == ()
    assert [review.verdict.value for review in policy.overlap_pair_reviews] == [
        "no_overlap",
        "no_overlap",
        "no_overlap",
    ]


def test_policy_is_canonical_complete_and_overlap_is_not_transitive() -> None:
    policy = _policy()

    assert [group.group_id for group in policy.intrinsic_groups] == ["lens-a", "lens-b"]
    assert policy.intrinsic_groups[0].camera_ids == ("camera-1", "camera-2")
    assert policy.lens_complete is True
    assert policy.overlap_complete is True
    assert policy.overlap_edges == (
        ("camera-1", "camera-2"),
        ("camera-2", "camera-3"),
    )
    assert ("camera-1", "camera-3") not in policy.overlap_edges
    assert policy.to_dict()["static_reconstruction_policy"] == (
        "all-enabled-cameras-per-scene-joint"
    )
    assert len(policy.sha256) == 64
    assert len(policy.intrinsic_policy_sha256) == 64


def test_overlap_revision_preserves_intrinsic_policy_identity() -> None:
    first = _policy()
    overlap_changed = _policy(overlap_13="overlap")

    assert overlap_changed.sha256 != first.sha256
    assert overlap_changed.intrinsic_policy_sha256 == first.intrinsic_policy_sha256


def test_policy_refuses_duplicate_membership_and_missing_pair() -> None:
    with pytest.raises(CameraPolicyError, match="only one intrinsic group"):
        SceneCameraPolicy.build(
            "project-a",
            "scene-a",
            CAMERAS,
            (
                {"group_id": "a", "lens_model": "A", "camera_ids": ["camera-1"]},
                {"group_id": "b", "lens_model": "B", "camera_ids": ["camera-1"]},
            ),
            _policy().to_dict()["overlap_pair_reviews"],
        )

    with pytest.raises(CameraPolicyError, match="every unordered camera pair"):
        SceneCameraPolicy.build(
            "project-a",
            "scene-a",
            CAMERAS,
            (),
            _policy().to_dict()["overlap_pair_reviews"][:-1],
        )


def test_repository_appends_revisions_and_rollback_selection(tmp_path: Path) -> None:
    repository = CameraPolicyRepository(
        tmp_path / "camera-policy.sqlite3", "project-a", "scene-a", CAMERAS
    )
    assert repository.status()["active_policy"] is None

    first = repository.apply(
        "policy-first",
        _policy(),
        expected_revision=None,
        confirm_impacts=False,
    )
    second_policy = _policy(overlap_13="overlap")
    impact = repository.impact(second_policy)
    assert impact["xr02_new_epoch_required"] is True
    assert impact["static_reconstruction_cohort_changed"] is False
    second = repository.apply(
        "policy-second",
        second_policy,
        expected_revision=first["revision"],
        confirm_impacts=True,
    )
    assert repository.by_sha256(second_policy.sha256) == second_policy

    rolled_back = repository.rollback(
        "policy-rollback",
        first["revision"],
        expected_revision=second["revision"],
        confirm_impacts=True,
    )
    status = repository.status()
    assert rolled_back["revision"] == first["revision"]
    assert status["active_revision"] == first["revision"]
    assert len(status["revisions"]) == 2
    first_history = next(item for item in status["revisions"] if item["revision"] == 1)
    assert first_history["selection_count"] == 2
    assert first_history["active"] is True

    reapplied = repository.apply(
        "policy-reapply",
        second_policy,
        expected_revision=first["revision"],
        confirm_impacts=True,
    )
    status = repository.status()
    assert reapplied["revision"] == second["revision"]
    assert status["active_revision"] == second["revision"]
    assert status["active_selection_id"] == reapplied["selection_id"]
    assert status["selection_reason"] == "reapplied"
    assert len(status["revisions"]) == 2
    second_history = next(item for item in status["revisions"] if item["revision"] == 2)
    assert second_history["selection_count"] == 2


def test_repository_refuses_reapplying_the_active_revision(tmp_path: Path) -> None:
    repository = CameraPolicyRepository(
        tmp_path / "camera-policy.sqlite3", "project-a", "scene-a", CAMERAS
    )
    repository.apply("policy-first", _policy(), expected_revision=None, confirm_impacts=False)

    with pytest.raises(CameraPolicyError, match="already active"):
        repository.apply(
            "policy-duplicate-active",
            _policy(),
            expected_revision=1,
            confirm_impacts=True,
        )


def test_stale_or_unconfirmed_write_is_atomic(tmp_path: Path) -> None:
    database = tmp_path / "camera-policy.sqlite3"
    repository = CameraPolicyRepository(database, "project-a", "scene-a", CAMERAS)
    repository.apply("policy-first", _policy(), expected_revision=None, confirm_impacts=False)
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    with pytest.raises(CameraPolicyError, match="confirm"):
        repository.apply(
            "policy-unconfirmed",
            _policy(overlap_13="overlap"),
            expected_revision=1,
            confirm_impacts=False,
        )
    assert repository.status()["active_revision"] == 1

    with pytest.raises(CameraPolicyError, match="stale"):
        repository.apply(
            "policy-stale",
            _policy(overlap_13="overlap"),
            expected_revision=None,
            confirm_impacts=True,
        )
    assert repository.status()["active_revision"] == 1
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_database_binding_refuses_a_different_roster(tmp_path: Path) -> None:
    path = tmp_path / "camera-policy.sqlite3"
    CameraPolicyRepository(path, "project-a", "scene-a", CAMERAS)
    with pytest.raises(CameraPolicyError, match="does not match"):
        CameraPolicyRepository(path, "project-a", "scene-a", ("camera-1",))
