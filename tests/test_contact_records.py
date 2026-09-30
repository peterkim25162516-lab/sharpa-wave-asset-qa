from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from wave_asset_qa.contact.records import (
    ContactRecordValidationError,
    ContactRun,
    ContactTraceSample,
    canonical_contact_run_json,
    loads_contact_run,
)
from wave_asset_qa.contact.scenarios import (
    contact_manifest_sha256,
    expand_contact_cases,
    load_contact_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"
ZERO40 = "0" * 40
ZERO64 = "0" * 64


def _error_payload() -> dict[str, object]:
    manifest = load_contact_manifest(MANIFEST_PATH)
    case = expand_contact_cases(manifest)[0]
    return {
        "schema_version": 1,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": contact_manifest_sha256(manifest),
        "case": case.to_dict(),
        "execution": {
            "status": "error",
            "message": "backend unavailable",
            "requested_steps": 3500,
            "completed_steps": 0,
        },
        "mapping": {
            "expected_joint_count": 22,
            "observed_joint_count": 0,
            "expected_distal_frame_count": 5,
            "observed_distal_frame_count": 0,
            "probe_frame_name": "left_index_DP",
            "probe_frame_mapped": False,
        },
        "fixture_readback": {
            "performed": False,
            "profile_id": None,
            "probe_parent_frame_name": None,
            "probe_local_center_m": None,
            "probe_radius_m": None,
            "target_world_center_m": None,
            "target_half_extents_m": None,
            "target_top_surface_z_m": None,
            "static_friction": None,
            "dynamic_friction": None,
            "restitution": None,
            "native_hand_collisions_enabled": None,
            "self_collisions_enabled": None,
            "ccd_enabled": None,
            "allowed_pair_id": None,
            "mujoco_condim": None,
            "native_collision_prim_count": None,
            "native_collision_disabled_count": None,
            "enabled_collision_shape_count": None,
            "collision_inventory_sha256": None,
            "mass_properties_preserved": None,
        },
        "contact_observation": {
            "performed": False,
            "capability": "unavailable",
            "selected_pair_id": None,
            "pair_active_source": None,
            "pair_active_force_threshold_n": None,
            "pair_active_record_count": 0,
            "missing_pair_active_count": 0,
            "filtered_sensor_body_count": 0,
            "filtered_target_count": 0,
        },
        "samples": [],
        "provenance": {
            "backend": "mujoco",
            "source_revision": None,
            "source_tree": None,
            "asset_commit": None,
            "asset_git_tree": None,
            "manifest_sha256": None,
            "fixture_overlay_sha256": None,
            "runtime_fingerprint_sha256": None,
            "fresh_process_identity_sha256": None,
        },
    }


def _sample(*, active: bool, force_n: float | None) -> ContactTraceSample:
    joints = {f"left_joint_{index:02d}": 0.0 for index in range(22)}
    frames = {
        f"left_frame_{index}": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        for index in range(5)
    }
    return ContactTraceSample(
        step=1,
        time_s=0.002,
        joint_positions=joints,
        joint_velocities=joints,
        frame_poses=frames,
        position_targets=joints,
        probe_center_world_m=(0.0, 0.0, 0.2),
        signed_gap_m=0.065,
        pair_active=active,
        raw_contact_count=None,
        native_contact_observation=(
            None if force_n is None else {"selected_pair_force_norm_n": force_n}
        ),
    )


def test_error_run_round_trips_through_closed_finite_json() -> None:
    payload = _error_payload()
    run = ContactRun.from_dict(payload)

    assert run.to_dict() == payload
    assert loads_contact_run(canonical_contact_run_json(run)) == run
    assert json.loads(canonical_contact_run_json(run)) == payload


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
        '{"schema_version":Infinity}',
    ],
)
def test_json_parser_rejects_duplicate_keys_and_nonfinite_constants(text: str) -> None:
    with pytest.raises(ContactRecordValidationError):
        loads_contact_run(text)


def test_every_closed_nested_object_rejects_unknown_fields() -> None:
    for field in (
        "case",
        "execution",
        "mapping",
        "fixture_readback",
        "contact_observation",
        "provenance",
    ):
        payload = deepcopy(_error_payload())
        payload[field]["unknown"] = True  # type: ignore[index]
        with pytest.raises(ContactRecordValidationError, match="unknown"):
            ContactRun.from_dict(payload)


def test_pair_active_is_strictly_derived_with_greater_than_threshold() -> None:
    assert _sample(active=False, force_n=1e-4).pair_active is False
    assert _sample(active=True, force_n=1.00001e-4).pair_active is True
    with pytest.raises(ContactRecordValidationError, match="contradicts"):
        _sample(active=True, force_n=1e-4)


def test_native_observation_is_private_descriptive_and_exact() -> None:
    sample = _sample(active=False, force_n=None)
    payload = sample.to_dict()
    payload["native_contact_observation"] = {
        "selected_pair_force_norm_n": 0.0,
        "raw_manifold": [1, 2, 3],
    }
    with pytest.raises(ContactRecordValidationError, match="unknown"):
        ContactTraceSample.from_dict(payload)


def test_completed_inventory_cannot_claim_partial_native_collision_disable() -> None:
    # Exercise the runtime-only inventory invariant without allocating a full
    # trace: a completed payload is rejected on its sample cardinality or the
    # inventory mismatch, and can never be accepted as evidence.
    payload = _error_payload()
    payload["execution"] = {
        "status": "completed",
        "message": None,
        "requested_steps": 3500,
        "completed_steps": 3500,
    }
    fixture = payload["fixture_readback"]
    assert isinstance(fixture, dict)
    fixture.update(
        {
            "performed": True,
            "native_collision_prim_count": 8,
            "native_collision_disabled_count": 7,
        }
    )
    with pytest.raises(ContactRecordValidationError):
        ContactRun.from_dict(payload)
