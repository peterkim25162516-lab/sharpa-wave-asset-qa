from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from wave_asset_qa.contact.contracts import (
    CONTACT_C0_MANIFEST_ID,
    CONTACT_C0_MANIFEST_SCHEMA_VERSION,
    CONTACT_C0_PAIR_ID,
    CONTACT_C0_RUN_SCHEMA_VERSION,
    ClaimBoundary,
    ContactCondition,
    ContactManifest,
)
from wave_asset_qa.contact.scenarios import (
    ContactTimestepVariant,
    canonical_contact_manifest_json,
    contact_manifest_sha256,
    expand_contact_cases,
    load_contact_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"
MANIFEST_SCHEMA_PATH = ROOT / "schemas" / "contact-c0-manifest.schema.json"
RUN_SCHEMA_PATH = ROOT / "schemas" / "contact-c0-run.schema.json"

MANIFEST_FILE_SHA256 = "73cc9dedffc1ea0600052fc1b78696eda292f1f34c7987ec1f433043e65bf215"
MANIFEST_SEMANTIC_SHA256 = "cb02398665dbed5722b6f9c44fca1c25a78ed495eb63cf5008e875672ceb930e"
MANIFEST_SCHEMA_FILE_SHA256 = "8a0101837f63abe483e1b93f509d41e024f5e137cabe19657e7171752e0c15ca"
RUN_SCHEMA_FILE_SHA256 = "08adc224e03a2dba52ed3b09868e88c60999dff4a10aaa8cf2fbda7e75822a17"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_contact_manifest_is_closed_frozen_and_round_trips() -> None:
    source_bytes = MANIFEST_PATH.read_bytes()
    source = _json(MANIFEST_PATH)
    manifest = load_contact_manifest(MANIFEST_PATH)

    assert sha256(source_bytes).hexdigest() == MANIFEST_FILE_SHA256
    assert contact_manifest_sha256(manifest) == MANIFEST_SEMANTIC_SHA256
    assert manifest.schema_version == CONTACT_C0_MANIFEST_SCHEMA_VERSION == 1
    assert CONTACT_C0_RUN_SCHEMA_VERSION == 1
    assert manifest.manifest_id == CONTACT_C0_MANIFEST_ID
    assert manifest.to_dict() == source
    assert ContactManifest.from_dict(manifest.to_dict()) == manifest
    assert json.loads(canonical_contact_manifest_json(manifest)) == source
    assert len(contact_manifest_sha256(manifest)) == 64
    assert manifest.expected_case_count == 32
    assert manifest.expected_joint_mapping_count == 44
    assert manifest.expected_distal_frame_mapping_count == 10
    assert manifest.fixture.probe.parent_frame_name(manifest.hands[0].side) == (
        "left_index_DP"
    )
    assert manifest.fixture.probe.parent_frame_name(manifest.hands[1].side) == (
        "right_index_DP"
    )


def test_fixture_and_pair_observation_profile_are_exact() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    fixture = manifest.fixture

    assert fixture.probe.local_center_m == (0.025, 0.0, 0.0)
    assert fixture.probe.radius_m == 0.005
    assert fixture.probe.preserve_mass_properties is True
    assert fixture.target.world_center_m == (0.0, 0.0, 0.120)
    assert fixture.target.half_extents_m == (0.150, 0.150, 0.010)
    assert fixture.target.top_surface_z_m == 0.130
    assert fixture.material.to_dict() == {
        "static_friction": 0.0,
        "dynamic_friction": 0.0,
        "restitution": 0.0,
    }
    policy = fixture.collision_policy
    assert policy.native_hand_collisions_enabled is False
    assert policy.self_collisions_enabled is False
    assert policy.ccd_enabled is False
    assert policy.allowed_pair_id == CONTACT_C0_PAIR_ID
    assert policy.mujoco_condim == 1
    assert policy.pair_active_source == "direct_filtered_selected_pair_force_norm_n"
    assert policy.pair_active_force_threshold_n == 1e-4
    assert policy.pair_active_threshold_freeze_basis == (
        "excluded_engineering_pilot_before_formal_source_freeze"
    )
    assert policy.excluded_pilot_scientific_evidence is False
    assert policy.raw_pair_force_cross_simulator_comparable is False
    assert policy.expected_enabled_collision_shape_count == 2
    assert policy.expected_filtered_sensor_body_count == 1
    assert policy.expected_filtered_target_count == 1


def test_matrix_is_exactly_32_fresh_process_cases_in_canonical_order() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    cases = expand_contact_cases(manifest)
    expected_steps = {
        ContactTimestepVariant.BASE: 3500,
        ContactTimestepVariant.HALVED: 7000,
    }

    assert len(cases) == len({case.case_id for case in cases}) == 32
    assert cases[0].case_id == "mujoco.left.press_hold_release.contact.base.r01"
    assert cases[-1].case_id == "ovphysx.right.press_hold_release.sham.halved.r02"
    assert sum(case.condition is ContactCondition.CONTACT for case in cases) == 16
    assert sum(case.condition is ContactCondition.SHAM for case in cases) == 16
    assert sum(
        case.timestep_variant is ContactTimestepVariant.BASE for case in cases
    ) == 16
    assert sum(
        case.timestep_variant is ContactTimestepVariant.HALVED for case in cases
    ) == 16
    assert {case.dt_s for case in cases} == {0.002, 0.001}
    assert manifest.scenario.steps == 3500
    assert all(
        round(manifest.scenario.duration_s / case.dt_s)
        == expected_steps[case.timestep_variant]
        for case in cases
    )
    assert all(case.pair_collision_enabled is (case.condition is ContactCondition.CONTACT) for case in cases)
    assert manifest.run_policy.repeat_count == 2
    assert manifest.run_policy.fresh_process_per_case is True
    assert manifest.run_policy.minimum_completion_fraction == 1.0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update({"extra": 1}), "unknown field"),
        (lambda data: data.pop("fixture"), "missing required"),
        (
            lambda data: data["fixture"]["probe"].update({"radius_m": 0.006}),
            "radius_m must be 0.005",
        ),
        (
            lambda data: data["fixture"]["collision_policy"].update(
                {"pair_active_force_threshold_n": 0.1}
            ),
            "pair_active_force_threshold_n must be 0.0001",
        ),
        (
            lambda data: data["conditions"].reverse(),
            "manifest.conditions must be",
        ),
        (
            lambda data: data["hands"].reverse(),
            "canonical left then right",
        ),
        (
            lambda data: data["run_policy"].update({"repeat_count": 1}),
            "repeat_count must be 2",
        ),
        (
            lambda data: data["thresholds"]["cross_simulator"].update(
                {"blocked_travel_max_abs_m": 0.02}
            ),
            "blocked_travel_max_abs_m must be 0.002",
        ),
        (
            lambda data: data["thresholds"]["admission"].update(
                {"target_readback_max_abs_error_rad": 1e-6}
            ),
            "unknown field",
        ),
        (
            lambda data: data["claim_boundary"].update(
                {"native_fingertip_geometry_validated": True}
            ),
            "native_fingertip_geometry_validated must be False",
        ),
    ],
)
def test_manifest_mutations_fail_closed(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    payload = deepcopy(_json(MANIFEST_PATH))
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        ContactManifest.from_dict(payload)


def test_claim_boundary_is_synthetic_only() -> None:
    boundary = load_contact_manifest(MANIFEST_PATH).claim_boundary

    assert boundary == ClaimBoundary(
        synthetic_fixture_only=True,
        native_fingertip_geometry_validated=False,
        full_isaac_sim=False,
        dual_hand=False,
        floating_base=False,
        hardware=False,
        sim2real=False,
    )


def test_admission_contains_only_run_record_verifiable_thresholds() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    admission = manifest.thresholds.admission.to_dict()

    assert set(admission) == {
        "fixture_geometry_max_abs_error_m",
        "material_zero_max_abs_error",
        "initial_gap_min_m",
        "sham_hold_mean_gap_max_m",
        "recovery_gap_min_m",
        "contact_hold_duty_min",
    }
    forbidden = {
        "target_readback_max_abs_error_rad",
        "effort_formula_max_abs_error_nm",
        "effort_clip_max_abs_error_nm",
        "maximum_effort_clip_count",
    }
    manifest_schema_text = MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8")
    run_schema_text = RUN_SCHEMA_PATH.read_text(encoding="utf-8")
    assert forbidden.isdisjoint(admission)
    assert all(name not in manifest_schema_text for name in forbidden)
    assert all(name not in run_schema_text for name in forbidden)


def test_plan_uses_strict_pair_threshold_and_disclaims_controller_gates() -> None:
    plan = (ROOT / "docs" / "CONTACT_C0_PLAN.md").read_text(encoding="utf-8")
    normalized_plan = " ".join(plan.split())

    assert "filtered_selected_pair_force_norm_n[k] > 1e-4 N" in plan
    assert "filtered_selected_pair_force_norm_n[k] >= 1e-4 N" not in plan
    assert "recovery minimum gap is at least `0.050 m`" in normalized_plan
    assert "recovery mean gap is at least" not in normalized_plan
    assert "none is a C0 admission gate" in plan
    assert "immediate target readback error is at most" not in plan


def test_published_schemas_are_draft_2020_12_and_recursively_closed() -> None:
    assert sha256(MANIFEST_SCHEMA_PATH.read_bytes()).hexdigest() == (
        MANIFEST_SCHEMA_FILE_SHA256
    )
    assert sha256(RUN_SCHEMA_PATH.read_bytes()).hexdigest() == RUN_SCHEMA_FILE_SHA256
    manifest_schema = _json(MANIFEST_SCHEMA_PATH)
    run_schema = _json(RUN_SCHEMA_PATH)

    for schema in (manifest_schema, run_schema):
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False

    assert manifest_schema["properties"]["schema_version"] == {"const": 1}
    collision = manifest_schema["$defs"]["fixture"]["properties"][
        "collision_policy"
    ]
    assert collision["additionalProperties"] is False
    assert collision["properties"]["pair_active_force_threshold_n"] == {
        "const": 0.0001
    }
    assert manifest_schema["$defs"]["scenario"]["additionalProperties"] is False
    assert manifest_schema["$defs"]["thresholds"]["additionalProperties"] is False
    assert manifest_schema["$defs"]["claimBoundary"]["additionalProperties"] is False

    assert run_schema["properties"]["schema_version"] == {"const": 1}
    for definition in (
        "case",
        "execution",
        "mapping",
        "fixtureReadback",
        "contactObservation",
        "sample",
        "provenance",
    ):
        assert run_schema["$defs"][definition]["additionalProperties"] is False
    sample = run_schema["$defs"]["sample"]
    assert sample["properties"]["pair_active"] == {"type": "boolean"}
    native = sample["properties"]["native_contact_observation"]["oneOf"][1]
    assert native["additionalProperties"] is False
    fixture_required = set(run_schema["$defs"]["fixtureReadback"]["required"])
    assert {
        "native_collision_prim_count",
        "native_collision_disabled_count",
        "enabled_collision_shape_count",
        "collision_inventory_sha256",
        "mass_properties_preserved",
    } <= fixture_required
    observation_required = set(
        run_schema["$defs"]["contactObservation"]["required"]
    )
    assert {"filtered_sensor_body_count", "filtered_target_count"} <= (
        observation_required
    )


def test_manifest_and_schemas_have_stable_nonempty_byte_identities() -> None:
    for path in (MANIFEST_PATH, MANIFEST_SCHEMA_PATH, RUN_SCHEMA_PATH):
        source = path.read_bytes()
        assert source.endswith(b"\n")
        assert len(source) > 1000
        assert len(sha256(source).hexdigest()) == 64
