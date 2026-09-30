from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from wave_asset_qa.contact.bundle import canonical_json_sha256
from wave_asset_qa.contact.scenarios import load_contact_manifest
from wave_asset_qa.parity.contracts import Simulator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"


def _load_finalizer():  # type: ignore[no-untyped-def]
    path = ROOT / "scripts" / "finalize_contact_c0.py"
    spec = importlib.util.spec_from_file_location("test_contact_c0_finalizer_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mujoco_private(*, condition: str = "contact") -> dict[str, object]:
    pair_enabled = condition == "contact"
    native_items = [
        {
            "ordinal": 0,
            "name": "native",
            "parent_body": "left_index_DP",
            "original_contype": 1,
            "original_conaffinity": 1,
            "effective_contype": 0,
            "effective_conaffinity": 0,
            "compiled_geom_id": 0,
            "compiled_contype": 0,
            "compiled_conaffinity": 0,
            "disabled": True,
        }
    ]
    mass = {
        "body_mass": 0.01,
        "body_inertia": [0.001, 0.001, 0.001],
        "body_ipos": [0.0, 0.0, 0.0],
        "body_iquat": [1.0, 0.0, 0.0, 0.0],
    }
    fixture = {
        "schema_version": 1,
        "backend": "mujoco",
        "overlay": {
            "method": "MjSpec_in_memory_pre_compile",
            "canonical_source_edited": False,
            "source_sha256_before": "a" * 64,
            "source_sha256_after": "a" * 64,
        },
        "geometry": {
            "probe": {
                "canonical_id": "synthetic_index_probe",
                "parent_frame": "left_index_DP",
                "compiled_geom_id": 1,
                "shape": "sphere",
                "local_center_m": [0.025, 0.0, 0.0],
                "radius_m": 0.005,
            },
            "target": {
                "canonical_id": "static_box",
                "compiled_geom_id": 2,
                "shape": "box",
                "center_world_m": [0.0, 0.0, 0.120],
                "half_extents_m": [0.150, 0.150, 0.010],
                "top_world_m": 0.130,
                "static": True,
            },
        },
        "collision": {
            "allowed_pair_id": "synthetic_index_probe__static_box",
            "condition": condition,
            "pair_collision_enabled": pair_enabled,
            "pair_mask_source": "MjSpec explicit pair presence",
            "compiled_explicit_pair_count": 1 if pair_enabled else 0,
            "compiled_condim": 1 if pair_enabled else None,
            "probe_contype": 0,
            "probe_conaffinity": 0,
            "target_contype": 0,
            "target_conaffinity": 0,
            "ccd_enabled": False,
        },
        "material": {
            "friction_coefficient": 0.0,
            "restitution_coefficient": 0.0,
            "compiled_pair_friction": [0.0] * 5 if pair_enabled else None,
            "compiled_pair_solref": [0.02, 1.0] if pair_enabled else None,
            "restitution_semantics": "unit-test frozen zero",
        },
        "mass_properties": {
            "preserved": True,
            "source": mass,
            "effective": deepcopy(mass),
            "synthetic_probe_density": 0.0,
        },
        "native_collision_disabled_inventory": {
            "count": 1,
            "all_disabled": True,
            "items": native_items,
        },
        "solver": {"dt_s": 0.002, "gravity_m_s2": [0.0, 0.0, 0.0]},
        "collision_inventory_hash_preimage": {
            "native_collision_items": deepcopy(native_items),
            "synthetic_enabled_shapes": ["synthetic_index_probe", "static_box"],
        },
    }
    runtime = {
        "backend": "mujoco",
        "backend_version": "3.12.0",
        "python_version": "3.12.6",
        "platform": "Windows-11-10.0.26200-SP0",
        "device": "cpu",
        "solver": "2",
        "integrator": "0",
        "control_path": "MjData.ctrl -> mj_step",
        "contact_path": "explicit pair -> mj_contactForce",
    }
    return {
        "fixture_overlay": fixture,
        "runtime_fingerprint": runtime,
        "unexpected_pair_observations": [],
    }


def _ov_private(*, condition: str = "contact") -> dict[str, object]:
    pair_enabled = condition == "contact"
    parent_path = "/World/Env_0/Robot/left_index_DP"
    probe_path = parent_path + "/waveqa_c0_index_probe"
    target_path = "/World/Env_0/waveqa_c0_target_box"
    native_items = [
        {
            "prim_path": "/World/Env_0/Robot/native",
            "original_collision_enabled": True,
            "effective_collision_enabled": False,
        }
    ]
    source_mass = {
        "physics:mass": {
            "valid": True,
            "authored": True,
            "value": 0.01,
            "type": "float",
        },
        "physics:density": {
            "valid": True,
            "authored": False,
            "value": None,
            "type": None,
        },
        "physics:centerOfMass": {
            "valid": True,
            "authored": True,
            "value": [0.0, 0.0, 0.0],
            "type": "Vec3f",
        },
        "physics:diagonalInertia": {
            "valid": True,
            "authored": True,
            "value": [0.001, 0.001, 0.001],
            "type": "Vec3f",
        },
        "physics:principalAxes": {
            "valid": True,
            "authored": True,
            "value": [0.0, 0.0, 0.0, 1.0],
            "type": "Quatf",
        },
    }
    runtime_mass = {
        "body_name": "left_index_DP",
        "body_prim_path": parent_path,
        "source": (
            "physx_5_9_0_native_mass_properties_via_ovphysx_get_physx_ptr_"
            "post_first_reset"
        ),
        "physx_version": "5.9.0",
        "helper_sha256": "5" * 64,
        "mass_kg": 0.01,
        "center_of_mass_pose_b": {
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "mass_space_inertia_diagonal_kg_m2": [0.001, 0.001, 0.001],
    }
    final_items = [
        {"prim_path": native_items[0]["prim_path"], "collision_enabled": False},
        {"prim_path": probe_path, "collision_enabled": True},
        {"prim_path": target_path, "collision_enabled": True},
    ]
    fixture = {
        "schema_version": 1,
        "backend": "ovphysx",
        "overlay": {
            "method": "anonymous_session_sublayer_pre_first_reset",
            "canonical_source_edited": False,
            "first_reset_count_at_readback": 0,
        },
        "geometry": {
            "probe": {
                "canonical_id": "synthetic_index_probe",
                "parent_prim_path": parent_path,
                "prim_path": probe_path,
                "shape": "sphere",
                "local_center_m": [0.025, 0.0, 0.0],
                "radius_m": 0.005,
            },
            "target": {
                "canonical_id": "static_box",
                "prim_path": target_path,
                "shape": "box",
                "center_world_m": [0.0, 0.0, 0.120],
                "half_extents_m": [0.150, 0.150, 0.010],
                "top_world_m": 0.130,
                "static": True,
            },
        },
        "collision": {
            "allowed_pair_id": "synthetic_index_probe__static_box",
            "pair_collision_enabled": pair_enabled,
            "pair_mask_source": "PhysicsFilteredPairsAPI",
            "filtered_pair_targets": [] if pair_enabled else [target_path],
            "probe_collision_enabled": True,
            "target_collision_enabled": True,
            "self_collision_enabled": False,
            "ccd_enabled": False,
            "ccd_readback_source": (
                "physx_5_9_0_native_flags_via_ovphysx_get_physx_ptr_"
                "post_first_reset"
            ),
            "ccd_physx_version": "5.9.0",
            "ccd_helper_sha256": "5" * 64,
            "ccd_scene_prim_path": "/World/physicsScene",
            "ccd_body_prim_path": parent_path,
            "ccd_scene_flags": 1,
            "ccd_body_flags": 8,
            "ccd_scene_mask": 2,
            "ccd_body_mask": 4,
            "ccd_scene_enabled": False,
            "ccd_body_enabled": False,
            "ccd_usd_input_source": "post_first_reset_live_composed_usd",
            "ccd_usd_scene_enabled": False,
            "ccd_usd_body_enabled": False,
        },
        "material": {
            "prim_path": "/World/Env_0/zero_material",
            "static_friction": 0.0,
            "dynamic_friction": 0.0,
            "friction_coefficient": 0.0,
            "restitution_coefficient": 0.0,
        },
        "mass_properties": {
            "preserved": True,
            "usd_composed_preserved": True,
            "runtime_verified": True,
            "source": source_mass,
            "effective": deepcopy(source_mass),
            "synthetic_probe_mass_api_applied": False,
            "runtime_effective": runtime_mass,
            "runtime_checks": {
                "body_name_exact": True,
                "mass_abs_error_kg": 0.0,
                "center_of_mass_max_abs_error_m": 0.0,
                "inertia_link_tensor_max_abs_error_kg_m2": 0.0,
                "inertia_symmetry_max_abs_error_kg_m2": 0.0,
                "runtime_inertia_min_eigenvalue_kg_m2": 0.001,
                "principal_axes_representation_angle_error_rad": 0.0,
                "expected_link_inertia_tensor_kg_m2": [
                    [0.001, 0.0, 0.0],
                    [0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.001],
                ],
                "runtime_link_inertia_tensor_kg_m2": [
                    [0.001, 0.0, 0.0],
                    [0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.001],
                ],
                "expected_inertia_eigenvalues_kg_m2": [0.001] * 3,
                "runtime_inertia_eigenvalues_kg_m2": [0.001] * 3,
                "tolerances": {
                    "mass_relative": 5.0e-5,
                    "mass_absolute_kg": 1.0e-8,
                    "center_of_mass_absolute_m": 1.0e-7,
                    "inertia_relative": 5.0e-5,
                    "inertia_absolute_kg_m2": 1.0e-12,
                },
            },
        },
        "native_collision_disabled_inventory": {
            "count": 1,
            "all_disabled": True,
            "items": native_items,
        },
        "final_collision_inventory": {
            "count": len(final_items),
            "enabled_count": 2,
            "enabled_paths": sorted([probe_path, target_path]),
            "items": final_items,
        },
        "solver": {
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "renderer": False,
            "camera": False,
            "ground_plane": False,
        },
        "sensor_readback": {
            "body_prim_path": parent_path,
            "filter_prim_paths_expr": [target_path],
            "filter_count": 1,
            "update_period_s": 0.0,
            "debug_visualization": False,
            "created_before_first_reset": True,
            "sample_update_force_recompute": True,
        },
        "collision_inventory_hash_preimage": {
            "native_collision_items": deepcopy(native_items),
            "synthetic_enabled_shapes": ["synthetic_index_probe", "static_box"],
        },
    }
    runtime = {
        "backend": "ovphysx",
        "python_version": "3.12.14",
        "platform": "Linux-unit-test",
        "device": "cuda:0",
        "dt_s": 0.002,
        "module_versions": {},
        "solver_config_type": "OvPhysxCfg",
        "control_path": "IdealPDActuator",
        "contact_path": "ContactSensor.data.force_matrix_w",
        "sensor_update_order": "step -> update -> sample",
        "kitless": True,
        "renderer": False,
        "camera": False,
        "scene_inventory_verification_phase": (
            "after_full_trajectory_before_cleanup"
        ),
        "forbidden_module_verification_phase": "after_runtime_cleanup",
        "forbidden_module_inventory": [],
        "camera_prim_paths": [],
        "created_sensor_types": ["ContactSensor"],
    }
    return {
        "fixture_overlay": fixture,
        "runtime_fingerprint": runtime,
        "unexpected_pair_observation": {
            "status": "proved_empty_by_exclusive_enabled_shape_inventory",
            "unexpected_pairs": [],
        },
    }


def _run_for(private: dict[str, object], *, backend: str):  # type: ignore[no-untyped-def]
    fixture = private["fixture_overlay"]
    runtime = private["runtime_fingerprint"]
    assert isinstance(fixture, dict) and isinstance(runtime, dict)
    preimage = fixture["collision_inventory_hash_preimage"]
    return SimpleNamespace(
        case=SimpleNamespace(simulator=backend),
        fixture_readback={
            "probe_parent_frame_name": "left_index_DP",
            "probe_local_center_m": [0.025, 0.0, 0.0],
            "probe_radius_m": 0.005,
            "target_world_center_m": [0.0, 0.0, 0.120],
            "target_half_extents_m": [0.150, 0.150, 0.010],
            "target_top_surface_z_m": 0.130,
            "static_friction": 0.0,
            "dynamic_friction": 0.0,
            "restitution": 0.0,
            "native_collision_prim_count": 1,
            "native_collision_disabled_count": 1,
            "enabled_collision_shape_count": 2,
            "collision_inventory_sha256": canonical_json_sha256(preimage),
        },
        contact_observation={
            "filtered_sensor_body_count": 1,
            "filtered_target_count": 1,
        },
        provenance={
            "fixture_overlay_sha256": canonical_json_sha256(fixture),
            "runtime_fingerprint_sha256": canonical_json_sha256(runtime),
        },
    )


def test_private_adapter_cross_binds_native_inventory_and_collision_preimage() -> None:
    finalizer = _load_finalizer()
    private = _mujoco_private()
    finalizer._validate_private_adapter(
        private, _run_for(private, backend="mujoco"), condition="contact"
    )

    mismatched_count_run = _run_for(private, backend="mujoco")
    mismatched_count_run.fixture_readback["native_collision_prim_count"] = 2
    with pytest.raises(finalizer.ContactC0FinalizationError, match="inventory"):
        finalizer._validate_private_adapter(
            private, mismatched_count_run, condition="contact"
        )

    mismatched_preimage = deepcopy(private)
    fixture = mismatched_preimage["fixture_overlay"]
    assert isinstance(fixture, dict)
    fixture["collision_inventory_hash_preimage"]["native_collision_items"] = []
    with pytest.raises(finalizer.ContactC0FinalizationError, match="contents"):
        finalizer._validate_private_adapter(
            mismatched_preimage,
            _run_for(mismatched_preimage, backend="mujoco"),
            condition="contact",
        )

    enabled_ccd = deepcopy(private)
    enabled_ccd_fixture = enabled_ccd["fixture_overlay"]
    assert isinstance(enabled_ccd_fixture, dict)
    enabled_ccd_fixture["collision"]["ccd_enabled"] = True
    with pytest.raises(finalizer.ContactC0FinalizationError, match="CCD"):
        finalizer._validate_private_adapter(
            enabled_ccd,
            _run_for(enabled_ccd, backend="mujoco"),
            condition="contact",
        )


@pytest.mark.parametrize("condition", ["contact", "sham"])
def test_ov_pair_mask_and_post_cook_mass_readback_are_bound(condition: str) -> None:
    finalizer = _load_finalizer()
    private = _ov_private(condition=condition)
    finalizer._validate_private_adapter(
        private, _run_for(private, backend="ovphysx"), condition=condition
    )

    unverified = deepcopy(private)
    unverified_fixture = unverified["fixture_overlay"]
    assert isinstance(unverified_fixture, dict)
    unverified_fixture["mass_properties"]["runtime_verified"] = False
    with pytest.raises(finalizer.ContactC0FinalizationError, match="runtime verification"):
        finalizer._validate_private_adapter(
            unverified,
            _run_for(unverified, backend="ovphysx"),
            condition=condition,
        )

    forged_ccd = deepcopy(private)
    forged_ccd_fixture = forged_ccd["fixture_overlay"]
    assert isinstance(forged_ccd_fixture, dict)
    forged_ccd_fixture["collision"]["ccd_body_enabled"] = True
    with pytest.raises(finalizer.ContactC0FinalizationError, match="collision boundary"):
        finalizer._validate_private_adapter(
            forged_ccd,
            _run_for(forged_ccd, backend="ovphysx"),
            condition=condition,
        )

    forged_native_flag = deepcopy(private)
    forged_native_fixture = forged_native_flag["fixture_overlay"]
    assert isinstance(forged_native_fixture, dict)
    forged_native_fixture["collision"]["ccd_body_flags"] = 12
    with pytest.raises(finalizer.ContactC0FinalizationError, match="collision boundary"):
        finalizer._validate_private_adapter(
            forged_native_flag,
            _run_for(forged_native_flag, backend="ovphysx"),
            condition=condition,
        )

    with pytest.raises(finalizer.ContactC0FinalizationError, match="collision boundary"):
        finalizer._validate_private_adapter(
            private,
            _run_for(private, backend="ovphysx"),
            condition=condition,
            native_helper_sha256="6" * 64,
        )

    forged_headless = deepcopy(private)
    forged_headless_runtime = forged_headless["runtime_fingerprint"]
    assert isinstance(forged_headless_runtime, dict)
    forged_headless_runtime["forbidden_module_inventory"] = ["omni.kit"]
    with pytest.raises(finalizer.ContactC0FinalizationError, match="headless"):
        finalizer._validate_private_adapter(
            forged_headless,
            _run_for(forged_headless, backend="ovphysx"),
            condition=condition,
        )

    forged_camera_prim = deepcopy(private)
    forged_camera_runtime = forged_camera_prim["runtime_fingerprint"]
    assert isinstance(forged_camera_runtime, dict)
    forged_camera_runtime["camera_prim_paths"] = ["/World/Camera"]
    with pytest.raises(finalizer.ContactC0FinalizationError, match="headless"):
        finalizer._validate_private_adapter(
            forged_camera_prim,
            _run_for(forged_camera_prim, backend="ovphysx"),
            condition=condition,
        )

    forged_sensor_type = deepcopy(private)
    forged_sensor_runtime = forged_sensor_type["runtime_fingerprint"]
    assert isinstance(forged_sensor_runtime, dict)
    forged_sensor_runtime["created_sensor_types"] = ["ContactSensor", "Camera"]
    with pytest.raises(finalizer.ContactC0FinalizationError, match="headless"):
        finalizer._validate_private_adapter(
            forged_sensor_type,
            _run_for(forged_sensor_type, backend="ovphysx"),
            condition=condition,
        )

    forged_scene_phase = deepcopy(private)
    forged_scene_phase_runtime = forged_scene_phase["runtime_fingerprint"]
    assert isinstance(forged_scene_phase_runtime, dict)
    forged_scene_phase_runtime["scene_inventory_verification_phase"] = (
        "after_runtime_cleanup"
    )
    with pytest.raises(finalizer.ContactC0FinalizationError, match="headless"):
        finalizer._validate_private_adapter(
            forged_scene_phase,
            _run_for(forged_scene_phase, backend="ovphysx"),
            condition=condition,
        )

    forged_module_phase = deepcopy(private)
    forged_module_phase_runtime = forged_module_phase["runtime_fingerprint"]
    assert isinstance(forged_module_phase_runtime, dict)
    forged_module_phase_runtime["forbidden_module_verification_phase"] = (
        "after_full_trajectory_before_cleanup"
    )
    with pytest.raises(finalizer.ContactC0FinalizationError, match="headless"):
        finalizer._validate_private_adapter(
            forged_module_phase,
            _run_for(forged_module_phase, backend="ovphysx"),
            condition=condition,
        )


def test_ov_final_inventory_rejects_a_third_enabled_collider() -> None:
    finalizer = _load_finalizer()
    private = _ov_private()
    fixture = private["fixture_overlay"]
    assert isinstance(fixture, dict)
    final_inventory = fixture["final_collision_inventory"]
    final_inventory["items"].append(
        {"prim_path": "/World/Env_0/unexpected", "collision_enabled": True}
    )
    final_inventory["count"] += 1
    final_inventory["enabled_count"] += 1
    final_inventory["enabled_paths"].append("/World/Env_0/unexpected")
    with pytest.raises(finalizer.ContactC0FinalizationError, match="enabled collision"):
        finalizer._validate_private_adapter(
            private, _run_for(private, backend="ovphysx"), condition="contact"
        )


def test_ov_ccd_scene_path_must_be_consistent_across_all_cases() -> None:
    finalizer = _load_finalizer()
    finalizer._validate_remote_ccd_scene_path_consistency({"/physicsScene"})
    with pytest.raises(finalizer.ContactC0FinalizationError, match="differs"):
        finalizer._validate_remote_ccd_scene_path_consistency(
            {"/physicsScene", "/World/physicsScene"}
        )


def test_ov_mass_proof_is_independently_recomputed_from_authored_source() -> None:
    finalizer = _load_finalizer()
    private = _ov_private()

    forged_runtime = deepcopy(private)
    forged_fixture = forged_runtime["fixture_overlay"]
    assert isinstance(forged_fixture, dict)
    forged_fixture["mass_properties"]["runtime_effective"]["mass_kg"] = 0.02
    with pytest.raises(finalizer.ContactC0FinalizationError, match="recompute"):
        finalizer._validate_private_adapter(
            forged_runtime,
            _run_for(forged_runtime, backend="ovphysx"),
            condition="contact",
        )

    # This difference is above 5e-5 of the authored 0.01 kg mass, but below
    # 5e-5 of the larger runtime mass.  The expected/authored denominator must
    # therefore reject it.
    authored_limit = deepcopy(private)
    authored_fixture = authored_limit["fixture_overlay"]
    assert isinstance(authored_fixture, dict)
    authored_mass = authored_fixture["mass_properties"]
    runtime_mass = 0.01000050001
    authored_mass["runtime_effective"]["mass_kg"] = runtime_mass
    authored_mass["runtime_checks"]["mass_abs_error_kg"] = runtime_mass - 0.01
    with pytest.raises(finalizer.ContactC0FinalizationError, match="tolerance proof"):
        finalizer._validate_private_adapter(
            authored_limit,
            _run_for(authored_limit, backend="ovphysx"),
            condition="contact",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("body_prim_path", "/World/Env_0/Robot/wrong_DP"),
        ("source", "ArticulationData post-reset readback"),
        ("physx_version", "5.8.0"),
        ("helper_sha256", "6" * 64),
    ],
)
def test_ov_native_mass_provenance_is_bound_to_ccd_helper(
    field: str, value: str
) -> None:
    finalizer = _load_finalizer()
    private = _ov_private()
    fixture = private["fixture_overlay"]
    assert isinstance(fixture, dict)
    fixture["mass_properties"]["runtime_effective"][field] = value
    with pytest.raises(finalizer.ContactC0FinalizationError, match="provenance"):
        finalizer._validate_private_adapter(
            private,
            _run_for(private, backend="ovphysx"),
            condition="contact",
        )


def test_ov_native_mass_diagonal_is_independently_recomputed() -> None:
    finalizer = _load_finalizer()
    private = _ov_private()
    fixture = private["fixture_overlay"]
    assert isinstance(fixture, dict)
    fixture["mass_properties"]["runtime_effective"][
        "mass_space_inertia_diagonal_kg_m2"
    ][0] = 0.002
    with pytest.raises(finalizer.ContactC0FinalizationError, match="recompute"):
        finalizer._validate_private_adapter(
            private,
            _run_for(private, backend="ovphysx"),
            condition="contact",
        )


def test_ov_native_mass_quaternion_must_be_unit_length() -> None:
    finalizer = _load_finalizer()
    private = _ov_private()
    fixture = private["fixture_overlay"]
    assert isinstance(fixture, dict)
    fixture["mass_properties"]["runtime_effective"]["center_of_mass_pose_b"][
        "quaternion_xyzw"
    ] = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(finalizer.ContactC0FinalizationError, match="unit length"):
        finalizer._validate_private_adapter(
            private,
            _run_for(private, backend="ovphysx"),
            condition="contact",
        )


def _write_minimal_local_topology(root: Path, case_id: str) -> None:
    (root / "launcher").mkdir(parents=True)
    (root / "cases" / case_id).mkdir(parents=True)
    (root / "evidence-manifest.json").write_text("{}\n", encoding="utf-8")
    for name in ("provenance.json", "status.json"):
        (root / "launcher" / name).write_text("{}\n", encoding="utf-8")
    case_root = root / "cases" / case_id
    for name, payload in {
        f"{case_id}.run.json": "{}\n",
        "private-process.json": "{}\n",
        "private-adapter-evidence.json": "{}\n",
        "stdout.txt": "normal worker output\n",
        "stderr.txt": "[Error] [omni.physx.cooking.plugin] registry is false.\n",
        "exit-code.txt": "0\n",
    }.items():
        (case_root / name).write_text(payload, encoding="utf-8")


def test_exact_topology_and_fatal_worker_log_gate(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    case_id = "mujoco.left.press_hold_release.contact.base.r01"
    _write_minimal_local_topology(tmp_path, case_id)

    records = finalizer._validate_backend_topology(
        tmp_path,
        backend=Simulator.MUJOCO,
        expected_case_ids={case_id},
    )
    assert len(records) == 2

    unexpected = tmp_path / "cases" / case_id / "unlisted.bin"
    unexpected.write_bytes(b"extra")
    with pytest.raises(finalizer.ContactC0FinalizationError, match="topology"):
        finalizer._validate_backend_topology(
            tmp_path,
            backend=Simulator.MUJOCO,
            expected_case_ids={case_id},
        )
    unexpected.unlink()

    stderr = tmp_path / "cases" / case_id / "stderr.txt"
    stderr.write_text("Traceback (most recent call last):\n", encoding="utf-8")
    with pytest.raises(finalizer.ContactC0FinalizationError, match="python_traceback"):
        finalizer._validate_backend_topology(
            tmp_path,
            backend=Simulator.MUJOCO,
            expected_case_ids={case_id},
        )


def test_existing_gate0_and_public_hardware_boundary_remain_closed() -> None:
    finalizer = _load_finalizer()
    existing = finalizer._validate_existing_gate0(ROOT)
    assert existing["trajectory_t1_science_status"] == "DIVERGENT"
    assert existing["pass_ready"] is False

    summary = {
        "claim_boundary": {
            "synthetic_fixture_only": True,
            "hardware_in_scope": False,
            "sim2real_claimed": False,
            "native_hand_collision_geometry_validated": False,
            "dual_hand_interaction_in_scope": False,
            "floating_base_in_scope": False,
            "full_isaac_sim_in_scope": False,
        },
        "existing_gate0": {
            "trajectory_t1_science_status": "DIVERGENT",
            "pass_ready": False,
            "unchanged_by_contact_c0": True,
        },
    }
    finalizer._audit_public(summary, "simulation only", ("private-session",))
    summary["claim_boundary"]["hardware_in_scope"] = True
    with pytest.raises(finalizer.ContactC0FinalizationError, match="boundary"):
        finalizer._audit_public(summary, "simulation only", ("private-session",))


@pytest.mark.parametrize(
    "private_runtime_key",
    [
        "camera_prim_paths",
        "created_sensor_types",
        "scene_inventory_verification_phase",
        "forbidden_module_verification_phase",
        "forbidden_module_inventory",
    ],
)
def test_public_audit_rejects_each_private_runtime_key(
    private_runtime_key: str,
) -> None:
    finalizer = _load_finalizer()
    summary = {
        "claim_boundary": {
            "synthetic_fixture_only": True,
            "hardware_in_scope": False,
            "sim2real_claimed": False,
            "native_hand_collision_geometry_validated": False,
            "dual_hand_interaction_in_scope": False,
            "floating_base_in_scope": False,
            "full_isaac_sim_in_scope": False,
        },
        "existing_gate0": {
            "trajectory_t1_science_status": "DIVERGENT",
            "pass_ready": False,
            "unchanged_by_contact_c0": True,
        },
        "leaked_runtime": {private_runtime_key: "private-value"},
    }
    with pytest.raises(finalizer.ContactC0FinalizationError, match="private material"):
        finalizer._audit_public(summary, "simulation only", ())


def test_launcher_provenance_is_fail_closed_on_nonexact_fields(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    manifest = load_contact_manifest(MANIFEST_PATH)
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / "provenance.json").write_text(
        '{"unexpected": true}\n', encoding="utf-8"
    )
    (launcher / "status.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(finalizer.ContactC0FinalizationError, match="fields are not exact"):
        finalizer._validate_local_launcher_identity(
            tmp_path,
            manifest=manifest,
            expected_case_ids=[],
            source_revision="1" * 40,
            source_tree="2" * 40,
            manifest_file_sha256="3" * 64,
            manifest_semantic_sha256="4" * 64,
            project_root=ROOT,
        )


_REMOTE_ROOT = "/data/home/exampleuser/sharpa-wave-asset-qa-gate0"
_REMOTE_CASE = "ovphysx.left.press_hold_release.contact.base.r01"
_REMOTE_RUN = "ovphysx-contact-c0-unit-a1"
_REMOTE_SESSION = "contact-c0-unit-remote-a1"
_REMOTE_REVISION = "1" * 40
_REMOTE_TREE = "2" * 40
_REMOTE_ASSET = "3" * 64
_REMOTE_SNAPSHOT = "4" * 64
_REMOTE_GPU_UUID = "GPU-a13e44aa-98b3-5a73-b7cc-9cc0c7434625"
_REMOTE_GPU_NAME = "NVIDIA A800-SXM4-40GB"
_REMOTE_DRIVER = "570.158.01"
_REMOTE_HELPER_PATH = (
    f"{_REMOTE_ROOT}/results/{_REMOTE_RUN}/launcher/native/"
    "libwaveqa_physx_ccd_readback.so"
)
_REMOTE_HELPER_SHA256 = "5" * 64


def _gpu_state_text(
    *,
    query_status: int = 0,
    memory_mib: int = 0,
    utilization: int = 0,
    processes: tuple[str, ...] = (),
) -> str:
    return "\n".join(
        [
            "recorded_at_utc=2026-08-31T00:00:00Z",
            f"query_status={query_status}",
            "gpu_index=3",
            f"gpu_uuid={_REMOTE_GPU_UUID}",
            f"gpu_name={_REMOTE_GPU_NAME}",
            f"driver_version={_REMOTE_DRIVER}",
            f"memory_used_mib={memory_mib}",
            f"utilization_percent={utilization}",
            "compute_processes_begin",
            *processes,
            "compute_processes_end",
            "",
        ]
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_remote_security_fixture(tmp_path: Path, finalizer):  # type: ignore[no-untyped-def]
    case_root = tmp_path / "case"
    case_root.mkdir()
    process_identity = {
        "platform": "posix",
        "boot_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "pid": 22002,
        "process_start_ticks": 123456,
    }
    fresh_hash = finalizer.os_process_identity_sha256(process_identity)
    project = f"{_REMOTE_ROOT}/project/{_REMOTE_TREE}"
    output = f"{_REMOTE_ROOT}/results/{_REMOTE_RUN}/cases/{_REMOTE_CASE}"
    command = {
        "schema_version": 1,
        "case_id": _REMOTE_CASE,
        "argv": [
            f"{_REMOTE_ROOT}/env/bin/python",
            "-P",
            f"{project}/scripts/run_contact_c0_ovphysx_worker.py",
            "--asset-root",
            f"{_REMOTE_ROOT}/assets",
            "--manifest",
            f"{project}/configs/parity/contact_c0.json",
            "--case-id",
            _REMOTE_CASE,
            "--output-dir",
            output,
            "--session-id",
            _REMOTE_SESSION,
            "--source-revision",
            _REMOTE_REVISION,
            "--source-tree",
            _REMOTE_TREE,
            "--asset-tree-sha256",
            _REMOTE_ASSET,
            "--device",
            "cuda:0",
        ],
        "environment": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "3",
            "WAVEQA_PHYSX_CCD_HELPER_PATH": _REMOTE_HELPER_PATH,
            "WAVEQA_PHYSX_CCD_HELPER_SHA256": _REMOTE_HELPER_SHA256,
            "worker_device": "cuda:0",
            "DISPLAY": None,
            "WAYLAND_DISPLAY": None,
        },
    }
    process = {
        "python_executable_realpath": (
            f"{_REMOTE_ROOT}/toolchain/python/cpython-3.12.14/bin/python3.12"
        ),
        "worker_module_realpath": (
            f"{project}/src/wave_asset_qa/contact/worker.py"
        ),
        "os_process_identity": process_identity,
    }
    group = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": _REMOTE_CASE,
        "transport_pid": 22000,
        "owned_process_group_id": 22000,
    }
    binding = {
        "schema_version": 1,
        "visibility": "private_not_for_publication",
        "case_id": _REMOTE_CASE,
        "worker_pid": 22002,
        "owned_process_group_id": 22000,
        "fresh_process_identity_sha256": fresh_hash,
    }
    _write_json(case_root / "command.json", command)
    _write_json(case_root / "process.json", group)
    _write_json(case_root / "worker-group-binding.json", binding)
    (case_root / "owned-process-group.txt").write_text("22000\n", encoding="ascii")
    for suffix in ("01", "02", "03"):
        (case_root / f"gpu-preflight-{suffix}.txt").write_text(
            _gpu_state_text(), encoding="utf-8"
        )
    (case_root / "gpu-postflight-01.txt").write_text(
        _gpu_state_text(), encoding="utf-8"
    )
    (case_root / "gpu-monitor.txt").write_text(
        "sample_at_utc=2026-08-31T00:00:01Z owner_pgid=22000\n"
        "compute_processes_begin\n"
        f"{_REMOTE_GPU_UUID}, 22002, 128\n"
        "compute_processes_end\n",
        encoding="utf-8",
    )
    return case_root, process, fresh_hash


def _validate_remote_security_fixture(finalizer, case_root, process, fresh_hash, campaign=None):  # type: ignore[no-untyped-def]
    finalizer._validate_remote_case_security_evidence(
        case_root,
        case_id=_REMOTE_CASE,
        run_id=_REMOTE_RUN,
        session_id=_REMOTE_SESSION,
        source_revision=_REMOTE_REVISION,
        source_tree=_REMOTE_TREE,
        asset_tree_sha256=_REMOTE_ASSET,
        gpu_index=3,
        gpu_uuid=_REMOTE_GPU_UUID,
        gpu_name=_REMOTE_GPU_NAME,
        driver_version=_REMOTE_DRIVER,
        native_helper_path=_REMOTE_HELPER_PATH,
        native_helper_sha256=_REMOTE_HELPER_SHA256,
        process=process,
        fresh_process_identity_sha256=fresh_hash,
        experimental_campaign=campaign,
    )


def test_candidate_remote_command_requires_explicit_mode_and_bound_query_index(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    root, process, fresh_hash = _write_remote_security_fixture(tmp_path, finalizer)
    command = json.loads((root / 'command.json').read_text())
    command['argv'][2] = command['argv'][2].replace('run_contact_c0_ovphysx_worker.py', 'run_contact_async_worker.py')
    command['argv'][3:3] = ['--backend', 'ovphysx']
    command['environment']['CUDA_VISIBLE_DEVICES'] = '3'
    _write_json(root / 'command.json', command)
    _validate_remote_security_fixture(finalizer, root, process, fresh_hash, 'contact-c0-async-v1')
    with pytest.raises(finalizer.ContactC0FinalizationError, match='binding differs'):
        _validate_remote_security_fixture(finalizer, root, process, fresh_hash)
    command['environment']['CUDA_VISIBLE_DEVICES'] = _REMOTE_GPU_UUID
    _write_json(root / 'command.json', command)
    with pytest.raises(finalizer.ContactC0FinalizationError, match='binding differs'):
        _validate_remote_security_fixture(finalizer, root, process, fresh_hash, 'contact-c0-async-v1')


def _write_remote_native_helper_fixture(tmp_path: Path, finalizer):  # type: ignore[no-untyped-def]
    native = tmp_path / "launcher" / "native"
    include = native / "include"
    (include / "foundation").mkdir(parents=True)
    for relative in (
        "PxArticulationLink.h",
        "PxScene.h",
        "PxRigidBody.h",
        "foundation/PxPhysicsVersion.h",
    ):
        path = include / relative
        path.write_text(f"// {relative}\n", encoding="utf-8")
    records = list(finalizer.regular_tree_records(include))
    headers_root = finalizer.inventory_root_sha256(records)
    archive = (
        f"{_REMOTE_ROOT}/dependencies/ovphysx-0.4.13/"
        "ovphysx-linux-x86_64-0.4.13.tar.gz"
    )
    remote_native = f"{_REMOTE_ROOT}/results/{_REMOTE_RUN}/launcher/native"
    remote_include = f"{remote_native}/include"
    headers_manifest_path = native / "headers-manifest.json"
    headers_manifest = {
        "schema_version": 1,
        "sdk_archive_path": archive,
        "sdk_archive_sha256": finalizer._OVPHYSX_SDK_ARCHIVE_SHA256,
        "include_root_in_archive": "ovphysx/include",
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "records": records,
        "root_sha256": headers_root,
    }
    _write_json(headers_manifest_path, headers_manifest)
    headers_manifest_sha = finalizer._file_sha256(headers_manifest_path)
    helper_path = native / "libwaveqa_physx_ccd_readback.so"
    helper_path.write_bytes(
        b"\x7fELF\x02\x01" + b"\x00" * 10 + b"\x03\x00\x3e\x00"
    )
    helper_sha = finalizer._file_sha256(helper_path)
    remote_source = (
        f"{_REMOTE_ROOT}/project/{_REMOTE_TREE}/"
        "src/wave_asset_qa/contact/native/physx_ccd_readback.cpp"
    )
    compiler = "/usr/bin/x86_64-linux-gnu-g++-11"
    build_argv = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        remote_include,
        remote_source,
        "-o",
        _REMOTE_HELPER_PATH,
    ]
    build_provenance_path = native / "build-provenance.json"
    build_provenance = {
        "schema_version": 1,
        "helper_id": "waveqa_physx_ccd_readback",
        "sdk_archive": {
            "package": "ovphysx",
            "version": "0.4.13",
            "path": archive,
            "sha256": finalizer._OVPHYSX_SDK_ARCHIVE_SHA256,
        },
        "headers": {
            "directory": remote_include,
            "manifest_path": f"{remote_native}/headers-manifest.json",
            "manifest_sha256": headers_manifest_sha,
            "include_root_in_archive": "ovphysx/include",
            "file_count": len(records),
            "total_bytes": sum(record["size"] for record in records),
            "root_sha256": headers_root,
        },
        "source": {
            "relative_path": (
                "src/wave_asset_qa/contact/native/physx_ccd_readback.cpp"
            ),
            "path": remote_source,
            "sha256": finalizer._file_sha256(
                ROOT
                / "src"
                / "wave_asset_qa"
                / "contact"
                / "native"
                / "physx_ccd_readback.cpp"
            ),
        },
        "compiler": {
            "path": compiler,
            "sha256": "6" * 64,
            "version": "g++ (Ubuntu 11.4.0) 11.4.0",
        },
        "build": {
            "argv": build_argv,
            "output_path": _REMOTE_HELPER_PATH,
            "output_sha256": helper_sha,
        },
    }
    _write_json(build_provenance_path, build_provenance)
    summary = {
        "build_provenance_path": f"{remote_native}/build-provenance.json",
        "build_provenance_sha256": finalizer._file_sha256(build_provenance_path),
        "helper_path": _REMOTE_HELPER_PATH,
        "helper_sha256": helper_sha,
    }
    return native, summary


def test_remote_gpu_selection_and_snapshot_checkpoints_are_independently_bound(
    tmp_path: Path,
) -> None:
    finalizer = _load_finalizer()
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / "gpu-selection.txt").write_text(_gpu_state_text(), encoding="utf-8")
    case_ids = [_REMOTE_CASE, _REMOTE_CASE.replace("r01", "r02")]
    (launcher / "snapshot-verifications.txt").write_text(
        "".join(
            f"2026-08-31T00:00:00Z checkpoint=before-{case_id} "
            f"snapshot_sha256={_REMOTE_SNAPSHOT}\n"
            for case_id in case_ids
        )
        + f"2026-08-31T00:00:00Z checkpoint=post-matrix snapshot_sha256={_REMOTE_SNAPSHOT}\n",
        encoding="utf-8",
    )
    finalizer._validate_remote_launcher_security_evidence(
        launcher,
        expected_case_ids=case_ids,
        snapshot_sha256=_REMOTE_SNAPSHOT,
        gpu_index=3,
        gpu_uuid=_REMOTE_GPU_UUID,
        gpu_name=_REMOTE_GPU_NAME,
        driver_version=_REMOTE_DRIVER,
    )

    (launcher / "gpu-selection.txt").write_text(
        _gpu_state_text(memory_mib=33), encoding="utf-8"
    )
    with pytest.raises(finalizer.ContactC0FinalizationError, match="strictly idle"):
        finalizer._validate_remote_launcher_security_evidence(
            launcher,
            expected_case_ids=case_ids,
            snapshot_sha256=_REMOTE_SNAPSHOT,
            gpu_index=3,
            gpu_uuid=_REMOTE_GPU_UUID,
            gpu_name=_REMOTE_GPU_NAME,
            driver_version=_REMOTE_DRIVER,
        )


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("command_gpu", "command/source/output-directory"),
        ("command_helper_path", "command/source/output-directory"),
        ("command_helper_sha", "command/source/output-directory"),
        ("command_output", "command/source/output-directory"),
        ("source_module", "immutable source tree"),
        ("preflight_busy", "preflight 02"),
        ("postflight_busy", "final GPU postflight"),
        ("owned_group", "owned process group"),
        ("owned_handshake", "handshake"),
        ("worker_binding", "worker-to-owned-process-group"),
        ("monitor_owner", "monitor owner-PGID"),
        ("monitor_device", "foreign-device"),
        ("monitor_pid", "bound worker"),
    ],
)
def test_remote_security_evidence_rejects_tampering(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    finalizer = _load_finalizer()
    case_root, process, fresh_hash = _write_remote_security_fixture(tmp_path, finalizer)
    if mutation == "command_gpu":
        command = json.loads((case_root / "command.json").read_text(encoding="utf-8"))
        command["environment"]["CUDA_VISIBLE_DEVICES"] = "4"
        _write_json(case_root / "command.json", command)
    elif mutation == "command_helper_path":
        command = json.loads((case_root / "command.json").read_text(encoding="utf-8"))
        command["environment"]["WAVEQA_PHYSX_CCD_HELPER_PATH"] += ".forged"
        _write_json(case_root / "command.json", command)
    elif mutation == "command_helper_sha":
        command = json.loads((case_root / "command.json").read_text(encoding="utf-8"))
        command["environment"]["WAVEQA_PHYSX_CCD_HELPER_SHA256"] = "6" * 64
        _write_json(case_root / "command.json", command)
    elif mutation == "command_output":
        command = json.loads((case_root / "command.json").read_text(encoding="utf-8"))
        position = command["argv"].index("--output-dir") + 1
        command["argv"][position] = f"{_REMOTE_ROOT}/results/forged"
        _write_json(case_root / "command.json", command)
    elif mutation == "source_module":
        process["worker_module_realpath"] = f"{_REMOTE_ROOT}/project/forged/worker.py"
    elif mutation == "preflight_busy":
        (case_root / "gpu-preflight-02.txt").write_text(
            _gpu_state_text(utilization=1), encoding="utf-8"
        )
    elif mutation == "postflight_busy":
        (case_root / "gpu-postflight-01.txt").write_text(
            _gpu_state_text(
                query_status=1,
                processes=(f"{_REMOTE_GPU_UUID}, 99999, 10",),
            ),
            encoding="utf-8",
        )
    elif mutation == "owned_group":
        group = json.loads((case_root / "process.json").read_text(encoding="utf-8"))
        group["owned_process_group_id"] = 22001
        _write_json(case_root / "process.json", group)
    elif mutation == "owned_handshake":
        (case_root / "owned-process-group.txt").write_text("22001\n", encoding="ascii")
    elif mutation == "worker_binding":
        binding = json.loads(
            (case_root / "worker-group-binding.json").read_text(encoding="utf-8")
        )
        binding["worker_pid"] = 22003
        _write_json(case_root / "worker-group-binding.json", binding)
    elif mutation == "monitor_owner":
        monitor = (case_root / "gpu-monitor.txt").read_text(encoding="utf-8")
        (case_root / "gpu-monitor.txt").write_text(
            monitor.replace("owner_pgid=22000", "owner_pgid=22001"),
            encoding="utf-8",
        )
    elif mutation == "monitor_device":
        monitor = (case_root / "gpu-monitor.txt").read_text(encoding="utf-8")
        (case_root / "gpu-monitor.txt").write_text(
            monitor.replace(_REMOTE_GPU_UUID, "GPU-0605114d-00c0-586d-b13a-3958164cefdf"),
            encoding="utf-8",
        )
    elif mutation == "monitor_pid":
        monitor = (case_root / "gpu-monitor.txt").read_text(encoding="utf-8")
        (case_root / "gpu-monitor.txt").write_text(
            monitor.replace(", 22002,", ", 22003,"),
            encoding="utf-8",
        )
    with pytest.raises(finalizer.ContactC0FinalizationError, match=expected_error):
        _validate_remote_security_fixture(finalizer, case_root, process, fresh_hash)


def test_remote_security_evidence_accepts_exact_launcher_records(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    case_root, process, fresh_hash = _write_remote_security_fixture(tmp_path, finalizer)
    _validate_remote_security_fixture(finalizer, case_root, process, fresh_hash)


def _validate_remote_native_fixture(finalizer, tmp_path: Path, summary):  # type: ignore[no-untyped-def]
    return finalizer._validate_remote_native_helper(
        tmp_path,
        run_id=_REMOTE_RUN,
        source_tree=_REMOTE_TREE,
        project_root=ROOT,
        launcher_summary_value=summary,
    )


def test_remote_native_helper_accepts_exact_build_chain(tmp_path: Path) -> None:
    finalizer = _load_finalizer()
    _, summary = _write_remote_native_helper_fixture(tmp_path, finalizer)
    assert _validate_remote_native_fixture(finalizer, tmp_path, summary) == {
        "path": _REMOTE_HELPER_PATH,
        "sha256": summary["helper_sha256"],
    }


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("helper_bytes", "build invocation/output"),
        ("header_bytes", "exact header inventory"),
        ("sdk_sha", "SDK archive provenance"),
        ("source_sha", "immutable source provenance"),
        ("compiler_path", "compiler identity"),
        ("build_argv", "build invocation/output"),
        ("launcher_summary", "launcher summary"),
    ],
)
def test_remote_native_helper_rejects_build_chain_tampering(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    finalizer = _load_finalizer()
    native, summary = _write_remote_native_helper_fixture(tmp_path, finalizer)
    build_path = native / "build-provenance.json"
    if mutation == "helper_bytes":
        with (native / "libwaveqa_physx_ccd_readback.so").open("ab") as handle:
            handle.write(b"forged")
    elif mutation == "header_bytes":
        with (native / "include" / "PxScene.h").open("a", encoding="utf-8") as handle:
            handle.write("// forged\n")
    elif mutation == "launcher_summary":
        summary["helper_sha256"] = "7" * 64
    else:
        build = json.loads(build_path.read_text(encoding="utf-8"))
        if mutation == "sdk_sha":
            build["sdk_archive"]["sha256"] = "7" * 64
        elif mutation == "source_sha":
            build["source"]["sha256"] = "7" * 64
        elif mutation == "compiler_path":
            build["compiler"]["path"] = "/tmp/g++"
        elif mutation == "build_argv":
            build["build"]["argv"].remove("-Werror")
        _write_json(build_path, build)
    with pytest.raises(finalizer.ContactC0FinalizationError, match=expected_error):
        _validate_remote_native_fixture(finalizer, tmp_path, summary)
