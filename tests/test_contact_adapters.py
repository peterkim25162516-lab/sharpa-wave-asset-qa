from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

import wave_asset_qa.contact.adapters as contact_adapters
from wave_asset_qa.contact.adapters import (
    ALLOWED_PAIR_ID,
    CONTACT_C0_PROFILE_ID,
    PAIR_ACTIVE_FORCE_THRESHOLD_N,
    PROBE_LOCAL_CENTER_M,
    PROBE_RADIUS_M,
    TARGET_CENTER_WORLD_M,
    TARGET_HALF_EXTENTS_M,
    TARGET_TOP_WORLD_M,
    ContactAdapterError,
    MuJoCoContactAdapter,
    OVPhysXContactAdapter,
    _DefaultOVPhysXBridge,
)
from wave_asset_qa.adapters.base import AdapterCapabilityError
from wave_asset_qa.contact.records import ContactRun
from wave_asset_qa.contact.scenarios import (
    analytic_signed_gap_m,
    expand_contact_cases,
    load_contact_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "parity" / "contact_c0.json"
ASSET_ROOT = PROJECT_ROOT / "external" / "lfstd"
SOURCE_REVISION = "1" * 40
SOURCE_TREE = "2" * 40
FRESH_IDENTITY = "3" * 64


def _canonical_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _case(*, simulator: str, condition: str):
    manifest = load_contact_manifest(MANIFEST_PATH)
    selected = next(
        case
        for case in expand_contact_cases(manifest)
        if case.simulator.value == simulator
        and case.hand.value == "left"
        and case.condition.value == condition
        and case.timestep_variant.value == "base"
        and case.repeat_index == 1
    )
    return manifest, selected


def _mujoco_adapter() -> MuJoCoContactAdapter:
    return MuJoCoContactAdapter(
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        fresh_process_identity_sha256=FRESH_IDENTITY,
    )


def test_contact_adapter_import_is_cpu_safe() -> None:
    script = """
import sys
import wave_asset_qa.contact.adapters
for prefix in ('mujoco', 'pxr', 'torch', 'warp', 'isaaclab', 'isaaclab_ovphysx', 'omni.kit', 'isaacsim'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    completed = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("dt_s", "expected_steps"),
    [(0.002, 3500), (0.001, 7000)],
)
def test_contact_adapter_steps_preserve_duration_across_dt_variants(
    dt_s: float,
    expected_steps: int,
) -> None:
    scenario = load_contact_manifest(MANIFEST_PATH).scenario

    assert contact_adapters._scenario_steps(scenario, dt_s=dt_s) == expected_steps


@pytest.mark.parametrize("dt_s", [0.003, 0.0010000001])
def test_contact_adapter_steps_reject_nonintegral_or_drifted_dt(dt_s: float) -> None:
    scenario = load_contact_manifest(MANIFEST_PATH).scenario

    with pytest.raises(ValueError, match="integer multiple"):
        contact_adapters._scenario_steps(scenario, dt_s=dt_s)


def test_contact_adapter_steps_reject_drifted_base_step_claim() -> None:
    scenario = SimpleNamespace(duration_s=7.0, dt_s=0.002, steps=3499)

    with pytest.raises(ValueError, match="steps contradicts"):
        contact_adapters._scenario_steps(scenario, dt_s=0.001)


@pytest.mark.parametrize(
    ("dt_s", "requested_steps", "step_index"),
    [
        (0.002, 3500, 3499),
        (0.002, 3500, 3500),
        (0.001, 7000, 6999),
        (0.001, 7000, 7000),
    ],
)
def test_contact_adapter_sample_time_uses_the_canonical_integer_grid(
    dt_s: float,
    requested_steps: int,
    step_index: int,
) -> None:
    expected_time_s = 7.0 if step_index == requested_steps else step_index * dt_s
    assert contact_adapters._canonical_sample_time_s(
        step_index=step_index,
        requested_steps=requested_steps,
        dt_s=dt_s,
        duration_s=7.0,
    ) == expected_time_s


@pytest.mark.parametrize(
    "arguments",
    [
        {"step_index": 0, "requested_steps": 0, "dt_s": 0.001, "duration_s": 7.0},
        {"step_index": 7001, "requested_steps": 7000, "dt_s": 0.001, "duration_s": 7.0},
        {"step_index": 3500, "requested_steps": 3500, "dt_s": 0.001, "duration_s": 7.0},
    ],
)
def test_contact_adapter_sample_time_rejects_inconsistent_grid(
    arguments: dict[str, int | float],
) -> None:
    with pytest.raises(ValueError):
        contact_adapters._canonical_sample_time_s(**arguments)


def test_physx_schemas_are_registered_and_verified_before_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Definition:
        def __init__(self, property_name: str) -> None:
            self._property_name = property_name

        def GetPropertyNames(self) -> list[str]:
            return [self._property_name]

    definitions = {
        "PhysxSceneAPI": Definition("physxScene:enableCCD"),
        "PhysxRigidBodyAPI": Definition("physxRigidBody:enableCCD"),
    }

    class Registry:
        @staticmethod
        def FindAppliedAPIPrimDefinition(name: str) -> Definition:
            return definitions[name]

        @staticmethod
        def GetSchemaKind(name: str) -> str:
            assert name in definitions
            return "SingleApplyAPI"

    class Manager:
        @staticmethod
        def _ensure_physx_schemas_registered() -> None:
            calls.append("register")

    monkeypatch.setattr(
        contact_adapters,
        "import_module",
        lambda name: SimpleNamespace(OvPhysxManager=Manager),
    )
    runtime = {"Usd": SimpleNamespace(SchemaRegistry=Registry)}

    contact_adapters._register_physx_schemas_before_stage(runtime)

    assert calls == ["register"]
    assert runtime["ov_manager_cls"] is Manager


def test_physx_schema_verification_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Definition:
        @staticmethod
        def GetPropertyNames() -> list[str]:
            return []

    registry = SimpleNamespace(
        FindAppliedAPIPrimDefinition=lambda name: Definition(),
        GetSchemaKind=lambda name: "SingleApplyAPI",
    )
    manager = SimpleNamespace(_ensure_physx_schemas_registered=lambda: None)
    monkeypatch.setattr(
        contact_adapters,
        "import_module",
        lambda name: SimpleNamespace(OvPhysxManager=manager),
    )
    runtime = {"Usd": SimpleNamespace(SchemaRegistry=lambda: registry)}

    with pytest.raises(
        AdapterCapabilityError,
        match="physx_schema_registration_before_stage",
    ):
        contact_adapters._register_physx_schemas_before_stage(runtime)


def test_native_physx_ccd_helper_requires_bound_path_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WAVEQA_PHYSX_CCD_HELPER_PATH", raising=False)
    monkeypatch.delenv("WAVEQA_PHYSX_CCD_HELPER_SHA256", raising=False)
    with pytest.raises(AdapterCapabilityError, match="native_physx_ccd_readback"):
        contact_adapters._load_physx_ccd_helper()


def test_native_physx_helper_verifies_hash_version_masks_and_mass_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ctypes

    payload = tmp_path / "libwaveqa_physx_ccd_readback.so"
    payload.write_bytes(b"pinned-native-helper")
    digest = sha256(payload.read_bytes()).hexdigest()
    monkeypatch.setenv("WAVEQA_PHYSX_CCD_HELPER_PATH", str(payload.resolve()))
    monkeypatch.setenv("WAVEQA_PHYSX_CCD_HELPER_SHA256", digest)

    class Symbol:
        def __init__(self, value: int) -> None:
            self.value = value
            self.restype: object | None = None
            self.argtypes: object | None = None

        def __call__(self, *args: object) -> int:
            del args
            return self.value

    library = SimpleNamespace(
        waveqa_physx_version=Symbol(0x05090000),
        waveqa_scene_ccd_mask=Symbol(2),
        waveqa_rigid_body_ccd_mask=Symbol(4),
        waveqa_scene_flags=Symbol(0),
        waveqa_rigid_body_flags=Symbol(0),
        waveqa_mass_properties_value_count=Symbol(11),
        waveqa_rigid_body_mass_properties=Symbol(0),
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)

    loaded, actual_digest = contact_adapters._load_physx_ccd_helper()

    assert loaded is library
    assert actual_digest == digest
    for name in (
        "waveqa_physx_version",
        "waveqa_scene_ccd_mask",
        "waveqa_rigid_body_ccd_mask",
        "waveqa_mass_properties_value_count",
    ):
        assert getattr(library, name).argtypes == []
    assert library.waveqa_scene_flags.argtypes == [ctypes.c_void_p]
    assert library.waveqa_rigid_body_flags.argtypes == [ctypes.c_void_p]
    assert library.waveqa_rigid_body_mass_properties.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32,
    ]
    for name in (
        "waveqa_physx_version",
        "waveqa_scene_ccd_mask",
        "waveqa_rigid_body_ccd_mask",
        "waveqa_scene_flags",
        "waveqa_rigid_body_flags",
        "waveqa_mass_properties_value_count",
        "waveqa_rigid_body_mass_properties",
    ):
        assert getattr(library, name).restype is ctypes.c_uint32

    library.waveqa_scene_ccd_mask.value = 8
    with pytest.raises(ContactAdapterError, match="flag masks differ"):
        contact_adapters._load_physx_ccd_helper()
    library.waveqa_scene_ccd_mask.value = 2

    library.waveqa_mass_properties_value_count.value = 10
    with pytest.raises(ContactAdapterError, match="mass-property layout differs"):
        contact_adapters._load_physx_ccd_helper()
    library.waveqa_mass_properties_value_count.value = 11

    monkeypatch.setenv("WAVEQA_PHYSX_CCD_HELPER_SHA256", "0" * 64)
    with pytest.raises(ContactAdapterError, match="SHA-256 differs"):
        contact_adapters._load_physx_ccd_helper()


def test_mujoco_in_memory_fixture_disables_native_collisions_and_preserves_mass() -> None:
    pytest.importorskip("mujoco")
    manifest, contact_case = _case(simulator="mujoco", condition="contact")
    _, sham_case = _case(simulator="mujoco", condition="sham")
    hand = manifest.hand(contact_case.hand)
    source = ASSET_ROOT / contact_case.model_path
    before = source.read_bytes()

    observed: dict[str, Mapping[str, Any]] = {}
    for label, case in (("contact", contact_case), ("sham", sham_case)):
        adapter = _mujoco_adapter()
        adapter.open_contact_case(
            hand,
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
        )
        try:
            private = adapter.fixture_readback
            native = private["native_collision_disabled_inventory"]
            assert native["count"] == 54
            assert native["all_disabled"] is True
            assert all(item["disabled"] for item in native["items"])
            assert private["collision"]["ccd_enabled"] is False
            assert private["mass_properties"]["preserved"] is True
            assert private["mass_properties"]["source"] == private[
                "mass_properties"
            ]["effective"]
            assert private["geometry"]["probe"]["local_center_m"] == list(
                PROBE_LOCAL_CENTER_M
            )
            assert private["geometry"]["probe"]["radius_m"] == PROBE_RADIUS_M
            assert private["geometry"]["target"]["center_world_m"] == list(
                TARGET_CENTER_WORLD_M
            )
            assert private["geometry"]["target"]["half_extents_m"] == list(
                TARGET_HALF_EXTENTS_M
            )
            observation = adapter.collect_contact_observation()
            assert observation["pair_active"] is False
            assert observation["unexpected_pairs"] == []
            observed[label] = dict(private)
        finally:
            adapter.close()

    assert source.read_bytes() == before
    assert observed["contact"]["collision"]["compiled_explicit_pair_count"] == 1
    assert observed["sham"]["collision"]["compiled_explicit_pair_count"] == 0
    assert observed["contact"]["collision"]["pair_collision_enabled"] is True
    assert observed["sham"]["collision"]["pair_collision_enabled"] is False
    # The geometry/inventory itself is identical; only the sole pair mask changes.
    assert (
        observed["contact"]["collision_inventory_hash_preimage"]
        == observed["sham"]["collision_inventory_hash_preimage"]
    )


@pytest.mark.parametrize("condition", ["contact", "sham"])
def test_mujoco_formal_run_uses_analytic_gap_and_selected_pair_force_threshold(
    condition: str,
) -> None:
    pytest.importorskip("mujoco")
    manifest, case = _case(simulator="mujoco", condition=condition)
    adapter = _mujoco_adapter()

    run = adapter.run_contact_case(
        manifest.hand(case.hand),
        manifest.scenario,
        case,
        dt_s=case.dt_s,
        asset_root=ASSET_ROOT,
    )

    assert isinstance(run, ContactRun)
    assert run.completed
    assert len(run.samples) == 3501
    assert run.fixture_readback["profile_id"] == CONTACT_C0_PROFILE_ID
    assert run.fixture_readback["enabled_collision_shape_count"] == 2
    assert run.fixture_readback["mass_properties_preserved"] is True
    assert run.contact_observation["filtered_sensor_body_count"] == 1
    assert run.contact_observation["filtered_target_count"] == 1
    for sample in run.samples:
        force_norm = sample.native_contact_observation[
            "selected_pair_force_norm_n"
        ]
        assert sample.pair_active is (force_norm > PAIR_ACTIVE_FORCE_THRESHOLD_N)
        assert sample.signed_gap_m == pytest.approx(
            analytic_signed_gap_m(sample.probe_center_world_m, manifest.fixture),
            abs=1e-15,
        )
    if condition == "contact":
        assert any(sample.pair_active for sample in run.samples)
    else:
        assert not any(sample.pair_active for sample in run.samples)
        assert max(
            sample.native_contact_observation["selected_pair_force_norm_n"]
            for sample in run.samples
        ) == 0.0

    private = adapter.last_private_evidence
    assert private["unexpected_pair_observations"] == []
    assert run.provenance["fixture_overlay_sha256"] == _canonical_sha(
        private["fixture_overlay"]
    )
    assert run.provenance["runtime_fingerprint_sha256"] == _canonical_sha(
        private["runtime_fingerprint"]
    )
    inventory_preimage = private["fixture_overlay"][
        "collision_inventory_hash_preimage"
    ]
    assert run.fixture_readback["collision_inventory_sha256"] == _canonical_sha(
        inventory_preimage
    )
    private["fixture_overlay"]["tamper"] = True
    assert "tamper" not in adapter.last_private_evidence["fixture_overlay"]


def test_mujoco_halved_run_uses_7000_intervals_and_exact_terminal_time() -> None:
    pytest.importorskip("mujoco")
    manifest = load_contact_manifest(MANIFEST_PATH)
    case = next(
        item
        for item in expand_contact_cases(manifest)
        if item.simulator.value == "mujoco"
        and item.hand.value == "left"
        and item.condition.value == "sham"
        and item.timestep_variant.value == "halved"
        and item.repeat_index == 1
    )
    adapter = _mujoco_adapter()

    run = adapter.run_contact_case(
        manifest.hand(case.hand),
        manifest.scenario,
        case,
        dt_s=case.dt_s,
        asset_root=ASSET_ROOT,
    )

    assert run.execution.requested_steps == 7000
    assert run.execution.completed_steps == 7000
    assert len(run.samples) == 7001
    assert run.samples[-2].time_s == 6999 * 0.001
    assert run.samples[-1].time_s == 7.0


class _FakeOVBridge:
    instances: list["_FakeOVBridge"] = []

    def __init__(self) -> None:
        self.reset_count = 0
        self.events: list[str] = []
        self.pair_collision_enabled = False
        self.step_index = 0
        self.joint_names: tuple[str, ...] = ()
        self.frame_names: tuple[str, ...] = ()
        self.targets: dict[str, float] = {}
        self.closed = False
        type(self).instances.append(self)

    def begin(
        self,
        *,
        source: Path,
        hand: object,
        dt_s: float,
        device: str,
    ) -> None:
        assert source.suffix == ".usda"
        assert device == "cpu"
        assert dt_s == 0.002
        self.events.append("begin")
        self.joint_names = tuple(hand.joint_names)
        self.frame_names = tuple(hand.distal_frame_names)
        self.targets = {name: 0.0 for name in self.joint_names}

    def author_contact_overlay(
        self,
        *,
        side: str,
        pair_collision_enabled: bool,
    ) -> Mapping[str, Any]:
        assert self.reset_count == 0
        self.events.append("overlay")
        self.pair_collision_enabled = pair_collision_enabled
        native_items = [
            {
                "prim_path": f"/World/Env_0/Robot/native_{index}",
                "original_collision_enabled": True,
                "effective_collision_enabled": False,
            }
            for index in range(26)
        ]
        return {
            "schema_version": 1,
            "backend": "ovphysx",
            "geometry": {
                "probe": {
                    "parent_prim_path": f"/World/Env_0/Robot/{side}_index_DP",
                    "prim_path": (
                        f"/World/Env_0/Robot/{side}_index_DP/synthetic_probe"
                    ),
                },
                "target": {"prim_path": "/World/Env_0/static_box"},
            },
            "collision": {
                "allowed_pair_id": ALLOWED_PAIR_ID,
                "pair_collision_enabled": pair_collision_enabled,
                "ccd_enabled": False,
            },
            "material": {
                "static_friction": 0.0,
                "dynamic_friction": 0.0,
                "restitution_coefficient": 0.0,
            },
            "mass_properties": {
                "preserved": False,
                "usd_composed_preserved": True,
                "runtime_verified": False,
                "source": {
                    "physics:mass": {
                        "valid": True,
                        "authored": True,
                        "value": 0.01,
                        "type": "float",
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
                        "value": [1.0e-6, 2.0e-6, 3.0e-6],
                        "type": "Vec3f",
                    },
                    "physics:principalAxes": {
                        "valid": True,
                        "authored": True,
                        "value": [0.0, 0.0, 0.0, 1.0],
                        "type": "Quatf",
                    },
                },
                "effective": {
                    "physics:mass": {
                        "valid": True,
                        "authored": True,
                        "value": 0.01,
                        "type": "float",
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
                        "value": [1.0e-6, 2.0e-6, 3.0e-6],
                        "type": "Vec3f",
                    },
                    "physics:principalAxes": {
                        "valid": True,
                        "authored": True,
                        "value": [0.0, 0.0, 0.0, 1.0],
                        "type": "Quatf",
                    },
                },
            },
            "native_collision_disabled_inventory": {
                "count": len(native_items),
                "all_disabled": True,
                "items": native_items,
            },
        }

    def create_filtered_contact_sensor(
        self,
        *,
        body_prim_path: str,
        target_prim_path: str,
    ) -> object:
        assert self.reset_count == 0
        self.events.append("sensor")
        assert body_prim_path.endswith("left_index_DP")
        return SimpleNamespace(
            cfg=SimpleNamespace(filter_prim_paths_expr=[target_prim_path])
        )

    def reset(self, *, hand: object) -> None:
        del hand
        self.events.append("reset")
        self.reset_count += 1

    def runtime_mass_properties(self, *, body_name: str) -> Mapping[str, Any]:
        assert self.reset_count == 1
        self.events.append("runtime_mass")
        return {
            "body_name": body_name,
            "body_prim_path": f"/World/Env_0/Robot/{body_name}",
            "source": (
                "physx_5_9_0_native_mass_properties_via_ovphysx_get_physx_ptr_"
                "post_first_reset"
            ),
            "physx_version": "5.9.0",
            "helper_sha256": "a" * 64,
            "mass_kg": 0.01,
            "center_of_mass_pose_b": {
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "mass_space_inertia_diagonal_kg_m2": [1.0e-6, 2.0e-6, 3.0e-6],
        }

    def runtime_ccd_readback(self) -> Mapping[str, Any]:
        assert self.reset_count == 1
        self.events.append("runtime_ccd")
        return {
            "source": (
                "physx_5_9_0_native_flags_via_ovphysx_get_physx_ptr_"
                "post_first_reset"
            ),
            "physx_version": "5.9.0",
            "helper_sha256": "a" * 64,
            "scene_prim_path": "/physicsScene",
            "body_prim_path": "/World/Env_0/Robot/left_index_DP",
            "scene_flags": 1,
            "body_flags": 8,
            "scene_ccd_mask": 2,
            "body_ccd_mask": 4,
            "scene_enabled": False,
            "body_enabled": False,
            "usd_input_source": "post_first_reset_live_composed_usd",
            "usd_scene_enabled": False,
            "usd_body_enabled": False,
        }

    def set_position_targets(self, targets: Mapping[str, float]) -> None:
        self.targets = dict(targets)

    def step(self, *, dt_s: float) -> None:
        assert dt_s == 0.002
        self.step_index += 1

    def state(self, *, hand: object) -> Mapping[str, Any]:
        del hand
        identity = [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
        return {
            "joint_positions": {name: 0.0 for name in self.joint_names},
            "joint_velocities": {name: 0.0 for name in self.joint_names},
            "frame_poses": {name: list(identity) for name in self.frame_names},
            "probe_center_world_m": [0.0, 0.0, 0.2],
        }

    def filtered_force_matrix(self, sensor: object) -> object:
        del sensor
        magnitude = (
            0.00554
            if self.pair_collision_enabled and self.step_index > 0
            else 0.0
        )
        return np.asarray([[[[0.0, 0.0, magnitude]]]], dtype=np.float32)

    def runtime_fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "ovphysx",
            "bridge": "mock-static-contract",
            "device": "cpu",
            "solver": "mock",
            "control_path": "mock",
            "contact_path": "force_matrix_w",
            "kitless": True,
            "renderer": False,
            "camera": False,
            "camera_prim_paths": [],
            "created_sensor_types": ["ContactSensor"],
        }

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


@pytest.mark.parametrize("condition", ["contact", "sham"])
def test_ov_mock_run_orders_overlay_sensor_before_reset_and_keeps_unique_filter(
    condition: str,
) -> None:
    manifest, case = _case(simulator="ovphysx", condition=condition)
    _FakeOVBridge.instances.clear()
    adapter = OVPhysXContactAdapter(
        runtime_bridge_factory=_FakeOVBridge,
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        fresh_process_identity_sha256=FRESH_IDENTITY,
    )

    run = adapter.run_contact_case(
        manifest.hand(case.hand),
        manifest.scenario,
        case,
        dt_s=case.dt_s,
        asset_root=ASSET_ROOT,
        device="cpu",
    )

    bridge = _FakeOVBridge.instances[-1]
    assert bridge.events[:4] == ["begin", "overlay", "sensor", "reset"]
    assert bridge.events[-1] == "close"
    assert bridge.reset_count == 1
    assert bridge.closed
    assert run.completed
    assert all(sample.signed_gap_m == pytest.approx(0.065) for sample in run.samples)
    assert run.contact_observation["filtered_sensor_body_count"] == 1
    assert run.contact_observation["filtered_target_count"] == 1
    if condition == "contact":
        assert not run.samples[0].pair_active
        assert all(sample.pair_active for sample in run.samples[1:])
    else:
        assert not any(sample.pair_active for sample in run.samples)
    sensor_readback = adapter.last_private_evidence["fixture_overlay"][
        "sensor_readback"
    ]
    assert sensor_readback["filter_count"] == 1
    assert sensor_readback["sample_update_force_recompute"] is True
    mass_readback = adapter.last_private_evidence["fixture_overlay"][
        "mass_properties"
    ]
    assert mass_readback["preserved"] is True
    assert mass_readback["usd_composed_preserved"] is True
    assert mass_readback["runtime_verified"] is True
    assert mass_readback["runtime_effective"]["body_name"] == "left_index_DP"
    assert mass_readback["runtime_effective"]["body_prim_path"] == (
        "/World/Env_0/Robot/left_index_DP"
    )
    assert mass_readback["runtime_effective"]["source"] == (
        "physx_5_9_0_native_mass_properties_via_ovphysx_get_physx_ptr_"
        "post_first_reset"
    )
    assert mass_readback["runtime_effective"]["physx_version"] == "5.9.0"
    assert mass_readback["runtime_effective"]["helper_sha256"] == "a" * 64
    assert mass_readback["runtime_effective"][
        "mass_space_inertia_diagonal_kg_m2"
    ] == [1.0e-6, 2.0e-6, 3.0e-6]
    collision_readback = adapter.last_private_evidence["fixture_overlay"][
        "collision"
    ]
    assert collision_readback["ccd_enabled"] is False
    assert collision_readback["ccd_scene_enabled"] is False
    assert collision_readback["ccd_body_enabled"] is False
    assert collision_readback["ccd_body_prim_path"].endswith("left_index_DP")
    assert collision_readback["ccd_readback_source"] == (
        "physx_5_9_0_native_flags_via_ovphysx_get_physx_ptr_post_first_reset"
    )
    assert collision_readback["ccd_physx_version"] == "5.9.0"
    assert collision_readback["ccd_helper_sha256"] == "a" * 64
    assert collision_readback["ccd_scene_flags"] == 1
    assert collision_readback["ccd_body_flags"] == 8
    assert collision_readback["ccd_usd_scene_enabled"] is False
    assert collision_readback["ccd_usd_body_enabled"] is False
    runtime = adapter.last_private_evidence["runtime_fingerprint"]
    assert runtime["scene_inventory_verification_phase"] == (
        "after_full_trajectory_before_cleanup"
    )
    assert runtime["forbidden_module_verification_phase"] == (
        "after_runtime_cleanup"
    )
    assert runtime["forbidden_module_inventory"] == []
    assert run.provenance["fixture_overlay_sha256"] == _canonical_sha(
        adapter.last_private_evidence["fixture_overlay"]
    )
    assert run.fixture_readback["collision_inventory_sha256"] == _canonical_sha(
        adapter.last_private_evidence["fixture_overlay"][
            "collision_inventory_hash_preimage"
        ]
    )


def test_default_ov_sensor_update_requires_force_recompute() -> None:
    calls: list[tuple[float, bool]] = []

    class Sensor:
        data = SimpleNamespace(
            force_matrix_w=np.asarray([[[[0.0, 0.0, 0.0]]]])
        )

        def update(self, dt_s: float, *, force_recompute: bool) -> None:
            calls.append((dt_s, force_recompute))

    bridge = _DefaultOVPhysXBridge()
    bridge._dt_s = 0.002
    matrix = bridge.filtered_force_matrix(Sensor())

    assert np.asarray(matrix).shape == (1, 1, 1, 3)
    assert calls == [(0.002, True)]

    class IncompatibleSensor:
        data = Sensor.data

        def update(self, dt_s: float) -> None:
            del dt_s

    with pytest.raises(
        AdapterCapabilityError,
        match="force_recompute",
    ):
        bridge.filtered_force_matrix(IncompatibleSensor())


def test_default_ov_runtime_fingerprint_reads_camera_prims_and_sensor_count() -> None:
    class CameraSchema:
        pass

    class Prim:
        def __init__(self, path: str, *, is_camera: bool) -> None:
            self._path = path
            self._is_camera = is_camera

        def IsA(self, schema: object) -> bool:
            return self._is_camera and schema is CameraSchema

        def GetPath(self) -> str:
            return self._path

    class Stage:
        def __init__(
            self,
            prims: list[Prim],
            *,
            all_prims: list[Prim] | None = None,
        ) -> None:
            self._prims = prims
            self._all_prims = prims if all_prims is None else all_prims

        def Traverse(self) -> tuple[Prim, ...]:
            return tuple(self._prims)

        def TraverseAll(self) -> tuple[Prim, ...]:
            return tuple(self._all_prims)

    bridge = _DefaultOVPhysXBridge()
    bridge._runtime = {"UsdGeom": SimpleNamespace(Camera=CameraSchema)}
    bridge._simulation = SimpleNamespace(
        stage=Stage([Prim("/World/Robot", is_camera=False)]),
        cfg=SimpleNamespace(physics=SimpleNamespace(), device="cuda:0"),
    )
    bridge._dt_s = 0.002
    bridge._contact_sensor_create_count = 1

    clean = bridge.runtime_fingerprint()
    assert clean["camera"] is False
    assert clean["camera_prim_paths"] == []
    assert clean["created_sensor_types"] == ["ContactSensor"]

    bridge._simulation.stage = Stage(
        [
            Prim("/World/Robot", is_camera=False),
            Prim("/World/ForbiddenCamera", is_camera=True),
        ]
    )
    with_camera = bridge.runtime_fingerprint()
    assert with_camera["camera"] is True
    assert with_camera["camera_prim_paths"] == ["/World/ForbiddenCamera"]

    inactive_camera = Prim("/World/InactiveCamera", is_camera=True)
    bridge._simulation.stage = Stage(
        [Prim("/World/Robot", is_camera=False)],
        all_prims=[
            Prim("/World/Robot", is_camera=False),
            inactive_camera,
        ],
    )
    assert all(
        not prim.IsA(CameraSchema)
        for prim in bridge._simulation.stage.Traverse()
    )
    exhaustive = bridge.runtime_fingerprint()
    assert exhaustive["camera"] is True
    assert exhaustive["camera_prim_paths"] == ["/World/InactiveCamera"]


def test_default_ov_runtime_fingerprint_requires_exhaustive_stage_traversal() -> None:
    class CameraSchema:
        pass

    bridge = _DefaultOVPhysXBridge()
    bridge._runtime = {"UsdGeom": SimpleNamespace(Camera=CameraSchema)}
    bridge._simulation = SimpleNamespace(
        stage=SimpleNamespace(Traverse=lambda: ()),
        cfg=SimpleNamespace(physics=SimpleNamespace(), device="cuda:0"),
    )
    bridge._dt_s = 0.002
    bridge._contact_sensor_create_count = 1

    with pytest.raises(ContactAdapterError, match="lacks TraverseAll"):
        bridge.runtime_fingerprint()


def test_default_ov_runtime_ccd_reads_live_physx_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = SimpleNamespace(identifier="anon:waveqa-c0")
    property_spec = SimpleNamespace(layer=overlay)

    class Attribute:
        @staticmethod
        def IsValid() -> bool:
            return True

        @staticmethod
        def HasAuthoredValueOpinion() -> bool:
            return True

        @staticmethod
        def Get() -> bool:
            return False

        @staticmethod
        def GetPropertyStack() -> list[object]:
            return [property_spec]

    class Prim:
        def __init__(self, schema: str, attribute_name: str) -> None:
            self.schema = schema
            self.attribute_name = attribute_name

        @staticmethod
        def IsValid() -> bool:
            return True

        def GetAppliedSchemas(self) -> list[str]:
            return [self.schema]

        def GetAttribute(self, name: str) -> Attribute:
            assert name == self.attribute_name
            return Attribute()

    scene_path = "/World/physicsScene"
    body_path = "/World/Env_0/Robot/left_index_DP"
    prims = {
        scene_path: Prim("PhysxSceneAPI", "physxScene:enableCCD"),
        body_path: Prim("PhysxRigidBodyAPI", "physxRigidBody:enableCCD"),
    }
    pointer_calls: list[tuple[str, int]] = []

    class PhysX:
        @staticmethod
        def get_physx_ptr(path: str, physx_type: int) -> int:
            pointer_calls.append((path, physx_type))
            return 101 if path == scene_path else 202

    class Helper:
        @staticmethod
        def waveqa_scene_flags(pointer: int) -> int:
            assert pointer == 101
            return 1

        @staticmethod
        def waveqa_rigid_body_flags(pointer: int) -> int:
            assert pointer == 202
            return 8

    bridge = _DefaultOVPhysXBridge()
    bridge.reset_count = 1
    bridge._ccd_scene_prim_path = scene_path
    bridge._ccd_body_prim_path = body_path
    bridge._overlay = overlay
    bridge._session_layer = SimpleNamespace(subLayerPaths=[overlay.identifier])
    bridge._simulation = SimpleNamespace(
        stage=SimpleNamespace(GetPrimAtPath=lambda path: prims[path])
    )
    bridge._runtime = {
        "ov_manager_cls": SimpleNamespace(_physx=PhysX()),
    }
    monkeypatch.setattr(
        contact_adapters,
        "import_module",
        lambda name: SimpleNamespace(
            PhysXType=SimpleNamespace(SCENE=1, LINK=9)
        ),
    )
    monkeypatch.setattr(
        contact_adapters,
        "_load_physx_ccd_helper",
        lambda: (Helper(), "b" * 64),
    )

    result = bridge.runtime_ccd_readback()

    assert pointer_calls == [(scene_path, 1), (body_path, 9)]
    assert result["source"] == (
        "physx_5_9_0_native_flags_via_ovphysx_get_physx_ptr_post_first_reset"
    )
    assert result["scene_flags"] == 1
    assert result["body_flags"] == 8
    assert result["scene_enabled"] is False
    assert result["body_enabled"] is False
    assert result["usd_scene_enabled"] is False
    assert result["usd_body_enabled"] is False


def test_default_ov_runtime_mass_reads_live_physx_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_path = "/World/Env_0/Robot/left_index_DP"
    pointer_calls: list[tuple[str, int]] = []
    helper_calls: list[tuple[int, int]] = []

    class PhysX:
        @staticmethod
        def get_physx_ptr(path: str, physx_type: int) -> int:
            pointer_calls.append((path, physx_type))
            return 202

    class Helper:
        @staticmethod
        def waveqa_rigid_body_mass_properties(
            pointer: int,
            output: object,
            count: int,
        ) -> int:
            helper_calls.append((pointer, count))
            values = (
                0.01,
                1.0e-6,
                2.0e-6,
                3.0e-6,
                0.001,
                -0.002,
                0.003,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            for index, value in enumerate(values):
                output[index] = value
            return 0

    bridge = _DefaultOVPhysXBridge()
    bridge.reset_count = 1
    bridge._simulation = SimpleNamespace()
    bridge._body_names = ("left_index_DP",)
    bridge._probe_body_name = "left_index_DP"
    bridge._ccd_body_prim_path = body_path
    bridge._runtime = {
        "ov_manager_cls": SimpleNamespace(_physx=PhysX()),
    }
    monkeypatch.setattr(
        contact_adapters,
        "import_module",
        lambda name: SimpleNamespace(PhysXType=SimpleNamespace(LINK=9)),
    )
    monkeypatch.setattr(
        contact_adapters,
        "_load_physx_ccd_helper",
        lambda: (Helper(), "b" * 64),
    )

    result = bridge.runtime_mass_properties(body_name="left_index_DP")

    assert pointer_calls == [(body_path, 9)]
    assert helper_calls == [(202, 11)]
    assert result == {
        "body_name": "left_index_DP",
        "body_prim_path": body_path,
        "source": (
            "physx_5_9_0_native_mass_properties_via_ovphysx_get_physx_ptr_"
            "post_first_reset"
        ),
        "physx_version": "5.9.0",
        "helper_sha256": "b" * 64,
        "mass_kg": 0.01,
        "center_of_mass_pose_b": {
            "position_m": [0.001, -0.002, 0.003],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "mass_space_inertia_diagonal_kg_m2": [1.0e-6, 2.0e-6, 3.0e-6],
    }


@pytest.mark.parametrize("status", [1, 2, 3, 4, 5, 6])
def test_default_ov_runtime_mass_rejects_native_helper_status(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    body_path = "/World/Env_0/Robot/left_index_DP"
    physx = SimpleNamespace(get_physx_ptr=lambda path, kind: 202)
    helper = SimpleNamespace(
        waveqa_rigid_body_mass_properties=lambda pointer, output, count: status
    )
    bridge = _DefaultOVPhysXBridge()
    bridge.reset_count = 1
    bridge._simulation = SimpleNamespace()
    bridge._body_names = ("left_index_DP",)
    bridge._probe_body_name = "left_index_DP"
    bridge._ccd_body_prim_path = body_path
    bridge._runtime = {"ov_manager_cls": SimpleNamespace(_physx=physx)}
    monkeypatch.setattr(
        contact_adapters,
        "import_module",
        lambda name: SimpleNamespace(PhysXType=SimpleNamespace(LINK=9)),
    )
    monkeypatch.setattr(
        contact_adapters,
        "_load_physx_ccd_helper",
        lambda: (helper, "b" * 64),
    )

    with pytest.raises(ContactAdapterError, match=f"status {status}"):
        bridge.runtime_mass_properties(body_name="left_index_DP")


def test_default_ov_runtime_mass_rejects_null_link_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_path = "/World/Env_0/Robot/left_index_DP"
    bridge = _DefaultOVPhysXBridge()
    bridge.reset_count = 1
    bridge._simulation = SimpleNamespace()
    bridge._body_names = ("left_index_DP",)
    bridge._probe_body_name = "left_index_DP"
    bridge._ccd_body_prim_path = body_path
    bridge._runtime = {
        "ov_manager_cls": SimpleNamespace(
            _physx=SimpleNamespace(get_physx_ptr=lambda path, kind: 0)
        )
    }
    monkeypatch.setattr(
        contact_adapters,
        "import_module",
        lambda name: SimpleNamespace(PhysXType=SimpleNamespace(LINK=9)),
    )

    with pytest.raises(ContactAdapterError, match="null PhysX mass-property pointer"):
        bridge.runtime_mass_properties(body_name="left_index_DP")


def test_default_ov_bridge_rejects_second_contact_sensor() -> None:
    created: list[object] = []

    class Sensors:
        @staticmethod
        def ContactSensorCfg(**kwargs: object) -> object:
            return SimpleNamespace(**kwargs)

        @staticmethod
        def ContactSensor(cfg: object) -> object:
            created.append(cfg)
            return SimpleNamespace(cfg=cfg)

    bridge = _DefaultOVPhysXBridge()
    bridge._runtime = {"sensors": Sensors}
    first = bridge.create_filtered_contact_sensor(
        body_prim_path="/World/Robot/left_index_DP",
        target_prim_path="/World/Target",
    )
    assert first is not None
    assert len(created) == 1
    with pytest.raises(ContactAdapterError, match="exactly one ContactSensor"):
        bridge.create_filtered_contact_sensor(
            body_prim_path="/World/Robot/left_index_DP",
            target_prim_path="/World/Target",
        )
    assert len(created) == 1


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("shape", "shape differs"),
        ("body", "body differs"),
        ("path", "prim path differs"),
        ("source", "source differs"),
        ("version", "version differs"),
        ("helper", "helper differs"),
        ("mass", "mass differs"),
        ("com", "center of mass differs"),
        ("inertia", "inertia differs"),
        ("axes", "inertia differs"),
        ("nonunit_axes", "not unit length"),
        ("nonpositive_inertia", "invalid shape or values"),
    ],
)
def test_ov_runtime_mass_property_mismatch_fails_closed(
    fault: str,
    message: str,
) -> None:
    class BadMassBridge(_FakeOVBridge):
        def runtime_mass_properties(self, *, body_name: str) -> Mapping[str, Any]:
            result = dict(super().runtime_mass_properties(body_name=body_name))
            result = json.loads(json.dumps(result))
            if fault == "shape":
                result["unexpected"] = True
            elif fault == "body":
                result["body_name"] = "wrong_body"
            elif fault == "path":
                result["body_prim_path"] = "/World/Env_0/Robot/wrong_body"
            elif fault == "source":
                result["source"] = "ambiguous-binding-readback"
            elif fault == "version":
                result["physx_version"] = "5.8.0"
            elif fault == "helper":
                result["helper_sha256"] = "b" * 64
            elif fault == "mass":
                result["mass_kg"] = 0.02
            elif fault == "com":
                result["center_of_mass_pose_b"]["position_m"][0] = 0.01
            elif fault == "inertia":
                result["mass_space_inertia_diagonal_kg_m2"][0] = 9.0e-6
            elif fault == "axes":
                root_half = 2.0 ** -0.5
                result["center_of_mass_pose_b"]["quaternion_xyzw"] = [
                    root_half,
                    0.0,
                    0.0,
                    root_half,
                ]
            elif fault == "nonunit_axes":
                result["center_of_mass_pose_b"]["quaternion_xyzw"] = [
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                ]
            else:
                result["mass_space_inertia_diagonal_kg_m2"][0] = 0.0
            return result

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(runtime_bridge_factory=BadMassBridge)
    with pytest.raises(ContactAdapterError, match=message):
        adapter.open_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert not adapter.is_open
    assert BadMassBridge.instances[-1].closed


def test_ov_runtime_mass_property_accepts_equivalent_principal_axis_reordering() -> None:
    class EquivalentAxesBridge(_FakeOVBridge):
        def runtime_mass_properties(self, *, body_name: str) -> Mapping[str, Any]:
            result = dict(super().runtime_mass_properties(body_name=body_name))
            result = json.loads(json.dumps(result))
            root_half = 2.0 ** -0.5
            result["center_of_mass_pose_b"]["quaternion_xyzw"] = [
                0.0,
                0.0,
                root_half,
                root_half,
            ]
            result["mass_space_inertia_diagonal_kg_m2"] = [
                2.0e-6,
                1.0e-6,
                3.0e-6,
            ]
            return result

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(runtime_bridge_factory=EquivalentAxesBridge)
    adapter.open_contact_case(
        manifest.hand(case.hand),
        manifest.scenario,
        case,
        dt_s=case.dt_s,
        asset_root=ASSET_ROOT,
        device="cpu",
    )
    try:
        mass = adapter.fixture_readback["mass_properties"]
        assert mass["runtime_verified"] is True
        assert mass["runtime_checks"][
            "inertia_link_tensor_max_abs_error_kg_m2"
        ] < 1.0e-18
        assert mass["runtime_checks"][
            "principal_axes_representation_angle_error_rad"
        ] == pytest.approx(np.pi / 2.0)
    finally:
        adapter.close()


def test_ov_runtime_mass_property_rejects_nonunit_authored_principal_axes() -> None:
    class NonUnitAuthoredAxesBridge(_FakeOVBridge):
        def author_contact_overlay(
            self,
            *,
            side: str,
            pair_collision_enabled: bool,
        ) -> Mapping[str, Any]:
            result = json.loads(
                json.dumps(
                    super().author_contact_overlay(
                        side=side,
                        pair_collision_enabled=pair_collision_enabled,
                    )
                )
            )
            for key in ("source", "effective"):
                result["mass_properties"][key]["physics:principalAxes"]["value"] = [
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                ]
            return result

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(
        runtime_bridge_factory=NonUnitAuthoredAxesBridge
    )
    with pytest.raises(ContactAdapterError, match="principalAxes.*not unit length"):
        adapter.open_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert NonUnitAuthoredAxesBridge.instances[-1].closed


def test_ov_runtime_mass_property_capability_is_mandatory() -> None:
    manifest, case = _case(simulator="ovphysx", condition="contact")
    bridge = _FakeOVBridge()
    bridge.runtime_mass_properties = None  # type: ignore[method-assign]
    adapter = OVPhysXContactAdapter(runtime_bridge_factory=lambda: bridge)
    with pytest.raises(AdapterCapabilityError, match="runtime_mass_properties"):
        adapter.open_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert bridge.closed


def test_ov_runtime_ccd_readback_capability_is_mandatory() -> None:
    manifest, case = _case(simulator="ovphysx", condition="contact")
    bridge = _FakeOVBridge()
    bridge.runtime_ccd_readback = None  # type: ignore[method-assign]
    adapter = OVPhysXContactAdapter(runtime_bridge_factory=lambda: bridge)
    with pytest.raises(AdapterCapabilityError, match="runtime_ccd_readback"):
        adapter.open_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert bridge.closed


def test_ov_runtime_ccd_true_fails_closed() -> None:
    class CcdEnabledBridge(_FakeOVBridge):
        def runtime_ccd_readback(self) -> Mapping[str, Any]:
            result = dict(super().runtime_ccd_readback())
            result["body_enabled"] = True
            return result

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(runtime_bridge_factory=CcdEnabledBridge)
    with pytest.raises(ContactAdapterError, match="CCD is not proved disabled"):
        adapter.open_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert CcdEnabledBridge.instances[-1].closed


@pytest.mark.parametrize(
    "module_name",
    [
        "omni.renderer.lazy_c0_test",
        "omni.replicator.lazy_c0_test",
        "omni.syntheticdata.lazy_c0_test",
    ],
)
def test_ov_lazy_visual_runtime_import_after_reset_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    class LazyResetBridge(_FakeOVBridge):
        def reset(self, *, hand: object) -> None:
            super().reset(hand=hand)
            monkeypatch.setitem(sys.modules, module_name, object())

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(runtime_bridge_factory=LazyResetBridge)
    with pytest.raises(AdapterCapabilityError, match="after OVPhysX C0 first reset"):
        adapter.open_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert LazyResetBridge.instances[-1].closed


def test_ov_passive_isaaclab_camera_class_import_is_not_camera_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PassiveCameraImportBridge(_FakeOVBridge):
        def reset(self, *, hand: object) -> None:
            super().reset(hand=hand)
            monkeypatch.setitem(
                sys.modules,
                "isaaclab.sensors.camera.camera_cfg",
                object(),
            )

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(
        runtime_bridge_factory=PassiveCameraImportBridge
    )
    adapter.open_contact_case(
        manifest.hand(case.hand),
        manifest.scenario,
        case,
        dt_s=case.dt_s,
        asset_root=ASSET_ROOT,
        device="cpu",
    )
    adapter.close()
    assert PassiveCameraImportBridge.instances[-1].closed


def test_ov_lazy_kit_import_during_trajectory_fails_before_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LazyStepBridge(_FakeOVBridge):
        def step(self, *, dt_s: float) -> None:
            super().step(dt_s=dt_s)
            if self.step_index == 1:
                monkeypatch.setitem(sys.modules, "omni.kit.lazy_c0_test", object())

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(
        runtime_bridge_factory=LazyStepBridge,
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        fresh_process_identity_sha256=FRESH_IDENTITY,
    )
    with pytest.raises(
        AdapterCapabilityError,
        match="after_full_trajectory_before_cleanup",
    ):
        adapter.run_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert LazyStepBridge.instances[-1].closed


def test_ov_lazy_renderer_import_during_cleanup_fails_before_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LazyCloseBridge(_FakeOVBridge):
        def close(self) -> None:
            super().close()
            monkeypatch.setitem(sys.modules, "omni.renderer.lazy_close_c0_test", object())

    manifest, case = _case(simulator="ovphysx", condition="contact")
    adapter = OVPhysXContactAdapter(
        runtime_bridge_factory=LazyCloseBridge,
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        fresh_process_identity_sha256=FRESH_IDENTITY,
    )
    with pytest.raises(
        AdapterCapabilityError,
        match="after OVPhysX C0 runtime cleanup",
    ):
        adapter.run_contact_case(
            manifest.hand(case.hand),
            manifest.scenario,
            case,
            dt_s=case.dt_s,
            asset_root=ASSET_ROOT,
            device="cpu",
        )
    assert LazyCloseBridge.instances[-1].closed
