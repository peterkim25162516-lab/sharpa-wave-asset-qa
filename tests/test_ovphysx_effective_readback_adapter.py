from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from wave_asset_qa.adapters import ovphysx
from wave_asset_qa.adapters.base import AdapterLifecycle, AdapterLifecycleError


class _CpuBinding:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.read_buffer: object | None = None

    def read(self, output: object) -> None:
        self.read_buffer = output
        output[...] = self.values  # type: ignore[index]


class _TensorRow:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> _TensorRow:
        return self

    def cpu(self) -> _TensorRow:
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _Tensor:
    shape = (1, 22)

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> _TensorRow:
        assert index == 0
        return _TensorRow(self.values)


class _Attribute:
    def __init__(self, value: object, *, authored: bool = True) -> None:
        self.value = value
        self.authored = authored

    def IsValid(self) -> bool:
        return True

    def Get(self) -> object:
        return self.value

    def HasAuthoredValueOpinion(self) -> bool:
        return self.authored


class _Prim:
    def __init__(
        self,
        path: str,
        *,
        kind: str,
        attributes: dict[str, _Attribute],
        name: str | None = None,
    ) -> None:
        self.path = path
        self.kind = kind
        self.attributes = attributes
        self.name = name or path.rsplit("/", 1)[-1]

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.name

    def GetTypeName(self) -> str:
        return "PhysicsRevoluteJoint" if self.kind == "revolute" else "Scope"

    def GetAttribute(self, name: str) -> _Attribute | None:
        return self.attributes.get(name)

    def IsA(self, schema: object) -> bool:
        return (self.kind, schema) in {
            ("scene", _UsdPhysics.Scene),
            ("revolute", _UsdPhysics.RevoluteJoint),
        }

    def HasAPI(self, schema: object) -> bool:
        return self.kind == "articulation" and schema is _UsdPhysics.ArticulationRootAPI


class _UsdPhysics:
    class Scene:
        pass

    class RevoluteJoint:
        pass

    class ArticulationRootAPI:
        pass


class _Stage:
    def __init__(self, prims: list[_Prim]) -> None:
        self.prims = prims

    def Traverse(self) -> tuple[_Prim, ...]:
        return tuple(self.prims)


def _patch_usd_physics(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    imports: list[str] = []

    def import_fake(name: str) -> object:
        imports.append(name)
        if name == "pxr.UsdPhysics":
            return _UsdPhysics
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(ovphysx, "import_module", import_fake)
    return imports


class _InterventionPropertySpec:
    def __init__(self, layer: _InterventionLayer, default: float) -> None:
        self.layer = layer
        self.default = default


class _InterventionLayer:
    _registry: dict[str, _InterventionLayer] = {}
    _counter = 0

    def __init__(self, label: str, *, anonymous: bool = True) -> None:
        type(self)._counter += 1
        self.identifier = f"anon:{label}:{type(self)._counter}"
        self.anonymous = anonymous
        self.subLayerPaths: list[str] = []
        self.properties: dict[str, _InterventionPropertySpec] = {}
        type(self)._registry[self.identifier] = self

    @classmethod
    def CreateAnonymous(cls, label: str) -> _InterventionLayer:
        return cls(label)

    def GetPropertyAtPath(self, path: object) -> _InterventionPropertySpec | None:
        return self.properties.get(str(path))

    def Clear(self) -> None:
        self.properties.clear()


class _InterventionEditTarget:
    def __init__(self, layer: _InterventionLayer) -> None:
        self.layer = layer

    def GetLayer(self) -> _InterventionLayer:
        return self.layer


class _InterventionEditContext:
    def __init__(
        self, stage: _InterventionStage, target: _InterventionEditTarget
    ) -> None:
        self.stage = stage
        self.target = target
        self.previous: _InterventionEditTarget | None = None

    def __enter__(self) -> None:
        self.previous = self.stage.edit_target
        self.stage.edit_target = self.target

    def __exit__(self, *_: object) -> None:
        assert self.previous is not None
        self.stage.edit_target = self.previous


class _InterventionUsd:
    EditTarget = _InterventionEditTarget
    EditContext = _InterventionEditContext


class _InterventionValueTypeNames:
    Float = "float"


class _InterventionSdf:
    Layer = _InterventionLayer
    ValueTypeNames = _InterventionValueTypeNames

    @staticmethod
    def Path(value: str) -> str:
        return value


class _InterventionAttribute:
    def __init__(
        self,
        stage: _InterventionStage,
        property_path: str,
        value: float,
        *,
        type_name: str = "float",
        authored: bool = True,
        fail_write: bool = False,
    ) -> None:
        self.stage = stage
        self.property_path = property_path
        self.base_value = value
        self.type_name = type_name
        self.authored = authored
        self.fail_write = fail_write
        self.write_count = 0

    def IsValid(self) -> bool:
        return True

    def Get(self) -> float:
        for identifier in self.stage.session_layer.subLayerPaths:
            layer = _InterventionLayer._registry[identifier]
            spec = layer.GetPropertyAtPath(self.property_path)
            if spec is not None:
                return float(spec.default)
        return self.base_value

    def Set(self, value: float) -> bool:
        self.write_count += 1
        if self.fail_write:
            return False
        layer = self.stage.GetEditTarget().GetLayer()
        layer.properties[self.property_path] = _InterventionPropertySpec(
            layer, float(value)
        )
        return True

    def GetTypeName(self) -> str:
        return self.type_name

    def HasAuthoredValueOpinion(self) -> bool:
        return self.authored

    def GetPropertyStack(self) -> list[_InterventionPropertySpec]:
        stack: list[_InterventionPropertySpec] = []
        for identifier in self.stage.session_layer.subLayerPaths:
            layer = _InterventionLayer._registry[identifier]
            spec = layer.GetPropertyAtPath(self.property_path)
            if spec is not None:
                stack.append(spec)
        stack.append(
            _InterventionPropertySpec(self.stage.root_layer, self.base_value)
        )
        return stack


class _InterventionPrim:
    def __init__(
        self,
        path: str,
        name: str,
        attribute: _InterventionAttribute,
        *,
        valid: bool = True,
        revolute: bool = True,
    ) -> None:
        self.path = path
        self.name = name
        self.attribute = attribute
        self.valid = valid
        self.revolute = revolute

    def IsValid(self) -> bool:
        return self.valid

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.name

    def IsA(self, schema: object) -> bool:
        return self.revolute and schema is _UsdPhysics.RevoluteJoint

    def GetAttribute(self, name: str) -> _InterventionAttribute | None:
        if name != "physxJoint:jointFriction":
            return None
        return self.attribute


class _InterventionStage:
    def __init__(self, names: tuple[str, ...], value: float = 0.125) -> None:
        self.root_layer = _InterventionLayer("root")
        self.session_layer = _InterventionLayer("session")
        self.edit_target = _InterventionEditTarget(self.root_layer)
        self.prims: dict[str, _InterventionPrim] = {}
        for name in names:
            path = f"/World/Env_0/Robot/joints/{name}"
            property_path = f"{path}.physxJoint:jointFriction"
            attribute = _InterventionAttribute(self, property_path, value)
            self.prims[path] = _InterventionPrim(path, name, attribute)

    def Traverse(self) -> tuple[_InterventionPrim, ...]:
        return tuple(self.prims.values())

    def GetPrimAtPath(self, path: str) -> _InterventionPrim | None:
        return self.prims.get(path)

    def GetSessionLayer(self) -> _InterventionLayer:
        return self.session_layer

    def GetEditTarget(self) -> _InterventionEditTarget:
        return self.edit_target

    def GetLayerStack(self) -> list[_InterventionLayer]:
        return [
            self.session_layer,
            *(
                _InterventionLayer._registry[identifier]
                for identifier in self.session_layer.subLayerPaths
            ),
            self.root_layer,
        ]


class _InterventionContextManager:
    def __init__(self) -> None:
        self.exited = False

    def __exit__(self, *_: object) -> None:
        self.exited = True


def _patch_intervention_pxr(monkeypatch: pytest.MonkeyPatch) -> None:
    _InterventionLayer._registry = {}
    _InterventionLayer._counter = 0

    def import_fake(name: str) -> object:
        modules = {
            "pxr.Sdf": _InterventionSdf,
            "pxr.Usd": _InterventionUsd,
            "pxr.UsdPhysics": _UsdPhysics,
        }
        if name not in modules:
            raise AssertionError(f"unexpected import: {name}")
        return modules[name]

    monkeypatch.setattr(ovphysx, "import_module", import_fake)


def _intervention_plan(
    names: tuple[str, ...], *, role: str
) -> dict[str, object]:
    expected = {name: 0.125 for name in names}
    return {
        "schema_version": 1,
        "role": role,
        "private_plan_sha256": "a" * 64,
        "expected_pre_values": expected,
        "write_values": dict(expected)
        if role == "sham"
        else {name: 0.0 for name in names},
    }


def _prepared_intervention_adapter(
    names: tuple[str, ...], *, role: str
) -> ovphysx.OVPhysXAdapter:
    adapter = ovphysx.OVPhysXAdapter(
        device="cpu",
        legacy_joint_friction_intervention=_intervention_plan(names, role=role),
    )
    adapter._validated_legacy_joint_friction_intervention = (
        ovphysx._validate_legacy_joint_friction_intervention_for_joints(
            adapter._legacy_joint_friction_intervention, names
        )
    )
    return adapter


def test_rank_safe_cpu_binding_preserves_friction_triples() -> None:
    raw = np.arange(66, dtype=np.float32).reshape(1, 22, 3)
    binding = _CpuBinding(raw)

    values = ovphysx._read_cpu_float_binding(
        binding,
        name="DOF_FRICTION_PROPERTIES",
        expected_shape=(1, 22, 3),
    )

    assert len(values) == 22
    assert all(isinstance(row, tuple) and len(row) == 3 for row in values)
    assert values[0] == (0.0, 1.0, 2.0)
    assert values[-1] == (63.0, 64.0, 65.0)
    assert binding.read_buffer is not None
    assert type(binding.read_buffer).__module__.startswith("numpy")


def test_cpu_binding_rejects_non_singleton_instance_contract() -> None:
    binding = _CpuBinding(np.zeros((2, 22), dtype=np.float32))

    with pytest.raises(RuntimeError, match="one articulation instance"):
        ovphysx._read_cpu_float_binding(
            binding,
            name="DOF_ARMATURE",
            expected_shape=(2, 22),
        )


def test_live_solver_readback_uses_generic_usd_attributes_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _patch_usd_physics(monkeypatch)
    scene = _Prim(
        "/World/physicsScene",
        kind="scene",
        attributes={
            "physxScene:solverType": _Attribute("TGS"),
            "physxScene:enableStabilization": _Attribute(True),
            "physxScene:minPositionIterationCount": _Attribute(1),
            "physxScene:maxPositionIterationCount": _Attribute(255),
            "physxScene:minVelocityIterationCount": _Attribute(0),
            "physxScene:maxVelocityIterationCount": _Attribute(255),
            "physxScene:timeStepsPerSecond": _Attribute(60, authored=False),
        },
    )
    articulation = _Prim(
        "/World/Env_0/Robot/hand",
        kind="articulation",
        attributes={
            "physxArticulation:solverPositionIterationCount": _Attribute(4),
            "physxArticulation:solverVelocityIterationCount": _Attribute(1),
            "physxArticulation:sleepThreshold": _Attribute(0.005),
            "physxArticulation:stabilizationThreshold": _Attribute(0.001),
            "physxArticulation:enabledSelfCollisions": _Attribute(False),
        },
    )
    simulation = SimpleNamespace(
        stage=_Stage([scene, articulation]),
        cfg=SimpleNamespace(
            dt=0.002,
            render_interval=1000,
            physics_prim_path="/World/physicsScene",
        ),
        get_physics_dt=lambda: 0.002,
    )

    result = ovphysx._read_live_composed_usd_solver_contract(
        simulation, requested_dt=0.002
    )

    assert imports == ["pxr.UsdPhysics"]
    assert "pxr.PhysxSchema" not in inspect.getsource(ovphysx)
    assert result["source"] == "composed_usd_input"
    assert result["compiled_runtime_readback"] is False
    assert result["schema_version"] == 2
    assert result["scene"]["solver_type"]["resolved_value"] == "TGS"  # type: ignore[index]
    assert result["scene"]["time_steps_per_second"][  # type: ignore[index]
        "has_authored_value_opinion"
    ] is False
    assert result["derived_clamped_requested_iterations"] == {
        "position": 4,
        "velocity": 1,
    }
    assert result["stepping"]["internal_solver_substeps"] is None  # type: ignore[index]


def test_live_solver_readback_records_missing_schema_attribute_without_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_usd_physics(monkeypatch)
    scene = _Prim(
        "/World/physicsScene",
        kind="scene",
        attributes={
            "physxScene:enableStabilization": _Attribute(True),
            "physxScene:minPositionIterationCount": _Attribute(1),
            "physxScene:maxPositionIterationCount": _Attribute(255),
            "physxScene:minVelocityIterationCount": _Attribute(0),
            "physxScene:maxVelocityIterationCount": _Attribute(255),
            "physxScene:timeStepsPerSecond": _Attribute(60),
        },
    )
    articulation = _Prim(
        "/World/Env_0/Robot/hand",
        kind="articulation",
        attributes={
            "physxArticulation:solverPositionIterationCount": _Attribute(4),
            "physxArticulation:solverVelocityIterationCount": _Attribute(1),
            "physxArticulation:sleepThreshold": _Attribute(0.005),
            "physxArticulation:stabilizationThreshold": _Attribute(0.001),
            "physxArticulation:enabledSelfCollisions": _Attribute(False),
        },
    )
    simulation = SimpleNamespace(
        stage=_Stage([scene, articulation]),
        cfg=SimpleNamespace(
            dt=0.002,
            render_interval=1000,
            physics_prim_path="/World/physicsScene",
        ),
        get_physics_dt=lambda: 0.002,
    )

    result = ovphysx._read_live_composed_usd_solver_contract(
        simulation, requested_dt=0.002
    )

    record = result["scene"]["solver_type"]  # type: ignore[index]
    assert record["resolved_available"] is False
    assert record["resolved_value"] is None
    assert record["has_authored_value_opinion"] is None
    assert "No default or compiled state was inferred" in record["unavailable_reason"]
    assert record["source_locator"].endswith("#physxScene:solverType")


def test_composed_joint_descriptors_keep_authored_resolved_layers_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_usd_physics(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    prims = [
        _Prim(
            f"/World/Env_0/Robot/joints/{name}",
            kind="revolute",
            name=name,
            attributes={
                "physics:axis": _Attribute("Z"),
                "physxJoint:armature": _Attribute(
                    index / 1000.0, authored=index != 0
                ),
                "physxJoint:jointFriction": _Attribute(index / 100.0),
            },
        )
        for index, name in enumerate(names)
    ]
    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    adapter._simulation = SimpleNamespace(stage=_Stage(prims))
    adapter._backend_joint_names = names

    records = adapter._read_composed_joint_descriptors()

    assert tuple(records) == names
    first = records[names[0]]
    last = records[names[-1]]
    assert first["armature_usd"] == {
        "attribute": "physxJoint:armature",
        "has_authored_value_opinion": False,
        "authored_value": None,
        "resolved_value": 0.0,
        "unit": "kg_m2_for_revolute_joint",
        "source": "composed_usd_input",
        "compiled_runtime_readback": False,
        "source_locator": (
            "/World/Env_0/Robot/joints/joint_00.physxJoint:armature"
        ),
    }
    assert last["legacy_joint_friction_usd"]["authored_value"] == pytest.approx(  # type: ignore[index]
        0.21
    )
    assert last["legacy_joint_friction_usd"][  # type: ignore[index]
        "runtime_binding_equality_check"
    ] == "not_performed"


def _prepared_joint_dynamics_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ovphysx.OVPhysXAdapter, object]:
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stiffness = [10.0 + index for index in range(22)]
    damping = [1.0 + index / 10.0 for index in range(22)]
    effort = [2.0 + index / 100.0 for index in range(22)]
    armature = [float(np.float32(index / 1000.0)) for index in range(22)]
    static_friction = [float(np.float32(index / 10.0)) for index in range(22)]
    dynamic_friction = [float(np.float32(index / 20.0)) for index in range(22)]
    viscous_friction = [float(np.float32(index / 30.0)) for index in range(22)]
    controller_type = type("IdealPDActuator", (), {})
    controller = controller_type()
    controller.is_implicit_model = False
    controller.joint_names = names
    controller.joint_indices = slice(None)
    controller.stiffness = _Tensor(stiffness)
    controller.damping = _Tensor(damping)
    controller.effort_limit = _Tensor(effort)
    controller.effort_limit_sim = _Tensor(effort)
    controller.armature = _Tensor(armature)
    controller.friction = _Tensor(static_friction)
    controller.dynamic_friction = _Tensor(dynamic_friction)
    controller.viscous_friction = _Tensor(viscous_friction)

    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    adapter._controller = controller
    adapter._controller_stiffness = tuple(stiffness)
    adapter._controller_damping = tuple(damping)
    adapter._controller_effort_limit = tuple(effort)
    adapter._controller_effort_limit_sim = tuple(effort)
    adapter._controller_armature = tuple(armature)
    adapter._controller_static_friction = tuple(static_friction)
    adapter._controller_dynamic_friction = tuple(dynamic_friction)
    adapter._controller_viscous_friction = tuple(viscous_friction)
    adapter._backend_joint_names = names
    adapter._joint_mapping = tuple(
        {
            "canonical_id": name,
            "backend_name": name,
            "index": index,
        }
        for index, name in enumerate(names)
    )
    adapter._tensor_bindings = {
        "DOF_STIFFNESS": _CpuBinding(np.zeros((1, 22))),
        "DOF_DAMPING": _CpuBinding(np.zeros((1, 22))),
        "DOF_ARMATURE": _CpuBinding(
            np.asarray([armature])
        ),
        "DOF_FRICTION_PROPERTIES": _CpuBinding(
            np.asarray(
                [
                    [
                        [index / 10.0, index / 20.0, index / 30.0]
                        for index in range(22)
                    ]
                ]
            )
        ),
    }
    descriptors = {
        name: {
            "joint_prim_path": f"/World/Env_0/Robot/{name}",
            "joint_type": "revolute",
            "joint_axis": "Z",
            "armature_usd": {"resolved_value": index / 1000.0},
            "legacy_joint_friction_usd": {
                "resolved_value": index / 10.0,
                "runtime_binding_equality_check": "not_performed",
            },
        }
        for index, name in enumerate(names)
    }
    monkeypatch.setattr(
        adapter, "_read_composed_joint_descriptors", lambda: descriptors
    )
    return adapter, controller


def test_joint_dynamics_readback_covers_all_dofs_and_checks_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, controller = _prepared_joint_dynamics_adapter(monkeypatch)

    result = adapter._capture_backend_joint_dynamics()

    assert result["joint_count"] == 22
    assert len(result["records"]) == 22
    assert result["controller_consistency_verified"] is True
    assert result["controller_joint_dynamics_binding_comparison_performed"] is True
    assert result["controller_joint_dynamics_binding_exact_match"] == {
        "overall": True,
        "armature": True,
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }
    record = result["records"][7]
    assert record["backend_index"] == 7
    assert record["armature"] == pytest.approx(0.007)
    assert record["friction_properties_raw"] == pytest.approx(
        [0.7, 0.35, 7 / 30]
    )
    assert record["armature_usd"]["resolved_value"] == pytest.approx(0.007)
    assert record["legacy_joint_friction_usd"][  # type: ignore[index]
        "runtime_binding_equality_check"
    ] == "not_performed"
    assert record["controller_armature"] == record["armature"]
    assert record["controller_static_friction"] == record[
        "friction_properties_raw"
    ][0]
    assert record["controller_binding_exact_match"] == {
        "armature": True,
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }

    controller.stiffness = _Tensor([99.0] * 22)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="changed between initialization"):
        adapter._capture_backend_joint_dynamics()


def test_joint_dynamics_preserves_binding_controller_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _controller = _prepared_joint_dynamics_adapter(monkeypatch)
    adapter._tensor_bindings["DOF_ARMATURE"] = _CpuBinding(
        np.ones((1, 22), dtype=np.float32)
    )

    result = adapter._capture_backend_joint_dynamics()

    assert result["controller_joint_dynamics_binding_comparison_performed"] is True
    assert result["controller_joint_dynamics_binding_exact_match"] == {
        "overall": False,
        "armature": False,
        "static_friction": True,
        "dynamic_friction": True,
        "viscous_friction": True,
    }
    assert result["records"][0]["armature"] == pytest.approx(1.0)
    assert result["records"][0]["controller_armature"] == pytest.approx(0.0)
    assert result["records"][0]["controller_binding_exact_match"]["armature"] is False


def test_effective_readback_is_detached_and_rejected_after_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    adapter._lifecycle = AdapterLifecycle.READY
    adapter._simulation = object()
    adapter._dt = 0.002
    monkeypatch.setattr(
        adapter,
        "_capture_backend_joint_dynamics",
        lambda: {"records": [{"armature": 0.1}]},
    )
    monkeypatch.setattr(
        ovphysx,
        "_read_live_composed_usd_solver_contract",
        lambda simulation, *, requested_dt: {
            "simulation_is_expected": simulation is adapter._simulation,
            "requested_dt_s": requested_dt,
        },
    )

    snapshot = adapter.effective_parameter_readback()
    snapshot["backend_joint_dynamics"]["records"][0]["armature"] = 9.9  # type: ignore[index]

    assert adapter._effective_parameter_snapshot["backend_joint_dynamics"][  # type: ignore[index]
        "records"
    ][0]["armature"] == 0.1
    adapter._step_index = 1
    with pytest.raises(AdapterLifecycleError, match="before the first trace step"):
        adapter.effective_parameter_readback()


def test_intervention_envelope_and_joint_coverage_are_strict() -> None:
    names = tuple(f"joint_{index:02d}" for index in range(22))
    extra = _intervention_plan(names, role="zero")
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="contain exactly"):
        ovphysx.OVPhysXAdapter(
            device="cpu", legacy_joint_friction_intervention=extra
        )

    adapter = ovphysx.OVPhysXAdapter(
        device="cpu",
        legacy_joint_friction_intervention=_intervention_plan(names, role="zero"),
    )
    incomplete_names = names[:-1]
    with pytest.raises(ValueError, match="exactly cover"):
        ovphysx._validate_legacy_joint_friction_intervention_for_joints(
            adapter._legacy_joint_friction_intervention, incomplete_names
        )


def test_intervention_roles_are_bound_by_exact_float32_bits() -> None:
    names = tuple(f"joint_{index:02d}" for index in range(22))
    sham = _intervention_plan(names, role="sham")
    sham["write_values"][names[0]] = 0.25  # type: ignore[index]
    adapter = ovphysx.OVPhysXAdapter(
        device="cpu", legacy_joint_friction_intervention=sham
    )
    with pytest.raises(ValueError, match="role sham"):
        ovphysx._validate_legacy_joint_friction_intervention_for_joints(
            adapter._legacy_joint_friction_intervention, names
        )

    zero = _intervention_plan(names, role="zero")
    zero["write_values"][names[0]] = -0.0  # type: ignore[index]
    adapter = ovphysx.OVPhysXAdapter(
        device="cpu", legacy_joint_friction_intervention=zero
    )
    with pytest.raises(ValueError, match="positive float32 zero"):
        ovphysx._validate_legacy_joint_friction_intervention_for_joints(
            adapter._legacy_joint_friction_intervention, names
        )


@pytest.mark.parametrize("role", ["sham", "zero"])
def test_intervention_uses_anonymous_overlay_and_survives_reset_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role=role)
    articulation = SimpleNamespace(is_initialized=False)

    adapter._apply_legacy_joint_friction_intervention(
        stage,
        articulation=articulation,
        joint_names=names,
        source=source,
    )
    articulation.is_initialized = True
    adapter._verify_legacy_joint_friction_intervention(
        stage, phase="post_reset", articulation=articulation
    )

    evidence = adapter._legacy_joint_friction_intervention_evidence
    assert evidence["role"] == role
    assert evidence["exact_manifest_coverage_verified"] is True
    assert evidence["all_properties_prevalidated_before_write"] is True
    assert evidence["post_write_verified"] is True
    assert evidence["post_reset_verified"] is True
    assert evidence["override_layer_anonymous"] is True
    assert evidence["source_asset_was_edit_target"] is False
    assert evidence["articulation_uninitialized_before_write"] is True
    assert evidence["articulation_initialized_after_reset"] is True
    assert len(evidence["records"]) == 22  # type: ignore[arg-type]
    expected = 0.125 if role == "sham" else 0.0
    assert all(
        prim.attribute.Get() == expected for prim in stage.prims.values()
    )


def test_intervention_type_drift_fails_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    first = stage.prims[f"/World/Env_0/Robot/joints/{names[0]}"]
    first.attribute.type_name = "double"
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role="zero")

    with pytest.raises(RuntimeError, match="is not float"):
        adapter._apply_legacy_joint_friction_intervention(
            stage,
            articulation=SimpleNamespace(is_initialized=False),
            joint_names=names,
            source=source,
        )

    assert stage.session_layer.subLayerPaths == []
    assert all(
        prim.attribute.write_count == 0 for prim in stage.prims.values()
    )
    assert adapter._legacy_joint_friction_intervention_evidence == {}


def test_intervention_refuses_an_already_initialized_articulation_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role="zero")

    with pytest.raises(RuntimeError, match="must be uninitialized"):
        adapter._apply_legacy_joint_friction_intervention(
            stage,
            articulation=SimpleNamespace(is_initialized=True),
            joint_names=names,
            source=source,
        )

    assert stage.session_layer.subLayerPaths == []
    assert all(prim.attribute.write_count == 0 for prim in stage.prims.values())
    assert adapter._legacy_joint_friction_intervention_evidence == {}


def test_post_reset_verification_requires_initialized_articulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role="zero")
    articulation = SimpleNamespace(is_initialized=False)
    adapter._apply_legacy_joint_friction_intervention(
        stage,
        articulation=articulation,
        joint_names=names,
        source=source,
    )

    with pytest.raises(RuntimeError, match="must be initialized"):
        adapter._verify_legacy_joint_friction_intervention(
            stage, phase="post_reset", articulation=articulation
        )

    evidence = adapter._legacy_joint_friction_intervention_evidence
    assert evidence["articulation_uninitialized_before_write"] is True
    assert evidence["articulation_initialized_after_reset"] is False
    assert evidence["post_reset_verified"] is False
    adapter._cleanup_legacy_joint_friction_intervention(stage)


def test_intervention_refuses_nonanonymous_original_edit_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    stage.root_layer.anonymous = False
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role="zero")

    with pytest.raises(RuntimeError, match="edit target must be anonymous"):
        adapter._apply_legacy_joint_friction_intervention(
            stage,
            articulation=SimpleNamespace(is_initialized=False),
            joint_names=names,
            source=source,
        )

    assert stage.session_layer.subLayerPaths == []
    assert all(prim.attribute.write_count == 0 for prim in stage.prims.values())
    assert adapter._legacy_joint_friction_intervention_evidence == {}


def test_intervention_partial_write_failure_rolls_back_every_opinion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    stage.prims[
        f"/World/Env_0/Robot/joints/{names[8]}"
    ].attribute.fail_write = True
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role="zero")

    with pytest.raises(RuntimeError, match="failed to author"):
        adapter._apply_legacy_joint_friction_intervention(
            stage,
            articulation=SimpleNamespace(is_initialized=False),
            joint_names=names,
            source=source,
        )

    assert stage.session_layer.subLayerPaths == []
    assert all(
        prim.attribute.Get() == 0.125 for prim in stage.prims.values()
    )
    assert (
        adapter._legacy_joint_friction_intervention_evidence["cleanup_status"]
        == "restored_pre_values"
    )


def test_release_removes_overlay_and_proves_source_hash_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_intervention_pxr(monkeypatch)
    names = tuple(f"joint_{index:02d}" for index in range(22))
    stage = _InterventionStage(names)
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = _prepared_intervention_adapter(names, role="zero")
    articulation = SimpleNamespace(is_initialized=False)
    adapter._apply_legacy_joint_friction_intervention(
        stage,
        articulation=articulation,
        joint_names=names,
        source=source,
    )
    articulation.is_initialized = True
    adapter._verify_legacy_joint_friction_intervention(
        stage, phase="post_reset", articulation=articulation
    )
    context = _InterventionContextManager()
    adapter._context_manager = context
    adapter._simulation = SimpleNamespace(stage=stage)

    adapter._release_runtime()

    evidence = adapter._legacy_joint_friction_intervention_evidence
    assert context.exited is True
    assert evidence["cleanup_status"] == "restored_pre_values"
    assert evidence["source_asset_sha256_unchanged"] is True
    assert evidence["source_asset_sha256_after_cleanup"] == evidence[
        "source_asset_sha256_before"
    ]
    assert stage.session_layer.subLayerPaths == []
    assert all(
        prim.attribute.Get() == 0.125 for prim in stage.prims.values()
    )


def test_simulation_configuration_v2_is_additive_and_denies_runtime_dt_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = {
        "verified": True,
        "requested_dt_s": 0.002,
        "cfg_dt_s": 0.002,
        "backend_dt_s": 0.002,
        "requested_gravity_m_s2": [0.0, 0.0, 0.0],
        "cfg_gravity_m_s2": [0.0, 0.0, 0.0],
        "physics_scene_gravity_m_s2": [0.0, 0.0, 0.0],
        "physics_prim_path": "/physicsScene",
    }
    unchanged = dict(legacy)
    v2 = ovphysx.OVPhysXAdapter._simulation_configuration_v2_from_legacy(
        legacy
    )

    assert legacy == unchanged
    assert v2["schema_version"] == 2
    assert v2["simulation_context_config_accessor_dt_s"] == 0.002
    assert v2["runtime_effective_dt_s"] is None
    assert v2["runtime_effective_dt_verified"] is False
    assert v2["dt_claim_scope"] == (
        "python_configuration_only_not_compiled_runtime"
    )

    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    adapter._source_path = source
    adapter._simulation_contract = dict(legacy)
    adapter._simulation_contract_v2 = dict(v2)
    monkeypatch.setattr(ovphysx, "_package_versions", lambda: {})
    provenance = adapter._provenance()
    assert provenance["simulation_configuration"] == legacy
    assert provenance["simulation_configuration_v2"] == v2


def _seed_effective_state(adapter: ovphysx.OVPhysXAdapter) -> None:
    adapter._runtime = {"sentinel": object()}
    adapter._simulation = object()
    adapter._articulation = object()
    adapter._source_path = Path("synthetic.usda")
    adapter._controller_armature = (0.1,)
    adapter._controller_static_friction = (0.2,)
    adapter._controller_dynamic_friction = (0.3,)
    adapter._controller_viscous_friction = (0.4,)
    adapter._backend_armature = (0.1,)
    adapter._backend_friction_properties = ((0.2, 0.3, 0.4),)
    adapter._tensor_bindings = {"DOF_ARMATURE": object()}
    adapter._backend_joint_dynamics = {"sentinel": True}
    adapter._physics_solver_contract = {"sentinel": True}
    adapter._effective_parameter_snapshot = {"sentinel": True}


def _assert_effective_state_released(adapter: ovphysx.OVPhysXAdapter) -> None:
    assert adapter._runtime == {}
    assert adapter._simulation is None
    assert adapter._articulation is None
    assert adapter._source_path is None
    assert adapter._controller_armature == ()
    assert adapter._controller_static_friction == ()
    assert adapter._controller_dynamic_friction == ()
    assert adapter._controller_viscous_friction == ()
    assert adapter._backend_armature == ()
    assert adapter._backend_friction_properties == ()
    assert adapter._tensor_bindings == {}
    assert adapter._backend_joint_dynamics == {}
    assert adapter._physics_solver_contract == {}
    assert adapter._effective_parameter_snapshot == {}
    assert adapter._open_gravity == (0.0, 0.0, 0.0)


def test_close_releases_all_effective_readback_state() -> None:
    class Context:
        exit_count = 0

        def __exit__(self, *args: object) -> None:
            del args
            self.exit_count += 1

    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    context = Context()
    adapter._context_manager = context
    adapter._lifecycle = AdapterLifecycle.READY
    _seed_effective_state(adapter)

    adapter.close()

    assert context.exit_count == 1
    assert adapter.lifecycle is AdapterLifecycle.CLOSED
    _assert_effective_state_released(adapter)


def test_close_failure_still_releases_state_and_marks_adapter_failed() -> None:
    class Context:
        def __exit__(self, *args: object) -> None:
            del args
            raise RuntimeError("synthetic close failure")

    adapter = ovphysx.OVPhysXAdapter(device="cpu")
    adapter._context_manager = Context()
    adapter._lifecycle = AdapterLifecycle.READY
    _seed_effective_state(adapter)

    with pytest.raises(RuntimeError, match="synthetic close failure"):
        adapter.close()

    assert adapter.lifecycle is AdapterLifecycle.FAILED
    _assert_effective_state_released(adapter)


def test_open_failure_after_context_entry_releases_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "hand.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    class Context:
        exit_count = 0

        def __enter__(self) -> object:
            return SimpleNamespace()

        def __exit__(self, *args: object) -> None:
            del args
            self.exit_count += 1

    context = Context()

    class SimulationCfg:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    def fail_after_context_entry(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic create_prim failure")

    sim_utils = SimpleNamespace(
        SimulationCfg=SimulationCfg,
        build_simulation_context=lambda **kwargs: context,
        create_prim=fail_after_context_entry,
    )
    runtime = {
        "torch": object(),
        "sim": sim_utils,
        "ov_physics": SimpleNamespace(OvPhysxCfg=lambda: object()),
    }
    monkeypatch.setattr(ovphysx, "_runtime_api", lambda device: runtime)
    adapter = ovphysx.OVPhysXAdapter(device="cpu", asset_root=tmp_path)
    _seed_effective_state(adapter)
    adapter._source_path = None

    manifest = {
        "side": "left",
        "mounting": "fixed_base",
        "control_mode": "position",
        "resolved_usd": source.name,
        "joint_names": [f"joint_{index:02d}" for index in range(22)],
        "distal_frame_names": [f"tip_{index}" for index in range(5)],
    }
    with pytest.raises(RuntimeError, match="synthetic create_prim failure"):
        adapter.open(manifest, dt_override=0.002)

    assert context.exit_count == 1
    assert adapter.lifecycle is AdapterLifecycle.FAILED
    _assert_effective_state_released(adapter)
