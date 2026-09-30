"""Conservative, lazy-import OVPhysX capability probe and adapter boundary.

Gate 0 targets Isaac Lab's kit-less OVPhysX path.  No Isaac Lab, OpenUSD,
Torch, Kit, renderer, or camera module is imported when this module itself is
imported. Runtime execution remains process-isolated because the pinned wheel
locks its device mode globally for the lifetime of one process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from importlib import import_module, metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import sys
from typing import Any

from wave_asset_qa.parity.contracts import ScenarioSpec, WaveformScenarioSpec
from wave_asset_qa.parity.scenarios import (
    CanonicalTargetSequenceDigest,
    canonical_position_targets,
    canonical_target_sequence_sha256,
)

from .base import (
    AdapterCapabilityError,
    AdapterLifecycle,
    AdapterLifecycleError,
    AdapterProbeResult,
    AdapterRunResult,
    TraceSample,
)


_MISSING = object()
_PINNED_OVPHYSX_WHEEL_VERSION = "0.4.13"
_DEVICE_PATTERN = re.compile(r"^(?:cpu|cuda:\d+)$")
_FORBIDDEN_KEYS = {
    "kit",
    "use_kit",
    "enable_kit",
    "renderer",
    "enable_renderer",
    "camera",
    "cameras",
    "enable_cameras",
    "livestream",
    "gui",
}
_FORBIDDEN_MODULE_PREFIXES = (
    "omni.kit",
    "omni.renderer",
    "isaacsim",
)
_SUPPORTED_SCENARIOS = {
    "zero_hold",
    "small_step",
    "gravity_settling",
    "offset_sine",
    "offset_linear_chirp",
}
_BACKEND_DRIVE_ZERO_ABS_TOL = 1e-8
_ZERO_AUXILIARY_TARGET_ABS_TOL = 1e-8
_EFFORT_FORMULA_VALIDATION_ABS_TOL = 1e-5
_EFFORT_CLIP_VALIDATION_ABS_TOL = 1e-6
_EFFECTIVE_PARAMETER_READBACK_CONTRACT_VERSION = 1
_BACKEND_JOINT_DYNAMICS_SCHEMA_VERSION = 1
_PHYSICS_SOLVER_CONTRACT_SCHEMA_VERSION = 2
_LEGACY_JOINT_FRICTION_INTERVENTION_SCHEMA_VERSION = 1
_LEGACY_JOINT_FRICTION_INTERVENTION_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "private_plan_sha256",
        "expected_pre_values",
        "write_values",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _float32_value_and_hex(value: object, *, name: str) -> tuple[float, str]:
    """Return the finite IEEE-754 binary32 value and its exact bit pattern."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    try:
        payload = struct.pack("!f", numeric)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name} must be representable as a finite float32") from error
    canonical = struct.unpack("!f", payload)[0]
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be representable as a finite float32")
    return canonical, payload.hex()


def _copy_legacy_joint_friction_intervention(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Validate and detach the intervention envelope from caller-owned data."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("legacy_joint_friction_intervention must be an object")
    if set(value) != _LEGACY_JOINT_FRICTION_INTERVENTION_KEYS:
        raise ValueError(
            "legacy_joint_friction_intervention must contain exactly: "
            + ", ".join(sorted(_LEGACY_JOINT_FRICTION_INTERVENTION_KEYS))
        )
    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _LEGACY_JOINT_FRICTION_INTERVENTION_SCHEMA_VERSION
    ):
        raise ValueError("legacy joint-friction intervention schema_version must be 1")
    role = value["role"]
    if role not in {"sham", "zero"}:
        raise ValueError("legacy joint-friction intervention role must be 'sham' or 'zero'")
    private_plan_sha256 = value["private_plan_sha256"]
    if (
        not isinstance(private_plan_sha256, str)
        or _SHA256_PATTERN.fullmatch(private_plan_sha256) is None
    ):
        raise ValueError("private_plan_sha256 must be a lowercase SHA-256 digest")
    detached: dict[str, object] = {
        "schema_version": schema_version,
        "role": role,
        "private_plan_sha256": private_plan_sha256,
    }
    for field in ("expected_pre_values", "write_values"):
        raw = value[field]
        if not isinstance(raw, Mapping):
            raise ValueError(f"legacy joint-friction {field} must be an object")
        copied: dict[str, float] = {}
        for raw_name, raw_value in raw.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"legacy joint-friction {field} keys must be names")
            canonical, _ = _float32_value_and_hex(
                raw_value, name=f"legacy joint-friction {field}.{raw_name}"
            )
            copied[raw_name] = canonical
        detached[field] = copied
    return detached


def _validate_legacy_joint_friction_intervention_for_joints(
    intervention: Mapping[str, object] | None,
    joint_names: tuple[str, ...],
) -> dict[str, object] | None:
    """Bind a detached sham/zero plan to the exact manifest joint inventory."""

    if intervention is None:
        return None
    expected = intervention["expected_pre_values"]
    write = intervention["write_values"]
    assert isinstance(expected, Mapping) and isinstance(write, Mapping)
    expected_names = set(joint_names)
    if set(expected) != expected_names or set(write) != expected_names:
        raise ValueError(
            "legacy joint-friction intervention must exactly cover the 22 manifest joints"
        )
    role = str(intervention["role"])
    normalized_expected: dict[str, float] = {}
    normalized_write: dict[str, float] = {}
    for name in joint_names:
        expected_value, expected_hex = _float32_value_and_hex(
            expected[name], name=f"expected_pre_values.{name}"
        )
        write_value, write_hex = _float32_value_and_hex(
            write[name], name=f"write_values.{name}"
        )
        if expected_value <= 0.0:
            raise ValueError(
                f"expected_pre_values.{name} must be strictly positive in Freeze B"
            )
        if role == "zero":
            if write_hex != "00000000":
                raise ValueError(
                    f"write_values.{name} must be positive float32 zero for role zero"
                )
        elif write_hex != expected_hex:
            raise ValueError(
                f"write_values.{name} must exactly match expected_pre_values for role sham"
            )
        normalized_expected[name] = expected_value
        normalized_write[name] = write_value
    return {
        "schema_version": int(intervention["schema_version"]),
        "role": role,
        "private_plan_sha256": str(intervention["private_plan_sha256"]),
        "expected_pre_values": normalized_expected,
        "write_values": normalized_write,
    }


def _field(source: object, *names: str, default: object = _MISSING) -> object:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    if default is _MISSING:
        raise ValueError(f"required field is missing: {', '.join(names)}")
    return default


def _plain_value(value: object) -> object:
    """Return an Enum-like value without importing the parity contracts."""

    return getattr(value, "value", value)


def _names(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique non-empty strings")
    return result


def _side(manifest: object) -> str:
    side = str(_plain_value(_field(manifest, "side", default=""))).lower()
    if side not in {"left", "right"}:
        raise ValueError("Wave hand side must be 'left' or 'right'")
    return side


def _path_entry(entry: object, asset_root: Path | None) -> Path | None:
    if entry is None:
        return None
    if isinstance(entry, (str, Path)):
        raw = Path(entry).expanduser()
    else:
        value = _field(
            entry,
            "path",
            "resolved_path",
            "source_path",
            "usd_path",
            default=None,
        )
        if value is None:
            return None
        raw = Path(str(value)).expanduser()
    if not raw.is_absolute() and asset_root is not None:
        raw = asset_root / raw
    return raw.resolve()


def _resolve_usd_path(
    manifest: object | None,
    *,
    asset_root: Path | None,
    resolved_usd: str | Path | None,
    hand: str | None = None,
) -> Path:
    direct = _path_entry(resolved_usd, asset_root)
    if direct is not None:
        return direct
    if manifest is None:
        raise ValueError("resolved USD mode requires --resolved-usd or a manifest USD path")

    direct = _path_entry(
        _field(manifest, "resolved_usd", "usd_path", "ovphysx_path", default=None),
        asset_root,
    )
    if direct is not None:
        return direct
    for container_name in ("ovphysx", "model_paths", "assets", "models"):
        container = _field(manifest, container_name, default=None)
        if container is None:
            continue
        if container_name == "ovphysx":
            candidate = _path_entry(container, asset_root)
            if candidate is not None:
                return candidate
        if container_name == "model_paths":
            candidate = _path_entry(
                _field(container, "ovphysx", "usd", default=None), asset_root
            )
            if candidate is not None:
                return candidate
        if isinstance(container, Mapping):
            for key in ("ovphysx", "usd", "resolved_usd"):
                candidate = _path_entry(container.get(key), asset_root)
                if candidate is not None:
                    return candidate
    hands = _field(manifest, "hands", default=None)
    if isinstance(hands, (list, tuple)):
        selected = [
            item
            for item in hands
            if hand is None or str(_field(item, "side", default="")).lower() == hand
        ]
        if hand is None and len(selected) != 1:
            raise ValueError(
                "a multi-hand canonical manifest requires an explicit hand selector"
            )
        if len(selected) != 1:
            raise ValueError(f"canonical manifest does not contain exactly one {hand!r} hand")
        model_paths = _field(selected[0], "model_paths", default=None)
        candidate = _path_entry(
            _field(model_paths, "ovphysx", "usd", default=None), asset_root
        )
        if candidate is not None:
            return candidate
    raise ValueError("manifest does not provide a resolved USD path")


def _requested(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "off", "headless"}
    if isinstance(value, (list, tuple, set, Mapping)):
        return bool(value)
    return bool(value)


def _reject_forbidden_features(value: object, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            current = f"{path}.{raw_key}"
            if key in _FORBIDDEN_KEYS and _requested(nested):
                raise AdapterCapabilityError(
                    "ovphysx",
                    "kitless_headless_only",
                    f"{current} requests a forbidden Kit/renderer/camera feature",
                )
            if key in {"render_mode", "rendering_mode"} and _requested(nested):
                raise AdapterCapabilityError(
                    "ovphysx",
                    "renderer_disabled",
                    f"{current} must be 'none' or 'headless'",
                )
            _reject_forbidden_features(nested, path=current)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden_features(nested, path=f"{path}[{index}]")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in (
        "isaaclab",
        "isaaclab-ovphysx",
        "ovphysx",
        "torch",
        "usd-core",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _base_provenance(device: str) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "display_present": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "packages": _package_versions(),
        "constraints": {
            "headless": True,
            "kit": False,
            "renderer": False,
            "camera": False,
        },
    }


def _validate_device(device: str) -> str:
    normalized = device.strip().lower()
    if not _DEVICE_PATTERN.fullmatch(normalized):
        raise ValueError("device must be 'cpu' or an explicit logical CUDA device such as 'cuda:0'")
    return normalized


def _scenario_steps(scenario: object, dt: float) -> int:
    duration = _field(scenario, "duration_s", "duration", default=None)
    if duration is not None and dt > 0.0:
        ratio = float(duration) / dt
        nearest = round(ratio)
        steps = (
            nearest
            if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-9)
            else int(math.ceil(ratio))
        )
        return max(0, steps)
    raw = _field(scenario, "steps", "num_steps", default=None)
    return max(0, int(raw)) if raw is not None else 0


def _read_cpu_float_binding(
    binding: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> tuple[Any, ...]:
    """Read an OVPhysX CPU-only binding through its supported NumPy path.

    The pinned OVPhysX articulation implementation marks joint properties as
    CPU-only bindings. Reading one into a CUDA Torch tensor bypasses Isaac
    Lab's pinned-host staging and fails on a GPU worker. A NumPy buffer matches
    the binding read path used by the locked upstream implementation and
    remains an independent backend readback.

    OVPhysX binding shapes always begin with the articulation-instance axis.
    Gate 0 has exactly one instance, so this helper removes that leading axis
    while preserving every remaining rank. In particular, ``(1, 22)`` reads
    return 22 floats and ``(1, 22, 3)`` reads return 22 friction triples.
    """

    actual_shape = tuple(int(value) for value in binding.shape)
    if len(expected_shape) < 2 or expected_shape[0] != 1:
        raise RuntimeError(
            f"{name} CPU binding contract requires one articulation instance"
        )
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"unexpected {name} binding shape: {actual_shape}, expected {expected_shape}"
        )
    numpy = import_module("numpy")
    buffer = numpy.full(actual_shape, numpy.nan, dtype=numpy.float32)
    binding.read(buffer)
    flat_values = tuple(float(value) for value in buffer.reshape(-1).tolist())
    expected_size = math.prod(expected_shape)
    if len(flat_values) != expected_size or not all(
        math.isfinite(value) for value in flat_values
    ):
        raise RuntimeError(f"{name} CPU binding readback is invalid")

    def convert(value: object) -> object:
        if isinstance(value, list):
            return tuple(convert(item) for item in value)
        return float(value)

    payload = convert(buffer[0].tolist())
    if not isinstance(payload, tuple):
        raise RuntimeError(f"{name} CPU binding readback rank is invalid")
    return payload


def _json_clone(value: Mapping[str, object]) -> dict[str, object]:
    """Return a detached JSON-safe copy while rejecting non-finite numbers."""

    cloned = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    if not isinstance(cloned, dict):
        raise RuntimeError("effective OVPhysX readback must be a JSON object")
    return cloned


def _strict_finite_number(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a non-boolean finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        constraint = f" >= {minimum}" if minimum is not None else ""
        raise RuntimeError(f"{name} must be finite{constraint}")
    return number


def _strict_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a non-boolean integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if not math.isfinite(numeric) or numeric != float(integer):
        raise RuntimeError(f"{name} must be an integer")
    if integer < minimum or integer > maximum:
        raise RuntimeError(f"{name} must be in [{minimum}, {maximum}]")
    return integer


def _composed_attribute(prim: Any, name: str) -> tuple[object, bool]:
    """Read an effective USD attribute and whether a value opinion was authored."""

    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.IsValid():
        raise RuntimeError(f"composed USD attribute is unavailable: {name}")
    value = attribute.Get()
    if value is None:
        raise RuntimeError(f"composed USD attribute has no effective value: {name}")
    authored = attribute.HasAuthoredValueOpinion()
    if not isinstance(authored, bool):
        raise RuntimeError(f"composed USD authored-state is invalid: {name}")
    return value, authored


def _probe_composed_solver_attribute(
    prim: Any, *, prim_path: str, name: str
) -> dict[str, object]:
    """Probe one solver input without inventing a USD schema fallback."""

    source_locator = f"live-composed-usd-probe:{prim_path}#{name}"
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.IsValid():
        return {
            "resolved_available": False,
            "resolved_value": None,
            "has_authored_value_opinion": None,
            "source_locator": source_locator,
            "unavailable_reason": (
                f"Live composed USD attribute {name} is absent on {prim_path}; "
                "the pinned kit-less API exposes no compiled solver getter. "
                "No default or compiled state was inferred."
            ),
        }
    value = attribute.Get()
    if value is None:
        return {
            "resolved_available": False,
            "resolved_value": None,
            "has_authored_value_opinion": None,
            "source_locator": source_locator,
            "unavailable_reason": (
                f"Live composed USD attribute {name} has no effective value on "
                f"{prim_path}; the pinned kit-less API exposes no compiled solver "
                "getter. No default or compiled state was inferred."
            ),
        }
    authored = attribute.HasAuthoredValueOpinion()
    if not isinstance(authored, bool):
        raise RuntimeError(f"composed USD authored-state is invalid: {name}")
    return {
        "resolved_available": True,
        "resolved_value": value,
        "has_authored_value_opinion": authored,
        "source_locator": source_locator,
        "unavailable_reason": None,
    }


def _read_live_composed_usd_solver_contract(
    simulation: Any,
    *,
    requested_dt: float,
    robot_prim_path: str = "/World/Env_0/Robot",
) -> dict[str, object]:
    """Read solver inputs from the live composed stage without overstating them.

    OVPhysX 0.4.13 has no Python tensor/getter for compiled scene or
    articulation solver settings. Present values are therefore recorded only
    as live composed-USD inputs visible around reset, not as compiled C++ state;
    absent attributes remain explicit unavailable records without defaults.
    """

    usd_physics = import_module("pxr.UsdPhysics")
    stage = simulation.stage
    prims = tuple(stage.Traverse())

    scene_prims = tuple(prim for prim in prims if prim.IsA(usd_physics.Scene))
    if len(scene_prims) != 1:
        raise RuntimeError(
            "live composed stage must contain exactly one USD PhysicsScene"
        )
    scene_prim = scene_prims[0]
    scene_path = str(scene_prim.GetPath())
    expected_scene_path = str(simulation.cfg.physics_prim_path)
    if scene_path != expected_scene_path:
        raise RuntimeError(
            "live composed PhysicsScene path differs from SimulationCfg"
        )
    robot_prefix = robot_prim_path.rstrip("/")
    articulation_prims = tuple(
        prim
        for prim in prims
        if (
            str(prim.GetPath()) == robot_prefix
            or str(prim.GetPath()).startswith(robot_prefix + "/")
        )
        and prim.HasAPI(usd_physics.ArticulationRootAPI)
    )
    if len(articulation_prims) != 1:
        raise RuntimeError(
            "live robot subtree must contain exactly one articulation root"
        )
    articulation_prim = articulation_prims[0]
    articulation_path = str(articulation_prim.GetPath())

    scene_raw: dict[str, dict[str, object]] = {
        "solver_type": _probe_composed_solver_attribute(
            scene_prim, prim_path=scene_path, name="physxScene:solverType"
        ),
        "enable_stabilization": _probe_composed_solver_attribute(
            scene_prim,
            prim_path=scene_path,
            name="physxScene:enableStabilization",
        ),
        "min_position_iteration_count": _probe_composed_solver_attribute(
            scene_prim,
            prim_path=scene_path,
            name="physxScene:minPositionIterationCount",
        ),
        "max_position_iteration_count": _probe_composed_solver_attribute(
            scene_prim,
            prim_path=scene_path,
            name="physxScene:maxPositionIterationCount",
        ),
        "min_velocity_iteration_count": _probe_composed_solver_attribute(
            scene_prim,
            prim_path=scene_path,
            name="physxScene:minVelocityIterationCount",
        ),
        "max_velocity_iteration_count": _probe_composed_solver_attribute(
            scene_prim,
            prim_path=scene_path,
            name="physxScene:maxVelocityIterationCount",
        ),
        "time_steps_per_second": _probe_composed_solver_attribute(
            scene_prim,
            prim_path=scene_path,
            name="physxScene:timeStepsPerSecond",
        ),
    }
    # Freeze A does not declare self-collision as an R1 output field, so it is
    # intentionally not sampled or implied by the repeatability claim here.
    articulation_raw: dict[str, dict[str, object]] = {
        "solver_position_iteration_count": _probe_composed_solver_attribute(
            articulation_prim,
            prim_path=articulation_path,
            name="physxArticulation:solverPositionIterationCount",
        ),
        "solver_velocity_iteration_count": _probe_composed_solver_attribute(
            articulation_prim,
            prim_path=articulation_path,
            name="physxArticulation:solverVelocityIterationCount",
        ),
        "sleep_threshold": _probe_composed_solver_attribute(
            articulation_prim,
            prim_path=articulation_path,
            name="physxArticulation:sleepThreshold",
        ),
        "stabilization_threshold": _probe_composed_solver_attribute(
            articulation_prim,
            prim_path=articulation_path,
            name="physxArticulation:stabilizationThreshold",
        ),
    }

    def optional_token(
        record: dict[str, object], *, name: str, allowed: set[str]
    ) -> str | None:
        if record["resolved_available"] is not True:
            return None
        token = str(record["resolved_value"])
        if token not in allowed:
            raise RuntimeError(f"{name} must be one of {sorted(allowed)}")
        record["resolved_value"] = token
        return token

    def optional_bool(record: dict[str, object], *, name: str) -> bool | None:
        if record["resolved_available"] is not True:
            return None
        value = record["resolved_value"]
        if not isinstance(value, bool):
            raise RuntimeError(f"{name} must be a boolean")
        return value

    def optional_integer(
        record: dict[str, object], *, name: str, minimum: int, maximum: int
    ) -> int | None:
        if record["resolved_available"] is not True:
            return None
        value = _strict_integer(
            record["resolved_value"], name=name, minimum=minimum, maximum=maximum
        )
        record["resolved_value"] = value
        return value

    def optional_number(
        record: dict[str, object], *, name: str, minimum: float
    ) -> float | None:
        if record["resolved_available"] is not True:
            return None
        value = _strict_finite_number(
            record["resolved_value"], name=name, minimum=minimum
        )
        record["resolved_value"] = value
        return value

    optional_token(
        scene_raw["solver_type"],
        name="PhysxScene solverType",
        allowed={"TGS", "PGS"},
    )
    optional_bool(
        scene_raw["enable_stabilization"],
        name="PhysxScene enableStabilization",
    )
    min_position = optional_integer(
        scene_raw["min_position_iteration_count"],
        name="PhysxScene minPositionIterationCount",
        minimum=1,
        maximum=255,
    )
    max_position = optional_integer(
        scene_raw["max_position_iteration_count"],
        name="PhysxScene maxPositionIterationCount",
        minimum=1,
        maximum=255,
    )
    min_velocity = optional_integer(
        scene_raw["min_velocity_iteration_count"],
        name="PhysxScene minVelocityIterationCount",
        minimum=0,
        maximum=255,
    )
    max_velocity = optional_integer(
        scene_raw["max_velocity_iteration_count"],
        name="PhysxScene maxVelocityIterationCount",
        minimum=0,
        maximum=255,
    )
    if (
        min_position is not None
        and max_position is not None
        and min_position > max_position
    ) or (
        min_velocity is not None
        and max_velocity is not None
        and min_velocity > max_velocity
    ):
        raise RuntimeError("PhysxScene solver iteration clamps are inverted")
    optional_integer(
        scene_raw["time_steps_per_second"],
        name="PhysxScene timeStepsPerSecond",
        minimum=1,
        maximum=2**32 - 1,
    )
    requested_position = optional_integer(
        articulation_raw["solver_position_iteration_count"],
        name="PhysxArticulation solverPositionIterationCount",
        minimum=1,
        maximum=255,
    )
    requested_velocity = optional_integer(
        articulation_raw["solver_velocity_iteration_count"],
        name="PhysxArticulation solverVelocityIterationCount",
        minimum=0,
        maximum=255,
    )
    optional_number(
        articulation_raw["sleep_threshold"],
        name="PhysxArticulation sleepThreshold",
        minimum=0.0,
    )
    optional_number(
        articulation_raw["stabilization_threshold"],
        name="PhysxArticulation stabilizationThreshold",
        minimum=0.0,
    )

    cfg_dt = _strict_finite_number(
        simulation.cfg.dt, name="SimulationCfg dt", minimum=0.0
    )
    config_accessor_dt = _strict_finite_number(
        simulation.get_physics_dt(),
        name="SimulationContext config-accessor dt",
        minimum=0.0,
    )
    if cfg_dt <= 0.0 or config_accessor_dt <= 0.0 or requested_dt <= 0.0:
        raise RuntimeError("OVPhysX timestep must be positive")
    if not math.isclose(cfg_dt, requested_dt, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("SimulationCfg timestep differs from requested timestep")
    if not math.isclose(
        config_accessor_dt, requested_dt, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "SimulationContext config-accessor timestep differs from requested timestep"
        )
    render_interval = _strict_integer(
        simulation.cfg.render_interval,
        name="SimulationCfg render_interval",
        minimum=1,
        maximum=2**31 - 1,
    )

    position_derived = None
    if (
        requested_position is not None
        and min_position is not None
        and max_position is not None
    ):
        position_derived = min(
            max(requested_position, min_position),
            max_position,
        )
    velocity_derived = None
    if (
        requested_velocity is not None
        and min_velocity is not None
        and max_velocity is not None
    ):
        velocity_derived = min(
            max(requested_velocity, min_velocity),
            max_velocity,
        )
    return {
        "schema_version": _PHYSICS_SOLVER_CONTRACT_SCHEMA_VERSION,
        "source": "composed_usd_input",
        "compiled_runtime_readback": False,
        "capture_phase": "post_runtime_reset_pre_trace_step",
        "physics_scene_path": scene_path,
        "articulation_root_path": articulation_path,
        "scene": scene_raw,
        "articulation": articulation_raw,
        "derived_clamped_requested_iterations": {
            "position": position_derived,
            "velocity": velocity_derived,
        },
        "derived_values_are_compiled_runtime_readback": False,
        "stepping": {
            "requested_dt_s": requested_dt,
            "cfg_dt_s": cfg_dt,
            "config_accessor_dt_s": config_accessor_dt,
            "config_accessor_semantics": (
                "SimulationContext.get_physics_dt_returns_SimulationCfg.dt_in_"
                "the_pinned_kitless_stack_not_compiled_backend_state"
            ),
            "backend_step_api": "ovphysx.PhysX.step_sync",
            "backend_step_calls_per_adapter_step": 1,
            "external_substeps": 1,
            "internal_solver_substeps": None,
            "internal_solver_substeps_status": "not_exposed_not_inferred",
            "physics_scene_time_steps_per_second_controls_effective_dt": False,
            "render_interval": render_interval,
            "render_interval_is_physics_substeps": False,
            "gpu_warmup_outside_trace": True,
            "gpu_warmup_timestep_status": "backend_internal_minimal_not_exposed",
        },
    }


def _read_gpu_float_binding(binding: Any, *, buffer: Any, warp: Any) -> Any:
    """Read a GPU binding into a reusable Warp buffer and expose a Torch view."""

    # Poison the reused buffer so a no-op/partial binding read cannot pass by
    # retaining the previous requested target. Warp and Torch use distinct
    # default streams, so establish host-visible ordering around the external
    # TensorBinding call. The binding API itself is synchronous, while these
    # explicit device barriers also order the preceding Warp fill and the
    # following Torch comparison.
    buffer.fill_(math.nan)
    warp.synchronize_device(buffer.device)
    binding.read(buffer)
    warp.synchronize_device(buffer.device)
    return warp.to_torch(buffer)


def _runtime_api(device: str) -> dict[str, Any]:
    """Load the verified kit-less API surface inside a worker process only."""

    if device != "cuda:0":
        raise AdapterCapabilityError(
            "ovphysx",
            "single_cuda_device",
            "Gate 0 OVPhysX workers require logical device 'cuda:0'",
        )
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise AdapterCapabilityError(
            "ovphysx",
            "headless_only",
            "DISPLAY and WAYLAND_DISPLAY must be unset",
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise AdapterCapabilityError(
            "ovphysx",
            "single_cuda_device",
            "CUDA_VISIBLE_DEVICES must contain exactly one physical GPU index",
        )

    module_names = {
        "torch": "torch",
        "warp": "warp",
        "sim": "isaaclab.sim",
        "actuators": "isaaclab.actuators",
        "assets_cfg": "isaaclab.assets",
        "version": "isaaclab.utils.version",
        "ov_root": "isaaclab_ovphysx",
        "ov_assets": "isaaclab_ovphysx.assets",
        "ov_physics": "isaaclab_ovphysx.physics",
    }
    modules: dict[str, Any] = {}
    for key, module_name in module_names.items():
        try:
            modules[key] = import_module(module_name)
        except Exception as error:
            raise AdapterCapabilityError(
                "ovphysx",
                "python_imports",
                f"failed to import {module_name}: {type(error).__name__}: {error}",
                hint="Use the pinned Python 3.12 kit-less OVPhysX environment.",
            ) from error

    if bool(modules["version"].has_kit()):
        raise AdapterCapabilityError(
            "ovphysx", "kitless_headless_only", "Isaac Lab reports a Kit runtime"
        )
    forbidden = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _FORBIDDEN_MODULE_PREFIXES
        )
    )
    if forbidden:
        raise AdapterCapabilityError(
            "ovphysx",
            "kitless_headless_only",
            "forbidden Kit/render modules were imported: " + ", ".join(forbidden[:10]),
        )
    if modules["torch"].cuda.device_count() != 1:
        raise AdapterCapabilityError(
            "ovphysx",
            "single_cuda_device",
            "the worker process must see exactly one logical CUDA device",
        )
    actual_version = _package_versions().get("ovphysx")
    if actual_version != _PINNED_OVPHYSX_WHEEL_VERSION:
        raise AdapterCapabilityError(
            "ovphysx",
            "ovphysx_wheel_version",
            f"ovphysx wheel must be {_PINNED_OVPHYSX_WHEEL_VERSION}, got {actual_version!r}",
        )
    return modules


def _load_import_capabilities() -> tuple[dict[str, Any], dict[str, bool]]:
    """Import only the documented kit-less modules and validate symbols."""

    modules: dict[str, Any] = {}
    for name in ("isaaclab", "isaaclab.sim", "isaaclab_ovphysx.physics"):
        try:
            modules[name] = import_module(name)
        except Exception as error:
            raise AdapterCapabilityError(
                "ovphysx",
                "python_imports",
                f"failed to import {name}: {type(error).__name__}: {error}",
                hint="Use the pinned Python 3.12 Isaac Lab kit-less OVPhysX environment.",
            ) from error

    sim = modules["isaaclab.sim"]
    ov_physics = modules["isaaclab_ovphysx.physics"]
    capabilities = {
        "isaaclab_import": True,
        "simulation_cfg_symbol": hasattr(sim, "SimulationCfg"),
        "build_simulation_context_symbol": hasattr(sim, "build_simulation_context"),
        "ovphysx_cfg_symbol": hasattr(ov_physics, "OvPhysxCfg"),
    }
    missing = [name for name, available in capabilities.items() if not available]
    if missing:
        raise AdapterCapabilityError(
            "ovphysx",
            "pinned_api_surface",
            "the installed packages lack required symbol(s): " + ", ".join(missing),
            hint="Confirm Isaac Lab v3.0.0-beta2.patch1 and its ov extra are installed together.",
        )

    forbidden = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_MODULE_PREFIXES)
    )
    if forbidden:
        raise AdapterCapabilityError(
            "ovphysx",
            "kitless_headless_only",
            "kit-less imports loaded forbidden module(s): " + ", ".join(forbidden[:10]),
        )
    capabilities["forbidden_modules_absent"] = True
    return modules, capabilities


def _import_provenance(modules: Mapping[str, Any]) -> dict[str, object]:
    """Describe, but do not invoke, the exact imported API surface."""

    description: dict[str, object] = {}
    symbol_names = {
        "isaaclab.sim": ("SimulationCfg", "build_simulation_context"),
        "isaaclab_ovphysx.physics": ("OvPhysxCfg",),
    }
    for module_name, module in modules.items():
        symbols: dict[str, object] = {}
        for symbol_name in symbol_names.get(module_name, ()):
            symbol = getattr(module, symbol_name, None)
            try:
                signature = str(inspect.signature(symbol)) if symbol is not None else None
            except (TypeError, ValueError):
                signature = None
            symbols[symbol_name] = {
                "present": symbol is not None,
                "module": getattr(symbol, "__module__", None),
                "signature": signature,
            }
        description[module_name] = {
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
            "symbols": symbols,
        }
    return description


def _inspect_usd(path: Path) -> dict[str, object]:
    """Open and fully compose an existing USD without starting simulation."""

    try:
        usd = import_module("pxr.Usd")
        usd_physics = import_module("pxr.UsdPhysics")
    except Exception as error:
        raise AdapterCapabilityError(
            "ovphysx",
            "openusd_schema_import",
            f"failed to import OpenUSD schemas: {type(error).__name__}: {error}",
        ) from error

    try:
        stage = usd.Stage.Open(str(path), load=usd.Stage.LoadAll)
    except Exception as error:
        raise AdapterCapabilityError(
            "ovphysx",
            "resolved_usd_open",
            f"OpenUSD could not open {path.name}: {type(error).__name__}: {error}",
        ) from error
    if stage is None:
        raise AdapterCapabilityError(
            "ovphysx",
            "resolved_usd_open",
            f"OpenUSD returned no stage for {path.name}",
        )

    prim_count = 0
    joint_paths: list[str] = []
    revolute_joint_names: list[str] = []
    fixed_joint_names: list[str] = []
    angular_drive_names: list[str] = []
    angular_drive_records: list[dict[str, object]] = []
    physx_velocity_joint_names: list[str] = []
    physx_velocity_records: list[dict[str, object]] = []
    physx_property_paths: list[str] = []
    distal_frame_names: list[str] = []
    articulation_roots: list[str] = []
    for prim in stage.Traverse():
        prim_count += 1
        prim_name = str(prim.GetName())
        prim_path = str(prim.GetPath())
        applied_schemas = {str(name) for name in prim.GetAppliedSchemas()}
        property_names = [str(prop.GetName()) for prop in prim.GetProperties()]
        if prim.IsA(usd_physics.Joint):
            joint_paths.append(prim_path)
        if prim.IsA(usd_physics.RevoluteJoint):
            revolute_joint_names.append(prim_name)
            if "PhysicsDriveAPI:angular" in applied_schemas:
                angular_drive_names.append(prim_name)
                drive = usd_physics.DriveAPI.Get(prim, "angular")
                angular_drive_records.append(
                    {
                        "joint_name": prim_name,
                        "type": str(drive.GetTypeAttr().Get()),
                        "stiffness": drive.GetStiffnessAttr().Get(),
                        "damping": drive.GetDampingAttr().Get(),
                        "max_force": drive.GetMaxForceAttr().Get(),
                        "target_position": drive.GetTargetPositionAttr().Get(),
                    }
                )
            if "physxJoint:maxJointVelocity" in property_names:
                physx_velocity_joint_names.append(prim_name)
                physx_velocity_records.append(
                    {
                        "joint_name": prim_name,
                        "max_joint_velocity": prim.GetAttribute(
                            "physxJoint:maxJointVelocity"
                        ).Get(),
                    }
                )
        if prim.IsA(usd_physics.FixedJoint):
            fixed_joint_names.append(prim_name)
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            articulation_roots.append(prim_path)
        if prim_name.endswith("_DP"):
            distal_frame_names.append(prim_name)
        physx_property_paths.extend(
            f"{prim_path}.{name}" for name in property_names if name.lower().startswith("physx")
        )
    default_prim = stage.GetDefaultPrim()
    return {
        "source_path": str(path),
        "source_sha256": sha256(path.read_bytes()).hexdigest(),
        "stage_load_policy": "load_all",
        "prim_count": prim_count,
        "joint_count": len(joint_paths),
        "joint_paths": joint_paths,
        "revolute_joint_count": len(revolute_joint_names),
        "revolute_joint_names": sorted(revolute_joint_names),
        "fixed_joint_count": len(fixed_joint_names),
        "fixed_joint_names": sorted(fixed_joint_names),
        "angular_drive_count": len(angular_drive_names),
        "angular_drive_joint_names": sorted(angular_drive_names),
        "angular_drive_records": sorted(
            angular_drive_records, key=lambda record: str(record["joint_name"])
        ),
        "physx_velocity_joint_count": len(physx_velocity_joint_names),
        "physx_velocity_joint_names": sorted(physx_velocity_joint_names),
        "physx_velocity_records": sorted(
            physx_velocity_records, key=lambda record: str(record["joint_name"])
        ),
        "physx_authored_property_count": len(physx_property_paths),
        "physx_authored_property_paths": sorted(physx_property_paths),
        "distal_frame_count": len(distal_frame_names),
        "distal_frame_names": sorted(distal_frame_names),
        "articulation_root_paths": articulation_roots,
        "default_prim_path": str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else None,
    }


def _standard_wave_schema_capabilities(inspection: Mapping[str, object]) -> dict[str, bool]:
    """Evaluate the fixed-base Wave Gate 0 schema invariants."""

    drive_records = inspection.get("angular_drive_records", ())
    velocity_records = inspection.get("physx_velocity_records", ())
    finite_drive_values = isinstance(drive_records, list) and len(drive_records) == 22
    positive_stiffness = finite_drive_values
    if finite_drive_values:
        for record in drive_records:
            if not isinstance(record, Mapping):
                finite_drive_values = positive_stiffness = False
                break
            values = tuple(
                record.get(name)
                for name in ("stiffness", "damping", "max_force", "target_position")
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                finite_drive_values = positive_stiffness = False
                break
            if float(record["stiffness"]) <= 0.0:
                positive_stiffness = False
    finite_velocity_values = isinstance(velocity_records, list) and len(velocity_records) == 22
    if finite_velocity_values:
        finite_velocity_values = all(
            isinstance(record, Mapping)
            and not isinstance(record.get("max_joint_velocity"), bool)
            and isinstance(record.get("max_joint_velocity"), (int, float))
            and math.isfinite(float(record["max_joint_velocity"]))
            and float(record["max_joint_velocity"]) > 0.0
            for record in velocity_records
        )

    return {
        "stage_fully_composed": inspection.get("stage_load_policy") == "load_all"
        and int(inspection.get("prim_count", 0)) > 0,
        "articulation_root_schema": len(inspection.get("articulation_root_paths", ())) == 1,
        "canonical_joint_count": int(inspection.get("revolute_joint_count", 0)) == 22,
        "position_drive_schema": int(inspection.get("angular_drive_count", 0)) == 22,
        "position_drive_values_finite": finite_drive_values,
        "position_drive_stiffness_positive": positive_stiffness,
        "physx_properties_authored": int(inspection.get("physx_velocity_joint_count", 0))
        == 22,
        "physx_max_joint_velocity_values_finite_positive": finite_velocity_values,
        "canonical_distal_frame_count": int(inspection.get("distal_frame_count", 0)) == 5,
    }


def probe_ovphysx(
    *,
    import_only: bool = True,
    manifest: object | None = None,
    asset_root: str | Path | None = None,
    resolved_usd: str | Path | None = None,
    hand: str | None = None,
    device: str = "cuda:0",
) -> AdapterProbeResult:
    """Probe the pinned kit-less environment without claiming runtime parity.

    Import-only success means only that the expected packages and symbols are
    present and no forbidden Kit/renderer module was loaded.  Resolved-USD
    mode additionally performs a non-simulating OpenUSD stage/schema check.
    This probe never claims that physics stepped. Actual stepping belongs to
    the process-isolated runtime probe and produces separate evidence.
    """

    mode = "import_only" if import_only else "resolved_usd_headless"
    capabilities: dict[str, bool] = {
        "kitless_contract": True,
        "renderer_disabled": True,
        "camera_disabled": True,
        "pinned_ovphysx_wheel_version": False,
        "resolved_usd_open": False,
        "stage_fully_composed": False,
        "articulation_root_schema": False,
        "canonical_joint_count": False,
        "position_drive_schema": False,
        "position_drive_values_finite": False,
        "position_drive_stiffness_positive": False,
        "physx_properties_authored": False,
        "physx_max_joint_velocity_values_finite_positive": False,
        "canonical_distal_frame_count": False,
        "physics_step": False,
    }
    provenance: dict[str, object] = {}
    try:
        normalized_device = _validate_device(device)
        if manifest is not None:
            _reject_forbidden_features(manifest)
        provenance = _base_provenance(normalized_device)
        modules, imported = _load_import_capabilities()
        capabilities.update(imported)
        provenance["imported_modules"] = _import_provenance(modules)
        package_versions = provenance["packages"]
        assert isinstance(package_versions, Mapping)
        actual_ovphysx_version = package_versions.get("ovphysx")
        if actual_ovphysx_version != _PINNED_OVPHYSX_WHEEL_VERSION:
            raise AdapterCapabilityError(
                "ovphysx",
                "ovphysx_wheel_version",
                (
                    "ovphysx wheel must be "
                    f"{_PINNED_OVPHYSX_WHEEL_VERSION}, got {actual_ovphysx_version!r}"
                ),
                hint="Install the wheel pinned by Isaac Lab v3.0.0-beta2.patch1.",
            )
        capabilities["pinned_ovphysx_wheel_version"] = True
        if import_only:
            return AdapterProbeResult(
                backend="ovphysx",
                mode=mode,
                status="available",
                message=(
                    "Pinned kit-less imports and required symbols are available; "
                    "USD loading and physics stepping were not exercised."
                ),
                capabilities=capabilities,
                provenance=provenance,
            )

        root = Path(asset_root).expanduser().resolve() if asset_root is not None else None
        if root is not None and not root.is_dir():
            raise FileNotFoundError(f"asset root does not exist: {root}")
        usd_path = _resolve_usd_path(
            manifest,
            asset_root=root,
            resolved_usd=resolved_usd,
            hand=hand,
        )
        if not usd_path.is_file():
            raise FileNotFoundError(f"resolved USD does not exist: {usd_path}")
        if usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError(f"resolved USD must use .usd, .usda or .usdc: {usd_path.name}")
        if root is not None:
            try:
                usd_path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"resolved USD escapes the approved asset root: {usd_path}") from error

        inspection = _inspect_usd(usd_path)
        provenance["usd_inspection"] = inspection
        capabilities["resolved_usd_open"] = True
        schema_capabilities = _standard_wave_schema_capabilities(inspection)
        capabilities.update(schema_capabilities)
        missing_schema = sorted(
            name for name, available in schema_capabilities.items() if not available
        )
        if missing_schema:
            raise AdapterCapabilityError(
                "ovphysx",
                "resolved_usd_schema",
                "the fully composed Wave USD failed invariant(s): "
                + ", ".join(missing_schema),
            )
        return AdapterProbeResult(
            backend="ovphysx",
            mode=mode,
            status="available",
            message=(
                "The fully composed Wave USD passed the non-simulating schema probe. "
                "Physics stepping is deliberately handled by the process-isolated runtime probe."
            ),
            capabilities=capabilities,
            provenance=provenance,
        )
    except AdapterCapabilityError as error:
        return AdapterProbeResult(
            backend="ovphysx",
            mode=mode,
            status="capability_error",
            message=str(error),
            capabilities=capabilities,
            provenance=provenance,
            error=error.to_dict(),
        )
    except Exception as error:
        return AdapterProbeResult(
            backend="ovphysx",
            mode=mode,
            status="error",
            message=f"{type(error).__name__}: {error}",
            capabilities=capabilities,
            provenance=provenance,
            error={"type": type(error).__name__, "message": str(error)},
        )


class OVPhysXAdapter:
    """One-shot kit-less OVPhysX adapter for use inside an isolated worker.

    Importing this module remains CPU-safe. Heavy packages and the global
    OVPhysX device state are loaded only by :meth:`open`, which the parity
    runner invokes in a fresh subprocess for every canonical case.
    """

    backend_name = "ovphysx"

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        asset_root: str | Path | None = None,
        legacy_joint_friction_intervention: Mapping[str, object] | None = None,
    ) -> None:
        self.device = _validate_device(device)
        self.asset_root = (
            Path(asset_root).expanduser().resolve() if asset_root is not None else None
        )
        self._legacy_joint_friction_intervention = (
            _copy_legacy_joint_friction_intervention(
                legacy_joint_friction_intervention
            )
        )
        self._validated_legacy_joint_friction_intervention: (
            dict[str, object] | None
        ) = None
        self._legacy_joint_friction_intervention_evidence: dict[str, object] = {}
        self._legacy_joint_friction_override_layer: Any = None
        self._legacy_joint_friction_session_layer: Any = None
        self._legacy_joint_friction_original_session_sublayers: tuple[str, ...] = ()
        self._legacy_joint_friction_override_records: dict[
            str, dict[str, object]
        ] = {}
        self._legacy_joint_friction_source_path: Path | None = None
        self._lifecycle = AdapterLifecycle.CREATED
        self._last_probe: AdapterProbeResult | None = None
        self._runtime: dict[str, Any] = {}
        self._context_manager: Any = None
        self._simulation: Any = None
        self._articulation: Any = None
        self._source_path: Path | None = None
        self._dt = 0.0
        self._step_index = 0
        self._side = ""
        self._joint_names: tuple[str, ...] = ()
        self._frame_names: tuple[str, ...] = ()
        self._backend_joint_names: tuple[str, ...] = ()
        self._backend_frame_names: tuple[str, ...] = ()
        self._joint_mapping: tuple[dict[str, object], ...] = ()
        self._frame_mapping: tuple[dict[str, object], ...] = ()
        self._position_targets: dict[str, float] = {}
        self._target_tensor: Any = None
        self._target_readback_count = 0
        self._target_readback_max_abs_error = 0.0
        self._target_readback_values: tuple[float, ...] = ()
        self._canonical_target_readback_max_abs_error = 0.0
        self._target_nonzero_readback_observed = False
        self._scheduled_target_sequence_digest: (
            CanonicalTargetSequenceDigest | None
        ) = None
        self._requested_target_sequence_digest: (
            CanonicalTargetSequenceDigest | None
        ) = None
        self._immediate_target_readback_sequence_digest: (
            CanonicalTargetSequenceDigest | None
        ) = None
        self._pre_step_applied_target_readback_sequence_digest: (
            CanonicalTargetSequenceDigest | None
        ) = None
        self._zero_velocity_target_verified = False
        self._zero_feedforward_effort_target_verified = False
        self._controller: Any = None
        self._controller_stiffness: tuple[float, ...] = ()
        self._controller_damping: tuple[float, ...] = ()
        self._controller_effort_limit: tuple[float, ...] = ()
        self._controller_effort_limit_sim: tuple[float, ...] = ()
        self._controller_armature: tuple[float, ...] = ()
        self._controller_static_friction: tuple[float, ...] = ()
        self._controller_dynamic_friction: tuple[float, ...] = ()
        self._controller_viscous_friction: tuple[float, ...] = ()
        self._backend_drive_stiffness: tuple[float, ...] = ()
        self._backend_drive_damping: tuple[float, ...] = ()
        self._backend_armature: tuple[float, ...] = ()
        self._backend_friction_properties: tuple[tuple[float, ...], ...] = ()
        self._tensor_bindings: dict[str, Any] = {}
        self._backend_joint_dynamics: dict[str, object] = {}
        self._physics_solver_contract: dict[str, object] = {}
        self._effective_parameter_snapshot: dict[str, object] = {}
        self._computed_effort_peak_abs: tuple[float, ...] = ()
        self._applied_effort_peak_abs: tuple[float, ...] = ()
        self._effort_observation_count = 0
        self._effort_formula_max_abs_error = 0.0
        self._effort_clip_max_abs_error = 0.0
        self._effort_clip_count = 0
        self._simulation_contract: dict[str, object] = {}
        self._simulation_contract_v2: dict[str, object] = {}
        self._tensor_binding_shapes: dict[str, list[int] | None] = {}
        self._open_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def lifecycle(self) -> AdapterLifecycle:
        return self._lifecycle

    @property
    def last_probe(self) -> AdapterProbeResult | None:
        return self._last_probe

    def probe(
        self,
        *,
        import_only: bool = True,
        manifest: object | None = None,
        asset_root: str | Path | None = None,
        resolved_usd: str | Path | None = None,
        hand: str | None = None,
    ) -> AdapterProbeResult:
        self._last_probe = probe_ovphysx(
            import_only=import_only,
            manifest=manifest,
            asset_root=asset_root,
            resolved_usd=resolved_usd,
            hand=hand,
            device=self.device,
        )
        return self._last_probe

    def _apply_legacy_joint_friction_intervention(
        self,
        stage: Any,
        *,
        articulation: Any,
        joint_names: tuple[str, ...],
        source: Path,
    ) -> None:
        """Author a fail-closed sham/zero opinion in an anonymous overlay.

        The source asset and the stage root layer are never edit targets.  All
        22 properties are validated before the first opinion is authored so a
        schema/path drift cannot yield a partially accepted intervention.
        """

        intervention = self._validated_legacy_joint_friction_intervention
        if intervention is None:
            return
        if getattr(articulation, "is_initialized", None) is not False:
            raise RuntimeError(
                "Wave articulation must be uninitialized before the legacy "
                "joint-friction intervention is written"
            )
        sdf = import_module("pxr.Sdf")
        usd = import_module("pxr.Usd")
        usd_physics = import_module("pxr.UsdPhysics")
        expected = intervention["expected_pre_values"]
        write = intervention["write_values"]
        assert isinstance(expected, Mapping) and isinstance(write, Mapping)

        source_sha256 = sha256(source.read_bytes()).hexdigest()
        robot_prefix = "/World/Env_0/Robot"
        target_prefix = robot_prefix + "/joints/"
        discovered: dict[str, str] = {}
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(target_prefix):
                continue
            if not prim.IsA(usd_physics.RevoluteJoint):
                continue
            name = str(prim.GetName())
            if name not in joint_names:
                continue
            if name in discovered:
                raise RuntimeError(
                    f"duplicate composed USD intervention joint name: {name}"
                )
            discovered[name] = prim_path
        if set(discovered) != set(joint_names):
            missing = sorted(set(joint_names) - set(discovered))
            extra = sorted(set(discovered) - set(joint_names))
            raise RuntimeError(
                "legacy joint-friction intervention traversal coverage differs "
                f"from the manifest: missing={missing}, extra={extra}"
            )

        records: dict[str, dict[str, object]] = {}
        evidence_records: list[dict[str, object]] = []
        for name in joint_names:
            prim_path = target_prefix + name
            if discovered[name] != prim_path:
                raise RuntimeError(
                    f"legacy joint-friction intervention path drift for {name!r}"
                )
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"intervention joint prim is invalid: {prim_path}")
            if str(prim.GetPath()) != prim_path or str(prim.GetName()) != name:
                raise RuntimeError(f"intervention joint identity drift: {prim_path}")
            if not prim.IsA(usd_physics.RevoluteJoint):
                raise RuntimeError(f"intervention joint is not revolute: {prim_path}")
            attribute = prim.GetAttribute("physxJoint:jointFriction")
            if attribute is None or not attribute.IsValid():
                raise RuntimeError(
                    f"legacy joint-friction attribute is unavailable: {prim_path}"
                )
            if attribute.GetTypeName() != sdf.ValueTypeNames.Float:
                raise RuntimeError(
                    f"legacy joint-friction attribute is not float: {prim_path}"
                )
            if attribute.HasAuthoredValueOpinion() is not True:
                raise RuntimeError(
                    f"legacy joint-friction has no authored opinion: {prim_path}"
                )
            observed_pre, observed_pre_hex = _float32_value_and_hex(
                attribute.Get(), name=f"composed pre-intervention {name}"
            )
            expected_pre, expected_pre_hex = _float32_value_and_hex(
                expected[name], name=f"expected pre-intervention {name}"
            )
            write_value, write_hex = _float32_value_and_hex(
                write[name], name=f"write intervention {name}"
            )
            if observed_pre_hex != expected_pre_hex:
                raise RuntimeError(
                    f"legacy joint-friction pre-value drift for {name!r}"
                )
            property_path = f"{prim_path}.physxJoint:jointFriction"
            records[name] = {
                "attribute": attribute,
                "prim_path": prim_path,
                "property_path": property_path,
                "observed_pre_value": observed_pre,
                "observed_pre_float32_hex": observed_pre_hex,
                "expected_pre_value": expected_pre,
                "expected_pre_float32_hex": expected_pre_hex,
                "write_value": write_value,
                "write_float32_hex": write_hex,
            }
            evidence_records.append(
                {
                    "canonical_id": name,
                    "prim_path": prim_path,
                    "property_path": property_path,
                    "observed_pre_value": observed_pre,
                    "observed_pre_float32_hex": observed_pre_hex,
                    "expected_pre_value": expected_pre,
                    "expected_pre_float32_hex": expected_pre_hex,
                    "write_value": write_value,
                    "write_float32_hex": write_hex,
                    "post_write_float32_hex": None,
                    "post_reset_float32_hex": None,
                    "post_cleanup_float32_hex": None,
                }
            )

        session_layer = stage.GetSessionLayer()
        if session_layer is None or not bool(getattr(session_layer, "anonymous", False)):
            raise RuntimeError("OVPhysX stage session layer must be anonymous")
        original_edit_layer = stage.GetEditTarget().GetLayer()
        if original_edit_layer is None:
            raise RuntimeError("OVPhysX stage has no current edit layer")
        if not bool(getattr(original_edit_layer, "anonymous", False)):
            raise RuntimeError(
                "OVPhysX stage edit target must be anonymous before Freeze B"
            )
        original_sublayers = tuple(str(value) for value in session_layer.subLayerPaths)
        overlay = sdf.Layer.CreateAnonymous(
            "freeze_b_legacy_joint_friction_override.usda"
        )
        if not bool(getattr(overlay, "anonymous", False)):
            raise RuntimeError("legacy joint-friction override layer is not anonymous")

        self._legacy_joint_friction_source_path = source
        self._legacy_joint_friction_override_records = records
        self._legacy_joint_friction_original_session_sublayers = original_sublayers
        self._legacy_joint_friction_session_layer = session_layer
        self._legacy_joint_friction_override_layer = overlay
        self._legacy_joint_friction_intervention_evidence = {
            "schema_version": _LEGACY_JOINT_FRICTION_INTERVENTION_SCHEMA_VERSION,
            "role": intervention["role"],
            "private_plan_sha256": intervention["private_plan_sha256"],
            "attribute": "physxJoint:jointFriction",
            "edit_strategy": "anonymous_overlay_as_strongest_session_sublayer",
            "source_asset_sha256_before": source_sha256,
            "source_asset_sha256_after_cleanup": None,
            "source_asset_sha256_unchanged": None,
            "joint_count": len(joint_names),
            "joint_order": list(joint_names),
            "exact_manifest_coverage_verified": True,
            "all_properties_prevalidated_before_write": True,
            "session_layer_anonymous": True,
            "original_edit_layer_anonymous": True,
            "override_layer_anonymous": True,
            "source_asset_was_edit_target": False,
            "articulation_uninitialized_before_write": True,
            "articulation_initialized_after_reset": False,
            "applied_before_first_simulation_reset": False,
            "post_write_verified": False,
            "post_reset_verified": False,
            "cleanup_status": "pending",
            "records": evidence_records,
        }

        try:
            session_layer.subLayerPaths.insert(0, overlay.identifier)
            if tuple(str(value) for value in session_layer.subLayerPaths) != (
                overlay.identifier,
                *original_sublayers,
            ):
                raise RuntimeError("failed to install the intervention overlay exactly once")
            with usd.EditContext(stage, usd.EditTarget(overlay)):
                for name in joint_names:
                    record = records[name]
                    if record["attribute"].Set(record["write_value"]) is not True:
                        raise RuntimeError(
                            f"failed to author legacy joint-friction override for {name!r}"
                        )
            if stage.GetEditTarget().GetLayer() != original_edit_layer:
                raise RuntimeError("USD edit target was not restored after intervention")
            self._verify_legacy_joint_friction_intervention(
                stage, phase="post_write"
            )
            self._legacy_joint_friction_intervention_evidence[
                "applied_before_first_simulation_reset"
            ] = True
        except Exception as apply_error:
            try:
                self._cleanup_legacy_joint_friction_intervention(stage)
            except Exception as cleanup_error:
                raise RuntimeError(
                    "legacy joint-friction intervention failed and rollback also failed: "
                    f"apply={type(apply_error).__name__}: {apply_error}; "
                    f"rollback={type(cleanup_error).__name__}: {cleanup_error}"
                ) from apply_error
            raise

    def _verify_legacy_joint_friction_intervention(
        self, stage: Any, *, phase: str, articulation: Any | None = None
    ) -> None:
        if phase not in {"post_write", "post_reset"}:
            raise ValueError("intervention verification phase is invalid")
        if phase == "post_reset" and (
            articulation is None
            or getattr(articulation, "is_initialized", None) is not True
        ):
            raise RuntimeError(
                "Wave articulation must be initialized for post-reset "
                "intervention verification"
            )
        overlay = self._legacy_joint_friction_override_layer
        session_layer = self._legacy_joint_friction_session_layer
        if overlay is None or session_layer is None:
            raise RuntimeError("legacy joint-friction intervention overlay is unavailable")
        expected_sublayers = (
            overlay.identifier,
            *self._legacy_joint_friction_original_session_sublayers,
        )
        if tuple(str(value) for value in session_layer.subLayerPaths) != expected_sublayers:
            raise RuntimeError("legacy joint-friction session sublayer stack drifted")
        if overlay not in stage.GetLayerStack():
            raise RuntimeError("legacy joint-friction overlay left the stage layer stack")
        sdf = import_module("pxr.Sdf")
        evidence_records = {
            str(record["canonical_id"]): record
            for record in self._legacy_joint_friction_intervention_evidence["records"]
        }
        for name, record in self._legacy_joint_friction_override_records.items():
            prim = stage.GetPrimAtPath(str(record["prim_path"]))
            if not prim or not prim.IsValid():
                raise RuntimeError(f"intervention joint disappeared after {phase}: {name}")
            attribute = prim.GetAttribute("physxJoint:jointFriction")
            if attribute is None or not attribute.IsValid():
                raise RuntimeError(
                    f"intervention attribute disappeared after {phase}: {name}"
                )
            observed, observed_hex = _float32_value_and_hex(
                attribute.Get(), name=f"{phase} legacy joint-friction {name}"
            )
            if observed_hex != record["write_float32_hex"]:
                raise RuntimeError(
                    f"legacy joint-friction override drifted after {phase}: {name}"
                )
            property_spec = overlay.GetPropertyAtPath(
                sdf.Path(str(record["property_path"]))
            )
            if property_spec is None or property_spec.layer != overlay:
                raise RuntimeError(
                    f"legacy joint-friction overlay spec is unavailable after {phase}: {name}"
                )
            _, spec_hex = _float32_value_and_hex(
                property_spec.default,
                name=f"{phase} override property spec {name}",
            )
            property_stack = attribute.GetPropertyStack()
            if (
                spec_hex != record["write_float32_hex"]
                or not property_stack
                or property_stack[0].layer != overlay
            ):
                raise RuntimeError(
                    f"legacy joint-friction override is not strongest after {phase}: {name}"
                )
            evidence_records[name][f"{phase}_float32_hex"] = observed_hex
            evidence_records[name][f"{phase}_value"] = observed
        self._legacy_joint_friction_intervention_evidence[
            f"{phase}_verified"
        ] = True
        if phase == "post_reset":
            self._legacy_joint_friction_intervention_evidence[
                "articulation_initialized_after_reset"
            ] = True

    def _cleanup_legacy_joint_friction_intervention(self, stage: Any) -> None:
        """Remove only the dedicated overlay and prove composed-value restoration."""

        overlay = self._legacy_joint_friction_override_layer
        session_layer = self._legacy_joint_friction_session_layer
        if overlay is None or session_layer is None:
            return
        expected_sublayers = (
            overlay.identifier,
            *self._legacy_joint_friction_original_session_sublayers,
        )
        stack_drifted = (
            tuple(str(value) for value in session_layer.subLayerPaths)
            != expected_sublayers
        )
        session_layer.subLayerPaths = list(
            self._legacy_joint_friction_original_session_sublayers
        )
        evidence_records = {
            str(record["canonical_id"]): record
            for record in self._legacy_joint_friction_intervention_evidence.get(
                "records", []
            )
        }
        restoration_errors: list[str] = []
        for name, record in self._legacy_joint_friction_override_records.items():
            prim = stage.GetPrimAtPath(str(record["prim_path"]))
            if not prim or not prim.IsValid():
                restoration_errors.append(f"{name}:missing_prim")
                continue
            attribute = prim.GetAttribute("physxJoint:jointFriction")
            if attribute is None or not attribute.IsValid():
                restoration_errors.append(f"{name}:missing_attribute")
                continue
            _, restored_hex = _float32_value_and_hex(
                attribute.Get(), name=f"post-cleanup legacy joint-friction {name}"
            )
            if restored_hex != record["observed_pre_float32_hex"]:
                restoration_errors.append(f"{name}:value_mismatch")
            if name in evidence_records:
                evidence_records[name]["post_cleanup_float32_hex"] = restored_hex
        if overlay in stage.GetLayerStack():
            restoration_errors.append("overlay_still_in_layer_stack")
        overlay.Clear()
        self._legacy_joint_friction_override_layer = None
        self._legacy_joint_friction_session_layer = None
        self._legacy_joint_friction_original_session_sublayers = ()
        self._legacy_joint_friction_override_records = {}
        if stack_drifted:
            restoration_errors.append("session_sublayer_stack_drift")
        if restoration_errors:
            self._legacy_joint_friction_intervention_evidence[
                "cleanup_status"
            ] = "error"
            raise RuntimeError(
                "legacy joint-friction intervention cleanup failed: "
                + ", ".join(restoration_errors)
            )
        self._legacy_joint_friction_intervention_evidence[
            "cleanup_status"
        ] = "restored_pre_values"

    def open(self, manifest: object, *, dt_override: float | None = None) -> None:
        if self._lifecycle not in {AdapterLifecycle.CREATED, AdapterLifecycle.CLOSED}:
            raise AdapterLifecycleError(
                f"cannot open OVPhysX adapter while it is {self._lifecycle.value}"
            )
        dt = 0.002 if dt_override is None else float(dt_override)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_override must be a positive finite number")
        if self.asset_root is None or not self.asset_root.is_dir():
            raise ValueError("OVPhysX asset_root must be an existing directory")
        self._validated_legacy_joint_friction_intervention = None
        self._legacy_joint_friction_intervention_evidence = {}
        self._legacy_joint_friction_override_layer = None
        self._legacy_joint_friction_session_layer = None
        self._legacy_joint_friction_original_session_sublayers = ()
        self._legacy_joint_friction_override_records = {}
        self._legacy_joint_friction_source_path = None

        mounting = str(_plain_value(_field(manifest, "mounting", default="fixed_base")))
        if mounting != "fixed_base":
            raise AdapterCapabilityError(
                self.backend_name, "fixed_base_only", "Gate 0 excludes floating-base models"
            )
        control_mode = str(
            _plain_value(_field(manifest, "control_mode", default="position"))
        )
        if control_mode != "position":
            raise AdapterCapabilityError(
                self.backend_name,
                "position_control_only",
                f"Gate 0 excludes control mode {control_mode!r}",
            )
        _reject_forbidden_features(manifest)
        source = _resolve_usd_path(
            manifest,
            asset_root=self.asset_root,
            resolved_usd=None,
        )
        if not source.is_file():
            raise FileNotFoundError(f"resolved USD does not exist: {source}")
        try:
            source.relative_to(self.asset_root)
        except ValueError as error:
            raise ValueError(f"resolved USD escapes the approved asset root: {source}") from error

        joint_names = _names(
            _field(manifest, "joint_names", "expected_joint_names", default=None),
            field_name="joint_names",
        )
        frame_names = _names(
            _field(
                manifest,
                "distal_frame_names",
                "expected_frame_names",
                default=None,
            ),
            field_name="distal_frame_names",
        )
        if len(joint_names) != 22 or len(frame_names) != 5:
            raise ValueError("Gate 0 requires exactly 22 joints and five distal frames")
        self._validated_legacy_joint_friction_intervention = (
            _validate_legacy_joint_friction_intervention_for_joints(
                self._legacy_joint_friction_intervention, joint_names
            )
        )

        try:
            runtime = _runtime_api(self.device)
            torch = runtime["torch"]
            sim_utils = runtime["sim"]
            simulation_cfg = sim_utils.SimulationCfg(
                physics=runtime["ov_physics"].OvPhysxCfg(),
                device=self.device,
                dt=dt,
                gravity=self._open_gravity,
                create_stage_in_memory=True,
                render_interval=1000,
            )
            context_manager = sim_utils.build_simulation_context(
                create_new_stage=True,
                device=self.device,
                sim_cfg=simulation_cfg,
                add_ground_plane=False,
                add_lighting=False,
                auto_add_lighting=False,
            )
            self._runtime = runtime
            self._context_manager = context_manager
            simulation = context_manager.__enter__()
            self._simulation = simulation
            if hasattr(simulation, "_app_control_on_stop_handle"):
                simulation._app_control_on_stop_handle = None

            sim_utils.create_prim("/World/Env_0", "Xform")
            articulation_cfg = runtime["assets_cfg"].ArticulationCfg(
                prim_path="/World/Env_.*/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=str(source)),
                actuators={
                    "all": runtime["actuators"].IdealPDActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=None,
                        damping=None,
                    )
                },
            )
            articulation = runtime["ov_assets"].Articulation(articulation_cfg)
            self._articulation = articulation
            self._apply_legacy_joint_friction_intervention(
                simulation.stage,
                articulation=articulation,
                joint_names=joint_names,
                source=source,
            )
            simulation.reset()
            if getattr(articulation, "is_initialized", None) is not True:
                raise RuntimeError("Wave articulation did not initialize")
            if self._validated_legacy_joint_friction_intervention is not None:
                self._verify_legacy_joint_friction_intervention(
                    simulation.stage,
                    phase="post_reset",
                    articulation=articulation,
                )
            if not articulation.is_fixed_base:
                raise AdapterCapabilityError(
                    self.backend_name,
                    "fixed_base_only",
                    "Wave articulation is not fixed-base",
                )

            backend_joint_names = tuple(str(name) for name in articulation.joint_names)
            backend_frame_names = tuple(str(name) for name in articulation.body_names)
            missing_joints = sorted(set(joint_names) - set(backend_joint_names))
            unexpected_joints = sorted(set(backend_joint_names) - set(joint_names))
            if missing_joints or unexpected_joints:
                raise RuntimeError(
                    "OVPhysX joint coverage differs from the manifest: "
                    f"missing={missing_joints}, unexpected={unexpected_joints}"
                )
            missing_frames = sorted(set(frame_names) - set(backend_frame_names))
            if missing_frames:
                raise RuntimeError("OVPhysX is missing distal frames: " + ", ".join(missing_frames))

            tensor_types = runtime["ov_root"].tensor_types
            binding_shapes: dict[str, list[int] | None] = {}
            tensor_bindings: dict[str, Any] = {}
            expected_binding_shapes = {
                "DOF_POSITION": (1, 22),
                "DOF_VELOCITY": (1, 22),
                "DOF_STIFFNESS": (1, 22),
                "DOF_DAMPING": (1, 22),
                "DOF_ARMATURE": (1, 22),
                "DOF_FRICTION_PROPERTIES": (1, 22, 3),
                "DOF_POSITION_TARGET": (1, 22),
            }
            for name, expected_shape in expected_binding_shapes.items():
                binding = articulation.root_view.get(getattr(tensor_types, name))
                tensor_bindings[name] = binding
                binding_shapes[name] = list(binding.shape) if binding is not None else None
                if binding is None or tuple(int(value) for value in binding.shape) != expected_shape:
                    raise RuntimeError(
                        f"unexpected {name} binding shape; expected {expected_shape}"
                    )

            self._simulation_contract = self._verify_simulation_configuration(
                simulation, requested_dt=dt, requested_gravity=self._open_gravity
            )
            self._simulation_contract_v2 = (
                self._simulation_configuration_v2_from_legacy(
                    self._simulation_contract
                )
            )
            controller = articulation.actuators.get("all")
            if controller is None or type(controller).__name__ != "IdealPDActuator":
                raise RuntimeError("OVPhysX did not initialize the requested IdealPDActuator")
            if controller.is_implicit_model is not False:
                raise RuntimeError("OVPhysX IdealPD controller is not an explicit actuator")
            if tuple(str(name) for name in controller.joint_names) != backend_joint_names:
                raise RuntimeError("IdealPD actuator joint order differs from the articulation")
            controller_joint_ids = controller.joint_indices
            if isinstance(controller_joint_ids, slice) and controller_joint_ids == slice(
                None
            ):
                resolved_controller_joint_ids = tuple(range(22))
            elif hasattr(controller_joint_ids, "detach"):
                resolved_controller_joint_ids = tuple(
                    int(value)
                    for value in controller_joint_ids.detach().cpu().tolist()
                )
            else:
                resolved_controller_joint_ids = tuple(int(value) for value in controller_joint_ids)
            if resolved_controller_joint_ids != tuple(range(22)):
                raise RuntimeError("IdealPD actuator does not cover all joints in backend order")

            def controller_values(tensor: Any, *, name: str) -> tuple[float, ...]:
                shape = tuple(int(value) for value in tensor.shape)
                if shape != (1, 22):
                    raise RuntimeError(
                        f"unexpected IdealPD {name} tensor shape: {shape}, expected (1, 22)"
                    )
                values = tuple(float(value) for value in tensor[0].detach().cpu().tolist())
                if len(values) != 22 or not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"IdealPD {name} tensor is invalid")
                return values

            self._controller = controller
            self._controller_stiffness = controller_values(
                controller.stiffness, name="stiffness"
            )
            self._controller_damping = controller_values(
                controller.damping, name="damping"
            )
            self._controller_effort_limit = controller_values(
                controller.effort_limit, name="effort_limit"
            )
            self._controller_effort_limit_sim = controller_values(
                controller.effort_limit_sim, name="effort_limit_sim"
            )
            self._controller_armature = controller_values(
                controller.armature, name="armature"
            )
            self._controller_static_friction = controller_values(
                controller.friction, name="friction"
            )
            self._controller_dynamic_friction = controller_values(
                controller.dynamic_friction, name="dynamic_friction"
            )
            self._controller_viscous_friction = controller_values(
                controller.viscous_friction, name="viscous_friction"
            )
            if (
                not all(value > 0.0 for value in self._controller_stiffness)
                or not all(value >= 0.0 for value in self._controller_damping)
                or not all(value > 0.0 for value in self._controller_effort_limit)
                or not all(value > 0.0 for value in self._controller_effort_limit_sim)
                or not all(value >= 0.0 for value in self._controller_armature)
                or not all(
                    value >= 0.0 for value in self._controller_static_friction
                )
                or not all(
                    value >= 0.0 for value in self._controller_dynamic_friction
                )
                or not all(
                    value >= 0.0 for value in self._controller_viscous_friction
                )
            ):
                raise RuntimeError("IdealPD controller parameters are invalid")

            side = _side(manifest)
            self._side = side
            self._source_path = source
            self._dt = dt
            self._joint_names = joint_names
            self._frame_names = frame_names
            self._backend_joint_names = backend_joint_names
            self._backend_frame_names = backend_frame_names
            self._joint_mapping = tuple(
                {
                    "canonical_id": name,
                    "backend": self.backend_name,
                    "scope": side,
                    "backend_name": name,
                    "index": backend_joint_names.index(name),
                    "sign": 1.0,
                    "offset": 0.0,
                    "unit": "rad",
                }
                for name in joint_names
            )
            self._frame_mapping = tuple(
                {
                    "canonical_id": name,
                    "backend": self.backend_name,
                    "scope": side,
                    "backend_name": name,
                    "index": backend_frame_names.index(name),
                    "sign": 1.0,
                    "offset": 0.0,
                    "unit": "xyz_m_qwxyz",
                }
                for name in frame_names
            )
            self._tensor_binding_shapes = binding_shapes
            self._tensor_bindings = tensor_bindings
            self._lifecycle = AdapterLifecycle.READY
            self.reset()
            self.effective_parameter_readback()
            assert bool(torch.isfinite(articulation.data.joint_pos.torch).all().item())
        except Exception as open_error:
            self._lifecycle = AdapterLifecycle.FAILED
            try:
                self._release_runtime()
            except Exception as cleanup_error:
                raise RuntimeError(
                    "OVPhysX open failed and runtime cleanup also failed: "
                    f"open={type(open_error).__name__}: {open_error}; "
                    f"cleanup={type(cleanup_error).__name__}: {cleanup_error}"
                ) from open_error
            raise

    @staticmethod
    def _verify_simulation_configuration(
        simulation: Any,
        *,
        requested_dt: float,
        requested_gravity: tuple[float, float, float],
    ) -> dict[str, object]:
        """Build the frozen v1 compatibility view used by Gate 0 validators.

        ``SimulationContext.get_physics_dt`` is a Python configuration
        accessor in the pinned stack, not a compiled-runtime timestep getter.
        The legacy ``backend_dt_s`` key is retained byte-for-byte for existing
        Gate 0 bundles; new scientific claims use the additive v2 view.
        """

        cfg_dt = float(simulation.cfg.dt)
        config_accessor_dt = float(simulation.get_physics_dt())
        cfg_gravity = tuple(float(value) for value in simulation.cfg.gravity)
        usd_physics = import_module("pxr.UsdPhysics")
        scene = usd_physics.Scene.Get(
            simulation.stage, str(simulation.cfg.physics_prim_path)
        )
        if not scene or not scene.GetPrim().IsValid():
            raise RuntimeError("OVPhysX PhysicsScene readback is unavailable")
        direction = tuple(float(value) for value in scene.GetGravityDirectionAttr().Get())
        magnitude = float(scene.GetGravityMagnitudeAttr().Get())
        scene_gravity = tuple(value * magnitude for value in direction)

        if not math.isclose(cfg_dt, requested_dt, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("SimulationCfg timestep differs from the requested timestep")
        if not math.isclose(
            config_accessor_dt, requested_dt, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                "OVPhysX timestep configuration accessor differs from the request"
            )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
            for actual, expected in zip(cfg_gravity, requested_gravity)
        ):
            raise RuntimeError("SimulationCfg gravity differs from the requested gravity")
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-5)
            for actual, expected in zip(scene_gravity, requested_gravity)
        ):
            raise RuntimeError("PhysicsScene gravity differs from the requested gravity")
        return {
            "verified": True,
            "requested_dt_s": requested_dt,
            "cfg_dt_s": cfg_dt,
            "backend_dt_s": config_accessor_dt,
            "requested_gravity_m_s2": list(requested_gravity),
            "cfg_gravity_m_s2": list(cfg_gravity),
            "physics_scene_gravity_m_s2": list(scene_gravity),
            "physics_prim_path": str(simulation.cfg.physics_prim_path),
        }

    @staticmethod
    def _simulation_configuration_v2_from_legacy(
        legacy: Mapping[str, object],
    ) -> dict[str, object]:
        """Relabel the pinned accessor without claiming compiled-runtime dt."""

        return {
            "schema_version": 2,
            "dt_claim_scope": "python_configuration_only_not_compiled_runtime",
            "requested_dt_s": float(legacy["requested_dt_s"]),
            "simulation_cfg_dt_s": float(legacy["cfg_dt_s"]),
            "simulation_context_config_accessor_dt_s": float(
                legacy["backend_dt_s"]
            ),
            "python_configuration_dt_exact_match": True,
            "runtime_effective_dt_s": None,
            "runtime_effective_dt_status": (
                "not_exposed_by_pinned_kitless_ovphysx"
            ),
            "runtime_effective_dt_verified": False,
            "requested_gravity_m_s2": list(legacy["requested_gravity_m_s2"]),
            "simulation_cfg_gravity_m_s2": list(legacy["cfg_gravity_m_s2"]),
            "physics_scene_gravity_m_s2": list(
                legacy["physics_scene_gravity_m_s2"]
            ),
            "physics_prim_path": str(legacy["physics_prim_path"]),
        }

    @staticmethod
    def _controller_tensor_values(tensor: Any, *, name: str) -> tuple[float, ...]:
        shape = tuple(int(value) for value in tensor.shape)
        if shape != (1, 22):
            raise RuntimeError(
                f"unexpected IdealPD {name} tensor shape: {shape}, expected (1, 22)"
            )
        values = tuple(float(value) for value in tensor[0].detach().cpu().tolist())
        if len(values) != 22 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"IdealPD {name} tensor is invalid")
        return values

    def _current_controller_parameters(self) -> dict[str, tuple[float, ...]]:
        controller = self._controller
        if controller is None:
            raise RuntimeError("OVPhysX IdealPD controller is unavailable")
        if type(controller).__name__ != "IdealPDActuator":
            raise RuntimeError("OVPhysX actuator model changed after initialization")
        if controller.is_implicit_model is not False:
            raise RuntimeError("OVPhysX IdealPD controller is no longer explicit")
        if tuple(str(name) for name in controller.joint_names) != self._backend_joint_names:
            raise RuntimeError("IdealPD actuator joint order changed after initialization")

        controller_joint_ids = controller.joint_indices
        if isinstance(controller_joint_ids, slice) and controller_joint_ids == slice(None):
            resolved_joint_ids = tuple(range(22))
        elif hasattr(controller_joint_ids, "detach"):
            resolved_joint_ids = tuple(
                int(value) for value in controller_joint_ids.detach().cpu().tolist()
            )
        else:
            resolved_joint_ids = tuple(int(value) for value in controller_joint_ids)
        if resolved_joint_ids != tuple(range(22)):
            raise RuntimeError("IdealPD actuator no longer covers all backend joints in order")

        values = {
            "stiffness": self._controller_tensor_values(
                controller.stiffness, name="stiffness"
            ),
            "damping": self._controller_tensor_values(
                controller.damping, name="damping"
            ),
            "effort_limit": self._controller_tensor_values(
                controller.effort_limit, name="effort_limit"
            ),
            "effort_limit_sim": self._controller_tensor_values(
                controller.effort_limit_sim, name="effort_limit_sim"
            ),
            "armature": self._controller_tensor_values(
                controller.armature, name="armature"
            ),
            "static_friction": self._controller_tensor_values(
                controller.friction, name="friction"
            ),
            "dynamic_friction": self._controller_tensor_values(
                controller.dynamic_friction, name="dynamic_friction"
            ),
            "viscous_friction": self._controller_tensor_values(
                controller.viscous_friction, name="viscous_friction"
            ),
        }
        if (
            not all(value > 0.0 for value in values["stiffness"])
            or not all(value >= 0.0 for value in values["damping"])
            or not all(value > 0.0 for value in values["effort_limit"])
            or not all(value > 0.0 for value in values["effort_limit_sim"])
            or not all(value >= 0.0 for value in values["armature"])
            or not all(value >= 0.0 for value in values["static_friction"])
            or not all(value >= 0.0 for value in values["dynamic_friction"])
            or not all(value >= 0.0 for value in values["viscous_friction"])
        ):
            raise RuntimeError("IdealPD controller parameters are invalid")
        return values

    def _read_composed_joint_descriptors(self) -> dict[str, dict[str, object]]:
        """Resolve joint metadata and authored properties from the live USD stage.

        The values returned by ``UsdAttribute.Get`` are composed values.  An
        authored value is reported only when USD confirms that a value opinion
        exists; otherwise its authored layer is ``None`` while the composed
        fallback remains explicit.  Neither legacy joint-friction scalar is
        compared with any slot of ``DOF_FRICTION_PROPERTIES`` because the
        pinned wrapper and PhysX documentation use conflicting semantics.
        """

        if self._simulation is None:
            raise RuntimeError("OVPhysX live USD stage is unavailable")
        usd_physics = import_module("pxr.UsdPhysics")
        robot_prefix = "/World/Env_0/Robot"
        records: dict[str, dict[str, object]] = {}
        for prim in self._simulation.stage.Traverse():
            prim_path = str(prim.GetPath())
            if not (
                prim_path == robot_prefix
                or prim_path.startswith(robot_prefix + "/")
            ):
                continue
            if not prim.IsA(usd_physics.RevoluteJoint):
                continue
            name = str(prim.GetName())
            if name not in self._backend_joint_names:
                continue
            if name in records:
                raise RuntimeError(f"duplicate composed USD joint name: {name}")
            axis_value, axis_authored = _composed_attribute(prim, "physics:axis")
            axis = str(axis_value)
            if axis not in {"X", "Y", "Z"}:
                raise RuntimeError(f"composed USD joint {name!r} has invalid axis {axis!r}")
            armature_value, armature_authored = _composed_attribute(
                prim, "physxJoint:armature"
            )
            friction_value, friction_authored = _composed_attribute(
                prim, "physxJoint:jointFriction"
            )
            armature = _strict_finite_number(
                armature_value,
                name=f"composed USD joint {name!r} armature",
                minimum=0.0,
            )
            legacy_joint_friction = _strict_finite_number(
                friction_value,
                name=f"composed USD joint {name!r} legacy joint friction",
                minimum=0.0,
            )
            records[name] = {
                "joint_prim_path": prim_path,
                "joint_type": "revolute",
                "joint_usd_type_name": str(prim.GetTypeName()),
                "joint_axis": axis,
                "joint_axis_authored": axis_authored,
                "joint_descriptor_source": "composed_usd_input",
                "joint_descriptor_is_compiled_runtime_readback": False,
                "armature_usd": {
                    "attribute": "physxJoint:armature",
                    "has_authored_value_opinion": armature_authored,
                    "authored_value": armature if armature_authored else None,
                    "resolved_value": armature,
                    "unit": "kg_m2_for_revolute_joint",
                    "source": "composed_usd_input",
                    "compiled_runtime_readback": False,
                    "source_locator": (
                        f"{prim_path}.physxJoint:armature"
                    ),
                },
                "legacy_joint_friction_usd": {
                    "attribute": "physxJoint:jointFriction",
                    "has_authored_value_opinion": friction_authored,
                    "authored_value": (
                        legacy_joint_friction if friction_authored else None
                    ),
                    "resolved_value": legacy_joint_friction,
                    "unit": "documentation_conflict_not_assigned",
                    "source": "composed_usd_input",
                    "compiled_runtime_readback": False,
                    "source_locator": (
                        f"{prim_path}.physxJoint:jointFriction"
                    ),
                    "runtime_binding_equality_check": "not_performed",
                },
            }
        missing = tuple(name for name in self._backend_joint_names if name not in records)
        if missing or len(records) != len(self._backend_joint_names):
            raise RuntimeError(
                "composed USD joint descriptors do not exactly cover backend order: "
                f"missing={list(missing)}"
            )
        return records

    def _capture_backend_joint_dynamics(self) -> dict[str, object]:
        required_bindings = {
            "DOF_STIFFNESS": (1, 22),
            "DOF_DAMPING": (1, 22),
            "DOF_ARMATURE": (1, 22),
            "DOF_FRICTION_PROPERTIES": (1, 22, 3),
        }
        missing_bindings = sorted(set(required_bindings) - set(self._tensor_bindings))
        if missing_bindings:
            raise RuntimeError(
                f"OVPhysX effective readback bindings are missing: {missing_bindings}"
            )

        stiffness_payload = _read_cpu_float_binding(
            self._tensor_bindings["DOF_STIFFNESS"],
            name="DOF_STIFFNESS",
            expected_shape=required_bindings["DOF_STIFFNESS"],
        )
        damping_payload = _read_cpu_float_binding(
            self._tensor_bindings["DOF_DAMPING"],
            name="DOF_DAMPING",
            expected_shape=required_bindings["DOF_DAMPING"],
        )
        armature_payload = _read_cpu_float_binding(
            self._tensor_bindings["DOF_ARMATURE"],
            name="DOF_ARMATURE",
            expected_shape=required_bindings["DOF_ARMATURE"],
        )
        friction_payload = _read_cpu_float_binding(
            self._tensor_bindings["DOF_FRICTION_PROPERTIES"],
            name="DOF_FRICTION_PROPERTIES",
            expected_shape=required_bindings["DOF_FRICTION_PROPERTIES"],
        )
        try:
            backend_stiffness = tuple(float(value) for value in stiffness_payload)
            backend_damping = tuple(float(value) for value in damping_payload)
            backend_armature = tuple(float(value) for value in armature_payload)
            backend_friction = tuple(
                tuple(float(value) for value in row) for row in friction_payload
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("OVPhysX joint-dynamics binding rank is invalid") from error
        if any(len(values) != 22 for values in (backend_stiffness, backend_damping, backend_armature)):
            raise RuntimeError("OVPhysX joint-dynamics binding length is invalid")
        if len(backend_friction) != 22 or any(len(row) != 3 for row in backend_friction):
            raise RuntimeError("OVPhysX friction binding rank is invalid")
        numeric_values = (
            *backend_stiffness,
            *backend_damping,
            *backend_armature,
            *(value for row in backend_friction for value in row),
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise RuntimeError("OVPhysX effective joint dynamics contain non-finite values")
        if any(value < 0.0 for value in numeric_values):
            raise RuntimeError("OVPhysX effective joint dynamics contain negative values")
        if any(
            abs(value) > _BACKEND_DRIVE_ZERO_ABS_TOL
            for value in (*backend_stiffness, *backend_damping)
        ):
            raise RuntimeError(
                "OVPhysX PhysX drive stiffness/damping were not zeroed for IdealPD"
            )

        controller_values = self._current_controller_parameters()
        expected_controller_values = {
            "stiffness": self._controller_stiffness,
            "damping": self._controller_damping,
            "effort_limit": self._controller_effort_limit,
            "effort_limit_sim": self._controller_effort_limit_sim,
            "armature": self._controller_armature,
            "static_friction": self._controller_static_friction,
            "dynamic_friction": self._controller_dynamic_friction,
            "viscous_friction": self._controller_viscous_friction,
        }
        for name, expected in expected_controller_values.items():
            if controller_values[name] != expected:
                raise RuntimeError(
                    f"IdealPD {name} changed between initialization and pre-step readback"
                )

        backend_to_controller = {
            "armature": (backend_armature, controller_values["armature"]),
            "static_friction": (
                tuple(row[0] for row in backend_friction),
                controller_values["static_friction"],
            ),
            "dynamic_friction": (
                tuple(row[1] for row in backend_friction),
                controller_values["dynamic_friction"],
            ),
            "viscous_friction": (
                tuple(row[2] for row in backend_friction),
                controller_values["viscous_friction"],
            ),
        }
        binding_exact_match = {
            name: backend_values == controller_buffer_values
            for name, (backend_values, controller_buffer_values) in (
                backend_to_controller.items()
            )
        }

        descriptors = self._read_composed_joint_descriptors()
        mapping_by_backend = {
            str(record["backend_name"]): record for record in self._joint_mapping
        }
        if set(mapping_by_backend) != set(self._backend_joint_names):
            raise RuntimeError("OVPhysX canonical mapping does not cover backend joint order")
        records: list[dict[str, object]] = []
        for index, backend_name in enumerate(self._backend_joint_names):
            mapping = mapping_by_backend[backend_name]
            records.append(
                {
                    "canonical_id": str(mapping["canonical_id"]),
                    "backend_name": backend_name,
                    "backend_index": index,
                    **descriptors[backend_name],
                    "armature": backend_armature[index],
                    "armature_unit": "kg_m2_for_revolute_joint",
                    "friction_properties_raw": list(backend_friction[index]),
                    "runtime_binding_source": "root_view_cpu_numpy_binding",
                    "controller_armature": controller_values["armature"][index],
                    "controller_static_friction": controller_values[
                        "static_friction"
                    ][index],
                    "controller_dynamic_friction": controller_values[
                        "dynamic_friction"
                    ][index],
                    "controller_viscous_friction": controller_values[
                        "viscous_friction"
                    ][index],
                    "controller_binding_exact_match": {
                        "armature": (
                            backend_armature[index]
                            == controller_values["armature"][index]
                        ),
                        "static_friction": (
                            backend_friction[index][0]
                            == controller_values["static_friction"][index]
                        ),
                        "dynamic_friction": (
                            backend_friction[index][1]
                            == controller_values["dynamic_friction"][index]
                        ),
                        "viscous_friction": (
                            backend_friction[index][2]
                            == controller_values["viscous_friction"][index]
                        ),
                    },
                    "controller_stiffness": controller_values["stiffness"][index],
                    "controller_damping": controller_values["damping"][index],
                    "controller_effort_limit": controller_values["effort_limit"][index],
                    "controller_effort_limit_sim": controller_values[
                        "effort_limit_sim"
                    ][index],
                    "backend_drive_stiffness": backend_stiffness[index],
                    "backend_drive_damping": backend_damping[index],
                    "velocity_limit": None,
                    "passive_joint_damping": None,
                    "gear": None,
                }
            )

        self._backend_drive_stiffness = backend_stiffness
        self._backend_drive_damping = backend_damping
        self._backend_armature = backend_armature
        self._backend_friction_properties = backend_friction
        return {
            "schema_version": _BACKEND_JOINT_DYNAMICS_SCHEMA_VERSION,
            "source": "ovphysx_tensor_binding_post_load",
            "runtime_effective_readback": True,
            "capture_phase": "post_reset_pre_trace_step",
            "scope": "all_backend_dofs",
            "joint_count": 22,
            "joint_order": list(self._backend_joint_names),
            "records": records,
            "controller_consistency_verified": True,
            "controller_joint_dynamics_binding_comparison_performed": True,
            "controller_joint_dynamics_binding_exact_match": {
                "overall": all(binding_exact_match.values()),
                **binding_exact_match,
            },
            "controller_joint_dynamics_binding_match_policy": "exact_float32_value",
            "controller_model": "IdealPDActuator",
            "controller_mode": "explicit_pd_effort",
            "backend_drive_zero_verified": True,
            "backend_drive_zero_abs_tolerance": _BACKEND_DRIVE_ZERO_ABS_TOL,
            "friction_semantics": {
                "raw_slot_order": ["static", "dynamic", "viscous"],
                "pinned_wrapper_labels": [
                    "joint_friction_coefficient",
                    "joint_dynamic_friction_coefficient",
                    "joint_viscous_friction_coefficient",
                ],
                "authoritative_api_labels": [
                    "static_friction_effort",
                    "dynamic_friction_effort",
                    "viscous_friction_coefficient",
                ],
                "semantic_conflict": True,
                "static_dynamic_units": "documentation_conflict_not_assigned",
                "viscous_unit": "N_m_s_per_rad_for_revolute_joint",
                "legacy_equality_check": "not_performed",
                "cross_engine_equivalence": "not_claimed",
            },
            "velocity_limit_readback": {
                "status": "not_captured_in_this_contract",
                "value": None,
            },
            "passive_joint_damping_readback": {
                "status": "not_exposed_separately_by_pinned_ovphysx",
                "value": None,
            },
            "gear_readback": {
                "status": "not_exposed_by_pinned_ovphysx_tensor_api",
                "value": None,
            },
        }

    def effective_parameter_readback(self) -> dict[str, object]:
        """Freshly capture effective joint tensors before the first trace step.

        Solver values are intentionally kept separate and labelled as live
        composed-USD inputs because the pinned Python backend does not expose
        the compiled C++ solver state. The method is fail-closed after the
        first adapter step so callers cannot accidentally present a late
        snapshot as the preregistered pre-step contract.
        """

        self._require_ready("read effective OVPhysX parameters")
        if self._step_index != 0:
            raise AdapterLifecycleError(
                "effective OVPhysX parameter readback is only valid before the first trace step"
            )
        backend_joint_dynamics = self._capture_backend_joint_dynamics()
        physics_solver_contract = _read_live_composed_usd_solver_contract(
            self._simulation, requested_dt=self._dt
        )
        snapshot = _json_clone(
            {
                "readback_contract_version": (
                    _EFFECTIVE_PARAMETER_READBACK_CONTRACT_VERSION
                ),
                "capture_phase": "post_reset_pre_trace_step",
                "step_index": self._step_index,
                "backend_joint_dynamics": backend_joint_dynamics,
                "physics_solver_contract": physics_solver_contract,
            }
        )
        self._backend_joint_dynamics = _json_clone(backend_joint_dynamics)
        self._physics_solver_contract = _json_clone(physics_solver_contract)
        self._effective_parameter_snapshot = snapshot
        return _json_clone(snapshot)

    def _canonical_position_target_readback(
        self,
        backend_values: Sequence[float],
    ) -> dict[str, float]:
        """Project one backend-order target vector into canonical joint order."""

        if len(backend_values) != len(self._backend_joint_names):
            raise RuntimeError("OVPhysX position-target readback length is invalid")
        mapping_by_name = {
            str(record["canonical_id"]): record for record in self._joint_mapping
        }
        if set(mapping_by_name) != set(self._joint_names):
            raise RuntimeError("OVPhysX position-target mapping coverage is invalid")
        canonical: dict[str, float] = {}
        for canonical_id in self._joint_names:
            record = mapping_by_name[canonical_id]
            index = int(record["index"])
            if index < 0 or index >= len(backend_values):
                raise RuntimeError("OVPhysX position-target mapping index is invalid")
            value = (
                float(record["sign"]) * float(backend_values[index])
                + float(record["offset"])
            )
            if not math.isfinite(value):
                raise RuntimeError("OVPhysX canonical position-target readback is invalid")
            canonical[canonical_id] = value
        return canonical

    def _verify_position_target(self) -> dict[str, float]:
        if self._articulation is None or self._target_tensor is None:
            raise RuntimeError("OVPhysX position-target buffer is unavailable")
        torch = self._runtime["torch"]
        readback = self._articulation.data.joint_pos_target.torch
        if tuple(int(value) for value in readback.shape) != (1, 22):
            raise RuntimeError("OVPhysX position-target buffer shape is invalid")
        delta = torch.max(torch.abs(readback - self._target_tensor))
        max_abs_error = float(delta.detach().cpu().item())
        if not math.isfinite(max_abs_error) or max_abs_error > 1e-6:
            raise RuntimeError(
                "OVPhysX articulation position-target buffer differs from the requested target"
            )
        values = tuple(float(value) for value in readback[0].detach().cpu().tolist())
        if len(values) != 22 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("OVPhysX articulation position-target buffer is invalid")
        canonical_values = self._canonical_position_target_readback(values)
        if set(self._position_targets) == set(self._joint_names):
            canonical_error = max(
                abs(canonical_values[name] - float(self._position_targets[name]))
                for name in self._joint_names
            )
            if not math.isfinite(canonical_error) or canonical_error > 1e-6:
                raise RuntimeError(
                    "OVPhysX canonical position-target readback differs from the "
                    "requested float64 target"
                )
            self._canonical_target_readback_max_abs_error = max(
                self._canonical_target_readback_max_abs_error,
                canonical_error,
            )
        self._target_readback_count += 1
        self._target_readback_max_abs_error = max(
            self._target_readback_max_abs_error, max_abs_error
        )
        self._target_readback_values = values
        self._target_nonzero_readback_observed = (
            self._target_nonzero_readback_observed
            or any(abs(value) > 1e-8 for value in values)
        )
        return canonical_values

    def _observe_explicit_effort(self) -> None:
        """Validate and accumulate the IdealPD torque path before simulation steps."""

        if self._articulation is None or self._controller is None:
            raise RuntimeError("OVPhysX IdealPD effort observation is unavailable")
        torch = self._runtime["torch"]
        data = self._articulation.data
        controller = self._controller
        joint_pos = data.joint_pos.torch
        joint_vel = data.joint_vel.torch
        joint_pos_target = data.joint_pos_target.torch
        joint_vel_target = data.joint_vel_target.torch
        joint_effort_target = data.joint_effort_target.torch
        computed = data.computed_torque.torch
        applied = data.applied_torque.torch
        tensors = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_pos_target": joint_pos_target,
            "joint_vel_target": joint_vel_target,
            "joint_effort_target": joint_effort_target,
            "computed_torque": computed,
            "applied_torque": applied,
            "controller_stiffness": controller.stiffness,
            "controller_damping": controller.damping,
            "controller_effort_limit": controller.effort_limit,
        }
        for name, tensor in tensors.items():
            if tuple(int(value) for value in tensor.shape) != (1, 22):
                raise RuntimeError(f"unexpected {name} tensor shape during effort validation")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"OVPhysX returned non-finite {name} tensor")

        expected_computed = (
            controller.stiffness * (joint_pos_target - joint_pos)
            + controller.damping * (joint_vel_target - joint_vel)
            + joint_effort_target
        )
        formula_error = float(
            torch.max(torch.abs(computed - expected_computed)).detach().cpu().item()
        )
        expected_applied = torch.maximum(
            torch.minimum(computed, controller.effort_limit),
            -controller.effort_limit,
        )
        clip_error = float(
            torch.max(torch.abs(applied - expected_applied)).detach().cpu().item()
        )
        if (
            not math.isfinite(formula_error)
            or formula_error > _EFFORT_FORMULA_VALIDATION_ABS_TOL
        ):
            raise RuntimeError("IdealPD computed torque does not satisfy the PD formula")
        if (
            not math.isfinite(clip_error)
            or clip_error > _EFFORT_CLIP_VALIDATION_ABS_TOL
        ):
            raise RuntimeError("IdealPD applied torque does not satisfy effort clipping")

        computed_abs = tuple(
            float(value) for value in torch.abs(computed[0]).detach().cpu().tolist()
        )
        applied_abs = tuple(
            float(value) for value in torch.abs(applied[0]).detach().cpu().tolist()
        )
        if not self._computed_effort_peak_abs:
            self._computed_effort_peak_abs = (0.0,) * 22
            self._applied_effort_peak_abs = (0.0,) * 22
        self._computed_effort_peak_abs = tuple(
            max(previous, current)
            for previous, current in zip(self._computed_effort_peak_abs, computed_abs)
        )
        self._applied_effort_peak_abs = tuple(
            max(previous, current)
            for previous, current in zip(self._applied_effort_peak_abs, applied_abs)
        )
        self._effort_observation_count += 1
        self._effort_formula_max_abs_error = max(
            self._effort_formula_max_abs_error, formula_error
        )
        self._effort_clip_max_abs_error = max(self._effort_clip_max_abs_error, clip_error)
        self._effort_clip_count += int(
            torch.count_nonzero(torch.abs(computed) > controller.effort_limit)
            .detach()
            .cpu()
            .item()
        )

    def _verify_zero_auxiliary_targets(self) -> None:
        """Verify the explicit PD velocity and feed-forward effort commands are zero."""

        if self._articulation is None:
            raise RuntimeError("OVPhysX auxiliary target buffers are unavailable")
        torch = self._runtime["torch"]
        velocity_target = self._articulation.data.joint_vel_target.torch
        effort_target = self._articulation.data.joint_effort_target.torch
        for name, tensor in (
            ("joint_vel_target", velocity_target),
            ("joint_effort_target", effort_target),
        ):
            if tuple(int(value) for value in tensor.shape) != (1, 22):
                raise RuntimeError(f"unexpected {name} tensor shape")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"OVPhysX returned non-finite {name} tensor")
            max_abs = float(torch.max(torch.abs(tensor)).detach().cpu().item())
            if max_abs > _ZERO_AUXILIARY_TARGET_ABS_TOL:
                raise RuntimeError(f"OVPhysX {name} is not zero")
        self._zero_velocity_target_verified = True
        self._zero_feedforward_effort_target_verified = True

    def reset(self) -> None:
        self._require_ready("reset")
        torch = self._runtime["torch"]
        self._target_readback_count = 0
        self._target_readback_max_abs_error = 0.0
        self._target_readback_values = ()
        self._canonical_target_readback_max_abs_error = 0.0
        self._target_nonzero_readback_observed = False
        self._scheduled_target_sequence_digest = None
        self._requested_target_sequence_digest = None
        self._immediate_target_readback_sequence_digest = None
        self._pre_step_applied_target_readback_sequence_digest = None
        self._zero_velocity_target_verified = False
        self._zero_feedforward_effort_target_verified = False
        self._computed_effort_peak_abs = (0.0,) * 22
        self._applied_effort_peak_abs = (0.0,) * 22
        self._effort_observation_count = 0
        self._effort_formula_max_abs_error = 0.0
        self._effort_clip_max_abs_error = 0.0
        self._effort_clip_count = 0
        zero = torch.zeros_like(self._articulation.data.joint_pos.torch)
        self._articulation.write_joint_position_to_sim_index(position=zero)
        self._articulation.write_joint_velocity_to_sim_index(velocity=zero)
        self._articulation.reset()
        self._target_tensor = zero.clone()
        self._articulation.set_joint_position_target_index(target=zero)
        self._articulation.set_joint_velocity_target_index(target=zero)
        self._articulation.set_joint_effort_target_index(target=zero)
        self._articulation.write_data_to_sim()
        self._verify_position_target()
        self._verify_zero_auxiliary_targets()
        self._observe_explicit_effort()
        self._position_targets = {name: 0.0 for name in self._joint_names}
        self._step_index = 0
        self._check_finite(step=0)

    def set_position_targets(self, targets: Mapping[str, float]) -> None:
        self._require_ready("set position targets")
        unknown = sorted(set(targets) - set(self._joint_names))
        if unknown:
            raise ValueError(f"unknown OVPhysX position target(s): {unknown}")
        target_tensor = self._target_tensor.clone()
        mapping_by_name = {record["canonical_id"]: record for record in self._joint_mapping}
        for canonical_id, raw_value in targets.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"position target for {canonical_id!r} must be a number")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"position target for {canonical_id!r} must be finite")
            record = mapping_by_name[canonical_id]
            backend_value = (value - float(record["offset"])) / float(record["sign"])
            target_tensor[0, int(record["index"])] = backend_value
            self._position_targets[canonical_id] = value
        self._articulation.set_joint_position_target_index(target=target_tensor)
        self._target_tensor = target_tensor
        canonical_readback = self._verify_position_target()
        if self._immediate_target_readback_sequence_digest is not None:
            self._immediate_target_readback_sequence_digest.append(
                canonical_readback
            )
        self._check_finite(step=self._step_index)

    def step(self) -> None:
        self._require_ready("step")
        self._articulation.write_data_to_sim()
        canonical_readback = self._verify_position_target()
        if self._pre_step_applied_target_readback_sequence_digest is not None:
            self._pre_step_applied_target_readback_sequence_digest.append(
                canonical_readback
            )
        self._verify_zero_auxiliary_targets()
        self._observe_explicit_effort()
        self._simulation.step(render=False)
        self._articulation.update(self._dt)
        self._step_index += 1
        self._check_finite(step=self._step_index)

    def sample(
        self,
        *,
        step: int,
        joint_names: tuple[str, ...] = (),
        frame_names: tuple[str, ...] = (),
    ) -> TraceSample:
        self._require_ready("sample")
        if step != self._step_index:
            raise ValueError(
                f"sample step {step} does not match completed OVPhysX step {self._step_index}"
            )
        selected_joints = joint_names or self._joint_names
        selected_frames = frame_names or self._frame_names
        if tuple(selected_joints) != self._joint_names:
            raise ValueError("OVPhysX samples require canonical joint order")
        if tuple(selected_frames) != self._frame_names:
            raise ValueError("OVPhysX samples require canonical distal-frame order")

        joint_pos = self._articulation.data.joint_pos.torch
        joint_vel = self._articulation.data.joint_vel.torch
        body_pose = self._articulation.data.body_link_pose_w.torch
        positions: dict[str, float] = {}
        for record in self._joint_mapping:
            backend_value = float(joint_pos[0, int(record["index"])].detach().cpu().item())
            positions[str(record["canonical_id"])] = (
                float(record["sign"]) * backend_value + float(record["offset"])
            )

        poses: dict[str, tuple[float, ...]] = {}
        for record in self._frame_mapping:
            index = int(record["index"])
            values = tuple(
                float(value)
                for value in body_pose[0, index].detach().cpu().tolist()
            )
            if len(values) != 7:
                raise RuntimeError("OVPhysX body pose tensor must contain xyz+xyzw")
            poses[str(record["canonical_id"])] = (
                values[0],
                values[1],
                values[2],
                values[6],
                values[3],
                values[4],
                values[5],
            )
        qpos = tuple(float(value) for value in joint_pos[0].detach().cpu().tolist())
        qvel = tuple(float(value) for value in joint_vel[0].detach().cpu().tolist())
        numeric = (
            *qpos,
            *qvel,
            *positions.values(),
            *self._position_targets.values(),
            *(value for pose in poses.values() for value in pose),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError(f"OVPhysX returned non-finite state at step {step}")
        return TraceSample(
            step=step,
            time_s=step * self._dt,
            qpos=qpos,
            qvel=qvel,
            joint_positions=positions,
            frame_poses=poses,
            position_targets=dict(self._position_targets),
            contact_count=None,
        )

    def _begin_target_sequence_audit(
        self,
        joint_names: Sequence[str],
    ) -> None:
        self._scheduled_target_sequence_digest = CanonicalTargetSequenceDigest(
            joint_names
        )
        self._requested_target_sequence_digest = CanonicalTargetSequenceDigest(
            joint_names
        )
        self._immediate_target_readback_sequence_digest = (
            CanonicalTargetSequenceDigest(joint_names)
        )
        self._pre_step_applied_target_readback_sequence_digest = (
            CanonicalTargetSequenceDigest(joint_names)
        )

    def _schedule_and_set_position_targets(
        self,
        targets: Mapping[str, float],
    ) -> None:
        scheduled = self._scheduled_target_sequence_digest
        requested = self._requested_target_sequence_digest
        immediate = self._immediate_target_readback_sequence_digest
        if scheduled is None or requested is None or immediate is None:
            raise RuntimeError("OVPhysX target-sequence audit is not initialized")
        scheduled.append(targets)
        requested.append(targets)
        before_count = immediate.count
        self.set_position_targets(targets)
        if immediate.count == before_count:
            # Lightweight unit-test harnesses may replace the runtime setter.
            # The production setter always appends the actual tensor readback.
            immediate.append(dict(self._position_targets))
        elif immediate.count != before_count + 1:
            raise RuntimeError(
                "OVPhysX immediate target readback audit count advanced unexpectedly"
            )

    def _step_with_target_readback_audit(self) -> None:
        pre_step = self._pre_step_applied_target_readback_sequence_digest
        if pre_step is None:
            raise RuntimeError("OVPhysX target-sequence audit is not initialized")
        before_count = pre_step.count
        fallback = dict(self._position_targets)
        self.step()
        if pre_step.count == before_count:
            # See the corresponding setter fallback above. Production step()
            # records after write_data_to_sim and before the physics advance.
            pre_step.append(fallback)
        elif pre_step.count != before_count + 1:
            raise RuntimeError(
                "OVPhysX pre-step target readback audit count advanced unexpectedly"
            )

    def _target_sequence_provenance(self) -> dict[str, object]:
        scheduled = self._scheduled_target_sequence_digest
        requested = self._requested_target_sequence_digest
        immediate = self._immediate_target_readback_sequence_digest
        pre_step = self._pre_step_applied_target_readback_sequence_digest
        if any(
            digest is None
            for digest in (scheduled, requested, immediate, pre_step)
        ):
            return {}
        assert scheduled is not None
        assert requested is not None
        assert immediate is not None
        assert pre_step is not None
        return {
            "target_sequence_digest_schema_version": 1,
            "target_sequence_digest_encoding": "utf8_json_lines_float_hex_v1",
            "target_sequence_digest_projection": "ieee754_binary32_roundtrip",
            "target_sequence_canonical_joint_names": list(scheduled.joint_names),
            "scheduled_target_sequence_semantics": (
                "q[0..N]; target q[k] is recorded at t_k and applies to "
                "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
            ),
            "scheduled_target_sequence_count": scheduled.count,
            "scheduled_target_sequence_sha256": scheduled.hexdigest(),
            "requested_target_sequence_semantics": (
                "q[0..N] passed to set_joint_position_target_index"
            ),
            "requested_target_sequence_count": requested.count,
            "requested_target_sequence_sha256": requested.hexdigest(),
            "immediate_target_readback_sequence_semantics": (
                "q[0..N] read immediately from joint_pos_target after each request"
            ),
            "immediate_target_readback_sequence_count": immediate.count,
            "immediate_target_readback_sequence_sha256": immediate.hexdigest(),
            "pre_step_applied_target_readback_sequence_semantics": (
                "q[0..N-1] read after write_data_to_sim and before each physics step"
            ),
            "pre_step_applied_target_readback_sequence_count": pre_step.count,
            "pre_step_applied_target_readback_sequence_sha256": pre_step.hexdigest(),
        }

    def _validate_complete_target_sequence_audit(
        self,
        scenario: ScenarioSpec | WaveformScenarioSpec,
        *,
        dt: float,
        requested_steps: int,
    ) -> None:
        evidence = self._target_sequence_provenance()
        terminal_count = requested_steps + 1
        count_fields = {
            "scheduled_target_sequence_count": terminal_count,
            "requested_target_sequence_count": terminal_count,
            "immediate_target_readback_sequence_count": terminal_count,
            "pre_step_applied_target_readback_sequence_count": requested_steps,
        }
        for field_name, expected in count_fields.items():
            if evidence.get(field_name) != expected:
                raise RuntimeError(
                    f"OVPhysX {field_name} is incomplete: "
                    f"{evidence.get(field_name)!r} != {expected}"
                )
        full_digest = canonical_target_sequence_sha256(
            scenario,
            self._joint_names,
            dt_s=dt,
            include_terminal=True,
        )
        prefix_digest = canonical_target_sequence_sha256(
            scenario,
            self._joint_names,
            dt_s=dt,
            include_terminal=False,
        )
        for field_name in (
            "scheduled_target_sequence_sha256",
            "requested_target_sequence_sha256",
            "immediate_target_readback_sequence_sha256",
        ):
            if evidence.get(field_name) != full_digest:
                raise RuntimeError(
                    f"OVPhysX {field_name} does not match the canonical schedule"
                )
        if (
            evidence.get("pre_step_applied_target_readback_sequence_sha256")
            != prefix_digest
        ):
            raise RuntimeError(
                "OVPhysX pre-step target readback does not match the canonical "
                "schedule prefix"
            )

    def run_scenario(
        self,
        manifest: object,
        scenario: object,
        *,
        dt_override: float | None = None,
    ) -> AdapterRunResult:
        scenario_id = str(_field(scenario, "scenario_id", "id", "name", default="scenario"))
        kind = str(_plain_value(_field(scenario, "kind", default="zero_hold")))
        requested_steps = 0
        completed_steps = 0
        samples: list[TraceSample] = []
        joint_names: tuple[str, ...] = ()
        frame_names: tuple[str, ...] = ()
        provenance: dict[str, object] = {}
        dt = 0.0
        result: AdapterRunResult
        try:
            if kind not in _SUPPORTED_SCENARIOS:
                raise AdapterCapabilityError(
                    self.backend_name,
                    "scenario_kind",
                    f"adapter supports only {sorted(_SUPPORTED_SCENARIOS)}, got {kind!r}",
                )
            raw_dt = (
                dt_override
                if dt_override is not None
                else _field(scenario, "dt_s", "dt", "timestep", default=0.0)
            )
            dt = float(raw_dt)
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("scenario dt must be a positive finite number")
            requested_steps = _scenario_steps(scenario, dt)
            if requested_steps < 1:
                raise ValueError("scenario must request at least one simulation step")
            raw_gravity = _field(scenario, "gravity_m_s2", "gravity", default=(0, 0, 0))
            if (
                not isinstance(raw_gravity, Sequence)
                or isinstance(raw_gravity, (str, bytes))
                or len(raw_gravity) != 3
            ):
                raise ValueError("gravity_m_s2 must contain exactly three numbers")
            gravity = tuple(float(value) for value in raw_gravity)
            if not all(math.isfinite(value) for value in gravity):
                raise ValueError("gravity_m_s2 must contain finite numbers")
            self._open_gravity = gravity  # consumed synchronously by open()
            self.open(manifest, dt_override=dt)
            joint_names = self._joint_names
            frame_names = self._frame_names
            canonical_scenario = isinstance(
                scenario,
                (ScenarioSpec, WaveformScenarioSpec),
            )
            if canonical_scenario:
                self._begin_target_sequence_audit(joint_names)
                baseline = canonical_position_targets(
                    scenario,
                    joint_names,
                    step_index=0,
                    dt_s=dt,
                )
                self._schedule_and_set_position_targets(baseline)
            else:
                baseline = {name: 0.0 for name in joint_names}
                self.set_position_targets(baseline)
            samples.append(
                self.sample(step=0, joint_names=joint_names, frame_names=frame_names)
            )

            onset = requested_steps + 1
            stepped = dict(baseline)
            if kind == "small_step":
                start_s = float(_field(scenario, "step_start_s", "step_start"))
                target = float(_field(scenario, "target_position_rad", default=0.0))
                if not math.isfinite(start_s) or not math.isfinite(target):
                    raise ValueError("small_step onset and target must be finite")
                ratio = start_s / dt
                onset = round(ratio)
                if not math.isclose(ratio, onset, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("small_step onset must align with the effective timestep")
                if onset < 1 or onset > requested_steps:
                    raise ValueError("small_step onset must be within the scenario")
                stepped = {name: target for name in joint_names}

            self._lifecycle = AdapterLifecycle.RUNNING
            for step_index in range(1, requested_steps + 1):
                # The target recorded at timestamp t becomes active for the
                # following interval [t, t + dt).  Stepping first prevents the
                # onset command from being integrated one base/halved interval
                # early and keeps dt-halving semantics aligned.
                if canonical_scenario:
                    self._step_with_target_readback_audit()
                else:
                    self.step()
                completed_steps = step_index
                if canonical_scenario:
                    targets = canonical_position_targets(
                        scenario,
                        joint_names,
                        step_index=step_index,
                        dt_s=dt,
                    )
                    self._schedule_and_set_position_targets(targets)
                elif (
                    kind == "small_step"
                    and step_index >= onset
                    and stepped != self._position_targets
                ):
                    self.set_position_targets(stepped)
                samples.append(
                    self.sample(
                        step=step_index,
                        joint_names=joint_names,
                        frame_names=frame_names,
                    )
                )
            if canonical_scenario:
                assert isinstance(scenario, (ScenarioSpec, WaveformScenarioSpec))
                self._validate_complete_target_sequence_audit(
                    scenario,
                    dt=dt,
                    requested_steps=requested_steps,
                )
            provenance = self._provenance()
            provenance.update(self._target_sequence_provenance())
            self._lifecycle = AdapterLifecycle.READY
            result = AdapterRunResult(
                backend=self.backend_name,
                scenario_id=scenario_id,
                status="completed",
                message="Kit-less fixed-base OVPhysX scenario completed with finite state.",
                dt=dt,
                requested_steps=requested_steps,
                completed_steps=completed_steps,
                joint_names=joint_names,
                frame_names=frame_names,
                samples=tuple(samples),
                provenance=provenance,
            )
        except Exception as error:
            if self._source_path is not None:
                provenance = self._provenance()
                provenance.update(self._target_sequence_provenance())
            self._lifecycle = AdapterLifecycle.FAILED
            detail = (
                error.to_dict()
                if isinstance(error, AdapterCapabilityError)
                else {"type": type(error).__name__, "message": str(error)}
            )
            result = AdapterRunResult(
                backend=self.backend_name,
                scenario_id=scenario_id,
                status="error",
                message=str(error),
                dt=dt,
                requested_steps=requested_steps,
                completed_steps=completed_steps,
                joint_names=joint_names,
                frame_names=frame_names,
                samples=tuple(samples),
                provenance=provenance,
                error=detail,
            )
        finally:
            self._open_gravity = (0.0, 0.0, 0.0)
            try:
                self.close()
            except Exception as close_error:
                if "result" in locals() and result.status == "completed":
                    result = AdapterRunResult(
                        backend=self.backend_name,
                        scenario_id=scenario_id,
                        status="error",
                        message=f"failed to close OVPhysX runtime: {close_error}",
                        dt=dt,
                        requested_steps=requested_steps,
                        completed_steps=completed_steps,
                        joint_names=joint_names,
                        frame_names=frame_names,
                        samples=tuple(samples),
                        provenance=provenance,
                        error={
                            "type": type(close_error).__name__,
                            "message": str(close_error),
                        },
                    )
            if (
                "result" in locals()
                and self._legacy_joint_friction_intervention_evidence
            ):
                final_provenance = dict(result.provenance)
                final_provenance["legacy_joint_friction_intervention"] = (
                    _json_clone(
                        self._legacy_joint_friction_intervention_evidence
                    )
                )
                result = replace(result, provenance=final_provenance)
        return result

    def close(self) -> None:
        try:
            self._release_runtime()
        except Exception:
            self._lifecycle = AdapterLifecycle.FAILED
            raise
        self._lifecycle = AdapterLifecycle.CLOSED

    def _require_ready(self, operation: str) -> None:
        if self._lifecycle not in {AdapterLifecycle.READY, AdapterLifecycle.RUNNING}:
            raise AdapterLifecycleError(
                f"cannot {operation} while OVPhysX adapter is {self._lifecycle.value}"
            )

    def _check_finite(self, *, step: int) -> None:
        torch = self._runtime["torch"]
        tensors = (
            self._articulation.data.joint_pos.torch,
            self._articulation.data.joint_vel.torch,
            self._articulation.data.body_link_pose_w.torch,
            self._articulation.data.joint_pos_target.torch,
            self._articulation.data.joint_vel_target.torch,
            self._articulation.data.joint_effort_target.torch,
            self._articulation.data.computed_torque.torch,
            self._articulation.data.applied_torque.torch,
            self._target_tensor,
        )
        if any(tensor is not None and not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
            raise FloatingPointError(f"OVPhysX returned non-finite tensor state at step {step}")

    def _provenance(self) -> dict[str, object]:
        if self._source_path is None:
            return {}
        torch = self._runtime.get("torch")
        provenance: dict[str, object] = {
            "backend": self.backend_name,
            "packages": _package_versions(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device": self.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_cuda_device_count": torch.cuda.device_count() if torch else None,
            "logical_cuda_device_name": torch.cuda.get_device_name(0) if torch else None,
            "source_path": str(self._source_path),
            "source_sha256": sha256(self._source_path.read_bytes()).hexdigest(),
            "source_revision": os.environ.get("WAVEQA_SOURCE_TREE"),
            "worker_pid": os.getpid(),
            "kitless": True,
            "renderer": False,
            "camera": False,
            "forbidden_modules": sorted(
                name
                for name in sys.modules
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in _FORBIDDEN_MODULE_PREFIXES
                )
            ),
            "backend_joint_names": list(self._backend_joint_names),
            "backend_frame_names": list(self._backend_frame_names),
            "joint_mapping": [dict(record) for record in self._joint_mapping],
            "frame_mapping": [dict(record) for record in self._frame_mapping],
            "tensor_binding_shapes": dict(self._tensor_binding_shapes),
            "simulation_configuration": dict(self._simulation_contract),
            "simulation_configuration_v2": dict(self._simulation_contract_v2),
            "actuation_contract_version": 2,
            "control_path": "explicit_pd_effort",
            "actuator_model": "IdealPDActuator",
            "controller_dof_stiffness": list(self._controller_stiffness),
            "controller_dof_damping": list(self._controller_damping),
            "controller_dof_effort_limit": list(self._controller_effort_limit),
            "controller_dof_effort_limit_sim": list(
                self._controller_effort_limit_sim
            ),
            "controller_parameter_source": "ideal_pd_actuator_tensor",
            "backend_dof_stiffness": list(self._backend_drive_stiffness),
            "backend_dof_damping": list(self._backend_drive_damping),
            "backend_dof_drive_readback_source": "root_view_cpu_numpy_binding",
            "backend_dof_drive_zero_verified": True,
            "backend_dof_drive_zero_abs_tolerance": _BACKEND_DRIVE_ZERO_ABS_TOL,
            "effective_parameter_readback": _json_clone(
                self._effective_parameter_snapshot
            )
            if self._effective_parameter_snapshot
            else {},
            "position_target_readback_verified": True,
            "position_target_readback_source": (
                "articulation_data_joint_pos_target_torch"
            ),
            "position_target_readback_count": self._target_readback_count,
            "position_target_readback_max_abs_error_rad": (
                self._target_readback_max_abs_error
            ),
            "position_target_canonical_readback_max_abs_error_rad": (
                self._canonical_target_readback_max_abs_error
            ),
            "position_target_readback_values_rad": list(self._target_readback_values),
            "position_target_nonzero_readback_observed": (
                self._target_nonzero_readback_observed
            ),
            "zero_velocity_target_verified": self._zero_velocity_target_verified,
            "zero_feedforward_effort_target_verified": (
                self._zero_feedforward_effort_target_verified
            ),
            "zero_auxiliary_target_abs_tolerance": (
                _ZERO_AUXILIARY_TARGET_ABS_TOL
            ),
            "computed_effort_peak_abs_nm": list(self._computed_effort_peak_abs),
            "applied_effort_peak_abs_nm": list(self._applied_effort_peak_abs),
            "effort_observation_count": self._effort_observation_count,
            "effort_formula_max_abs_error_nm": self._effort_formula_max_abs_error,
            "effort_clip_max_abs_error_nm": self._effort_clip_max_abs_error,
            "effort_clip_count": self._effort_clip_count,
            "effort_command_source": (
                "articulation_data_computed_and_applied_torque_torch"
            ),
            "effort_formula_validation_abs_tolerance_nm": (
                _EFFORT_FORMULA_VALIDATION_ABS_TOL
            ),
            "effort_clip_validation_abs_tolerance_nm": (
                _EFFORT_CLIP_VALIDATION_ABS_TOL
            ),
            "contact_check_performed": False,
            "contact_observation_capability": "not_evaluated",
        }
        if self._legacy_joint_friction_intervention_evidence:
            provenance["legacy_joint_friction_intervention"] = _json_clone(
                self._legacy_joint_friction_intervention_evidence
            )
        provenance.update(self._target_sequence_provenance())
        return provenance

    def _release_runtime(self) -> None:
        context_manager = self._context_manager
        self._context_manager = None
        release_errors: list[str] = []
        source_path = self._legacy_joint_friction_source_path
        try:
            if self._legacy_joint_friction_override_layer is not None:
                try:
                    if self._simulation is None:
                        raise RuntimeError(
                            "live stage is unavailable for intervention cleanup"
                        )
                    self._cleanup_legacy_joint_friction_intervention(
                        self._simulation.stage
                    )
                except Exception as error:
                    self._legacy_joint_friction_intervention_evidence[
                        "cleanup_status"
                    ] = "error"
                    release_errors.append(
                        f"cleanup={type(error).__name__}: {error}"
                    )
            try:
                if context_manager is not None:
                    context_manager.__exit__(None, None, None)
            except Exception as error:
                release_errors.append(
                    f"context_exit={type(error).__name__}: {error}"
                )
            if source_path is not None:
                try:
                    after_sha256 = sha256(source_path.read_bytes()).hexdigest()
                    before_sha256 = self._legacy_joint_friction_intervention_evidence.get(
                        "source_asset_sha256_before"
                    )
                    unchanged = after_sha256 == before_sha256
                    self._legacy_joint_friction_intervention_evidence[
                        "source_asset_sha256_after_cleanup"
                    ] = after_sha256
                    self._legacy_joint_friction_intervention_evidence[
                        "source_asset_sha256_unchanged"
                    ] = unchanged
                    if not unchanged:
                        release_errors.append("source_asset_sha256_changed")
                except Exception as error:
                    release_errors.append(
                        f"source_hash={type(error).__name__}: {error}"
                    )
        finally:
            self._runtime = {}
            self._simulation = None
            self._articulation = None
            self._source_path = None
            self._dt = 0.0
            self._step_index = 0
            self._side = ""
            self._joint_names = ()
            self._frame_names = ()
            self._backend_joint_names = ()
            self._backend_frame_names = ()
            self._joint_mapping = ()
            self._frame_mapping = ()
            self._position_targets = {}
            self._target_tensor = None
            self._target_readback_count = 0
            self._target_readback_max_abs_error = 0.0
            self._target_readback_values = ()
            self._canonical_target_readback_max_abs_error = 0.0
            self._target_nonzero_readback_observed = False
            self._scheduled_target_sequence_digest = None
            self._requested_target_sequence_digest = None
            self._immediate_target_readback_sequence_digest = None
            self._pre_step_applied_target_readback_sequence_digest = None
            self._zero_velocity_target_verified = False
            self._zero_feedforward_effort_target_verified = False
            self._controller = None
            self._controller_stiffness = ()
            self._controller_damping = ()
            self._controller_effort_limit = ()
            self._controller_effort_limit_sim = ()
            self._controller_armature = ()
            self._controller_static_friction = ()
            self._controller_dynamic_friction = ()
            self._controller_viscous_friction = ()
            self._backend_drive_stiffness = ()
            self._backend_drive_damping = ()
            self._backend_armature = ()
            self._backend_friction_properties = ()
            self._tensor_bindings = {}
            self._backend_joint_dynamics = {}
            self._physics_solver_contract = {}
            self._effective_parameter_snapshot = {}
            self._computed_effort_peak_abs = ()
            self._applied_effort_peak_abs = ()
            self._effort_observation_count = 0
            self._effort_formula_max_abs_error = 0.0
            self._effort_clip_max_abs_error = 0.0
            self._effort_clip_count = 0
            self._simulation_contract = {}
            self._simulation_contract_v2 = {}
            self._tensor_binding_shapes = {}
            self._open_gravity = (0.0, 0.0, 0.0)
            self._validated_legacy_joint_friction_intervention = None
            self._legacy_joint_friction_override_layer = None
            self._legacy_joint_friction_session_layer = None
            self._legacy_joint_friction_original_session_sublayers = ()
            self._legacy_joint_friction_override_records = {}
            self._legacy_joint_friction_source_path = None
        if release_errors:
            raise RuntimeError("OVPhysX runtime release failed: " + "; ".join(release_errors))


def load_manifest_json(path: str | Path) -> object:
    """Read a probe manifest without interpreting it as executable input."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["OVPhysXAdapter", "load_manifest_json", "probe_ovphysx"]
