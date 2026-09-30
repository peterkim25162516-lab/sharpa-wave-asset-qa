"""Backend cores for the synthetic Contacts Gate C0 fixture.

The module is deliberately safe to import in a CPU-only Python process.  In
particular, MuJoCo, OpenUSD, Torch, Isaac Lab, and OVPhysX are all imported
only when the corresponding adapter is opened.  The C0 fixture is authored in
an in-memory model/stage overlay; canonical MJCF and USD files are never edit
targets.

The two backends share one public operation::

    run_contact_case(hand, scenario, case, *, dt_s, asset_root, device=None)

``hand``, ``scenario``, and ``case`` are intentionally duck typed so that the
backend core does not own the versioned manifest contract.  Results are
constructed through :mod:`wave_asset_qa.contact.records` at call time.  A
JSON-ready mapping is returned only while that module is not yet installed,
which keeps this leaf module usable during staged local development.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from wave_asset_qa.adapters.base import AdapterCapabilityError

if TYPE_CHECKING:  # pragma: no cover - typing only; runtime import is delayed.
    from wave_asset_qa.contact.records import ContactRun
else:
    ContactRun = Any


ALLOWED_PAIR_ID = "synthetic_index_probe__static_box"
CONTACT_C0_MANIFEST_ID = "wavesimparity-contact-c0"
CONTACT_C0_PROFILE_ID = "synthetic_sphere_box_v1"
PAIR_ACTIVE_SOURCE = "direct_filtered_selected_pair_force_norm_n"
PAIR_ACTIVE_FORCE_THRESHOLD_N = 1.0e-4
PROBE_GEOM_NAME = "waveqa_c0_synthetic_index_probe"
TARGET_GEOM_NAME = "waveqa_c0_static_box"
PAIR_NAME = "waveqa_c0_synthetic_index_probe__static_box"
PROBE_LOCAL_CENTER_M = (0.025, 0.0, 0.0)
PROBE_RADIUS_M = 0.005
TARGET_CENTER_WORLD_M = (0.0, 0.0, 0.120)
TARGET_HALF_EXTENTS_M = (0.150, 0.150, 0.010)
TARGET_TOP_WORLD_M = 0.130
FRICTION_COEFFICIENT = 0.0
RESTITUTION_COEFFICIENT = 0.0
GRAVITY_M_S2 = (0.0, 0.0, 0.0)
MUJOCO_CONDIM = 1
_MUJOCO_SOLREF_NO_RESTITUTION = (0.02, 1.0)
_FORBIDDEN_MODULE_PREFIXES = (
    "omni.kit",
    "omni.renderer",
    "omni.replicator",
    "omni.syntheticdata",
    "isaacsim",
)
_MISSING = object()
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHYSX_CCD_HELPER_PATH_ENV = "WAVEQA_PHYSX_CCD_HELPER_PATH"
_PHYSX_CCD_HELPER_SHA256_ENV = "WAVEQA_PHYSX_CCD_HELPER_SHA256"
_PHYSX_5_9_0_VERSION = 0x05090000
_PHYSX_SCENE_CCD_MASK = 1 << 1
_PHYSX_RIGID_BODY_CCD_MASK = 1 << 2
_PHYSX_MASS_PROPERTIES_VALUE_COUNT = 11
_PHYSX_MASS_QUATERNION_NORM_ABS_TOLERANCE = 1.0e-5
_PHYSX_NATIVE_CCD_READBACK_SOURCE = (
    "physx_5_9_0_native_flags_via_ovphysx_get_physx_ptr_post_first_reset"
)
_PHYSX_NATIVE_MASS_READBACK_SOURCE = (
    "physx_5_9_0_native_mass_properties_via_ovphysx_get_physx_ptr_"
    "post_first_reset"
)


class ContactAdapterError(RuntimeError):
    """A C0 fixture, observation, or lifecycle invariant was violated."""


def _load_physx_ccd_helper() -> tuple[object, str]:
    """Load the run-bound PhysX 5.9 reader without importing it on CPU paths."""

    import ctypes

    raw_path = os.environ.get(_PHYSX_CCD_HELPER_PATH_ENV)
    expected_sha256 = os.environ.get(_PHYSX_CCD_HELPER_SHA256_ENV)
    if not raw_path or not expected_sha256 or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise AdapterCapabilityError(
            "ovphysx",
            "native_physx_ccd_readback",
            "the native PhysX CCD helper path and SHA-256 must be bound by the launcher",
        )
    candidate = Path(raw_path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ContactAdapterError("native PhysX CCD helper must be an absolute regular file")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ContactAdapterError("native PhysX CCD helper is unavailable") from error
    if resolved != candidate or not resolved.is_file():
        raise ContactAdapterError("native PhysX CCD helper path is not canonical")
    actual_sha256 = sha256(resolved.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ContactAdapterError("native PhysX CCD helper SHA-256 differs")
    try:
        library = ctypes.CDLL(str(resolved))
        for name in (
            "waveqa_physx_version",
            "waveqa_scene_ccd_mask",
            "waveqa_rigid_body_ccd_mask",
            "waveqa_scene_flags",
            "waveqa_rigid_body_flags",
            "waveqa_mass_properties_value_count",
            "waveqa_rigid_body_mass_properties",
        ):
            symbol = getattr(library, name)
            symbol.restype = ctypes.c_uint32
        for name in (
            "waveqa_physx_version",
            "waveqa_scene_ccd_mask",
            "waveqa_rigid_body_ccd_mask",
            "waveqa_mass_properties_value_count",
        ):
            getattr(library, name).argtypes = []
        library.waveqa_scene_flags.argtypes = [ctypes.c_void_p]
        library.waveqa_rigid_body_flags.argtypes = [ctypes.c_void_p]
        library.waveqa_rigid_body_mass_properties.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint32,
        ]
    except (AttributeError, OSError) as error:
        raise AdapterCapabilityError(
            "ovphysx",
            "native_physx_ccd_readback",
            f"failed to load the pinned PhysX flag reader: {type(error).__name__}: {error}",
        ) from error
    if int(library.waveqa_physx_version()) != _PHYSX_5_9_0_VERSION:
        raise ContactAdapterError("native CCD helper PhysX version is not 5.9.0")
    if (
        int(library.waveqa_scene_ccd_mask()) != _PHYSX_SCENE_CCD_MASK
        or int(library.waveqa_rigid_body_ccd_mask()) != _PHYSX_RIGID_BODY_CCD_MASK
    ):
        raise ContactAdapterError("native CCD helper flag masks differ from PhysX 5.9")
    if (
        int(library.waveqa_mass_properties_value_count())
        != _PHYSX_MASS_PROPERTIES_VALUE_COUNT
    ):
        raise ContactAdapterError("native PhysX mass-property layout differs")
    return library, actual_sha256


def _register_physx_schemas_before_stage(runtime: dict[str, Any]) -> None:
    """Register and prove the two C0 PhysX APIs before stage construction."""

    try:
        manager_module = import_module(
            "isaaclab_ovphysx.physics.ovphysx_manager"
        )
        manager_cls = manager_module.OvPhysxManager
        ensure_registered = manager_cls._ensure_physx_schemas_registered
        if not callable(ensure_registered):
            raise TypeError("schema registration hook is not callable")
        ensure_registered()
        registry = runtime["Usd"].SchemaRegistry()
        requirements = {
            "PhysxSceneAPI": "physxScene:enableCCD",
            "PhysxRigidBodyAPI": "physxRigidBody:enableCCD",
        }
        for schema_name, property_name in requirements.items():
            definition = registry.FindAppliedAPIPrimDefinition(schema_name)
            property_names = (
                tuple(str(value) for value in definition.GetPropertyNames())
                if definition
                else ()
            )
            if (
                not definition
                or str(registry.GetSchemaKind(schema_name)) != "SingleApplyAPI"
                or property_name not in property_names
            ):
                raise RuntimeError(
                    f"{schema_name} is not a registered SingleApplyAPI with "
                    f"{property_name}"
                )
    except Exception as error:
        raise AdapterCapabilityError(
            "ovphysx",
            "physx_schema_registration_before_stage",
            (
                "failed to register and verify the pinned PhysxSchema APIs before "
                f"stage construction: {type(error).__name__}: {error}"
            ),
        ) from error
    runtime["ov_manager_cls"] = manager_cls


class _OVRuntimeBridge(Protocol):
    """Small injectable boundary around the pinned kit-less OV stack.

    A bridge must spawn the articulation but must not call the first physics
    reset in ``begin``.  This lets the adapter author and verify the USD
    overlay and instantiate the filtered sensor before any PhysX cooking.
    """

    reset_count: int

    def begin(
        self,
        *,
        source: Path,
        hand: object,
        dt_s: float,
        device: str,
    ) -> None: ...

    def author_contact_overlay(
        self,
        *,
        side: str,
        pair_collision_enabled: bool,
    ) -> Mapping[str, Any]: ...

    def create_filtered_contact_sensor(
        self,
        *,
        body_prim_path: str,
        target_prim_path: str,
    ) -> object: ...

    def reset(self, *, hand: object) -> None: ...

    def runtime_mass_properties(self, *, body_name: str) -> Mapping[str, Any]: ...

    def runtime_ccd_readback(self) -> Mapping[str, Any]: ...

    def set_position_targets(self, targets: Mapping[str, float]) -> None: ...

    def step(self, *, dt_s: float) -> None: ...

    def state(self, *, hand: object) -> Mapping[str, Any]: ...

    def filtered_force_matrix(self, sensor: object) -> object: ...

    def close(self) -> None: ...


def _field(source: object, *names: str, default: object = _MISSING) -> object:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    if default is _MISSING:
        raise ValueError(
            "required field is missing (accepted names: " + ", ".join(names) + ")"
        )
    return default


def _plain(value: object) -> object:
    return getattr(value, "value", value)


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= 0")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return numeric


def _names(source: object, *fields: str) -> tuple[str, ...]:
    value = _field(source, *fields, default=())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{fields[0]} must be an ordered sequence of names")
    result = tuple(str(item) for item in value)
    if len(result) != len(set(result)) or any(not item for item in result):
        raise ValueError(f"{fields[0]} must contain unique non-empty names")
    return result


def _hand_side(hand: object) -> str:
    side = str(_plain(_field(hand, "side", "hand", default=""))).lower()
    if side not in {"left", "right"}:
        raise ValueError("hand side must be 'left' or 'right'")
    return side


def _case_condition(case: object) -> str:
    value = str(_plain(_field(case, "condition", default=""))).lower()
    if value not in {"contact", "sham"}:
        raise ValueError("contact case condition must be 'contact' or 'sham'")
    return value


def _pair_collision_enabled(case: object) -> bool:
    direct = _field(case, "pair_collision_enabled", default=None)
    if direct is not None:
        if not isinstance(direct, bool):
            raise ValueError("case.pair_collision_enabled must be boolean")
        enabled = direct
    else:
        enabled = _case_condition(case) == "contact"
    if enabled != (_case_condition(case) == "contact"):
        raise ValueError(
            "contact/sham condition must differ only through pair_collision_enabled"
        )
    return enabled


def _case_id(case: object) -> str:
    value = _field(case, "case_id", "id", default=None)
    if value is None:
        sim = str(_plain(_field(case, "simulator", "sim", default="unknown")))
        hand = str(_plain(_field(case, "hand", "side", default="unknown")))
        scenario = str(_field(case, "scenario_id", default="unknown"))
        condition = _case_condition(case)
        variant = str(_plain(_field(case, "variant", default="base")))
        repeat = int(_field(case, "repeat", "repeat_index", default=1))
        return f"{sim}.{hand}.{scenario}.{condition}.{variant}.r{repeat:02d}"
    if not isinstance(value, str) or not value:
        raise ValueError("case id must be a non-empty string")
    return value


def _json_ready_case(case: object) -> Mapping[str, Any]:
    to_dict = getattr(case, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            source = dict(value)
        else:
            source = {}
    elif isinstance(case, Mapping):
        source = dict(case)
    else:
        source = {
            name: getattr(case, name)
            for name in (
                "case_id",
                "simulator",
                "sim",
                "hand",
                "scenario_id",
                "condition",
                "timestep_variant",
                "variant",
                "repeat_index",
                "repeat",
                "dt_s",
                "model_path",
            )
            if hasattr(case, name)
        }
    simulator = source.get("simulator", source.get("sim"))
    timestep_variant = source.get("timestep_variant", source.get("variant"))
    repeat_index = source.get("repeat_index", source.get("repeat"))
    result = {
        "case_id": str(source.get("case_id", _case_id(case))),
        "simulator": str(_plain(simulator)),
        "hand": str(_plain(source.get("hand"))),
        "scenario_id": str(source.get("scenario_id")),
        "condition": _case_condition(case),
        "timestep_variant": str(_plain(timestep_variant)),
        "repeat_index": int(repeat_index),
        "dt_s": float(source.get("dt_s")),
        "model_path": str(source.get("model_path")),
    }
    if result["case_id"] != _case_id(case):
        raise ValueError("case.case_id differs from its frozen components")
    return result


def _model_path(
    hand: object,
    case: object,
    *,
    backend: str,
    asset_root: str | Path,
) -> Path:
    raw = _field(case, "model_path", default=None)
    if raw is None:
        paths = _field(hand, "model_paths")
        raw = _field(paths, backend)
    path = Path(str(raw)).expanduser()
    root = Path(asset_root).expanduser().resolve()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{backend} model path escapes asset_root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"{backend} model does not exist: {path}")
    return path


def _scenario_steps(scenario: object, *, dt_s: float) -> int:
    dt = _finite_positive(dt_s, name="dt_s")
    duration = _finite_positive(
        _field(scenario, "duration_s"), name="scenario.duration_s"
    )
    rounded = round(duration / dt)
    if not math.isclose(rounded * dt, duration, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("scenario duration must be an integer multiple of dt_s")

    # ``ContactScenarioSpec.steps`` describes only its frozen base timestep.
    # Treat it as a consistency assertion, never as the requested case count:
    # halving the case dt must keep the same physical duration and double N.
    base_steps = _field(scenario, "steps", default=None)
    if base_steps is not None:
        if (
            isinstance(base_steps, bool)
            or not isinstance(base_steps, int)
            or base_steps <= 0
        ):
            raise ValueError("scenario.steps must be a positive integer")
        base_dt = _finite_positive(
            _field(scenario, "dt_s"), name="scenario.dt_s"
        )
        if not math.isclose(
            base_steps * base_dt,
            duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "scenario.steps contradicts scenario duration and base dt"
            )
    return rounded


def _canonical_sample_time_s(
    *,
    step_index: int,
    requested_steps: int,
    dt_s: float,
    duration_s: float,
) -> float:
    if (
        isinstance(requested_steps, bool)
        or not isinstance(requested_steps, int)
        or requested_steps <= 0
    ):
        raise ValueError("requested_steps must be a positive integer")
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or step_index > requested_steps
    ):
        raise ValueError("sample step must lie within the requested interval count")
    dt = _finite_positive(dt_s, name="dt_s")
    duration = _finite_positive(duration_s, name="scenario.duration_s")
    if not math.isclose(
        requested_steps * dt,
        duration,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("requested steps, dt, and duration are inconsistent")
    # The integer grid is authoritative.  Snapping the terminal sample to the
    # frozen duration avoids exposing integrator accumulation roundoff such as
    # MuJoCo's 7.000000000000672 after 7,000 increments of 0.001 s.
    return duration if step_index == requested_steps else step_index * dt


def _canonical_targets(
    scenario: object,
    joint_names: tuple[str, ...],
    *,
    step_index: int,
    dt_s: float,
) -> dict[str, float]:
    try:
        module = import_module("wave_asset_qa.contact.scenarios")
        function = getattr(module, "canonical_contact_position_targets")
    except (AttributeError, ImportError) as error:
        raise ContactAdapterError(
            "canonical contact schedule is unavailable; "
            "wave_asset_qa.contact.scenarios must define "
            "canonical_contact_position_targets"
        ) from error
    raw = function(
        scenario,
        joint_names,
        step_index=step_index,
        dt_s=dt_s,
    )
    if not isinstance(raw, Mapping) or set(raw) != set(joint_names):
        raise ContactAdapterError(
            "canonical contact target schedule must exactly cover all joints"
        )
    result: dict[str, float] = {}
    for name in joint_names:
        value = float(raw[name])
        if not math.isfinite(value):
            raise FloatingPointError(
                f"canonical contact target is non-finite for {name!r}"
            )
        result[name] = value
    return result


def _observation_threshold(
    scenario: object,
    case: object,
    explicit: float | None,
) -> float:
    if explicit is not None:
        return _finite_nonnegative(explicit, name="observation_force_threshold_n")
    containers = (case, scenario)
    for container in containers:
        direct = _field(
            container,
            "observation_force_threshold_n",
            "force_threshold_n",
            default=None,
        )
        if direct is not None:
            return _finite_nonnegative(direct, name="observation force threshold")
        nested = _field(container, "observation", "contact_observation", default=None)
        if nested is not None:
            value = _field(
                nested,
                "force_threshold_n",
                "pair_active_force_threshold_n",
                "normal_force_threshold_n",
                default=None,
            )
            if value is not None:
                return _finite_nonnegative(
                    value, name="observation force threshold"
                )
    return PAIR_ACTIVE_FORCE_THRESHOLD_N


def _analytic_signed_gap(probe_center_world_m: Sequence[float]) -> float:
    """Portable C0 gap: sphere bottom relative to the box top plane."""

    if len(probe_center_world_m) != 3:
        raise ValueError("probe center must contain three values")
    center_z = float(probe_center_world_m[2])
    gap = center_z - TARGET_TOP_WORLD_M - PROBE_RADIUS_M
    if not math.isfinite(gap):
        raise FloatingPointError("analytic contact gap is non-finite")
    return gap


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _canonical_manifest_identity() -> tuple[str, str, str]:
    """Read the checked-in frozen manifest used by formal source workers."""

    project_root = Path(__file__).resolve().parents[3]
    path = project_root / "configs" / "parity" / "contact_c0.json"
    if not path.is_file():
        raise ContactAdapterError(f"frozen Contact C0 manifest is absent: {path}")
    contracts = import_module("wave_asset_qa.contact.scenarios")
    manifest = contracts.load_contact_manifest(path)
    digest = str(contracts.contact_manifest_sha256(manifest))
    manifest_id = str(getattr(manifest, "manifest_id", ""))
    if manifest_id != CONTACT_C0_MANIFEST_ID:
        raise ContactAdapterError("checked-in Contact C0 manifest id drifted")
    provenance = getattr(manifest, "provenance")
    return digest, str(provenance.commit), str(provenance.asset_git_tree)


def _closed_fixture_readback(
    *,
    side: str,
    detailed: Mapping[str, Any],
) -> dict[str, Any]:
    """Project private backend evidence into the closed public run schema."""

    native = detailed.get("native_collision_disabled_inventory")
    if not isinstance(native, Mapping) or native.get("all_disabled") is not True:
        raise ContactAdapterError("native collision disable inventory is incomplete")
    geometry = detailed.get("geometry")
    collision = detailed.get("collision")
    material = detailed.get("material")
    mass_properties = detailed.get("mass_properties")
    if (
        not isinstance(geometry, Mapping)
        or not isinstance(collision, Mapping)
        or not isinstance(material, Mapping)
        or not isinstance(mass_properties, Mapping)
        or mass_properties.get("preserved") is not True
        or collision.get("ccd_enabled") is not False
    ):
        raise ContactAdapterError(
            "fixture geometry/material/collision readback is incomplete"
        )
    raw_items = native.get("items")
    if not isinstance(raw_items, list):
        raise ContactAdapterError("native collision inventory items are missing")
    native_count = int(native.get("count", -1))
    if native_count != len(raw_items) or native_count <= 0:
        raise ContactAdapterError("native collision inventory count differs")
    disabled_count = 0
    for item in raw_items:
        if not isinstance(item, Mapping):
            raise ContactAdapterError("native collision inventory item is invalid")
        if item.get("disabled") is True or item.get(
            "effective_collision_enabled"
        ) is False:
            disabled_count += 1
    if disabled_count != native_count:
        raise ContactAdapterError("native collision inventory is not fully disabled")
    collision_inventory_preimage = detailed.get(
        "collision_inventory_hash_preimage",
        {
            "native_collision_items": raw_items,
            "synthetic_enabled_shapes": [
                "synthetic_index_probe",
                "static_box",
            ],
        },
    )
    return {
        "performed": True,
        "profile_id": CONTACT_C0_PROFILE_ID,
        "probe_parent_frame_name": f"{side}_index_DP",
        "probe_local_center_m": list(PROBE_LOCAL_CENTER_M),
        "probe_radius_m": PROBE_RADIUS_M,
        "target_world_center_m": list(TARGET_CENTER_WORLD_M),
        "target_half_extents_m": list(TARGET_HALF_EXTENTS_M),
        "target_top_surface_z_m": TARGET_TOP_WORLD_M,
        "static_friction": FRICTION_COEFFICIENT,
        "dynamic_friction": FRICTION_COEFFICIENT,
        "restitution": RESTITUTION_COEFFICIENT,
        "native_hand_collisions_enabled": False,
        "self_collisions_enabled": False,
        "ccd_enabled": bool(collision["ccd_enabled"]),
        "allowed_pair_id": ALLOWED_PAIR_ID,
        "mujoco_condim": MUJOCO_CONDIM,
        "native_collision_prim_count": native_count,
        "native_collision_disabled_count": disabled_count,
        "enabled_collision_shape_count": 2,
        "collision_inventory_sha256": _sha256_json(
            collision_inventory_preimage
        ),
        "mass_properties_preserved": bool(mass_properties["preserved"]),
    }


def _construct_contact_run(payload: Mapping[str, Any]) -> ContactRun:
    """Construct the versioned record without importing it at module import."""

    try:
        records = import_module("wave_asset_qa.contact.records")
    except ImportError:
        # Temporary development seam explicitly permitted by the C0 task.  It
        # disappears automatically as soon as records.py is present.
        return dict(payload)
    run_type = getattr(records, "ContactRun", None)
    if run_type is None:
        return dict(payload)
    from_dict = getattr(run_type, "from_dict", None)
    if callable(from_dict):
        return from_dict(payload)
    return run_type(**payload)


def _formal_source_identity(
    source_revision: str | None,
    source_tree: str | None,
    fresh_process_identity_sha256: str | None,
) -> tuple[str, str, str]:
    if (
        not isinstance(source_revision, str)
        or _GIT_OBJECT_RE.fullmatch(source_revision) is None
    ):
        raise ContactAdapterError(
            "formal run requires a lowercase 40-character source_revision"
        )
    if not isinstance(source_tree, str) or _GIT_OBJECT_RE.fullmatch(source_tree) is None:
        raise ContactAdapterError(
            "formal run requires a lowercase 40-character source_tree"
        )
    if (
        not isinstance(fresh_process_identity_sha256, str)
        or _SHA256_RE.fullmatch(fresh_process_identity_sha256) is None
    ):
        raise ContactAdapterError(
            "formal run requires a lowercase SHA-256 fresh process identity"
        )
    return source_revision, source_tree, fresh_process_identity_sha256


def _forbidden_module_inventory() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _FORBIDDEN_MODULE_PREFIXES
        )
    )


def _assert_no_forbidden_modules(*, phase: str) -> None:
    forbidden = _forbidden_module_inventory()
    if forbidden:
        raise AdapterCapabilityError(
            "ovphysx",
            "kitless_headless_only",
            f"forbidden Kit/renderer/camera modules were present {phase}: "
            + ", ".join(forbidden[:10]),
        )


def _normalize_filtered_force_vector(value: object) -> tuple[float, float, float]:
    """Return the sole 3-vector from a 1x1 filtered force matrix.

    Leading singleton dimensions are accepted because Isaac Lab revisions use
    either ``(env, body, filter, xyz)`` or ``(env, filter, xyz)``.  Any other
    coverage is rejected rather than summed or silently selecting an entry.
    """

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        array = np.asarray(value.numpy(), dtype=float)
    else:
        array = np.asarray(value, dtype=float)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (3,):
        raise ContactAdapterError(
            "filtered force_matrix must resolve to exactly one 3-vector; "
            f"got shape {tuple(np.asarray(value).shape)}"
        )
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("OVPhysX filtered force vector is non-finite")
    return (float(array[0]), float(array[1]), float(array[2]))


class MuJoCoContactAdapter:
    """CPU MuJoCo implementation using an in-memory :class:`MjSpec` overlay."""

    backend_name = "mujoco"

    def __init__(
        self,
        *,
        source_revision: str | None = None,
        source_tree: str | None = None,
        fresh_process_identity_sha256: str | None = None,
    ) -> None:
        self._formal_source_revision = source_revision
        self._formal_source_tree = source_tree
        self._formal_fresh_process_identity_sha256 = (
            fresh_process_identity_sha256
        )
        self._last_private_evidence: dict[str, Any] = {}
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._source: Path | None = None
        self._source_sha256: str | None = None
        self._dt_s = 0.0
        self._side = ""
        self._condition = ""
        self._pair_collision_enabled = False
        self._joint_names: tuple[str, ...] = ()
        self._frame_names: tuple[str, ...] = ()
        self._joint_qpos_address: dict[str, int] = {}
        self._joint_dof_address: dict[str, int] = {}
        self._joint_actuator: dict[str, int] = {}
        self._position_targets: dict[str, float] = {}
        self._probe_geom_id = -1
        self._target_geom_id = -1
        self._fixture_readback: dict[str, Any] = {}

    @property
    def is_open(self) -> bool:
        return self._model is not None and self._data is not None

    @property
    def fixture_readback(self) -> Mapping[str, Any]:
        if not self.is_open:
            raise ContactAdapterError("MuJoCo contact adapter is not open")
        return self._fixture_readback

    @property
    def last_private_evidence(self) -> Mapping[str, Any]:
        """Detached hash preimages retained after ``close`` for the worker."""

        return deepcopy(self._last_private_evidence)

    def open_contact_case(
        self,
        hand: object,
        scenario: object,
        case: object,
        *,
        dt_s: float,
        asset_root: str | Path,
    ) -> None:
        del scenario  # The frozen geometry is independent of the trajectory.
        if self.is_open:
            raise ContactAdapterError("MuJoCo contact adapter is already open")
        self._last_private_evidence = {}
        dt = _finite_positive(dt_s, name="dt_s")
        side = _hand_side(hand)
        condition = _case_condition(case)
        pair_enabled = _pair_collision_enabled(case)
        source = _model_path(hand, case, backend="mujoco", asset_root=asset_root)
        source_before = sha256(source.read_bytes()).hexdigest()

        try:
            mujoco = import_module("mujoco")
        except ImportError as error:
            raise AdapterCapabilityError(
                "mujoco",
                "python_package",
                "the 'mujoco' package is not installed",
                hint="Install the project's simulation extra.",
            ) from error

        try:
            index_body_name = f"{side}_index_DP"
            baseline_model = mujoco.MjModel.from_xml_path(str(source))
            baseline_body_id = int(
                mujoco.mj_name2id(
                    baseline_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    index_body_name,
                )
            )
            if baseline_body_id < 0:
                raise ContactAdapterError(
                    f"MuJoCo model must contain exactly named body {index_body_name!r}"
                )
            baseline_mass_properties = {
                "body_mass": float(baseline_model.body_mass[baseline_body_id]),
                "body_inertia": [
                    float(value) for value in baseline_model.body_inertia[baseline_body_id]
                ],
                "body_ipos": [
                    float(value) for value in baseline_model.body_ipos[baseline_body_id]
                ],
                "body_iquat": [
                    float(value) for value in baseline_model.body_iquat[baseline_body_id]
                ],
            }
            spec = mujoco.MjSpec.from_file(str(source))
            native_inventory: list[dict[str, Any]] = []
            native_geoms = tuple(spec.geoms)
            for ordinal, geom in enumerate(native_geoms):
                parent = getattr(geom, "parent", None)
                parent_name = str(getattr(parent, "name", "")) or None
                name = str(getattr(geom, "name", "")) or None
                before_contype = int(geom.contype)
                before_conaffinity = int(geom.conaffinity)
                geom.contype = 0
                geom.conaffinity = 0
                native_inventory.append(
                    {
                        "ordinal": ordinal,
                        "name": name,
                        "parent_body": parent_name,
                        "original_contype": before_contype,
                        "original_conaffinity": before_conaffinity,
                        "effective_contype": int(geom.contype),
                        "effective_conaffinity": int(geom.conaffinity),
                        "disabled": int(geom.contype) == 0
                        and int(geom.conaffinity) == 0,
                    }
                )

            index_body = spec.body(index_body_name)
            if index_body is None or str(getattr(index_body, "name", "")) != index_body_name:
                raise ContactAdapterError(
                    f"MuJoCo model must contain exactly named body {index_body_name!r}"
                )

            index_body.add_geom(
                name=PROBE_GEOM_NAME,
                type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
                pos=PROBE_LOCAL_CENTER_M,
                size=(PROBE_RADIUS_M, 0.0, 0.0),
                contype=0,
                conaffinity=0,
                condim=MUJOCO_CONDIM,
                friction=(0.0, 0.0, 0.0),
                solref=_MUJOCO_SOLREF_NO_RESTITUTION,
                density=0.0,
            )
            spec.worldbody.add_geom(
                name=TARGET_GEOM_NAME,
                type=int(mujoco.mjtGeom.mjGEOM_BOX),
                pos=TARGET_CENTER_WORLD_M,
                size=TARGET_HALF_EXTENTS_M,
                contype=0,
                conaffinity=0,
                condim=MUJOCO_CONDIM,
                friction=(0.0, 0.0, 0.0),
                solref=_MUJOCO_SOLREF_NO_RESTITUTION,
                density=0.0,
            )
            if pair_enabled:
                spec.add_pair(
                    name=PAIR_NAME,
                    geomname1=PROBE_GEOM_NAME,
                    geomname2=TARGET_GEOM_NAME,
                    condim=MUJOCO_CONDIM,
                    friction=(0.0, 0.0, 0.0, 0.0, 0.0),
                    solref=_MUJOCO_SOLREF_NO_RESTITUTION,
                )
            spec.option.gravity = GRAVITY_M_S2
            spec.option.timestep = dt
            ccd_disable_mask = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(
                mujoco.mjtDisableBit.mjDSBL_MULTICCD
            )
            spec.option.disableflags = int(spec.option.disableflags) | ccd_disable_mask
            model = spec.compile()
            data = mujoco.MjData(model)

            if sha256(source.read_bytes()).hexdigest() != source_before:
                raise ContactAdapterError("canonical MJCF changed during in-memory injection")
            if not np.all(np.asarray(model.opt.gravity, dtype=float) == 0.0):
                raise ContactAdapterError("compiled MuJoCo gravity readback is not zero")
            if float(model.opt.timestep) != dt:
                raise ContactAdapterError("compiled MuJoCo timestep readback differs")
            compiled_ccd_enabled = (
                int(model.opt.disableflags) & ccd_disable_mask
            ) != ccd_disable_mask
            if compiled_ccd_enabled:
                raise ContactAdapterError("compiled MuJoCo CCD disable flags differ")

            probe_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROBE_GEOM_NAME)
            )
            target_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TARGET_GEOM_NAME)
            )
            if probe_id < 0 or target_id < 0 or probe_id == target_id:
                raise ContactAdapterError("synthetic MuJoCo geoms did not compile uniquely")
            compiled_body_id = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, index_body_name
                )
            )
            compiled_mass_properties = {
                "body_mass": float(model.body_mass[compiled_body_id]),
                "body_inertia": [
                    float(value) for value in model.body_inertia[compiled_body_id]
                ],
                "body_ipos": [
                    float(value) for value in model.body_ipos[compiled_body_id]
                ],
                "body_iquat": [
                    float(value) for value in model.body_iquat[compiled_body_id]
                ],
            }
            if compiled_mass_properties != baseline_mass_properties:
                raise ContactAdapterError(
                    "synthetic MuJoCo probe changed index_DP mass properties"
                )

            native_count = len(native_inventory)
            compiled_native_masks: list[dict[str, Any]] = []
            for ordinal, record in enumerate(native_inventory):
                contype = int(model.geom_contype[ordinal])
                conaffinity = int(model.geom_conaffinity[ordinal])
                compiled_native_masks.append(
                    {
                        **record,
                        "compiled_geom_id": ordinal,
                        "compiled_contype": contype,
                        "compiled_conaffinity": conaffinity,
                        "disabled": contype == 0 and conaffinity == 0,
                    }
                )
            if native_count == 0 or not all(
                bool(item["disabled"]) for item in compiled_native_masks
            ):
                raise ContactAdapterError("not all native MuJoCo collisions were disabled")
            if int(model.geom_contype[probe_id]) != 0 or int(
                model.geom_conaffinity[probe_id]
            ) != 0:
                raise ContactAdapterError("synthetic probe must use explicit-pair-only masks")
            if int(model.geom_contype[target_id]) != 0 or int(
                model.geom_conaffinity[target_id]
            ) != 0:
                raise ContactAdapterError("synthetic target must use explicit-pair-only masks")
            expected_pairs = 1 if pair_enabled else 0
            if int(model.npair) != expected_pairs:
                raise ContactAdapterError(
                    "MuJoCo explicit pair inventory differs from contact/sham mask"
                )
            if pair_enabled:
                pair_ids = {
                    int(model.pair_geom1[0]),
                    int(model.pair_geom2[0]),
                }
                if pair_ids != {probe_id, target_id}:
                    raise ContactAdapterError("compiled explicit pair targets wrong geoms")
                if int(model.pair_dim[0]) != MUJOCO_CONDIM:
                    raise ContactAdapterError("compiled explicit pair condim differs")
                if not np.all(np.asarray(model.pair_friction[0], dtype=float) == 0.0):
                    raise ContactAdapterError("compiled explicit pair friction is not zero")

            joint_names = _names(hand, "joint_names")
            frame_names = _names(hand, "distal_frame_names", "frame_names")
            qpos_addresses: dict[str, int] = {}
            dof_addresses: dict[str, int] = {}
            for name in joint_names:
                joint_id = int(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                )
                if joint_id < 0:
                    raise ContactAdapterError(f"MuJoCo is missing joint {name!r}")
                joint_type = int(model.jnt_type[joint_id])
                if joint_type not in {
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                }:
                    raise ContactAdapterError(f"joint {name!r} is not one-DOF")
                qpos_addresses[name] = int(model.jnt_qposadr[joint_id])
                dof_addresses[name] = int(model.jnt_dofadr[joint_id])

            actuators: dict[str, int] = {}
            for actuator_id in range(int(model.nu)):
                joint_id = int(model.actuator_trnid[actuator_id, 0])
                if joint_id < 0 or joint_id >= int(model.njnt):
                    continue
                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                if name and str(name) in qpos_addresses and str(name) not in actuators:
                    actuators[str(name)] = actuator_id
            missing_actuators = sorted(set(joint_names) - set(actuators))
            if missing_actuators:
                raise ContactAdapterError(
                    "MuJoCo joints lack actuators: " + ", ".join(missing_actuators)
                )
            for name in frame_names:
                if int(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                ) < 0:
                    raise ContactAdapterError(f"MuJoCo is missing frame {name!r}")

            mujoco.mj_resetData(model, data)
            if int(model.nu):
                data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)

            pair_friction = (
                [float(value) for value in model.pair_friction[0]]
                if pair_enabled
                else None
            )
            pair_solref = (
                [float(value) for value in model.pair_solref[0]]
                if pair_enabled
                else None
            )
            self._mujoco = mujoco
            self._model = model
            self._data = data
            self._source = source
            self._source_sha256 = source_before
            self._dt_s = dt
            self._side = side
            self._condition = condition
            self._pair_collision_enabled = pair_enabled
            self._joint_names = joint_names
            self._frame_names = frame_names
            self._joint_qpos_address = qpos_addresses
            self._joint_dof_address = dof_addresses
            self._joint_actuator = actuators
            self._position_targets = {name: 0.0 for name in joint_names}
            self._probe_geom_id = probe_id
            self._target_geom_id = target_id
            self._fixture_readback = {
                "schema_version": 1,
                "backend": "mujoco",
                "overlay": {
                    "method": "MjSpec_in_memory_pre_compile",
                    "canonical_source_edited": False,
                    "source_sha256_before": source_before,
                    "source_sha256_after": sha256(source.read_bytes()).hexdigest(),
                },
                "geometry": {
                    "probe": {
                        "canonical_id": "synthetic_index_probe",
                        "parent_frame": index_body_name,
                        "compiled_geom_id": probe_id,
                        "shape": "sphere",
                        "local_center_m": list(PROBE_LOCAL_CENTER_M),
                        "radius_m": PROBE_RADIUS_M,
                    },
                    "target": {
                        "canonical_id": "static_box",
                        "compiled_geom_id": target_id,
                        "shape": "box",
                        "center_world_m": list(TARGET_CENTER_WORLD_M),
                        "half_extents_m": list(TARGET_HALF_EXTENTS_M),
                        "top_world_m": TARGET_TOP_WORLD_M,
                        "static": True,
                    },
                },
                "collision": {
                    "allowed_pair_id": ALLOWED_PAIR_ID,
                    "condition": condition,
                    "pair_collision_enabled": pair_enabled,
                    "pair_mask_source": (
                        "MjSpec explicit pair presence; all geom bitmasks are zero"
                    ),
                    "compiled_explicit_pair_count": int(model.npair),
                    "compiled_condim": (
                        int(model.pair_dim[0]) if pair_enabled else None
                    ),
                    "probe_contype": int(model.geom_contype[probe_id]),
                    "probe_conaffinity": int(model.geom_conaffinity[probe_id]),
                    "target_contype": int(model.geom_contype[target_id]),
                    "target_conaffinity": int(model.geom_conaffinity[target_id]),
                    "ccd_enabled": compiled_ccd_enabled,
                },
                "material": {
                    "friction_coefficient": FRICTION_COEFFICIENT,
                    "restitution_coefficient": RESTITUTION_COEFFICIENT,
                    "compiled_pair_friction": pair_friction,
                    "compiled_pair_solref": pair_solref,
                    "restitution_semantics": (
                        "protocol coefficient is zero; MuJoCo uses critically "
                        "damped solref rather than a restitution coefficient"
                    ),
                },
                "mass_properties": {
                    "preserved": True,
                    "source": baseline_mass_properties,
                    "effective": compiled_mass_properties,
                    "synthetic_probe_density": 0.0,
                },
                "native_collision_disabled_inventory": {
                    "count": native_count,
                    "all_disabled": True,
                    "items": compiled_native_masks,
                },
                "solver": {
                    "dt_s": float(model.opt.timestep),
                    "gravity_m_s2": [float(value) for value in model.opt.gravity],
                },
            }
            self._fixture_readback["collision_inventory_hash_preimage"] = {
                "native_collision_items": compiled_native_masks,
                "synthetic_enabled_shapes": [
                    "synthetic_index_probe",
                    "static_box",
                ],
            }
            self._last_private_evidence = {
                "fixture_overlay": self._fixture_readback,
                "runtime_fingerprint": {
                    "backend": "mujoco",
                    "backend_version": str(getattr(mujoco, "__version__", "unknown")),
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "device": "cpu",
                    "solver": str(model.opt.solver),
                    "integrator": str(model.opt.integrator),
                    "control_path": "MjData.ctrl -> mj_step",
                    "contact_path": (
                        "explicit MjSpec pair -> data.contact -> mj_contactForce"
                    ),
                },
                "unexpected_pair_observations": [],
            }
        except Exception:
            self.close()
            raise

    def set_position_targets(self, targets: Mapping[str, float]) -> None:
        if not self.is_open:
            raise ContactAdapterError("MuJoCo contact adapter is not open")
        if set(targets) != set(self._joint_names):
            raise ValueError("position target mapping must exactly cover all joints")
        for name in self._joint_names:
            value = float(targets[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite position target for {name!r}")
            self._data.ctrl[self._joint_actuator[name]] = value
            self._position_targets[name] = value

    def step(self) -> None:
        if not self.is_open:
            raise ContactAdapterError("MuJoCo contact adapter is not open")
        self._mujoco.mj_step(self._model, self._data)
        for name in ("qpos", "qvel", "qacc", "ctrl", "xpos", "xquat"):
            if not np.all(np.isfinite(np.asarray(getattr(self._data, name)))):
                raise FloatingPointError(f"non-finite MuJoCo {name}")

    def collect_contact_observation(self) -> Mapping[str, Any]:
        """Collect the sole intended pair and fail-visible unexpected contacts."""

        if not self.is_open:
            raise ContactAdapterError("MuJoCo contact adapter is not open")
        intended_indices: list[int] = []
        unexpected: list[dict[str, Any]] = []
        intended_ids = {self._probe_geom_id, self._target_geom_id}
        for contact_index in range(int(self._data.ncon)):
            contact = self._data.contact[contact_index]
            ids = {int(contact.geom1), int(contact.geom2)}
            if ids == intended_ids:
                intended_indices.append(contact_index)
                continue
            names: list[str | None] = []
            for geom_id in (int(contact.geom1), int(contact.geom2)):
                name = self._mujoco.mj_id2name(
                    self._model, self._mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                names.append(str(name) if name else None)
            unexpected.append(
                {
                    "contact_index": contact_index,
                    "geom_ids": [int(contact.geom1), int(contact.geom2)],
                    "geom_names": names,
                    "signed_distance_m": float(contact.dist),
                }
            )

        force_records: list[dict[str, Any]] = []
        selected_pair_force_world = np.zeros(3, dtype=np.float64)
        for contact_index in intended_indices:
            contact = self._data.contact[contact_index]
            force = np.zeros(6, dtype=np.float64)
            self._mujoco.mj_contactForce(
                self._model, self._data, contact_index, force
            )
            if not np.all(np.isfinite(force)):
                raise FloatingPointError("MuJoCo mj_contactForce returned non-finite data")
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            point = np.asarray(contact.pos, dtype=float)
            # MuJoCo expresses the first three components in the contact-local
            # basis whose rows are stored in contact.frame.  Sum the selected
            # pair in world coordinates before taking the frozen force norm.
            selected_pair_force_world += frame.T @ force[:3]
            force_records.append(
                {
                    "contact_index": contact_index,
                    "geom_ids": [int(contact.geom1), int(contact.geom2)],
                    "signed_distance_m": float(contact.dist),
                    "point_world_m": [float(value) for value in point],
                    "contact_frame_row_major": [
                        float(value) for value in frame.reshape(-1)
                    ],
                    "constraint_force_local_6d": [float(value) for value in force],
                }
            )

        selected_pair_force_norm = float(
            np.linalg.norm(selected_pair_force_world)
        )
        if not math.isfinite(selected_pair_force_norm):
            raise FloatingPointError("MuJoCo selected-pair force norm is non-finite")
        if unexpected:
            observations = self._last_private_evidence.get(
                "unexpected_pair_observations"
            )
            if isinstance(observations, list):
                observations.extend(unexpected)
            raise ContactAdapterError(
                "MuJoCo observed a contact outside the sole synthetic pair"
            )

        fromto = np.zeros(6, dtype=np.float64)
        signed_gap = float(
            self._mujoco.mj_geomDistance(
                self._model,
                self._data,
                self._probe_geom_id,
                self._target_geom_id,
                1.0e6,
                fromto,
            )
        )
        if not math.isfinite(signed_gap) or not np.all(np.isfinite(fromto)):
            raise FloatingPointError("MuJoCo signed geom distance is non-finite")
        pair_active = selected_pair_force_norm > PAIR_ACTIVE_FORCE_THRESHOLD_N
        if not self._pair_collision_enabled and pair_active:
            raise ContactAdapterError(
                "sham selected-pair force exceeded the frozen activity threshold"
            )
        return {
            "pair_id": ALLOWED_PAIR_ID,
            "pair_active": pair_active,
            "pair_active_source": PAIR_ACTIVE_SOURCE,
            "pair_active_force_threshold_n": PAIR_ACTIVE_FORCE_THRESHOLD_N,
            "pair_collision_enabled": self._pair_collision_enabled,
            "signed_gap_m": signed_gap,
            "signed_gap_source": "mujoco.mj_geomDistance",
            "signed_gap_convention": "positive_separation_zero_touch_negative_penetration",
            "closest_points_world_m": [float(value) for value in fromto],
            "raw_contact_count": int(self._data.ncon),
            "intended_pair_contact_count": len(intended_indices),
            "unexpected_pairs": unexpected,
            "raw_force": {
                "status": "observed_descriptive_only",
                "source_api": "mujoco.mj_contactForce",
                "units": "N,N*m",
                "frame": "per-contact MuJoCo contact-local frame",
                "semantics": (
                    "six constraint-force components; not normalized into a "
                    "cross-backend force or impulse"
                ),
                "contacts": force_records,
                "selected_pair_force_world_n": [
                    float(value) for value in selected_pair_force_world
                ],
                "selected_pair_force_norm_n": selected_pair_force_norm,
            },
        }

    def _trace_sample(self, *, step: int, canonical_time_s: float) -> Mapping[str, Any]:
        observation = self.collect_contact_observation()
        self._mujoco.mj_kinematics(self._model, self._data)
        positions = {
            name: float(self._data.qpos[self._joint_qpos_address[name]])
            for name in self._joint_names
        }
        velocities = {
            name: float(self._data.qvel[self._joint_dof_address[name]])
            for name in self._joint_names
        }
        poses: dict[str, list[float]] = {}
        for name in self._frame_names:
            body_id = int(
                self._mujoco.mj_name2id(
                    self._model, self._mujoco.mjtObj.mjOBJ_BODY, name
                )
            )
            poses[name] = [
                *[float(value) for value in self._data.xpos[body_id]],
                *[float(value) for value in self._data.xquat[body_id]],
            ]
        probe_center = [
            float(value) for value in self._data.geom_xpos[self._probe_geom_id]
        ]
        analytic_gap = _analytic_signed_gap(probe_center)
        numeric = (
            *positions.values(),
            *velocities.values(),
            *probe_center,
            *(value for pose in poses.values() for value in pose),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError(f"non-finite MuJoCo trace state at step {step}")
        runtime_time_s = float(self._data.time)
        if not math.isclose(
            runtime_time_s,
            canonical_time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContactAdapterError(
                "MuJoCo runtime time differs from the canonical integer grid"
            )
        return {
            "step": step,
            "time_s": canonical_time_s,
            "joint_positions": positions,
            "joint_velocities": velocities,
            "frame_poses": poses,
            "position_targets": dict(self._position_targets),
            "probe_center_world_m": probe_center,
            "signed_gap_m": analytic_gap,
            "pair_active": observation["pair_active"],
            "raw_contact_count": observation["raw_contact_count"],
            "native_contact_observation": {
                "selected_pair_force_norm_n": observation["raw_force"][
                    "selected_pair_force_norm_n"
                ]
            },
        }

    def run_contact_case(
        self,
        hand: object,
        scenario: object,
        case: object,
        *,
        dt_s: float,
        asset_root: str | Path,
        device: str | None = None,
    ) -> ContactRun:
        if device not in {None, "cpu"}:
            raise ValueError("MuJoCo C0 is CPU-only; device must be None or 'cpu'")
        dt = _finite_positive(dt_s, name="dt_s")
        requested_steps = _scenario_steps(scenario, dt_s=dt)
        samples: list[Mapping[str, Any]] = []
        fixture_readback: Mapping[str, Any] = {}
        source: Path | None = None
        source_digest: str | None = None
        backend_version: str | None = None
        try:
            self.open_contact_case(
                hand,
                scenario,
                case,
                dt_s=dt,
                asset_root=asset_root,
            )
            source = self._source
            source_digest = self._source_sha256
            backend_version = str(getattr(self._mujoco, "__version__", "unknown"))
            for step_index in range(requested_steps + 1):
                targets = _canonical_targets(
                    scenario,
                    self._joint_names,
                    step_index=step_index,
                    dt_s=dt,
                )
                self.set_position_targets(targets)
                canonical_time_s = _canonical_sample_time_s(
                    step_index=step_index,
                    requested_steps=requested_steps,
                    dt_s=dt,
                    duration_s=float(_field(scenario, "duration_s")),
                )
                samples.append(
                    self._trace_sample(
                        step=step_index,
                        canonical_time_s=canonical_time_s,
                    )
                )
                if step_index < requested_steps:
                    self.step()
            fixture_readback = dict(self._fixture_readback)
            if source is None or source_digest is None:
                raise ContactAdapterError("MuJoCo provenance was not initialized")
            if sha256(source.read_bytes()).hexdigest() != source_digest:
                raise ContactAdapterError("canonical MJCF changed during C0 run")
        finally:
            self.close()

        unexpected_count = sum(
            0 for _sample in samples
        )
        del unexpected_count
        manifest_digest, asset_commit, asset_tree = _canonical_manifest_identity()
        source_revision, source_tree, fresh_identity = _formal_source_identity(
            self._formal_source_revision,
            self._formal_source_tree,
            self._formal_fresh_process_identity_sha256,
        )
        closed_fixture = _closed_fixture_readback(
            side=_hand_side(hand), detailed=fixture_readback
        )
        runtime_fingerprint = _sha256_json(
            self._last_private_evidence["runtime_fingerprint"]
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "manifest_id": CONTACT_C0_MANIFEST_ID,
            "manifest_sha256": manifest_digest,
            "case": dict(_json_ready_case(case)),
            "execution": {
                "status": "completed",
                "message": None,
                "requested_steps": requested_steps,
                "completed_steps": requested_steps,
            },
            "mapping": {
                "expected_joint_count": 22,
                "observed_joint_count": len(_names(hand, "joint_names")),
                "expected_distal_frame_count": 5,
                "observed_distal_frame_count": len(
                    _names(hand, "distal_frame_names", "frame_names")
                ),
                "probe_frame_name": f"{_hand_side(hand)}_index_DP",
                "probe_frame_mapped": True,
            },
            "fixture_readback": closed_fixture,
            "contact_observation": {
                "performed": True,
                "capability": "direct_filtered_pair_force_threshold",
                "selected_pair_id": ALLOWED_PAIR_ID,
                "pair_active_source": PAIR_ACTIVE_SOURCE,
                "pair_active_force_threshold_n": PAIR_ACTIVE_FORCE_THRESHOLD_N,
                "pair_active_record_count": len(samples),
                "missing_pair_active_count": 0,
                "filtered_sensor_body_count": 1,
                "filtered_target_count": 1,
            },
            "samples": samples,
            "provenance": {
                "backend": "mujoco",
                "source_revision": source_revision,
                "source_tree": source_tree,
                "asset_commit": asset_commit,
                "asset_git_tree": asset_tree,
                "manifest_sha256": manifest_digest,
                "fixture_overlay_sha256": _sha256_json(
                    self._last_private_evidence["fixture_overlay"]
                ),
                "runtime_fingerprint_sha256": runtime_fingerprint,
                "fresh_process_identity_sha256": fresh_identity,
            },
        }
        return _construct_contact_run(payload)

    def close(self) -> None:
        self._mujoco = None
        self._model = None
        self._data = None
        self._source = None
        self._source_sha256 = None
        self._dt_s = 0.0
        self._side = ""
        self._condition = ""
        self._pair_collision_enabled = False
        self._joint_names = ()
        self._frame_names = ()
        self._joint_qpos_address = {}
        self._joint_dof_address = {}
        self._joint_actuator = {}
        self._position_targets = {}
        self._probe_geom_id = -1
        self._target_geom_id = -1
        self._fixture_readback = {}


class OVPhysXContactAdapter:
    """Kit-less OVPhysX C0 core with a pre-first-reset USD overlay.

    The default runtime bridge imports the pinned stack lazily.  Tests can
    inject a bridge factory to exercise ordering, filtering, and observation
    semantics without importing OpenUSD, Torch, Isaac Lab, or OVPhysX.
    """

    backend_name = "ovphysx"

    def __init__(
        self,
        *,
        runtime_bridge_factory: Callable[[], _OVRuntimeBridge] | None = None,
        observation_force_threshold_n: float | None = None,
        source_revision: str | None = None,
        source_tree: str | None = None,
        fresh_process_identity_sha256: str | None = None,
    ) -> None:
        self._formal_source_revision = source_revision
        self._formal_source_tree = source_tree
        self._formal_fresh_process_identity_sha256 = (
            fresh_process_identity_sha256
        )
        self._last_private_evidence: dict[str, Any] = {}
        self._bridge_factory = runtime_bridge_factory
        self._explicit_threshold = observation_force_threshold_n
        if observation_force_threshold_n is not None:
            _finite_nonnegative(
                observation_force_threshold_n,
                name="observation_force_threshold_n",
            )
        self._bridge: _OVRuntimeBridge | None = None
        self._sensor: object | None = None
        self._source: Path | None = None
        self._source_sha256: str | None = None
        self._fixture_readback: dict[str, Any] = {}
        self._threshold_n: float | None = None
        self._pair_collision_enabled = False
        self._position_targets: dict[str, float] = {}
        self._device = ""

    @property
    def is_open(self) -> bool:
        return self._bridge is not None and self._sensor is not None

    @property
    def fixture_readback(self) -> Mapping[str, Any]:
        if not self.is_open:
            raise ContactAdapterError("OVPhysX contact adapter is not open")
        return self._fixture_readback

    @property
    def last_private_evidence(self) -> Mapping[str, Any]:
        """Detached private hash preimages retained after runtime cleanup."""

        return deepcopy(self._last_private_evidence)

    def _runtime_fingerprint_snapshot(
        self,
        *,
        bridge: _OVRuntimeBridge,
        phase: str,
    ) -> dict[str, Any]:
        """Bind the runtime fingerprint to an explicit fail-closed lifecycle point."""

        _assert_no_forbidden_modules(phase=phase)
        reader = getattr(bridge, "runtime_fingerprint", None)
        if callable(reader):
            fingerprint = dict(reader())
        else:
            fingerprint = {
                "backend": "ovphysx",
                "bridge_type": type(bridge).__name__,
                "python_version": platform.python_version(),
                "device": self._device,
                "control_path": "runtime_bridge.set_position_targets -> step",
                "contact_path": "ContactSensor.data.force_matrix_w",
            }
        inventory = _forbidden_module_inventory()
        if inventory:
            raise ContactAdapterError(
                "forbidden module inventory changed after headless verification"
            )
        fingerprint["scene_inventory_verification_phase"] = phase
        fingerprint["forbidden_module_verification_phase"] = phase
        fingerprint["forbidden_module_inventory"] = inventory
        if self._bridge_factory is None and (
            fingerprint.get("kitless") is not True
            or fingerprint.get("renderer") is not False
            or fingerprint.get("camera") is not False
            or fingerprint.get("camera_prim_paths") != []
            or fingerprint.get("created_sensor_types") != ["ContactSensor"]
        ):
            raise AdapterCapabilityError(
                "ovphysx",
                "kitless_headless_only",
                "runtime scene created a Kit, renderer, or camera path",
            )
        return fingerprint

    def open_contact_case(
        self,
        hand: object,
        scenario: object,
        case: object,
        *,
        dt_s: float,
        asset_root: str | Path,
        device: str,
    ) -> None:
        if self.is_open or self._bridge is not None:
            raise ContactAdapterError("OVPhysX contact adapter is already open")
        self._last_private_evidence = {}
        if device != "cuda:0" and self._bridge_factory is None:
            raise AdapterCapabilityError(
                "ovphysx",
                "single_cuda_device",
                "the formal C0 worker requires logical device 'cuda:0'",
            )
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            raise AdapterCapabilityError(
                "ovphysx",
                "headless_only",
                "DISPLAY and WAYLAND_DISPLAY must be unset",
            )
        _assert_no_forbidden_modules(phase="before OVPhysX C0 imports")
        dt = _finite_positive(dt_s, name="dt_s")
        side = _hand_side(hand)
        pair_enabled = _pair_collision_enabled(case)
        threshold = _observation_threshold(
            scenario, case, self._explicit_threshold
        )
        source = _model_path(hand, case, backend="ovphysx", asset_root=asset_root)
        source_digest = sha256(source.read_bytes()).hexdigest()
        factory = self._bridge_factory or _DefaultOVPhysXBridge
        bridge = factory()
        self._bridge = bridge
        self._device = device
        try:
            bridge.begin(source=source, hand=hand, dt_s=dt, device=device)
            _assert_no_forbidden_modules(phase="after OVPhysX C0 imports")
            if int(bridge.reset_count) != 0:
                raise ContactAdapterError(
                    "OV runtime called physics reset before the C0 USD overlay"
                )
            readback = dict(
                bridge.author_contact_overlay(
                    side=side,
                    pair_collision_enabled=pair_enabled,
                )
            )
            if int(bridge.reset_count) != 0:
                raise ContactAdapterError(
                    "OV runtime reset while authoring the C0 USD overlay"
                )
            native = readback.get("native_collision_disabled_inventory")
            if not isinstance(native, Mapping) or not bool(native.get("all_disabled")):
                raise ContactAdapterError(
                    "OV overlay did not prove all native collisions disabled"
                )
            collision = readback.get("collision")
            if not isinstance(collision, Mapping):
                raise ContactAdapterError("OV overlay collision readback is missing")
            if collision.get("allowed_pair_id") != ALLOWED_PAIR_ID:
                raise ContactAdapterError("OV overlay allowed-pair id drifted")
            if collision.get("pair_collision_enabled") is not pair_enabled:
                raise ContactAdapterError("OV overlay pair mask readback differs")
            body_path = str(
                _field(readback.get("geometry", {}).get("probe", {}), "parent_prim_path")
            )
            target_path = str(
                _field(readback.get("geometry", {}).get("target", {}), "prim_path")
            )
            sensor = bridge.create_filtered_contact_sensor(
                body_prim_path=body_path,
                target_prim_path=target_path,
            )
            sensor_cfg = getattr(sensor, "cfg", None)
            filter_paths = getattr(sensor_cfg, "filter_prim_paths_expr", None)
            if filter_paths is None and isinstance(sensor, Mapping):
                filter_paths = sensor.get("filter_prim_paths_expr")
            if filter_paths is None or [str(value) for value in filter_paths] != [
                target_path
            ]:
                raise ContactAdapterError(
                    "OV ContactSensor must retain exactly the synthetic target filter"
                )
            readback["sensor_readback"] = {
                "body_prim_path": body_path,
                "filter_prim_paths_expr": [target_path],
                "filter_count": 1,
                "update_period_s": 0.0,
                "debug_visualization": False,
                "created_before_first_reset": True,
                "sample_update_force_recompute": True,
            }
            readback["collision_inventory_hash_preimage"] = {
                "native_collision_items": readback[
                    "native_collision_disabled_inventory"
                ]["items"],
                "synthetic_enabled_shapes": [
                    "synthetic_index_probe",
                    "static_box",
                ],
            }
            if int(bridge.reset_count) != 0:
                raise ContactAdapterError(
                    "OV runtime reset before the filtered ContactSensor existed"
                )
            bridge.reset(hand=hand)
            if int(bridge.reset_count) != 1:
                raise ContactAdapterError(
                    "OV runtime must perform exactly one first reset after overlay/sensor"
                )
            _assert_no_forbidden_modules(phase="after OVPhysX C0 first reset")
            runtime_ccd_reader = getattr(bridge, "runtime_ccd_readback", None)
            if not callable(runtime_ccd_reader):
                raise AdapterCapabilityError(
                    "ovphysx",
                    "runtime_ccd_readback",
                    "OV bridge exposes no post-reset PhysX CCD readback",
                )
            runtime_ccd = dict(runtime_ccd_reader())
            expected_ccd_keys = {
                "source",
                "physx_version",
                "helper_sha256",
                "scene_prim_path",
                "body_prim_path",
                "scene_flags",
                "body_flags",
                "scene_ccd_mask",
                "body_ccd_mask",
                "scene_enabled",
                "body_enabled",
                "usd_input_source",
                "usd_scene_enabled",
                "usd_body_enabled",
            }
            if set(runtime_ccd) != expected_ccd_keys:
                raise ContactAdapterError("OV runtime CCD readback shape differs")
            if (
                runtime_ccd.get("source")
                != _PHYSX_NATIVE_CCD_READBACK_SOURCE
                or runtime_ccd.get("physx_version") != "5.9.0"
                or _SHA256_RE.fullmatch(
                    str(runtime_ccd.get("helper_sha256", ""))
                )
                is None
                or not str(runtime_ccd.get("scene_prim_path", "")).startswith("/")
                or runtime_ccd.get("body_prim_path") != body_path
                or not isinstance(runtime_ccd.get("scene_flags"), int)
                or isinstance(runtime_ccd.get("scene_flags"), bool)
                or not isinstance(runtime_ccd.get("body_flags"), int)
                or isinstance(runtime_ccd.get("body_flags"), bool)
                or runtime_ccd.get("scene_ccd_mask") != _PHYSX_SCENE_CCD_MASK
                or runtime_ccd.get("body_ccd_mask")
                != _PHYSX_RIGID_BODY_CCD_MASK
                or bool(
                    int(runtime_ccd.get("scene_flags", 0))
                    & _PHYSX_SCENE_CCD_MASK
                )
                is not runtime_ccd.get("scene_enabled")
                or bool(
                    int(runtime_ccd.get("body_flags", 0))
                    & _PHYSX_RIGID_BODY_CCD_MASK
                )
                is not runtime_ccd.get("body_enabled")
                or runtime_ccd.get("scene_enabled") is not False
                or runtime_ccd.get("body_enabled") is not False
                or runtime_ccd.get("usd_input_source")
                != "post_first_reset_live_composed_usd"
                or runtime_ccd.get("usd_scene_enabled") is not False
                or runtime_ccd.get("usd_body_enabled") is not False
            ):
                raise ContactAdapterError("OV runtime CCD is not proved disabled")
            collision.update(
                {
                    "ccd_enabled": False,
                    "ccd_readback_source": runtime_ccd["source"],
                    "ccd_physx_version": runtime_ccd["physx_version"],
                    "ccd_helper_sha256": runtime_ccd["helper_sha256"],
                    "ccd_scene_prim_path": runtime_ccd["scene_prim_path"],
                    "ccd_body_prim_path": runtime_ccd["body_prim_path"],
                    "ccd_scene_flags": runtime_ccd["scene_flags"],
                    "ccd_body_flags": runtime_ccd["body_flags"],
                    "ccd_scene_mask": runtime_ccd["scene_ccd_mask"],
                    "ccd_body_mask": runtime_ccd["body_ccd_mask"],
                    "ccd_scene_enabled": runtime_ccd["scene_enabled"],
                    "ccd_body_enabled": runtime_ccd["body_enabled"],
                    "ccd_usd_input_source": runtime_ccd["usd_input_source"],
                    "ccd_usd_scene_enabled": runtime_ccd["usd_scene_enabled"],
                    "ccd_usd_body_enabled": runtime_ccd["usd_body_enabled"],
                }
            )
            runtime_mass_reader = getattr(bridge, "runtime_mass_properties", None)
            if not callable(runtime_mass_reader):
                raise AdapterCapabilityError(
                    "ovphysx",
                    "runtime_mass_properties",
                    "OV bridge exposes no post-cook mass/COM/inertia readback",
                )
            runtime_mass_properties = dict(
                runtime_mass_reader(body_name=f"{side}_index_DP")
            )
            mass_properties = readback.get("mass_properties")
            if not isinstance(mass_properties, dict):
                raise ContactAdapterError("OV overlay mass-property readback is missing")
            mass_properties.update(
                _verify_ov_runtime_mass_properties(
                    source=mass_properties.get("source"),
                    runtime=runtime_mass_properties,
                    expected_body_name=f"{side}_index_DP",
                    expected_body_prim_path=body_path,
                    expected_helper_sha256=str(runtime_ccd["helper_sha256"]),
                )
            )
            if sha256(source.read_bytes()).hexdigest() != source_digest:
                raise ContactAdapterError("canonical USD changed during session overlay")
            self._sensor = sensor
            self._source = source
            self._source_sha256 = source_digest
            self._fixture_readback = readback
            self._threshold_n = threshold
            self._pair_collision_enabled = pair_enabled
            self._position_targets = {
                name: 0.0 for name in _names(hand, "joint_names")
            }
            runtime_fingerprint = self._runtime_fingerprint_snapshot(
                bridge=bridge,
                phase="after_first_reset_before_trajectory",
            )
            self._last_private_evidence = {
                "fixture_overlay": readback,
                "runtime_fingerprint": runtime_fingerprint,
                "unexpected_pair_observation": {
                    "status": "proved_empty_by_exclusive_enabled_shape_inventory",
                    "unexpected_pairs": [],
                },
            }
        except Exception:
            self.close()
            raise

    def set_position_targets(self, targets: Mapping[str, float]) -> None:
        if not self.is_open:
            raise ContactAdapterError("OVPhysX contact adapter is not open")
        if set(targets) != set(self._position_targets):
            raise ValueError("position target mapping must exactly cover all joints")
        normalized: dict[str, float] = {}
        for name in self._position_targets:
            value = float(targets[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite position target for {name!r}")
            normalized[name] = value
        self._bridge.set_position_targets(normalized)
        self._position_targets = normalized

    def step(self, *, dt_s: float) -> None:
        if not self.is_open:
            raise ContactAdapterError("OVPhysX contact adapter is not open")
        self._bridge.step(dt_s=dt_s)

    def collect_contact_observation(self) -> Mapping[str, Any]:
        """Read the exact 1x1 filtered sensor vector used for pair activation."""

        if not self.is_open or self._threshold_n is None:
            raise ContactAdapterError("OVPhysX contact adapter is not open")
        raw_matrix = self._bridge.filtered_force_matrix(self._sensor)
        vector = _normalize_filtered_force_vector(raw_matrix)
        norm = math.sqrt(sum(component * component for component in vector))
        pair_active = norm > self._threshold_n
        if not self._pair_collision_enabled and pair_active:
            raise ContactAdapterError(
                "sham filtered sensor exceeded the frozen pair-active threshold"
            )
        return {
            "pair_id": ALLOWED_PAIR_ID,
            "pair_active": pair_active,
            "pair_active_source": PAIR_ACTIVE_SOURCE,
            "pair_collision_enabled": self._pair_collision_enabled,
            "signed_gap_m": None,
            "signed_gap_source": "unavailable_in_filtered_ContactSensor",
            "signed_gap_convention": "positive_separation_zero_touch_negative_penetration",
            "raw_contact_count": None,
            "intended_pair_contact_count": None,
            "unexpected_pairs": [],
            "unexpected_pair_source": (
                "exclusive synthetic collision inventory; ContactSensor does not "
                "expose a global manifold stream"
            ),
            "raw_force": {
                "status": "observed_descriptive_only",
                "source_api": "ContactSensor.data.force_matrix_w",
                "units": "N",
                "frame": "world",
                "filtered_vector_w_n": list(vector),
                "filtered_vector_norm_n": norm,
                "pair_active_threshold_n": self._threshold_n,
                "comparison": "strict_greater_than",
                "semantics": (
                    "filtered sensor aggregate only; not a manifold force, "
                    "contact count, penetration, or raw solver impulse"
                ),
            },
        }

    def _trace_sample(
        self,
        hand: object,
        *,
        step: int,
        canonical_time_s: float,
    ) -> Mapping[str, Any]:
        state = self._bridge.state(hand=hand)
        observation = self.collect_contact_observation()
        positions = dict(_field(state, "joint_positions"))
        velocities = dict(_field(state, "joint_velocities"))
        poses = dict(_field(state, "frame_poses"))
        probe_center = tuple(float(value) for value in _field(state, "probe_center_world_m"))
        if len(probe_center) != 3 or not all(
            math.isfinite(value) for value in probe_center
        ):
            raise FloatingPointError("OVPhysX probe center readback is invalid")
        analytic_gap = _analytic_signed_gap(probe_center)
        return {
            "step": step,
            "time_s": canonical_time_s,
            "joint_positions": positions,
            "joint_velocities": velocities,
            "frame_poses": poses,
            "position_targets": dict(self._position_targets),
            "probe_center_world_m": list(probe_center),
            "signed_gap_m": analytic_gap,
            "pair_active": observation["pair_active"],
            "raw_contact_count": None,
            "native_contact_observation": {
                "selected_pair_force_norm_n": observation["raw_force"][
                    "filtered_vector_norm_n"
                ]
            },
        }

    def run_contact_case(
        self,
        hand: object,
        scenario: object,
        case: object,
        *,
        dt_s: float,
        asset_root: str | Path,
        device: str | None = None,
    ) -> ContactRun:
        dt = _finite_positive(dt_s, name="dt_s")
        resolved_device = "cuda:0" if device is None else str(device)
        requested_steps = _scenario_steps(scenario, dt_s=dt)
        samples: list[Mapping[str, Any]] = []
        fixture_readback: Mapping[str, Any] = {}
        source: Path | None = None
        source_digest: str | None = None
        threshold = 0.0
        try:
            self.open_contact_case(
                hand,
                scenario,
                case,
                dt_s=dt,
                asset_root=asset_root,
                device=resolved_device,
            )
            source = self._source
            source_digest = self._source_sha256
            threshold = float(self._threshold_n)
            for step_index in range(requested_steps + 1):
                targets = _canonical_targets(
                    scenario,
                    _names(hand, "joint_names"),
                    step_index=step_index,
                    dt_s=dt,
                )
                self.set_position_targets(targets)
                canonical_time_s = _canonical_sample_time_s(
                    step_index=step_index,
                    requested_steps=requested_steps,
                    dt_s=dt,
                    duration_s=float(_field(scenario, "duration_s")),
                )
                samples.append(
                    self._trace_sample(
                        hand,
                        step=step_index,
                        canonical_time_s=canonical_time_s,
                    )
                )
                if step_index < requested_steps:
                    self.step(dt_s=dt)
            bridge = self._bridge
            if bridge is None:
                raise ContactAdapterError("OVPhysX runtime disappeared during trajectory")
            self._last_private_evidence["runtime_fingerprint"] = (
                self._runtime_fingerprint_snapshot(
                    bridge=bridge,
                    phase="after_full_trajectory_before_cleanup",
                )
            )
            fixture_readback = dict(self._fixture_readback)
            if source is None or source_digest is None:
                raise ContactAdapterError("OVPhysX provenance was not initialized")
            if sha256(source.read_bytes()).hexdigest() != source_digest:
                raise ContactAdapterError("canonical USD changed during C0 run")
        finally:
            self.close()

        _assert_no_forbidden_modules(phase="after OVPhysX C0 runtime cleanup")
        final_runtime_fingerprint = dict(
            self._last_private_evidence["runtime_fingerprint"]
        )
        final_runtime_fingerprint["forbidden_module_verification_phase"] = (
            "after_runtime_cleanup"
        )
        final_runtime_fingerprint["forbidden_module_inventory"] = (
            _forbidden_module_inventory()
        )
        self._last_private_evidence["runtime_fingerprint"] = (
            final_runtime_fingerprint
        )

        manifest_digest, asset_commit, asset_tree = _canonical_manifest_identity()
        source_revision, source_tree, fresh_identity = _formal_source_identity(
            self._formal_source_revision,
            self._formal_source_tree,
            self._formal_fresh_process_identity_sha256,
        )
        closed_fixture = _closed_fixture_readback(
            side=_hand_side(hand), detailed=fixture_readback
        )
        runtime_fingerprint = _sha256_json(
            self._last_private_evidence["runtime_fingerprint"]
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "manifest_id": CONTACT_C0_MANIFEST_ID,
            "manifest_sha256": manifest_digest,
            "case": dict(_json_ready_case(case)),
            "execution": {
                "status": "completed",
                "message": None,
                "requested_steps": requested_steps,
                "completed_steps": requested_steps,
            },
            "mapping": {
                "expected_joint_count": 22,
                "observed_joint_count": len(_names(hand, "joint_names")),
                "expected_distal_frame_count": 5,
                "observed_distal_frame_count": len(
                    _names(hand, "distal_frame_names", "frame_names")
                ),
                "probe_frame_name": f"{_hand_side(hand)}_index_DP",
                "probe_frame_mapped": True,
            },
            "fixture_readback": closed_fixture,
            "contact_observation": {
                "performed": True,
                "capability": "direct_filtered_pair_force_threshold",
                "selected_pair_id": ALLOWED_PAIR_ID,
                "pair_active_source": PAIR_ACTIVE_SOURCE,
                "pair_active_force_threshold_n": threshold,
                "pair_active_record_count": len(samples),
                "missing_pair_active_count": 0,
                "filtered_sensor_body_count": 1,
                "filtered_target_count": 1,
            },
            "samples": samples,
            "provenance": {
                "backend": "ovphysx",
                "source_revision": source_revision,
                "source_tree": source_tree,
                "asset_commit": asset_commit,
                "asset_git_tree": asset_tree,
                "manifest_sha256": manifest_digest,
                "fixture_overlay_sha256": _sha256_json(
                    self._last_private_evidence["fixture_overlay"]
                ),
                "runtime_fingerprint_sha256": runtime_fingerprint,
                "fresh_process_identity_sha256": fresh_identity,
            },
        }
        return _construct_contact_run(payload)

    def close(self) -> None:
        bridge = self._bridge
        self._bridge = None
        self._sensor = None
        self._source = None
        self._source_sha256 = None
        self._fixture_readback = {}
        self._threshold_n = None
        self._pair_collision_enabled = False
        self._position_targets = {}
        self._device = ""
        if bridge is not None:
            bridge.close()


class _DefaultOVPhysXBridge:
    """Pinned Isaac Lab/OVPhysX bridge, imported only in a formal worker."""

    def __init__(self) -> None:
        self.reset_count = 0
        self._runtime: dict[str, Any] = {}
        self._context: Any | None = None
        self._simulation: Any | None = None
        self._articulation: Any | None = None
        self._dt_s = 0.0
        self._joint_names: tuple[str, ...] = ()
        self._backend_joint_names: tuple[str, ...] = ()
        self._body_names: tuple[str, ...] = ()
        self._target_tensor: Any | None = None
        self._probe_body_name = ""
        self._overlay: Any | None = None
        self._session_layer: Any | None = None
        self._original_session_sublayers: tuple[str, ...] = ()
        self._ccd_scene_prim_path = ""
        self._ccd_body_prim_path = ""
        self._contact_sensor_create_count = 0

    def begin(
        self,
        *,
        source: Path,
        hand: object,
        dt_s: float,
        device: str,
    ) -> None:
        if self._simulation is not None:
            raise ContactAdapterError("OV bridge already began")
        try:
            from wave_asset_qa.adapters.ovphysx import _runtime_api

            runtime = dict(_runtime_api(device))
            runtime["sensors"] = import_module("isaaclab.sensors")
            runtime["Usd"] = import_module("pxr.Usd")
            runtime["UsdGeom"] = import_module("pxr.UsdGeom")
            runtime["UsdPhysics"] = import_module("pxr.UsdPhysics")
            runtime["UsdShade"] = import_module("pxr.UsdShade")
            runtime["Sdf"] = import_module("pxr.Sdf")
            _register_physx_schemas_before_stage(runtime)
        except Exception as error:
            if isinstance(error, AdapterCapabilityError):
                raise
            raise AdapterCapabilityError(
                "ovphysx",
                "contact_sensor_imports",
                f"failed to import kit-less contact APIs: {type(error).__name__}: {error}",
            ) from error
        _assert_no_forbidden_modules(phase="after contact bridge imports")
        sensors = runtime["sensors"]
        if not all(
            hasattr(sensors, name) for name in ("ContactSensor", "ContactSensorCfg")
        ):
            raise AdapterCapabilityError(
                "ovphysx",
                "filtered_contact_sensor",
                "isaaclab.sensors lacks ContactSensor or ContactSensorCfg",
            )
        sim = runtime["sim"]
        simulation_cfg = sim.SimulationCfg(
            physics=runtime["ov_physics"].OvPhysxCfg(),
            device=device,
            dt=dt_s,
            gravity=GRAVITY_M_S2,
            create_stage_in_memory=True,
            render_interval=1000,
        )
        context = sim.build_simulation_context(
            create_new_stage=True,
            device=device,
            sim_cfg=simulation_cfg,
            add_ground_plane=False,
            add_lighting=False,
            auto_add_lighting=False,
        )
        simulation = context.__enter__()
        if hasattr(simulation, "_app_control_on_stop_handle"):
            simulation._app_control_on_stop_handle = None
        sim.create_prim("/World/Env_0", "Xform")
        try:
            spawn = sim.UsdFileCfg(
                usd_path=str(source),
                activate_contact_sensors=True,
            )
        except TypeError as error:
            context.__exit__(type(error), error, error.__traceback__)
            raise AdapterCapabilityError(
                "ovphysx",
                "activate_contact_sensors",
                "pinned UsdFileCfg does not accept activate_contact_sensors=True",
            ) from error
        articulation_cfg = runtime["assets_cfg"].ArticulationCfg(
            prim_path="/World/Env_.*/Robot",
            spawn=spawn,
            actuators={
                "all": runtime["actuators"].IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=None,
                    damping=None,
                )
            },
        )
        articulation = runtime["ov_assets"].Articulation(articulation_cfg)
        self._runtime = runtime
        self._context = context
        self._simulation = simulation
        self._articulation = articulation
        self._dt_s = dt_s
        self._joint_names = _names(hand, "joint_names")
        self._probe_body_name = f"{_hand_side(hand)}_index_DP"

    def author_contact_overlay(
        self,
        *,
        side: str,
        pair_collision_enabled: bool,
    ) -> Mapping[str, Any]:
        if self.reset_count != 0 or self._simulation is None:
            raise ContactAdapterError("OV overlay must be authored before first reset")
        session_layer = self._simulation.stage.GetSessionLayer()
        if session_layer is None:
            raise ContactAdapterError("OV C0 stage has no session layer")
        original_sublayers = tuple(session_layer.subLayerPaths)
        try:
            readback, overlay_state = _author_ovphysx_contact_overlay(
                stage=self._simulation.stage,
                runtime=self._runtime,
                side=side,
                pair_collision_enabled=pair_collision_enabled,
            )
        except Exception:
            session_layer.subLayerPaths[:] = list(original_sublayers)
            raise
        self._overlay = overlay_state["overlay"]
        self._session_layer = overlay_state["session_layer"]
        self._original_session_sublayers = tuple(
            overlay_state["original_session_sublayers"]
        )
        self._ccd_scene_prim_path = str(overlay_state["ccd_scene_prim_path"])
        self._ccd_body_prim_path = str(overlay_state["ccd_body_prim_path"])
        return readback

    def create_filtered_contact_sensor(
        self,
        *,
        body_prim_path: str,
        target_prim_path: str,
    ) -> object:
        if self.reset_count != 0:
            raise ContactAdapterError("ContactSensor must exist before first reset")
        cfg = self._runtime["sensors"].ContactSensorCfg(
            prim_path=body_prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            filter_prim_paths_expr=[target_prim_path],
        )
        if self._contact_sensor_create_count != 0:
            raise ContactAdapterError("OV bridge may create exactly one ContactSensor")
        sensor = self._runtime["sensors"].ContactSensor(cfg)
        self._contact_sensor_create_count = 1
        return sensor

    def reset(self, *, hand: object) -> None:
        if self.reset_count != 0:
            raise ContactAdapterError("OV bridge first reset may run only once")
        if self._contact_sensor_create_count != 1:
            raise ContactAdapterError(
                "OV bridge must create exactly one ContactSensor before reset"
            )
        self._simulation.reset()
        self.reset_count = 1
        articulation = self._articulation
        if getattr(articulation, "is_initialized", None) is not True:
            raise ContactAdapterError("OV articulation did not initialize")
        self._backend_joint_names = tuple(str(name) for name in articulation.joint_names)
        if set(self._backend_joint_names) != set(self._joint_names):
            raise ContactAdapterError("OV joint inventory differs from contact hand")
        self._body_names = tuple(str(name) for name in articulation.body_names)
        frame_names = _names(hand, "distal_frame_names", "frame_names")
        if not set(frame_names).issubset(self._body_names):
            raise ContactAdapterError("OV distal-frame inventory is incomplete")
        if self._probe_body_name not in self._body_names:
            raise ContactAdapterError("OV index_DP sensor body is absent")
        torch = self._runtime["torch"]
        zero = torch.zeros_like(articulation.data.joint_pos.torch)
        articulation.write_joint_position_to_sim_index(position=zero)
        articulation.write_joint_velocity_to_sim_index(velocity=zero)
        articulation.reset()
        articulation.set_joint_position_target_index(target=zero)
        articulation.set_joint_velocity_target_index(target=zero)
        articulation.set_joint_effort_target_index(target=zero)
        articulation.write_data_to_sim()
        self._target_tensor = zero.clone()

    def runtime_mass_properties(self, *, body_name: str) -> Mapping[str, Any]:
        """Read the cooked link's effective mass properties from live PhysX.

        C0 adds the probe as a child collision shape without a MassAPI.  The
        authored USD opinions are therefore necessary but not sufficient
        evidence that PhysX preserved the original link mass.  The native
        readback deliberately keeps PhysX's mass-space diagonal inertia and
        center-of-mass local pose separate, so their frame semantics remain
        explicit when the link-frame tensor is reconstructed.
        """

        if self.reset_count != 1 or self._simulation is None:
            raise ContactAdapterError(
                "OV runtime mass properties require exactly one completed reset"
            )
        if body_name not in self._body_names or body_name != self._probe_body_name:
            raise ContactAdapterError(
                f"OV runtime mass body {body_name!r} is absent"
            )
        if (
            not self._ccd_body_prim_path
            or self._ccd_body_prim_path.rsplit("/", 1)[-1] != body_name
        ):
            raise ContactAdapterError("OV runtime mass-property prim path differs")
        manager_cls = self._runtime.get("ov_manager_cls")
        physx = getattr(manager_cls, "_physx", None)
        if physx is None:
            raise AdapterCapabilityError(
                "ovphysx",
                "native_physx_mass_property_readback",
                "the live OvPhysxManager has no PhysX instance after first reset",
            )
        try:
            ovphysx = import_module("ovphysx")
            body_pointer = int(
                physx.get_physx_ptr(
                    self._ccd_body_prim_path,
                    ovphysx.PhysXType.LINK,
                )
            )
        except Exception as error:
            raise AdapterCapabilityError(
                "ovphysx",
                "native_physx_mass_property_readback",
                (
                    "failed to resolve the live PhysX articulation-link pointer: "
                    f"{type(error).__name__}: {error}"
                ),
            ) from error
        if body_pointer == 0:
            raise ContactAdapterError(
                "OV runtime returned a null PhysX mass-property pointer"
            )

        import ctypes

        helper, helper_sha256 = _load_physx_ccd_helper()
        output_type = ctypes.c_double * _PHYSX_MASS_PROPERTIES_VALUE_COUNT
        output = output_type()
        status = int(
            helper.waveqa_rigid_body_mass_properties(
                body_pointer,
                output,
                _PHYSX_MASS_PROPERTIES_VALUE_COUNT,
            )
        )
        if status != 0:
            raise ContactAdapterError(
                f"native PhysX mass-property helper failed with status {status}"
            )
        values = [float(value) for value in output]
        mass = values[0]
        inertia_diagonal = values[1:4]
        com_position = values[4:7]
        com_quaternion = values[7:11]
        if (
            len(values) != _PHYSX_MASS_PROPERTIES_VALUE_COUNT
            or not all(math.isfinite(value) for value in values)
            or mass <= 0.0
            or any(value <= 0.0 for value in inertia_diagonal)
            or abs(
                math.sqrt(sum(value * value for value in com_quaternion)) - 1.0
            )
            > _PHYSX_MASS_QUATERNION_NORM_ABS_TOLERANCE
        ):
            raise ContactAdapterError(
                "native PhysX mass-property readback contains invalid values"
            )
        return {
            "body_name": body_name,
            "body_prim_path": self._ccd_body_prim_path,
            "source": _PHYSX_NATIVE_MASS_READBACK_SOURCE,
            "physx_version": "5.9.0",
            "helper_sha256": helper_sha256,
            "mass_kg": mass,
            "center_of_mass_pose_b": {
                "position_m": com_position,
                "quaternion_xyzw": com_quaternion,
            },
            "mass_space_inertia_diagonal_kg_m2": inertia_diagonal,
        }

    def runtime_ccd_readback(self) -> Mapping[str, Any]:
        """Read effective CCD flags from the live PhysX 5.9 scene and link."""

        if self.reset_count != 1 or self._simulation is None:
            raise ContactAdapterError(
                "OV runtime CCD readback requires exactly one completed reset"
            )
        if not self._ccd_scene_prim_path or not self._ccd_body_prim_path:
            raise ContactAdapterError("OV runtime CCD prim paths are unavailable")
        stage = self._simulation.stage
        scene_prim = stage.GetPrimAtPath(self._ccd_scene_prim_path)
        body_prim = stage.GetPrimAtPath(self._ccd_body_prim_path)
        if not scene_prim.IsValid() or not body_prim.IsValid():
            raise ContactAdapterError("OV runtime CCD prim readback is invalid")
        if "PhysxSceneAPI" not in tuple(scene_prim.GetAppliedSchemas()) or (
            "PhysxRigidBodyAPI" not in tuple(body_prim.GetAppliedSchemas())
        ):
            raise ContactAdapterError("OV runtime CCD schemas are not composed")
        scene_attr = scene_prim.GetAttribute("physxScene:enableCCD")
        body_attr = body_prim.GetAttribute("physxRigidBody:enableCCD")
        if (
            not scene_attr.IsValid()
            or not body_attr.IsValid()
            or not scene_attr.HasAuthoredValueOpinion()
            or not body_attr.HasAuthoredValueOpinion()
        ):
            raise ContactAdapterError("OV runtime CCD attributes are not authored")
        usd_scene_enabled = scene_attr.Get()
        usd_body_enabled = body_attr.Get()
        if not isinstance(usd_scene_enabled, bool) or not isinstance(
            usd_body_enabled, bool
        ):
            raise ContactAdapterError("OV runtime CCD attributes are not Boolean")
        if usd_scene_enabled or usd_body_enabled:
            raise ContactAdapterError("OV runtime CCD USD inputs are not disabled")
        if self._overlay is None or self._session_layer is None:
            raise ContactAdapterError("OV runtime CCD overlay provenance is unavailable")
        if (
            not self._session_layer.subLayerPaths
            or self._session_layer.subLayerPaths[0] != self._overlay.identifier
        ):
            raise ContactAdapterError("OV runtime CCD overlay is not strongest")
        for attribute in (scene_attr, body_attr):
            stack = tuple(attribute.GetPropertyStack())
            if (
                not stack
                or getattr(stack[0], "layer", None) != self._overlay
            ):
                raise ContactAdapterError(
                    "OV runtime CCD property is not sourced from the strongest overlay"
                )

        manager_cls = self._runtime.get("ov_manager_cls")
        physx = getattr(manager_cls, "_physx", None)
        if physx is None:
            raise AdapterCapabilityError(
                "ovphysx",
                "native_physx_ccd_readback",
                "the live OvPhysxManager has no PhysX instance after first reset",
            )
        try:
            ovphysx = import_module("ovphysx")
            scene_pointer = int(
                physx.get_physx_ptr(
                    self._ccd_scene_prim_path,
                    ovphysx.PhysXType.SCENE,
                )
            )
            body_pointer = int(
                physx.get_physx_ptr(
                    self._ccd_body_prim_path,
                    ovphysx.PhysXType.LINK,
                )
            )
        except Exception as error:
            raise AdapterCapabilityError(
                "ovphysx",
                "native_physx_ccd_readback",
                (
                    "failed to resolve live PhysX scene/link pointers: "
                    f"{type(error).__name__}: {error}"
                ),
            ) from error
        if scene_pointer == 0 or body_pointer == 0:
            raise ContactAdapterError("OV runtime returned a null PhysX CCD pointer")
        helper, helper_sha256 = _load_physx_ccd_helper()
        scene_flags = int(helper.waveqa_scene_flags(scene_pointer))
        body_flags = int(helper.waveqa_rigid_body_flags(body_pointer))
        if scene_flags == 0xFFFFFFFF or body_flags == 0xFFFFFFFF:
            raise ContactAdapterError("native PhysX CCD helper rejected a pointer")
        scene_enabled = bool(scene_flags & _PHYSX_SCENE_CCD_MASK)
        body_enabled = bool(body_flags & _PHYSX_RIGID_BODY_CCD_MASK)
        if scene_enabled or body_enabled:
            raise ContactAdapterError("native PhysX CCD flags are enabled")
        return {
            "source": _PHYSX_NATIVE_CCD_READBACK_SOURCE,
            "physx_version": "5.9.0",
            "helper_sha256": helper_sha256,
            "scene_prim_path": self._ccd_scene_prim_path,
            "body_prim_path": self._ccd_body_prim_path,
            "scene_flags": scene_flags,
            "body_flags": body_flags,
            "scene_ccd_mask": _PHYSX_SCENE_CCD_MASK,
            "body_ccd_mask": _PHYSX_RIGID_BODY_CCD_MASK,
            "scene_enabled": scene_enabled,
            "body_enabled": body_enabled,
            "usd_input_source": "post_first_reset_live_composed_usd",
            "usd_scene_enabled": usd_scene_enabled,
            "usd_body_enabled": usd_body_enabled,
        }

    def set_position_targets(self, targets: Mapping[str, float]) -> None:
        tensor = self._target_tensor.clone()
        indices = {name: index for index, name in enumerate(self._backend_joint_names)}
        for name in self._joint_names:
            tensor[0, indices[name]] = float(targets[name])
        self._articulation.set_joint_position_target_index(target=tensor)
        self._target_tensor = tensor

    def step(self, *, dt_s: float) -> None:
        if dt_s != self._dt_s:
            raise ContactAdapterError("OV bridge timestep drifted")
        self._articulation.write_data_to_sim()
        self._simulation.step(render=False)
        self._articulation.update(dt_s)

    def state(self, *, hand: object) -> Mapping[str, Any]:
        articulation = self._articulation
        positions_tensor = articulation.data.joint_pos.torch
        velocity_tensor = articulation.data.joint_vel.torch
        pose_tensor = articulation.data.body_link_pose_w.torch
        joint_indices = {
            name: index for index, name in enumerate(self._backend_joint_names)
        }
        body_indices = {name: index for index, name in enumerate(self._body_names)}
        positions = {
            name: float(positions_tensor[0, joint_indices[name]].detach().cpu().item())
            for name in self._joint_names
        }
        velocities = {
            name: float(velocity_tensor[0, joint_indices[name]].detach().cpu().item())
            for name in self._joint_names
        }
        frames: dict[str, list[float]] = {}
        for name in _names(hand, "distal_frame_names", "frame_names"):
            values = [
                float(value)
                for value in pose_tensor[0, body_indices[name]].detach().cpu().tolist()
            ]
            if len(values) != 7:
                raise ContactAdapterError("OV body pose must be xyz+xyzw")
            frames[name] = [
                values[0],
                values[1],
                values[2],
                values[6],
                values[3],
                values[4],
                values[5],
            ]
        probe_pose = [
            float(value)
            for value in pose_tensor[
                0, body_indices[self._probe_body_name]
            ].detach().cpu().tolist()
        ]
        probe_center = _transform_point_xyzw(
            translation=probe_pose[:3],
            quaternion_xyzw=probe_pose[3:],
            local_point=PROBE_LOCAL_CENTER_M,
        )
        return {
            "joint_positions": positions,
            "joint_velocities": velocities,
            "frame_poses": frames,
            "probe_center_world_m": probe_center,
        }

    def filtered_force_matrix(self, sensor: object) -> object:
        update = getattr(sensor, "update", None)
        if not callable(update):
            raise AdapterCapabilityError(
                "ovphysx",
                "contact_sensor_force_recompute",
                "ContactSensor exposes no update method",
            )
        try:
            update(self._dt_s, force_recompute=True)
        except TypeError as error:
            raise AdapterCapabilityError(
                "ovphysx",
                "contact_sensor_force_recompute",
                "ContactSensor.update does not accept force_recompute=True",
            ) from error
        data = getattr(sensor, "data", None)
        matrix = getattr(data, "force_matrix_w", None)
        if matrix is None:
            raise AdapterCapabilityError(
                "ovphysx",
                "filtered_force_matrix",
                "ContactSensor.data.force_matrix_w is unavailable",
            )
        return matrix

    def runtime_fingerprint(self) -> Mapping[str, Any]:
        module_versions: dict[str, str | None] = {}
        for key in (
            "torch",
            "warp",
            "sim",
            "assets_cfg",
            "sensors",
            "ov_root",
            "ov_assets",
            "ov_physics",
        ):
            module = self._runtime.get(key)
            if module is not None:
                module_versions[key] = getattr(module, "__version__", None)
                if module_versions[key] is not None:
                    module_versions[key] = str(module_versions[key])
        cfg = getattr(self._simulation, "cfg", None)
        physics = getattr(cfg, "physics", None)
        device = str(getattr(cfg, "device", "cuda:0"))
        forbidden_modules = _forbidden_module_inventory()
        stage = getattr(self._simulation, "stage", None)
        if stage is None:
            raise ContactAdapterError("OV runtime stage is unavailable for camera proof")
        UsdGeom = self._runtime.get("UsdGeom")
        if UsdGeom is None or not hasattr(UsdGeom, "Camera"):
            raise ContactAdapterError("OV runtime lacks UsdGeom.Camera for camera proof")
        traverse_all = getattr(stage, "TraverseAll", None)
        if not callable(traverse_all):
            raise ContactAdapterError(
                "OV runtime stage lacks TraverseAll for exhaustive camera proof"
            )
        camera_prim_paths = sorted(
            str(prim.GetPath())
            for prim in tuple(traverse_all())
            if prim.IsA(UsdGeom.Camera)
        )
        return {
            "backend": "ovphysx",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "dt_s": self._dt_s,
            "module_versions": module_versions,
            "solver_config_type": type(physics).__name__,
            "control_path": (
                "IdealPDActuator target -> write_data_to_sim -> "
                "SimulationContext.step(render=False)"
            ),
            "contact_path": (
                "ContactSensor(update_period=0, filter_count=1)."
                "data.force_matrix_w"
            ),
            "sensor_update_order": (
                "simulation.step -> articulation.update -> sensor.update -> sample"
            ),
            "kitless": not any(
                name == "omni.kit"
                or name.startswith("omni.kit.")
                or name == "isaacsim"
                or name.startswith("isaacsim.")
                for name in forbidden_modules
            ),
            "renderer": any(
                name == "omni.renderer" or name.startswith("omni.renderer.")
                for name in forbidden_modules
            ),
            "camera": any(
                name == "omni.replicator"
                or name.startswith("omni.replicator.")
                or name == "omni.syntheticdata"
                or name.startswith("omni.syntheticdata.")
                for name in forbidden_modules
            )
            or bool(camera_prim_paths),
            "camera_prim_paths": camera_prim_paths,
            "created_sensor_types": (
                ["ContactSensor"] if self._contact_sensor_create_count == 1 else []
            ),
        }

    def close(self) -> None:
        errors: list[str] = []
        stage = getattr(self._simulation, "stage", None)
        if stage is not None and self._session_layer is not None:
            try:
                self._session_layer.subLayerPaths[:] = list(
                    self._original_session_sublayers
                )
                if tuple(self._session_layer.subLayerPaths) != tuple(
                    self._original_session_sublayers
                ):
                    errors.append("session_sublayer_restore_failed")
                if self._overlay is not None and self._overlay in stage.GetLayerStack():
                    errors.append("overlay_still_in_layer_stack")
            except Exception as error:  # pragma: no cover - runtime-only cleanup.
                errors.append(f"overlay_cleanup:{type(error).__name__}:{error}")
        context = self._context
        self._runtime = {}
        self._context = None
        self._simulation = None
        self._articulation = None
        self._target_tensor = None
        self._overlay = None
        self._session_layer = None
        self._original_session_sublayers = ()
        self._ccd_scene_prim_path = ""
        self._ccd_body_prim_path = ""
        self._contact_sensor_create_count = 0
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception as error:  # pragma: no cover - runtime-only cleanup.
                errors.append(f"context_close:{type(error).__name__}:{error}")
        if errors:
            raise ContactAdapterError("OV bridge cleanup failed: " + ", ".join(errors))


def _encode_usd_mass_value(value: object) -> Any:
    """Encode scalar/vector/quaternion USD values without a pxr dependency."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    get_real = getattr(value, "GetReal", None)
    get_imaginary = getattr(value, "GetImaginary", None)
    if callable(get_real) and callable(get_imaginary):
        imaginary = get_imaginary()
        try:
            xyz = [float(imaginary[index]) for index in range(3)]
            return [*xyz, float(get_real())]
        except (IndexError, TypeError, ValueError) as error:
            raise ContactAdapterError(
                "USD principalAxes quaternion could not be encoded"
            ) from error
    try:
        return [float(item) for item in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return repr(value)


def _usd_mass_property_readback(prim: object) -> dict[str, Any]:
    """Return authored mass inputs without depending on Gf JSON encoders."""

    names = (
        "physics:mass",
        "physics:density",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    )
    result: dict[str, Any] = {}
    for name in names:
        attribute = prim.GetAttribute(name)
        valid = bool(attribute) and bool(attribute.IsValid())
        authored = valid and bool(attribute.HasAuthoredValueOpinion())
        value = attribute.Get() if valid else None
        encoded = _encode_usd_mass_value(value)
        result[name] = {
            "valid": valid,
            "authored": authored,
            "value": encoded,
            "type": type(value).__name__ if value is not None else None,
        }
    return result


def _mass_source_value(
    source: object,
    name: str,
    *,
    length: int | None = None,
) -> float | list[float]:
    if not isinstance(source, Mapping):
        raise ContactAdapterError("OV authored mass-property source is missing")
    record = source.get(name)
    if not isinstance(record, Mapping) or record.get("valid") is not True:
        raise ContactAdapterError(f"OV authored mass property {name} is invalid")
    value = record.get("value")
    if length is None:
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise ContactAdapterError(
                f"OV authored mass property {name} is not scalar"
            ) from error
        if not math.isfinite(scalar):
            raise ContactAdapterError(
                f"OV authored mass property {name} is non-finite"
            )
        return scalar
    if not isinstance(value, list) or len(value) != length:
        raise ContactAdapterError(
            f"OV authored mass property {name} must have length {length}"
        )
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ContactAdapterError(
            f"OV authored mass property {name} is non-finite"
        )
    return vector


def _normalized_quaternion_xyzw(value: Sequence[float], *, name: str) -> np.ndarray:
    if len(value) != 4:
        raise ContactAdapterError(f"{name} quaternion must have four components")
    quaternion = np.asarray([float(item) for item in value], dtype=np.float64)
    if not np.all(np.isfinite(quaternion)):
        raise ContactAdapterError(f"{name} quaternion is non-finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ContactAdapterError(f"{name} quaternion has zero norm")
    if abs(norm - 1.0) > _PHYSX_MASS_QUATERNION_NORM_ABS_TOLERANCE:
        raise ContactAdapterError(f"{name} quaternion is not unit length")
    return quaternion / norm


def _quaternion_rotation_matrix_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    """Return the active 3x3 rotation for one normalized xyzw quaternion."""

    qx, qy, qz, qw = _normalized_quaternion_xyzw(
        quaternion, name="inertia-frame"
    )
    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _verify_ov_runtime_mass_properties(
    *,
    source: object,
    runtime: Mapping[str, Any],
    expected_body_name: str,
    expected_body_prim_path: str,
    expected_helper_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless cooked OVPhysX mass, COM, and inertia match USD."""

    expected_mass = float(_mass_source_value(source, "physics:mass"))
    expected_com = np.asarray(
        _mass_source_value(source, "physics:centerOfMass", length=3),
        dtype=np.float64,
    )
    expected_diagonal = np.asarray(
        _mass_source_value(source, "physics:diagonalInertia", length=3),
        dtype=np.float64,
    )
    if expected_mass <= 0.0 or np.any(expected_diagonal <= 0.0):
        raise ContactAdapterError("OV authored mass/inertia must be positive")
    expected_quaternion = _normalized_quaternion_xyzw(
        _mass_source_value(source, "physics:principalAxes", length=4),
        name="OV authored principalAxes",
    )

    expected_runtime_keys = {
        "body_name",
        "body_prim_path",
        "source",
        "physx_version",
        "helper_sha256",
        "mass_kg",
        "center_of_mass_pose_b",
        "mass_space_inertia_diagonal_kg_m2",
    }
    if set(runtime) != expected_runtime_keys:
        raise ContactAdapterError("OV runtime mass-property readback shape differs")
    if runtime.get("body_name") != expected_body_name:
        raise ContactAdapterError("OV runtime mass body differs from index_DP")
    if runtime.get("body_prim_path") != expected_body_prim_path:
        raise ContactAdapterError("OV runtime mass body prim path differs")
    if runtime.get("source") != _PHYSX_NATIVE_MASS_READBACK_SOURCE:
        raise ContactAdapterError("OV runtime mass-property source differs")
    if runtime.get("physx_version") != "5.9.0":
        raise ContactAdapterError("OV runtime mass-property PhysX version differs")
    helper_sha256 = str(runtime.get("helper_sha256", ""))
    if (
        _SHA256_RE.fullmatch(helper_sha256) is None
        or helper_sha256 != expected_helper_sha256
    ):
        raise ContactAdapterError("OV runtime mass-property helper differs")
    try:
        actual_mass = float(runtime["mass_kg"])
        com_pose = runtime["center_of_mass_pose_b"]
        actual_com = np.asarray(com_pose["position_m"], dtype=np.float64)
        actual_quaternion = _normalized_quaternion_xyzw(
            com_pose["quaternion_xyzw"],
            name="OV runtime COM pose",
        )
        actual_diagonal = np.asarray(
            runtime["mass_space_inertia_diagonal_kg_m2"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContactAdapterError(
            "OV runtime mass-property readback is incomplete"
        ) from error
    if (
        not math.isfinite(actual_mass)
        or actual_com.shape != (3,)
        or actual_diagonal.shape != (3,)
        or not np.all(np.isfinite(actual_com))
        or not np.all(np.isfinite(actual_diagonal))
        or np.any(actual_diagonal <= 0.0)
    ):
        raise ContactAdapterError(
            "OV runtime mass-property readback has invalid shape or values"
        )

    tolerances = {
        "mass_relative": 5.0e-5,
        "mass_absolute_kg": 1.0e-8,
        "center_of_mass_absolute_m": 1.0e-7,
        "inertia_relative": 5.0e-5,
        "inertia_absolute_kg_m2": 1.0e-12,
    }
    mass_error = abs(actual_mass - expected_mass)
    mass_limit = max(
        tolerances["mass_absolute_kg"],
        tolerances["mass_relative"] * abs(expected_mass),
    )
    com_error = float(np.max(np.abs(actual_com - expected_com)))
    actual_eigenvalues = np.sort(actual_diagonal)
    expected_rotation = _quaternion_rotation_matrix_xyzw(expected_quaternion)
    actual_rotation = _quaternion_rotation_matrix_xyzw(actual_quaternion)
    expected_link_inertia = (
        expected_rotation @ np.diag(expected_diagonal) @ expected_rotation.T
    )
    actual_link_inertia = (
        actual_rotation @ np.diag(actual_diagonal) @ actual_rotation.T
    )
    symmetry_error = float(
        np.max(np.abs(actual_link_inertia - actual_link_inertia.T))
    )
    inertia_error = float(
        np.max(np.abs(actual_link_inertia - expected_link_inertia))
    )
    inertia_limit = max(
        tolerances["inertia_absolute_kg_m2"],
        tolerances["inertia_relative"]
        * float(np.max(np.abs(expected_link_inertia))),
    )
    quaternion_dot = float(abs(np.dot(expected_quaternion, actual_quaternion)))
    quaternion_dot = min(1.0, max(0.0, quaternion_dot))
    axes_angle_error = 2.0 * math.acos(quaternion_dot)
    if mass_error > mass_limit:
        raise ContactAdapterError("OV runtime mass differs from authored USD")
    if com_error > tolerances["center_of_mass_absolute_m"]:
        raise ContactAdapterError("OV runtime center of mass differs from authored USD")
    if inertia_error > inertia_limit:
        raise ContactAdapterError("OV runtime inertia differs from authored USD")

    return {
        "preserved": True,
        "usd_composed_preserved": True,
        "runtime_verified": True,
        "runtime_effective": deepcopy(dict(runtime)),
        "runtime_checks": {
            "body_name_exact": True,
            "mass_abs_error_kg": mass_error,
            "center_of_mass_max_abs_error_m": com_error,
            "inertia_link_tensor_max_abs_error_kg_m2": inertia_error,
            "inertia_symmetry_max_abs_error_kg_m2": symmetry_error,
            "runtime_inertia_min_eigenvalue_kg_m2": float(
                np.min(actual_eigenvalues)
            ),
            "principal_axes_representation_angle_error_rad": axes_angle_error,
            "expected_link_inertia_tensor_kg_m2": (
                expected_link_inertia.tolist()
            ),
            "runtime_link_inertia_tensor_kg_m2": actual_link_inertia.tolist(),
            "expected_inertia_eigenvalues_kg_m2": np.sort(
                expected_diagonal
            ).tolist(),
            "runtime_inertia_eigenvalues_kg_m2": actual_eigenvalues.tolist(),
            "tolerances": tolerances,
        },
    }


def _author_ovphysx_contact_overlay(
    *,
    stage: object,
    runtime: Mapping[str, Any],
    side: str,
    pair_collision_enabled: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Author and read back the C0 overlay before the first physics reset."""

    Usd = runtime["Usd"]
    UsdGeom = runtime["UsdGeom"]
    UsdPhysics = runtime["UsdPhysics"]
    UsdShade = runtime["UsdShade"]
    Sdf = runtime["Sdf"]
    session_layer = stage.GetSessionLayer()
    if session_layer is None or not bool(getattr(session_layer, "anonymous", False)):
        raise ContactAdapterError("OV C0 requires an anonymous session layer")
    original_sublayers = tuple(session_layer.subLayerPaths)
    robot_prefix = "/World/Env_0/Robot"
    index_name = f"{side}_index_DP"
    stage_prims = tuple(stage.Traverse())
    scene_prims = tuple(prim for prim in stage_prims if prim.IsA(UsdPhysics.Scene))
    if len(scene_prims) != 1:
        session_layer.subLayerPaths[:] = list(original_sublayers)
        raise ContactAdapterError(
            "OV composed stage must expose exactly one PhysicsScene for CCD"
        )
    scene_prim = scene_prims[0]
    scene_path = str(scene_prim.GetPath())
    native_prims: list[object] = []
    index_prims: list[object] = []
    for prim in stage_prims:
        path = str(prim.GetPath())
        if path == robot_prefix or path.startswith(robot_prefix + "/"):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                native_prims.append(prim)
            if str(prim.GetName()) == index_name:
                index_prims.append(prim)
    if not native_prims:
        raise ContactAdapterError("OV source exposes no native collision prims")
    if len(index_prims) != 1:
        raise ContactAdapterError(
            f"OV source must resolve exactly one {index_name!r} prim"
        )
    index_prim = index_prims[0]
    index_path = str(index_prim.GetPath())
    if not index_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ContactAdapterError("OV index_DP prim is not a rigid body")
    mass_properties_before = _usd_mass_property_readback(index_prim)
    for required_name in (
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ):
        required = mass_properties_before[required_name]
        if required["valid"] is not True or required["value"] is None:
            raise ContactAdapterError(
                f"OV index_DP lacks required mass readback {required_name}"
            )

    probe_path = index_path + "/" + PROBE_GEOM_NAME
    target_path = "/World/Env_0/" + TARGET_GEOM_NAME
    material_path = "/World/Env_0/waveqa_c0_zero_contact_material"
    native_inventory: list[dict[str, Any]] = []
    overlay = Sdf.Layer.CreateAnonymous("waveqa-c0-contact-overlay.usda")
    session_layer.subLayerPaths.insert(0, overlay.identifier)
    if (
        not session_layer.subLayerPaths
        or session_layer.subLayerPaths[0] != overlay.identifier
    ):
        session_layer.subLayerPaths[:] = list(original_sublayers)
        raise ContactAdapterError("OV C0 overlay is not the strongest session sublayer")
    original_edit_layer = stage.GetEditTarget().GetLayer()
    try:
        with Usd.EditContext(stage, Usd.EditTarget(overlay)):
            if not scene_prim.AddAppliedSchema("PhysxSceneAPI"):
                raise ContactAdapterError("OV could not apply PhysxSceneAPI")
            if not index_prim.AddAppliedSchema("PhysxRigidBodyAPI"):
                raise ContactAdapterError("OV could not apply PhysxRigidBodyAPI")
            scene_ccd_input = scene_prim.GetAttribute("physxScene:enableCCD")
            body_ccd_input = index_prim.GetAttribute("physxRigidBody:enableCCD")
            if (
                not scene_ccd_input.IsValid()
                or not body_ccd_input.IsValid()
                or scene_ccd_input.GetTypeName() != Sdf.ValueTypeNames.Bool
                or body_ccd_input.GetTypeName() != Sdf.ValueTypeNames.Bool
                or scene_ccd_input.IsCustom()
                or body_ccd_input.IsCustom()
            ):
                raise ContactAdapterError(
                    "OV registered CCD schema attributes are invalid"
                )
            if not scene_ccd_input.Set(False) or not body_ccd_input.Set(False):
                raise ContactAdapterError("OV could not author CCD disable opinions")
            for prim in native_prims:
                collision = UsdPhysics.CollisionAPI(prim)
                attr = collision.GetCollisionEnabledAttr()
                authored = attr.Get()
                original_enabled = True if authored is None else bool(authored)
                collision.CreateCollisionEnabledAttr(False).Set(False)
                native_inventory.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "original_collision_enabled": original_enabled,
                        "effective_collision_enabled": False,
                    }
                )

            probe = UsdGeom.Sphere.Define(stage, probe_path)
            probe.CreateRadiusAttr(PROBE_RADIUS_M)
            UsdGeom.Xformable(probe.GetPrim()).AddTranslateOp().Set(
                PROBE_LOCAL_CENTER_M
            )
            probe_collision = UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
            probe_collision.CreateCollisionEnabledAttr(True).Set(True)

            target = UsdGeom.Cube.Define(stage, target_path)
            target.CreateSizeAttr(2.0)
            target_xform = UsdGeom.Xformable(target.GetPrim())
            target_xform.AddTranslateOp().Set(TARGET_CENTER_WORLD_M)
            target_xform.AddScaleOp().Set(TARGET_HALF_EXTENTS_M)
            target_collision = UsdPhysics.CollisionAPI.Apply(target.GetPrim())
            target_collision.CreateCollisionEnabledAttr(True).Set(True)

            material = UsdShade.Material.Define(stage, material_path)
            material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            material_api.CreateStaticFrictionAttr(FRICTION_COEFFICIENT).Set(
                FRICTION_COEFFICIENT
            )
            material_api.CreateDynamicFrictionAttr(FRICTION_COEFFICIENT).Set(
                FRICTION_COEFFICIENT
            )
            material_api.CreateRestitutionAttr(RESTITUTION_COEFFICIENT).Set(
                RESTITUTION_COEFFICIENT
            )
            for prim in (probe.GetPrim(), target.GetPrim()):
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                try:
                    binding.Bind(material, materialPurpose="physics")
                except TypeError:
                    binding.Bind(
                        material,
                        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                        materialPurpose="physics",
                    )

            if not pair_collision_enabled:
                filtered = UsdPhysics.FilteredPairsAPI.Apply(probe.GetPrim())
                relationship = filtered.CreateFilteredPairsRel()
                relationship.AddTarget(Sdf.Path(target_path))
    except Exception:
        session_layer.subLayerPaths[:] = list(original_sublayers)
        raise
    finally:
        if stage.GetEditTarget().GetLayer() != original_edit_layer:
            raise ContactAdapterError("OV overlay changed the caller's edit target")

    for record in native_inventory:
        prim = stage.GetPrimAtPath(record["prim_path"])
        effective = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        record["effective_collision_enabled"] = bool(effective)
    if not all(not bool(item["effective_collision_enabled"]) for item in native_inventory):
        session_layer.subLayerPaths[:] = list(original_sublayers)
        raise ContactAdapterError("OV native collision disable readback failed")

    scene_ccd_attr = scene_prim.GetAttribute("physxScene:enableCCD")
    body_ccd_attr = index_prim.GetAttribute("physxRigidBody:enableCCD")
    scene_stack = tuple(scene_ccd_attr.GetPropertyStack())
    body_stack = tuple(body_ccd_attr.GetPropertyStack())
    scene_spec = overlay.GetPrimAtPath(scene_path)
    body_spec = overlay.GetPrimAtPath(index_path)

    def _api_schema_items(spec: object) -> tuple[str, ...]:
        if spec is None or not spec.HasInfo("apiSchemas"):
            return ()
        value = spec.GetInfo("apiSchemas")
        getter = getattr(value, "GetAddedOrExplicitItems", None)
        if not callable(getter):
            return ()
        return tuple(str(item) for item in getter())

    checks = {
        "overlay_is_strongest_session_sublayer": bool(
            session_layer.subLayerPaths
            and session_layer.subLayerPaths[0] == overlay.identifier
        ),
        "scene_schema_composed": "PhysxSceneAPI"
        in tuple(scene_prim.GetAppliedSchemas()),
        "body_schema_composed": "PhysxRigidBodyAPI"
        in tuple(index_prim.GetAppliedSchemas()),
        "scene_schema_authored_in_overlay": "PhysxSceneAPI"
        in _api_schema_items(scene_spec),
        "body_schema_authored_in_overlay": "PhysxRigidBodyAPI"
        in _api_schema_items(body_spec),
        "scene_attribute_valid": scene_ccd_attr.IsValid(),
        "body_attribute_valid": body_ccd_attr.IsValid(),
        "scene_attribute_boolean": (
            scene_ccd_attr.GetTypeName() == Sdf.ValueTypeNames.Bool
            and not scene_ccd_attr.IsCustom()
        ),
        "body_attribute_boolean": (
            body_ccd_attr.GetTypeName() == Sdf.ValueTypeNames.Bool
            and not body_ccd_attr.IsCustom()
        ),
        "scene_attribute_authored": scene_ccd_attr.HasAuthoredValueOpinion(),
        "body_attribute_authored": body_ccd_attr.HasAuthoredValueOpinion(),
        "scene_attribute_overlay_strongest": bool(
            scene_stack and getattr(scene_stack[0], "layer", None) == overlay
        ),
        "body_attribute_overlay_strongest": bool(
            body_stack and getattr(body_stack[0], "layer", None) == overlay
        ),
        "scene_attribute_false": scene_ccd_attr.Get() is False,
        "body_attribute_false": body_ccd_attr.Get() is False,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if failed_checks:
        session_layer.subLayerPaths[:] = list(original_sublayers)
        raise ContactAdapterError(
            "OV PhysX CCD disable readback failed: " + ", ".join(failed_checks)
        )

    final_collision_inventory: list[dict[str, Any]] = []
    for prim in tuple(stage.Traverse()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        enabled = True if value is None else bool(value)
        final_collision_inventory.append(
            {"prim_path": str(prim.GetPath()), "collision_enabled": enabled}
        )
    enabled_paths = sorted(
        item["prim_path"]
        for item in final_collision_inventory
        if item["collision_enabled"]
    )
    if enabled_paths != sorted((probe_path, target_path)):
        session_layer.subLayerPaths[:] = list(original_sublayers)
        raise ContactAdapterError(
            "OV final collision inventory must enable only probe and target"
        )

    probe_prim = stage.GetPrimAtPath(probe_path)
    target_prim = stage.GetPrimAtPath(target_path)
    mass_properties_after = _usd_mass_property_readback(index_prim)
    if mass_properties_after != mass_properties_before:
        raise ContactAdapterError("OV overlay changed index_DP mass properties")
    probe_radius = float(UsdGeom.Sphere(probe_prim).GetRadiusAttr().Get())
    target_size = float(UsdGeom.Cube(target_prim).GetSizeAttr().Get())
    probe_enabled = bool(
        UsdPhysics.CollisionAPI(probe_prim).GetCollisionEnabledAttr().Get()
    )
    target_enabled = bool(
        UsdPhysics.CollisionAPI(target_prim).GetCollisionEnabledAttr().Get()
    )
    if probe_radius != PROBE_RADIUS_M or target_size != 2.0:
        raise ContactAdapterError("OV synthetic geometry readback differs")
    if not probe_enabled or not target_enabled:
        raise ContactAdapterError("OV synthetic collision shapes are not enabled")
    if probe_prim.HasAPI(UsdPhysics.MassAPI):
        raise ContactAdapterError("OV synthetic probe unexpectedly authors MassAPI")
    filtered_targets: list[str] = []
    if probe_prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        filtered_api = UsdPhysics.FilteredPairsAPI(probe_prim)
        filtered_targets = [
            str(path) for path in filtered_api.GetFilteredPairsRel().GetTargets()
        ]
    expected_filtered = [] if pair_collision_enabled else [target_path]
    if filtered_targets != expected_filtered:
        raise ContactAdapterError("OV unique pair-filter readback differs")

    material_prim = stage.GetPrimAtPath(material_path)
    material_api = UsdPhysics.MaterialAPI(material_prim)
    static_friction = float(material_api.GetStaticFrictionAttr().Get())
    dynamic_friction = float(material_api.GetDynamicFrictionAttr().Get())
    restitution = float(material_api.GetRestitutionAttr().Get())
    if (static_friction, dynamic_friction, restitution) != (0.0, 0.0, 0.0):
        raise ContactAdapterError("OV zero material readback differs")

    readback = {
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
                "parent_prim_path": index_path,
                "prim_path": probe_path,
                "shape": "sphere",
                "local_center_m": list(PROBE_LOCAL_CENTER_M),
                "radius_m": probe_radius,
            },
            "target": {
                "canonical_id": "static_box",
                "prim_path": target_path,
                "shape": "box",
                "center_world_m": list(TARGET_CENTER_WORLD_M),
                "half_extents_m": list(TARGET_HALF_EXTENTS_M),
                "top_world_m": TARGET_TOP_WORLD_M,
                "static": not target_prim.HasAPI(UsdPhysics.RigidBodyAPI),
            },
        },
        "collision": {
            "allowed_pair_id": ALLOWED_PAIR_ID,
            "pair_collision_enabled": pair_collision_enabled,
            "pair_mask_source": "PhysicsFilteredPairsAPI on synthetic probe only",
            "filtered_pair_targets": filtered_targets,
            "probe_collision_enabled": probe_enabled,
            "target_collision_enabled": target_enabled,
            "self_collision_enabled": False,
            "ccd_enabled": bool(scene_ccd_attr.Get() or body_ccd_attr.Get()),
            "ccd_readback_source": (
                "pre_first_reset_composed_physx_ccd_inputs_pending_native_readback"
            ),
            "ccd_scene_prim_path": scene_path,
            "ccd_body_prim_path": index_path,
            "ccd_scene_enabled": bool(scene_ccd_attr.Get()),
            "ccd_body_enabled": bool(body_ccd_attr.Get()),
        },
        "material": {
            "prim_path": material_path,
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "friction_coefficient": FRICTION_COEFFICIENT,
            "restitution_coefficient": restitution,
        },
        "mass_properties": {
            "preserved": False,
            "usd_composed_preserved": True,
            "runtime_verified": False,
            "source": mass_properties_before,
            "effective": mass_properties_after,
            "synthetic_probe_mass_api_applied": False,
        },
        "native_collision_disabled_inventory": {
            "count": len(native_inventory),
            "all_disabled": True,
            "items": native_inventory,
        },
        "final_collision_inventory": {
            "count": len(final_collision_inventory),
            "enabled_count": len(enabled_paths),
            "enabled_paths": enabled_paths,
            "items": final_collision_inventory,
        },
        "solver": {
            "gravity_m_s2": list(GRAVITY_M_S2),
            "renderer": False,
            "camera": False,
            "ground_plane": False,
        },
    }
    overlay_state = {
        "overlay": overlay,
        "session_layer": session_layer,
        "original_session_sublayers": original_sublayers,
        "ccd_scene_prim_path": scene_path,
        "ccd_body_prim_path": index_path,
    }
    return readback, overlay_state


def _transform_point_xyzw(
    *,
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
    local_point: Sequence[float],
) -> tuple[float, float, float]:
    """Transform one point without importing a GPU math package."""

    if len(translation) != 3 or len(quaternion_xyzw) != 4 or len(local_point) != 3:
        raise ValueError("invalid rigid transform shape")
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm == 0.0:
        raise FloatingPointError("invalid OV quaternion readback")
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    vector = np.asarray(tuple(float(value) for value in local_point), dtype=float)
    qvec = np.asarray((qx, qy, qz), dtype=float)
    rotated = vector + 2.0 * np.cross(qvec, np.cross(qvec, vector) + qw * vector)
    world = rotated + np.asarray(tuple(float(value) for value in translation))
    if not np.all(np.isfinite(world)):
        raise FloatingPointError("non-finite transformed OV probe center")
    return (float(world[0]), float(world[1]), float(world[2]))


__all__ = [
    "ALLOWED_PAIR_ID",
    "FRICTION_COEFFICIENT",
    "GRAVITY_M_S2",
    "MUJOCO_CONDIM",
    "MuJoCoContactAdapter",
    "OVPhysXContactAdapter",
    "PAIR_NAME",
    "PROBE_GEOM_NAME",
    "PROBE_LOCAL_CENTER_M",
    "PROBE_RADIUS_M",
    "RESTITUTION_COEFFICIENT",
    "TARGET_CENTER_WORLD_M",
    "TARGET_GEOM_NAME",
    "TARGET_HALF_EXTENTS_M",
    "TARGET_TOP_WORLD_M",
    "ContactAdapterError",
]
