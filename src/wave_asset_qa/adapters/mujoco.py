"""Headless fixed-base MuJoCo adapter for WaveSimParity Gate 0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib import import_module
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from wave_asset_qa.parity.contracts import ScenarioSpec, WaveformScenarioSpec
from wave_asset_qa.parity.mapping import (
    MAPPING_SCHEMA_VERSION,
    FrameMapping,
    JointMapping,
)
from wave_asset_qa.parity.scenarios import (
    CanonicalTargetSequenceDigest,
    canonical_initial_positions,
    canonical_position_targets,
    canonical_target_sequence_sha256,
)

from .base import (
    AdapterCapabilityError,
    AdapterLifecycle,
    AdapterLifecycleError,
    AdapterRunResult,
    TraceSample,
)


_MISSING = object()
_SUPPORTED_SCENARIOS = {
    "zero_hold",
    "small_step",
    "gravity_settling",
    "offset_sine",
    "offset_linear_chirp",
}
_DIRECT_JOINT_GEAR = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _field(source: object, *names: str, default: object = _MISSING) -> object:
    """Read a field from a mapping or dataclass-like object."""

    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    if default is _MISSING:
        joined = ", ".join(names)
        raise ValueError(f"required field is missing (accepted names: {joined})")
    return default


def _plain_value(value: object) -> object:
    """Return the value of Enum-like objects without importing their type."""

    candidate = getattr(value, "value", value)
    return candidate


def _enum_label(enum_type: object, value: object, *, prefix: str) -> str:
    """Return a stable lower-case label for a compiled MuJoCo enum value."""

    member = enum_type(int(value))  # type: ignore[operator]
    name = str(getattr(member, "name", member))
    return name.removeprefix(prefix).lower()


def _mapping(value: object, *, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping from joint name to number")
    result: dict[str, float] = {}
    for name, raw in value.items():
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{name!r}] must be finite")
        result[str(name)] = number
    return result


def _names(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError("name selection must be a string or sequence of strings")
    return tuple(str(item) for item in value)


def _path_from_entry(entry: object, asset_root: Path | None = None) -> Path | None:
    if isinstance(entry, (str, Path)):
        raw = Path(entry).expanduser()
        if not raw.is_absolute() and asset_root is not None:
            raw = asset_root / raw
        return raw.resolve()
    if entry is None:
        return None
    raw = _field(
        entry,
        "path",
        "resolved_path",
        "source_path",
        "model_path",
        "mjcf_path",
        default=None,
    )
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute() and asset_root is not None:
        path = asset_root / path
    return path.resolve()


def _resolve_mjcf_path(manifest: object, asset_root: Path | None = None) -> Path:
    direct = _path_from_entry(
        _field(manifest, "mjcf_path", "mujoco_path", "model_path", default=None),
        asset_root,
    )
    if direct is not None:
        return direct

    for container_name in ("mujoco", "model_paths", "assets", "models"):
        container = _field(manifest, container_name, default=None)
        if container is None:
            continue
        if container_name == "mujoco":
            candidate = _path_from_entry(container, asset_root)
            if candidate is not None:
                return candidate
        if container_name == "model_paths":
            candidate = _path_from_entry(
                _field(container, "mujoco", "mjcf", default=None), asset_root
            )
            if candidate is not None:
                return candidate
        if isinstance(container, Mapping):
            for key in ("mujoco", "mjcf"):
                candidate = _path_from_entry(container.get(key), asset_root)
                if candidate is not None:
                    return candidate

    if isinstance(manifest, (str, Path)):
        return _path_from_entry(manifest, asset_root)  # type: ignore[return-value]
    raise ValueError("manifest does not provide a fixed-base MuJoCo MJCF path")


def _manifest_names(manifest: object, kind: str) -> tuple[str, ...]:
    direct = _field(
        manifest,
        f"expected_{kind}_names",
        f"trace_{kind}_names",
        *("distal_frame_names",) if kind == "frame" else (),
        default=None,
    )
    if direct is not None:
        return _names(direct)
    mapping = _field(manifest, "mapping", "canonical_mapping", default=None)
    if mapping is not None:
        nested = _field(mapping, f"{kind}_names", f"expected_{kind}_names", default=None)
        if nested is not None:
            return _names(nested)
    return ()


class MuJoCoAdapter:
    """A small, CPU-only adapter for canonical fixed-base scenarios.

    Direct lifecycle use is supported through ``open``/``reset``/``step`` and
    ``close``.  ``run_scenario`` is the safer one-shot entry point: it owns the
    complete lifecycle and always closes simulator state before returning.
    """

    backend_name = "mujoco"

    def __init__(
        self,
        *,
        asset_root: str | Path | None = None,
        dof_frictionloss_scale: float = 1.0,
        joint_total_damping_targets: Mapping[str, float] | None = None,
    ) -> None:
        if (
            isinstance(dof_frictionloss_scale, bool)
            or not isinstance(dof_frictionloss_scale, (int, float))
        ):
            raise ValueError("dof_frictionloss_scale must be a finite number >= 0")
        try:
            frictionloss_scale = float(dof_frictionloss_scale)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "dof_frictionloss_scale must be a finite number >= 0"
            ) from exc
        if not math.isfinite(frictionloss_scale) or frictionloss_scale < 0.0:
            raise ValueError("dof_frictionloss_scale must be a finite number >= 0")

        if joint_total_damping_targets is None:
            damping_targets: dict[str, float] = {}
        elif not isinstance(joint_total_damping_targets, Mapping):
            raise ValueError(
                "joint_total_damping_targets must map canonical joint names "
                "to finite numbers >= 0"
            )
        else:
            damping_targets = {}
            for joint_name, raw_target in joint_total_damping_targets.items():
                if not isinstance(joint_name, str) or not joint_name:
                    raise ValueError(
                        "joint_total_damping_targets keys must be non-empty "
                        "canonical joint names"
                    )
                if isinstance(raw_target, bool) or not isinstance(
                    raw_target, (int, float)
                ):
                    raise ValueError(
                        f"joint_total_damping_targets[{joint_name!r}] must be "
                        "a finite number >= 0"
                    )
                try:
                    target = float(raw_target)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"joint_total_damping_targets[{joint_name!r}] must be "
                        "a finite number >= 0"
                    ) from exc
                if not math.isfinite(target) or target < 0.0:
                    raise ValueError(
                        f"joint_total_damping_targets[{joint_name!r}] must be "
                        "a finite number >= 0"
                    )
                damping_targets[joint_name] = 0.0 if target == 0.0 else target

        self._lifecycle = AdapterLifecycle.CREATED
        self._asset_root = (
            Path(asset_root).expanduser().resolve() if asset_root is not None else None
        )
        self._dof_frictionloss_scale = (
            0.0 if frictionloss_scale == 0.0 else frictionloss_scale
        )
        self._joint_total_damping_targets = damping_targets
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._source_path: Path | None = None
        self._joint_names: tuple[str, ...] = ()
        self._joint_qpos_address: dict[str, int] = {}
        self._joint_actuator: dict[str, int] = {}
        self._position_targets: dict[str, float] = {}
        self._mapping_scope = "unscoped"
        self._original_dof_frictionloss: tuple[float, ...] = ()
        self._joint_total_damping_target_joint_names: tuple[str, ...] = ()
        self._joint_total_damping_records: tuple[dict[str, object], ...] = ()

    @property
    def lifecycle(self) -> AdapterLifecycle:
        return self._lifecycle

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def dt(self) -> float:
        self._require_open("read timestep")
        return float(self._model.opt.timestep)

    def _require_open(self, operation: str) -> None:
        if self._lifecycle not in {AdapterLifecycle.READY, AdapterLifecycle.RUNNING}:
            raise AdapterLifecycleError(
                f"cannot {operation} while MuJoCo adapter is {self._lifecycle.value}"
            )

    def open(self, manifest: object, *, dt_override: float | None = None) -> None:
        if self._lifecycle not in {AdapterLifecycle.CREATED, AdapterLifecycle.CLOSED}:
            raise AdapterLifecycleError(
                f"cannot open MuJoCo adapter while it is {self._lifecycle.value}"
            )

        mounting = _plain_value(_field(manifest, "mounting", default="fixed_base"))
        fixed_base = bool(_field(manifest, "fixed_base", default=mounting == "fixed_base"))
        if not fixed_base:
            raise AdapterCapabilityError(
                self.backend_name,
                "fixed_base_only",
                "Gate 0 excludes floating-base models.",
            )
        control_mode = _plain_value(_field(manifest, "control_mode", default="position"))
        if control_mode != "position":
            raise AdapterCapabilityError(
                self.backend_name,
                "position_control_only",
                f"Gate 0 excludes control mode {control_mode!r}.",
            )

        source = _resolve_mjcf_path(manifest, self._asset_root)
        if not source.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {source}")
        if source.suffix.lower() not in {".xml", ".mjcf"}:
            raise ValueError(f"MuJoCo source must be .xml or .mjcf, got: {source.name}")

        if dt_override is not None:
            dt_override = float(dt_override)
            if not math.isfinite(dt_override) or dt_override <= 0.0:
                raise ValueError("dt_override must be a positive finite number")

        try:
            mujoco = import_module("mujoco")
        except ImportError as error:
            raise AdapterCapabilityError(
                self.backend_name,
                "python_package",
                "the 'mujoco' package is not installed",
                hint="Install the project's 'simulation' extra.",
            ) from error

        try:
            model = mujoco.MjModel.from_xml_path(str(source))
            if dt_override is not None:
                model.opt.timestep = dt_override
            original_dof_frictionloss = np.asarray(
                model.dof_frictionloss, dtype=float
            ).copy()
            if not np.all(np.isfinite(original_dof_frictionloss)):
                raise FloatingPointError(
                    "compiled MuJoCo model contains non-finite dof_frictionloss"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                effective_dof_frictionloss = (
                    original_dof_frictionloss * self._dof_frictionloss_scale
                )
            if not np.all(np.isfinite(effective_dof_frictionloss)):
                raise FloatingPointError(
                    "dof_frictionloss_scale produces non-finite compiled values"
                )
            model.dof_frictionloss[:] = effective_dof_frictionloss
            self._validate_fixed_base(model, mujoco)
            joint_names, addresses = self._discover_joints(model, mujoco)
            actuators = self._discover_position_actuators(model, mujoco)
            missing_actuators = sorted(set(joint_names) - set(actuators))
            if missing_actuators:
                raise AdapterCapabilityError(
                    self.backend_name,
                    "position_actuator_mapping",
                    f"{len(missing_actuators)} joint(s) lack a bound actuator: "
                    + ", ".join(missing_actuators[:5]),
                )

            expected_joints = _manifest_names(manifest, "joint")
            if expected_joints:
                missing = sorted(set(expected_joints) - set(joint_names))
                unexpected = sorted(set(joint_names) - set(expected_joints))
                if missing or unexpected:
                    raise ValueError(
                        "MuJoCo joint set differs from the manifest; "
                        f"missing={missing}, unexpected={unexpected}"
                    )

            expected_frames = _manifest_names(manifest, "frame")
            missing_frames = [
                name
                for name in expected_frames
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
            ]
            if missing_frames:
                raise ValueError(f"MuJoCo is missing manifest frame(s): {missing_frames}")

            (
                damping_target_joint_names,
                damping_records,
            ) = self._apply_joint_total_damping_targets(
                model,
                mujoco,
                joint_names,
                actuators,
            )

            self._mujoco = mujoco
            self._model = model
            self._data = mujoco.MjData(model)
            self._source_path = source
            self._joint_names = joint_names
            self._joint_qpos_address = addresses
            self._joint_actuator = actuators
            self._position_targets = {name: 0.0 for name in joint_names}
            self._original_dof_frictionloss = tuple(
                float(value) for value in original_dof_frictionloss
            )
            self._joint_total_damping_target_joint_names = (
                damping_target_joint_names
            )
            self._joint_total_damping_records = damping_records
            raw_scope = _plain_value(_field(manifest, "side", default=None))
            if raw_scope is None and expected_joints:
                prefixes = {name.split("_", 1)[0] for name in expected_joints}
                raw_scope = next(iter(prefixes)) if len(prefixes) == 1 else None
            self._mapping_scope = str(raw_scope) if raw_scope is not None else "unscoped"
            self._lifecycle = AdapterLifecycle.READY
            self.reset()
        except Exception:
            self._lifecycle = AdapterLifecycle.FAILED
            self._mujoco = None
            self._model = None
            self._data = None
            self._source_path = None
            self._joint_names = ()
            self._joint_qpos_address = {}
            self._joint_actuator = {}
            self._position_targets = {}
            self._mapping_scope = "unscoped"
            self._original_dof_frictionloss = ()
            self._joint_total_damping_target_joint_names = ()
            self._joint_total_damping_records = ()
            raise

    @staticmethod
    def _validate_fixed_base(model: object, mujoco: object) -> None:
        unsupported: list[str] = []
        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        ball_type = int(mujoco.mjtJoint.mjJNT_BALL)
        for joint_id in range(int(model.njnt)):
            if int(model.jnt_type[joint_id]) in {free_type, ball_type}:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                unsupported.append(name or f"joint_{joint_id}")
        if unsupported:
            raise AdapterCapabilityError(
                "mujoco",
                "fixed_base_only",
                "free or ball root joints were found: " + ", ".join(unsupported),
            )

    @staticmethod
    def _discover_joints(model: object, mujoco: object) -> tuple[tuple[str, ...], dict[str, int]]:
        names: list[str] = []
        addresses: dict[str, int] = {}
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for joint_id in range(int(model.njnt)):
            joint_type = int(model.jnt_type[joint_id])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                raise ValueError(f"MuJoCo joint {joint_id} has no name")
            if joint_type not in {hinge, slide}:
                raise AdapterCapabilityError(
                    "mujoco",
                    "one_dof_joint_mapping",
                    f"joint {name!r} is not hinge or slide",
                )
            names.append(str(name))
            addresses[str(name)] = int(model.jnt_qposadr[joint_id])
        return tuple(names), addresses

    @staticmethod
    def _discover_position_actuators(model: object, mujoco: object) -> dict[str, int]:
        result: dict[str, int] = {}
        affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
        joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
        for actuator_id in range(int(model.nu)):
            gain = float(model.actuator_gainprm[actuator_id, 0])
            position_bias = float(model.actuator_biasprm[actuator_id, 1])
            is_position_servo = (
                int(model.actuator_biastype[actuator_id]) == affine_bias
                and gain != 0.0
                and math.isclose(position_bias, -gain, rel_tol=1e-9, abs_tol=1e-12)
            )
            if not is_position_servo:
                continue
            transmission = int(model.actuator_trntype[actuator_id])
            gear = np.asarray(model.actuator_gear[actuator_id], dtype=float)
            direct_gear = np.asarray(_DIRECT_JOINT_GEAR, dtype=float)
            if transmission != joint_transmission or not np.array_equal(gear, direct_gear):
                actuator_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                )
                raise AdapterCapabilityError(
                    "mujoco",
                    "direct_position_actuator",
                    "Gate 0 requires a direct joint transmission with unity gear; "
                    f"actuator {actuator_name or actuator_id!r} has "
                    f"transmission={transmission}, gear={gear.tolist()}",
                )
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or joint_id >= int(model.njnt):
                continue
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name and str(joint_name) not in result:
                result[str(joint_name)] = actuator_id
        return result

    def _apply_joint_total_damping_targets(
        self,
        model: object,
        mujoco: object,
        joint_names: tuple[str, ...],
        actuators: Mapping[str, int],
    ) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
        unknown = sorted(set(self._joint_total_damping_targets) - set(joint_names))
        if unknown:
            raise ValueError(
                "joint_total_damping_targets contains unknown canonical joint(s): "
                + ", ".join(unknown)
            )

        target_joint_names = tuple(
            name for name in joint_names if name in self._joint_total_damping_targets
        )
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        records: list[dict[str, object]] = []
        for canonical_id in joint_names:
            actuator_id = actuators[canonical_id]
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            dof_index = int(model.jnt_dofadr[joint_id])
            original_gainprm = np.asarray(
                model.actuator_gainprm[actuator_id], dtype=float
            ).copy()
            original_biasprm = np.asarray(
                model.actuator_biasprm[actuator_id], dtype=float
            ).copy()
            if not np.all(np.isfinite(original_gainprm)) or not np.all(
                np.isfinite(original_biasprm)
            ):
                raise FloatingPointError(
                    f"non-finite compiled actuator parameters for {canonical_id!r}"
                )
            original_controller_kv = -float(
                model.actuator_biasprm[actuator_id, 2]
            )
            joint_damping = float(model.dof_damping[dof_index])
            original_total = original_controller_kv + joint_damping
            if not all(
                math.isfinite(value)
                for value in (
                    original_controller_kv,
                    joint_damping,
                    original_total,
                )
            ):
                raise FloatingPointError(
                    f"non-finite compiled damping parameter for {canonical_id!r}"
                )

            override_applied = canonical_id in self._joint_total_damping_targets
            requested_total = (
                self._joint_total_damping_targets[canonical_id]
                if override_applied
                else original_total
            )
            effective_controller_kv = (
                requested_total - joint_damping
                if override_applied
                else original_controller_kv
            )
            if override_applied and effective_controller_kv < 0.0:
                raise ValueError(
                    f"joint_total_damping_targets[{canonical_id!r}] must be >= "
                    f"compiled joint damping {joint_damping}"
                )
            if effective_controller_kv == 0.0:
                effective_controller_kv = 0.0
            if override_applied:
                model.actuator_biasprm[actuator_id, 2] = -effective_controller_kv

            effective_controller_readback = -float(
                model.actuator_biasprm[actuator_id, 2]
            )
            effective_total = effective_controller_readback + joint_damping
            effective_gainprm = np.asarray(
                model.actuator_gainprm[actuator_id], dtype=float
            ).copy()
            effective_biasprm = np.asarray(
                model.actuator_biasprm[actuator_id], dtype=float
            ).copy()
            changed_gainprm_indices = tuple(
                int(index)
                for index in np.flatnonzero(original_gainprm != effective_gainprm)
            )
            changed_biasprm_indices = tuple(
                int(index)
                for index in np.flatnonzero(original_biasprm != effective_biasprm)
            )
            if changed_gainprm_indices or any(
                index != 2 for index in changed_biasprm_indices
            ):
                raise RuntimeError(
                    "joint total damping override changed actuator parameters "
                    f"outside biasprm[2] for {canonical_id!r}"
                )
            if not all(
                math.isfinite(value)
                for value in (effective_controller_readback, effective_total)
            ):
                raise FloatingPointError(
                    f"non-finite effective damping parameter for {canonical_id!r}"
                )
            joint_type = int(model.jnt_type[joint_id])
            damping_unit = "N*m*s/rad" if joint_type == hinge_type else "N*s/m"
            records.append(
                {
                    "canonical_id": canonical_id,
                    "joint_id": joint_id,
                    "dof_index": dof_index,
                    "actuator_id": actuator_id,
                    "override_applied": override_applied,
                    "original_actuator_gainprm": tuple(
                        float(value) for value in original_gainprm
                    ),
                    "effective_actuator_gainprm": tuple(
                        float(value) for value in effective_gainprm
                    ),
                    "original_actuator_biasprm": tuple(
                        float(value) for value in original_biasprm
                    ),
                    "effective_actuator_biasprm": tuple(
                        float(value) for value in effective_biasprm
                    ),
                    "changed_gainprm_indices": changed_gainprm_indices,
                    "changed_biasprm_indices": changed_biasprm_indices,
                    "original_controller_damping_kv": original_controller_kv,
                    "joint_damping": joint_damping,
                    "original_total_damping": original_total,
                    "requested_total_damping": requested_total,
                    "effective_controller_damping_kv": (
                        effective_controller_readback
                    ),
                    "effective_total_damping": effective_total,
                    "unit": damping_unit,
                }
            )
        return target_joint_names, tuple(records)

    def _compiled_control_parameters(
        self,
        selected_joints: tuple[str, ...],
    ) -> dict[str, object]:
        """Describe compiled controller, joint, and solver configuration.

        These are configuration readbacks from ``MjModel``.  They are not a
        measurement or estimate of actuator torque produced during a step.
        """

        if len(set(selected_joints)) != len(selected_joints):
            raise ValueError("canonical joint selection contains duplicate names")

        hinge_type = int(self._mujoco.mjtJoint.mjJNT_HINGE)
        slide_type = int(self._mujoco.mjtJoint.mjJNT_SLIDE)
        joint_transmission = int(self._mujoco.mjtTrn.mjTRN_JOINT)
        expected_gear = np.asarray(_DIRECT_JOINT_GEAR, dtype=float)
        joint_records: list[dict[str, object]] = []

        for canonical_id in selected_joints:
            if canonical_id not in self._joint_actuator:
                raise ValueError(
                    f"canonical joint {canonical_id!r} has no compiled position actuator"
                )
            actuator_id = self._joint_actuator[canonical_id]
            joint_id = int(self._model.actuator_trnid[actuator_id, 0])
            dof_index = int(self._model.jnt_dofadr[joint_id])
            transmission = int(self._model.actuator_trntype[actuator_id])
            gear = np.asarray(self._model.actuator_gear[actuator_id], dtype=float)
            if not np.all(np.isfinite(gear)):
                raise FloatingPointError(
                    f"non-finite compiled actuator gear for {canonical_id!r}"
                )
            if transmission != joint_transmission or not np.array_equal(
                gear, expected_gear
            ):
                raise AdapterCapabilityError(
                    self.backend_name,
                    "direct_position_actuator",
                    "Gate 0 requires a direct joint transmission with unity gear; "
                    f"joint {canonical_id!r} has transmission={transmission}, "
                    f"gear={gear.tolist()}",
                )

            joint_type = int(self._model.jnt_type[joint_id])
            if joint_type == hinge_type:
                joint_type_name = "hinge"
                coordinate_unit = "rad"
                effort_unit = "N*m"
                stiffness_unit = "N*m/rad"
                damping_unit = "N*m*s/rad"
                armature_unit = "kg*m^2"
            elif joint_type == slide_type:
                joint_type_name = "slide"
                coordinate_unit = "m"
                effort_unit = "N"
                stiffness_unit = "N/m"
                damping_unit = "N*s/m"
                armature_unit = "kg"
            else:  # Guard the one-DOF invariant if discovery changes later.
                raise AdapterCapabilityError(
                    self.backend_name,
                    "one_dof_joint_mapping",
                    f"joint {canonical_id!r} is not hinge or slide",
                )

            controller_stiffness_kp = float(
                self._model.actuator_gainprm[actuator_id, 0]
            )
            # MuJoCo's affine position-servo bias is -kp*q - kv*qvel.
            controller_damping_kv = -float(
                self._model.actuator_biasprm[actuator_id, 2]
            )
            forcerange = tuple(
                float(value) for value in self._model.actuator_forcerange[actuator_id]
            )
            joint_armature = float(self._model.dof_armature[dof_index])
            joint_damping = float(self._model.dof_damping[dof_index])
            joint_frictionloss = float(self._model.dof_frictionloss[dof_index])
            numeric_values = (
                controller_stiffness_kp,
                controller_damping_kv,
                *forcerange,
                joint_armature,
                joint_damping,
                joint_frictionloss,
            )
            if not all(math.isfinite(value) for value in numeric_values):
                raise FloatingPointError(
                    f"non-finite compiled control parameter for {canonical_id!r}"
                )

            actuator_name = self._mujoco.mj_id2name(
                self._model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            joint_records.append(
                {
                    "canonical_id": canonical_id,
                    "joint_type": joint_type_name,
                    "joint_id": joint_id,
                    "dof_index": dof_index,
                    "actuator_id": actuator_id,
                    "actuator_name": str(actuator_name) if actuator_name else None,
                    "transmission_type": "joint",
                    "direct_gear": True,
                    "controller_stiffness_kp": controller_stiffness_kp,
                    "controller_damping_kv": controller_damping_kv,
                    "gear": [float(value) for value in gear],
                    "forcelimited": bool(
                        self._model.actuator_forcelimited[actuator_id]
                    ),
                    "forcerange": list(forcerange),
                    "joint_armature": joint_armature,
                    "joint_damping": joint_damping,
                    "joint_frictionloss": joint_frictionloss,
                    "units": {
                        "joint_coordinate": coordinate_unit,
                        "controller_stiffness_kp": stiffness_unit,
                        "controller_damping_kv": damping_unit,
                        "gear": "dimensionless",
                        "forcerange": effort_unit,
                        "joint_armature": armature_unit,
                        "joint_damping": damping_unit,
                        "joint_frictionloss": effort_unit,
                    },
                }
            )

        option = self._model.opt
        gravity = tuple(float(value) for value in option.gravity)
        solver_values = (
            float(option.timestep),
            float(option.tolerance),
            *gravity,
        )
        if not all(math.isfinite(value) for value in solver_values):
            raise FloatingPointError("non-finite compiled MuJoCo solver parameter")

        return {
            "schema_version": 1,
            "source": "mujoco_compiled_mjmodel_arrays",
            "semantics": "compiled_configuration_only_not_measured_torque",
            "contains_measured_or_realized_torque": False,
            "joint_order": list(selected_joints),
            "field_sources": {
                "controller_stiffness_kp": "model.actuator_gainprm[actuator_id, 0]",
                "controller_damping_kv": "-model.actuator_biasprm[actuator_id, 2]",
                "gear": "model.actuator_gear[actuator_id]",
                "forcelimited": "model.actuator_forcelimited[actuator_id]",
                "forcerange": "model.actuator_forcerange[actuator_id]",
                "joint_armature": "model.dof_armature[dof_index]",
                "joint_damping": "model.dof_damping[dof_index]",
                "joint_frictionloss": "model.dof_frictionloss[dof_index]",
            },
            "joints": joint_records,
            "solver": {
                "source": "model.opt",
                "solver": _enum_label(
                    self._mujoco.mjtSolver,
                    option.solver,
                    prefix="mjSOL_",
                ),
                "solver_enum_value": int(option.solver),
                "integrator": _enum_label(
                    self._mujoco.mjtIntegrator,
                    option.integrator,
                    prefix="mjINT_",
                ),
                "integrator_enum_value": int(option.integrator),
                "timestep": float(option.timestep),
                "iterations": int(option.iterations),
                "ls_iterations": int(option.ls_iterations),
                "tolerance": float(option.tolerance),
                "gravity": list(gravity),
                "units": {
                    "timestep": "s",
                    "iterations": "count",
                    "ls_iterations": "count",
                    "tolerance": "dimensionless",
                    "gravity": "m/s^2",
                },
                "field_sources": {
                    "solver": "model.opt.solver",
                    "integrator": "model.opt.integrator",
                    "timestep": "model.opt.timestep",
                    "iterations": "model.opt.iterations",
                    "ls_iterations": "model.opt.ls_iterations",
                    "tolerance": "model.opt.tolerance",
                    "gravity": "model.opt.gravity",
                },
            },
        }

    def reset(self) -> None:
        self._require_open("reset")
        self._mujoco.mj_resetData(self._model, self._data)
        self._position_targets = {name: 0.0 for name in self._joint_names}
        self.set_position_targets(self._position_targets)
        self._mujoco.mj_forward(self._model, self._data)
        self._check_finite(step=0)

    def set_joint_positions(self, positions: Mapping[str, float]) -> None:
        """Set named initial joint positions, then recompute derived state."""

        self._require_open("set initial joint positions")
        values = _mapping(positions, field_name="initial_positions")
        unknown = sorted(set(values) - set(self._joint_qpos_address))
        if unknown:
            raise ValueError(f"unknown MuJoCo joint position(s): {unknown}")
        for name, value in values.items():
            self._data.qpos[self._joint_qpos_address[name]] = value
        self._data.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, self._data)
        self._check_finite(step=0)

    def set_position_targets(self, targets: Mapping[str, float]) -> None:
        self._require_open("set position targets")
        values = _mapping(targets, field_name="position_targets")
        unknown = sorted(set(values) - set(self._joint_actuator))
        if unknown:
            raise ValueError(f"joint(s) have no MuJoCo position actuator: {unknown}")
        for name, value in values.items():
            self._data.ctrl[self._joint_actuator[name]] = value
            self._position_targets[name] = value
        self._check_array("ctrl", self._data.ctrl, step=0)

    def step(self) -> None:
        self._require_open("step")
        self._mujoco.mj_step(self._model, self._data)
        self._check_finite(step=-1)

    @staticmethod
    def _check_array(name: str, values: object, *, step: int) -> None:
        array = np.asarray(values)
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"non-finite MuJoCo {name} at step {step}")

    def _check_finite(self, *, step: int) -> None:
        for name in ("qpos", "qvel", "qacc", "ctrl", "act", "xpos", "xquat"):
            self._check_array(name, getattr(self._data, name), step=step)

    def sample(
        self,
        *,
        step: int,
        joint_names: tuple[str, ...] = (),
        frame_names: tuple[str, ...] = (),
    ) -> TraceSample:
        self._require_open("sample")
        selected_joints = joint_names or self._joint_names
        unknown_joints = sorted(set(selected_joints) - set(self._joint_qpos_address))
        if unknown_joints:
            raise ValueError(f"cannot trace unknown MuJoCo joint(s): {unknown_joints}")

        poses: dict[str, tuple[float, ...]] = {}
        if frame_names:
            # mj_step integrates qpos after forward dynamics, so derived Cartesian
            # body poses still describe the pre-integration state.  Refresh only
            # kinematics here to keep frame poses aligned with this sample's qpos
            # without advancing or otherwise changing the simulated state.
            self._mujoco.mj_kinematics(self._model, self._data)
        for name in frame_names:
            body_id = self._mujoco.mj_name2id(
                self._model, self._mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                raise ValueError(f"cannot trace unknown MuJoCo body frame: {name}")
            pose = tuple(float(value) for value in self._data.xpos[body_id]) + tuple(
                float(value) for value in self._data.xquat[body_id]
            )
            if not all(math.isfinite(value) for value in pose):
                raise FloatingPointError(f"non-finite MuJoCo frame pose for {name!r} at step {step}")
            poses[name] = pose

        positions = {
            name: float(self._data.qpos[self._joint_qpos_address[name]])
            for name in selected_joints
        }
        return TraceSample(
            step=step,
            time_s=float(self._data.time),
            qpos=tuple(float(value) for value in self._data.qpos),
            qvel=tuple(float(value) for value in self._data.qvel),
            joint_positions=positions,
            frame_poses=poses,
            position_targets={
                name: self._position_targets[name]
                for name in selected_joints
                if name in self._position_targets
            },
            contact_count=int(self._data.ncon),
        )

    def _provenance(
        self,
        *,
        joint_names: tuple[str, ...] = (),
        frame_names: tuple[str, ...] = (),
    ) -> dict[str, object]:
        assert self._source_path is not None
        selected_joints = joint_names or self._joint_names
        backend_joint_names = tuple(
            name
            for name, _index in sorted(
                self._joint_qpos_address.items(), key=lambda item: item[1]
            )
        )
        backend_frame_names = tuple(
            str(name)
            for body_id in range(int(self._model.nbody))
            if (
                name := self._mujoco.mj_id2name(
                    self._model, self._mujoco.mjtObj.mjOBJ_BODY, body_id
                )
            )
        )
        if len(backend_frame_names) != int(self._model.nbody):
            raise RuntimeError("MuJoCo body index domain contains an unnamed body")
        joint_mapping = tuple(
            JointMapping(
                canonical_id=name,
                backend=self.backend_name,
                scope=self._mapping_scope,
                backend_name=name,
                index=self._joint_qpos_address[name],
                sign=1.0,
                offset=0.0,
                unit="rad",
            )
            for name in selected_joints
        )
        frame_mapping = tuple(
            FrameMapping(
                canonical_id=name,
                backend=self.backend_name,
                scope=self._mapping_scope,
                backend_name=name,
                index=index,
                sign=1.0,
                offset=0.0,
                unit="xyz_m_qwxyz",
            )
            for name in frame_names
            for index in (
                self._mujoco.mj_name2id(
                    self._model, self._mujoco.mjtObj.mjOBJ_BODY, name
                ),
            )
        )
        compiled_control_parameters = self._compiled_control_parameters(
            selected_joints
        )
        effective_dof_frictionloss = tuple(
            float(value) for value in self._model.dof_frictionloss
        )
        if len(self._original_dof_frictionloss) != int(self._model.nv):
            raise RuntimeError(
                "original dof_frictionloss audit snapshot does not cover all compiled DOFs"
            )
        damping_record_order = tuple(
            str(record["canonical_id"])
            for record in self._joint_total_damping_records
        )
        if damping_record_order != self._joint_names:
            raise RuntimeError(
                "joint total damping audit snapshot does not cover compiled joint order"
            )
        return {
            "backend": self.backend_name,
            "backend_version": getattr(self._mujoco, "__version__", None),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "source_path": str(self._source_path),
            "source_sha256": sha256(self._source_path.read_bytes()).hexdigest(),
            "device": "cpu",
            "headless": True,
            "renderer": False,
            "dof_frictionloss_scale": self._dof_frictionloss_scale,
            "dof_frictionloss_mode": (
                "canonical_no_override"
                if self._dof_frictionloss_scale == 1.0
                else "diagnostic_scaled_override"
            ),
            "dof_frictionloss_scope": "all_compiled_dofs",
            "dof_frictionloss_order": "model_dof_index_ascending",
            "dof_frictionloss_dof_indices": list(range(int(self._model.nv))),
            "dof_frictionloss_original": list(self._original_dof_frictionloss),
            "dof_frictionloss_effective": list(effective_dof_frictionloss),
            "joint_total_damping_mode": (
                "diagnostic_total_damping_override"
                if self._joint_total_damping_targets
                else "canonical_no_override"
            ),
            "joint_total_damping_scope": "all_compiled_canonical_joints",
            "joint_total_damping_order": "model_joint_index_ascending",
            "joint_total_damping_semantics": (
                "total=actuator_controller_kv+model_dof_damping"
            ),
            "joint_total_damping_target_joint_names": list(
                self._joint_total_damping_target_joint_names
            ),
            "joint_total_damping_records": [
                {
                    **record,
                    "original_actuator_gainprm": list(
                        record["original_actuator_gainprm"]
                    ),
                    "effective_actuator_gainprm": list(
                        record["effective_actuator_gainprm"]
                    ),
                    "original_actuator_biasprm": list(
                        record["original_actuator_biasprm"]
                    ),
                    "effective_actuator_biasprm": list(
                        record["effective_actuator_biasprm"]
                    ),
                    "changed_gainprm_indices": list(
                        record["changed_gainprm_indices"]
                    ),
                    "changed_biasprm_indices": list(
                        record["changed_biasprm_indices"]
                    ),
                }
                for record in self._joint_total_damping_records
            ],
            "mapping_schema_version": MAPPING_SCHEMA_VERSION,
            "backend_joint_names": list(backend_joint_names),
            "backend_frame_names": list(backend_frame_names),
            "frame_pose_source": "mujoco_data_xpos_xquat_after_mj_kinematics",
            "frame_pose_state_alignment": "current_sample_qpos",
            "joint_mapping": [entry.to_dict() for entry in joint_mapping],
            "frame_mapping": [entry.to_dict() for entry in frame_mapping],
            "compiled_control_parameters": compiled_control_parameters,
        }

    def _trace_names(self, manifest: object, scenario: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
        joints = _names(
            _field(scenario, "trace_joint_names", "joint_names", default=None)
        ) or _manifest_names(manifest, "joint") or self._joint_names
        frames = _names(
            _field(scenario, "trace_frame_names", "frame_names", default=None)
        ) or _manifest_names(manifest, "frame")
        if not frames:
            frames = tuple(
                name
                for body_id in range(1, int(self._model.nbody))
                if (name := self._mujoco.mj_id2name(
                    self._model, self._mujoco.mjtObj.mjOBJ_BODY, body_id
                ))
                and str(name).endswith("_DP")
            )
        return joints, tuple(str(name) for name in frames)

    @staticmethod
    def _step_count(scenario: object, dt: float) -> int:
        raw_duration = _field(scenario, "duration_s", "duration", default=None)
        if raw_duration is not None:
            duration = float(raw_duration)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("scenario duration must be a positive finite number")
            ratio = duration / dt
            nearest = round(ratio)
            steps = (
                nearest
                if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-9)
                else int(math.ceil(ratio))
            )
        else:
            raw_steps = _field(scenario, "steps", "num_steps", default=None)
            steps = int(raw_steps) if raw_steps is not None else 1
        if steps < 1:
            raise ValueError("scenario must request at least one simulation step")
        return steps

    def _scenario_targets(
        self,
        scenario: object,
        *,
        kind: str,
        initial: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        baseline = {name: float(initial.get(name, 0.0)) for name in self._joint_names}
        explicit = _mapping(
            _field(scenario, "position_targets", "targets", default=None),
            field_name="position_targets",
        )
        if kind in {"zero_hold", "gravity_settling"}:
            hold = dict(baseline)
            hold.update(explicit)
            return hold, hold

        stepped = dict(baseline)
        if explicit:
            stepped.update(explicit)
        else:
            selected = _names(
                _field(scenario, "target_joint_names", "target_joints", default=None)
            )
            if not selected:
                selected = self._joint_names
            amplitude = float(
                _field(
                    scenario,
                    "target_position_rad",
                    "amplitude",
                    "step_amplitude",
                    default=0.05,
                )
            )
            if not math.isfinite(amplitude):
                raise ValueError("small_step amplitude must be finite")
            for name in selected:
                if name not in stepped:
                    raise ValueError(f"small_step references unknown joint: {name}")
                stepped[name] = baseline[name] + amplitude
        return baseline, stepped

    def run_scenario(
        self,
        manifest: object,
        scenario: object,
        *,
        dt_override: float | None = None,
    ) -> AdapterRunResult:
        if self._lifecycle not in {AdapterLifecycle.CREATED, AdapterLifecycle.CLOSED}:
            raise AdapterLifecycleError(
                "one-shot run_scenario requires a newly created or closed adapter"
            )

        scenario_id = str(_field(scenario, "scenario_id", "id", "name", default="scenario"))
        kind = str(_plain_value(_field(scenario, "kind", "scenario_kind", default="zero_hold")))
        requested_steps = 0
        completed_steps = 0
        effective_dt = float(dt_override) if dt_override is not None else 0.0
        samples: list[TraceSample] = []
        joint_names: tuple[str, ...] = ()
        frame_names: tuple[str, ...] = ()
        provenance: dict[str, object] = {}
        scheduled_target_digest: CanonicalTargetSequenceDigest | None = None

        try:
            if kind not in _SUPPORTED_SCENARIOS:
                raise AdapterCapabilityError(
                    self.backend_name,
                    "scenario_kind",
                    f"adapter supports only {sorted(_SUPPORTED_SCENARIOS)}, got {kind!r}",
                )
            scenario_dt = _field(scenario, "dt_s", "dt", "timestep", default=None)
            chosen_dt = dt_override if dt_override is not None else scenario_dt
            self.open(manifest, dt_override=float(chosen_dt) if chosen_dt is not None else None)
            effective_dt = self.dt
            requested_steps = self._step_count(scenario, effective_dt)
            joint_names, frame_names = self._trace_names(manifest, scenario)
            if isinstance(scenario, (ScenarioSpec, WaveformScenarioSpec)):
                scheduled_target_digest = CanonicalTargetSequenceDigest(joint_names)
            initial = (
                canonical_initial_positions(scenario, joint_names)
                if isinstance(scenario, (ScenarioSpec, WaveformScenarioSpec))
                else _mapping(
                    _field(scenario, "initial_positions", "q0", default=None),
                    field_name="initial_positions",
                )
            )
            if initial:
                self.set_joint_positions(initial)
            gravity = _field(scenario, "gravity_m_s2", "gravity", default=None)
            if gravity is not None:
                if not isinstance(gravity, Sequence) or isinstance(gravity, str) or len(gravity) != 3:
                    raise ValueError("gravity_m_s2 must contain exactly three numbers")
                gravity_values = np.asarray(tuple(float(item) for item in gravity), dtype=float)
                self._check_array("gravity", gravity_values, step=0)
                self._model.opt.gravity[:] = gravity_values
                self._mujoco.mj_forward(self._model, self._data)
            if isinstance(scenario, (ScenarioSpec, WaveformScenarioSpec)):
                baseline = canonical_position_targets(
                    scenario,
                    joint_names,
                    step_index=0,
                    dt_s=effective_dt,
                )
                assert scheduled_target_digest is not None
                scheduled_target_digest.append(baseline)
                stepped = dict(baseline)
            else:
                baseline, stepped = self._scenario_targets(
                    scenario, kind=kind, initial=initial
                )
            self.set_position_targets(baseline)
            self._mujoco.mj_forward(self._model, self._data)
            self._check_finite(step=0)
            samples.append(
                self.sample(step=0, joint_names=joint_names, frame_names=frame_names)
            )

            onset_raw = _field(scenario, "onset_step", default=None)
            if onset_raw is None:
                start_s = _field(scenario, "step_start_s", "step_start", default=None)
                onset = (
                    int(round(float(start_s) / effective_dt))
                    if start_s is not None
                    else max(1, requested_steps // 4)
                )
            else:
                onset = int(onset_raw)
            if onset < 0 or onset > requested_steps:
                raise ValueError("small_step onset must be within the scenario step range")
            expect_no_contacts = bool(
                _field(
                    scenario,
                    "expect_no_contacts",
                    default=not bool(_field(scenario, "allow_contacts", default=False)),
                )
            )
            self._lifecycle = AdapterLifecycle.RUNNING
            for step_index in range(1, requested_steps + 1):
                # Advance to the sample timestamp under the command recorded at
                # the previous timestamp.  A command change at t therefore
                # affects [t, t + dt), rather than leaking one integration
                # interval early (and by a different amount after dt-halving).
                self._mujoco.mj_step(self._model, self._data)
                self._check_finite(step=step_index)
                if expect_no_contacts and int(self._data.ncon) != 0:
                    raise RuntimeError(
                        f"unexpected MuJoCo contact(s) at step {step_index}: {int(self._data.ncon)}"
                    )
                completed_steps = step_index
                if isinstance(scenario, (ScenarioSpec, WaveformScenarioSpec)):
                    targets = canonical_position_targets(
                        scenario,
                        joint_names,
                        step_index=step_index,
                        dt_s=effective_dt,
                    )
                    assert scheduled_target_digest is not None
                    scheduled_target_digest.append(targets)
                    if targets != self._position_targets:
                        self.set_position_targets(targets)
                elif kind == "small_step" and step_index >= onset:
                    self.set_position_targets(stepped)
                samples.append(
                    self.sample(
                        step=step_index,
                        joint_names=joint_names,
                        frame_names=frame_names,
                    )
                )
            if scheduled_target_digest is not None:
                assert isinstance(
                    scenario,
                    (ScenarioSpec, WaveformScenarioSpec),
                )
                if scheduled_target_digest.count != requested_steps + 1:
                    raise RuntimeError(
                        "MuJoCo scheduled target-sequence audit is incomplete"
                    )
                expected_schedule_sha256 = canonical_target_sequence_sha256(
                    scenario,
                    joint_names,
                    dt_s=effective_dt,
                    include_terminal=True,
                )
                if scheduled_target_digest.hexdigest() != expected_schedule_sha256:
                    raise RuntimeError(
                        "MuJoCo scheduled target sequence does not match the "
                        "canonical scenario"
                    )
            provenance = self._provenance(
                joint_names=joint_names,
                frame_names=frame_names,
            )
            if scheduled_target_digest is not None:
                provenance.update(
                    {
                        "target_sequence_digest_schema_version": 1,
                        "target_sequence_digest_encoding": (
                            "utf8_json_lines_float_hex_v1"
                        ),
                        "target_sequence_digest_projection": (
                            "ieee754_binary32_roundtrip"
                        ),
                        "target_sequence_canonical_joint_names": list(joint_names),
                        "scheduled_target_sequence_semantics": (
                            "q[0..N]; target q[k] is recorded at t_k and applies to "
                            "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
                        ),
                        "scheduled_target_sequence_count": (
                            scheduled_target_digest.count
                        ),
                        "scheduled_target_sequence_sha256": (
                            scheduled_target_digest.hexdigest()
                        ),
                    }
                )
            self._lifecycle = AdapterLifecycle.READY
            result = AdapterRunResult(
                backend=self.backend_name,
                scenario_id=scenario_id,
                status="completed",
                message="Headless fixed-base MuJoCo scenario completed with finite state.",
                dt=effective_dt,
                requested_steps=requested_steps,
                completed_steps=completed_steps,
                joint_names=joint_names,
                frame_names=frame_names,
                samples=tuple(samples),
                provenance=provenance,
            )
        except Exception as error:
            if self._source_path is not None and self._mujoco is not None:
                provenance = self._provenance(
                    joint_names=joint_names,
                    frame_names=frame_names,
                )
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
                dt=effective_dt,
                requested_steps=requested_steps,
                completed_steps=completed_steps,
                joint_names=joint_names,
                frame_names=frame_names,
                samples=tuple(samples),
                provenance=provenance,
                error=detail,
            )
        finally:
            self.close()
        return result

    def close(self) -> None:
        self._data = None
        self._model = None
        self._mujoco = None
        self._source_path = None
        self._joint_names = ()
        self._joint_qpos_address = {}
        self._joint_actuator = {}
        self._position_targets = {}
        self._mapping_scope = "unscoped"
        self._original_dof_frictionloss = ()
        self._joint_total_damping_target_joint_names = ()
        self._joint_total_damping_records = ()
        self._lifecycle = AdapterLifecycle.CLOSED


__all__ = ["MuJoCoAdapter"]
