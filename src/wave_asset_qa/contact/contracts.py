"""Closed, simulator-neutral contracts for WaveSimParity Contact Gate C0.

C0 deliberately uses a synthetic sphere against a static box.  The frozen
fixture isolates one normal-contact pair and therefore does not validate the
native fingertip collision meshes.  Every public ``from_dict`` constructor is
fail-closed: missing fields, unknown fields, non-finite numbers, reordered
canonical entries, or values outside the frozen profile are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, ClassVar, Mapping, TypeVar

from wave_asset_qa.parity.contracts import (
    AssetProvenance,
    HandSide,
    HandSpec,
    Simulator,
)


CONTACT_C0_MANIFEST_SCHEMA_VERSION = 1
CONTACT_C0_RUN_SCHEMA_VERSION = 1
CONTACT_C0_MANIFEST_ID = "wavesimparity-contact-c0"
CONTACT_C0_SCENARIO_ID = "press_hold_release"
CONTACT_C0_PAIR_ID = "synthetic_index_probe__static_box"

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class ContactCondition(str, Enum):
    CONTACT = "contact"
    SHAM = "sham"


class ContactScenarioKind(str, Enum):
    SMOOTH_PRESS_HOLD_RELEASE = "smooth_press_hold_release"


class SmoothstepProfile(str, Enum):
    QUINTIC = "quintic_10u3_minus_15u4_plus_6u5"


EnumT = TypeVar("EnumT", bound=Enum)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _exact_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _identifier(value: object, context: str) -> str:
    result = _string(value, context)
    if not _IDENTIFIER_RE.fullmatch(result):
        raise ValueError(f"{context} must match {_IDENTIFIER_RE.pattern}")
    return result


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _enum(value: object, enum_type: type[EnumT], context: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{context} must be one of: {choices}") from exc


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must be an array of three numbers")
    result = tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(value))


def _enum_tuple(
    value: object,
    enum_type: type[EnumT],
    context: str,
) -> tuple[EnumT, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return tuple(
        _enum(item, enum_type, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


def _require_exact(value: object, expected: object, context: str) -> None:
    if value != expected:
        raise ValueError(f"{context} must be {expected!r}")


@dataclass(frozen=True, slots=True)
class SphereProbeSpec:
    parent_frame_suffix: str
    local_center_m: tuple[float, float, float]
    radius_m: float
    preserve_mass_properties: bool

    def __post_init__(self) -> None:
        _require_exact(self.parent_frame_suffix, "index_DP", "fixture.probe.parent_frame_suffix")
        _require_exact(self.local_center_m, (0.025, 0.0, 0.0), "fixture.probe.local_center_m")
        _require_exact(_number(self.radius_m, "fixture.probe.radius_m"), 0.005, "fixture.probe.radius_m")
        _require_exact(
            _boolean(self.preserve_mass_properties, "fixture.probe.preserve_mass_properties"),
            True,
            "fixture.probe.preserve_mass_properties",
        )

    def parent_frame_name(self, side: HandSide) -> str:
        if not isinstance(side, HandSide):
            raise ValueError("side must be a HandSide")
        return f"{side.value}_{self.parent_frame_suffix}"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "sphere",
            "parent_frame_suffix": self.parent_frame_suffix,
            "local_center_m": list(self.local_center_m),
            "radius_m": self.radius_m,
            "preserve_mass_properties": self.preserve_mass_properties,
        }

    @classmethod
    def from_dict(cls, value: object) -> SphereProbeSpec:
        data = _mapping(value, "fixture.probe")
        _exact_keys(
            data,
            {
                "kind",
                "parent_frame_suffix",
                "local_center_m",
                "radius_m",
                "preserve_mass_properties",
            },
            "fixture.probe",
        )
        _require_exact(data["kind"], "sphere", "fixture.probe.kind")
        return cls(
            parent_frame_suffix=_string(
                data["parent_frame_suffix"], "fixture.probe.parent_frame_suffix"
            ),
            local_center_m=_vector3(data["local_center_m"], "fixture.probe.local_center_m"),
            radius_m=_number(data["radius_m"], "fixture.probe.radius_m"),
            preserve_mass_properties=_boolean(
                data["preserve_mass_properties"],
                "fixture.probe.preserve_mass_properties",
            ),
        )


@dataclass(frozen=True, slots=True)
class StaticBoxSpec:
    world_center_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]
    top_surface_z_m: float

    def __post_init__(self) -> None:
        _require_exact(self.world_center_m, (0.0, 0.0, 0.120), "fixture.target.world_center_m")
        _require_exact(
            self.half_extents_m,
            (0.150, 0.150, 0.010),
            "fixture.target.half_extents_m",
        )
        _require_exact(
            _number(self.top_surface_z_m, "fixture.target.top_surface_z_m"),
            0.130,
            "fixture.target.top_surface_z_m",
        )
        if not math.isclose(
            self.world_center_m[2] + self.half_extents_m[2],
            self.top_surface_z_m,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("fixture target top surface must equal center z plus half-height")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "static_box",
            "world_center_m": list(self.world_center_m),
            "half_extents_m": list(self.half_extents_m),
            "top_surface_z_m": self.top_surface_z_m,
        }

    @classmethod
    def from_dict(cls, value: object) -> StaticBoxSpec:
        data = _mapping(value, "fixture.target")
        _exact_keys(
            data,
            {"kind", "world_center_m", "half_extents_m", "top_surface_z_m"},
            "fixture.target",
        )
        _require_exact(data["kind"], "static_box", "fixture.target.kind")
        return cls(
            world_center_m=_vector3(
                data["world_center_m"], "fixture.target.world_center_m"
            ),
            half_extents_m=_vector3(
                data["half_extents_m"], "fixture.target.half_extents_m"
            ),
            top_surface_z_m=_number(
                data["top_surface_z_m"], "fixture.target.top_surface_z_m"
            ),
        )


@dataclass(frozen=True, slots=True)
class ContactMaterialSpec:
    static_friction: float
    dynamic_friction: float
    restitution: float

    def __post_init__(self) -> None:
        for name, value in (
            ("static_friction", self.static_friction),
            ("dynamic_friction", self.dynamic_friction),
            ("restitution", self.restitution),
        ):
            _require_exact(
                _number(value, f"fixture.material.{name}"),
                0.0,
                f"fixture.material.{name}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactMaterialSpec:
        data = _mapping(value, "fixture.material")
        _exact_keys(
            data,
            {"static_friction", "dynamic_friction", "restitution"},
            "fixture.material",
        )
        return cls(
            static_friction=_number(
                data["static_friction"], "fixture.material.static_friction"
            ),
            dynamic_friction=_number(
                data["dynamic_friction"], "fixture.material.dynamic_friction"
            ),
            restitution=_number(data["restitution"], "fixture.material.restitution"),
        )


@dataclass(frozen=True, slots=True)
class CollisionPolicySpec:
    native_hand_collisions_enabled: bool
    self_collisions_enabled: bool
    ccd_enabled: bool
    allowed_pair_id: str
    mujoco_condim: int
    pair_active_source: str
    pair_active_force_threshold_n: float
    pair_active_threshold_freeze_basis: str
    excluded_pilot_scientific_evidence: bool
    raw_pair_force_cross_simulator_comparable: bool
    expected_enabled_collision_shape_count: int
    expected_filtered_sensor_body_count: int
    expected_filtered_target_count: int

    def __post_init__(self) -> None:
        _require_exact(
            _boolean(
                self.native_hand_collisions_enabled,
                "fixture.collision_policy.native_hand_collisions_enabled",
            ),
            False,
            "fixture.collision_policy.native_hand_collisions_enabled",
        )
        _require_exact(
            _boolean(
                self.self_collisions_enabled,
                "fixture.collision_policy.self_collisions_enabled",
            ),
            False,
            "fixture.collision_policy.self_collisions_enabled",
        )
        _require_exact(
            _boolean(self.ccd_enabled, "fixture.collision_policy.ccd_enabled"),
            False,
            "fixture.collision_policy.ccd_enabled",
        )
        _require_exact(
            self.allowed_pair_id,
            CONTACT_C0_PAIR_ID,
            "fixture.collision_policy.allowed_pair_id",
        )
        _require_exact(
            _integer(self.mujoco_condim, "fixture.collision_policy.mujoco_condim"),
            1,
            "fixture.collision_policy.mujoco_condim",
        )
        _require_exact(
            self.pair_active_source,
            "direct_filtered_selected_pair_force_norm_n",
            "fixture.collision_policy.pair_active_source",
        )
        _require_exact(
            _number(
                self.pair_active_force_threshold_n,
                "fixture.collision_policy.pair_active_force_threshold_n",
            ),
            1e-4,
            "fixture.collision_policy.pair_active_force_threshold_n",
        )
        _require_exact(
            self.pair_active_threshold_freeze_basis,
            "excluded_engineering_pilot_before_formal_source_freeze",
            "fixture.collision_policy.pair_active_threshold_freeze_basis",
        )
        _require_exact(
            _boolean(
                self.excluded_pilot_scientific_evidence,
                "fixture.collision_policy.excluded_pilot_scientific_evidence",
            ),
            False,
            "fixture.collision_policy.excluded_pilot_scientific_evidence",
        )
        _require_exact(
            _boolean(
                self.raw_pair_force_cross_simulator_comparable,
                "fixture.collision_policy.raw_pair_force_cross_simulator_comparable",
            ),
            False,
            "fixture.collision_policy.raw_pair_force_cross_simulator_comparable",
        )
        for name, expected in (
            ("expected_enabled_collision_shape_count", 2),
            ("expected_filtered_sensor_body_count", 1),
            ("expected_filtered_target_count", 1),
        ):
            _require_exact(
                _integer(
                    getattr(self, name), f"fixture.collision_policy.{name}"
                ),
                expected,
                f"fixture.collision_policy.{name}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "native_hand_collisions_enabled": self.native_hand_collisions_enabled,
            "self_collisions_enabled": self.self_collisions_enabled,
            "ccd_enabled": self.ccd_enabled,
            "allowed_pair_id": self.allowed_pair_id,
            "mujoco_condim": self.mujoco_condim,
            "pair_active_source": self.pair_active_source,
            "pair_active_force_threshold_n": self.pair_active_force_threshold_n,
            "pair_active_threshold_freeze_basis": (
                self.pair_active_threshold_freeze_basis
            ),
            "excluded_pilot_scientific_evidence": (
                self.excluded_pilot_scientific_evidence
            ),
            "raw_pair_force_cross_simulator_comparable": (
                self.raw_pair_force_cross_simulator_comparable
            ),
            "expected_enabled_collision_shape_count": (
                self.expected_enabled_collision_shape_count
            ),
            "expected_filtered_sensor_body_count": (
                self.expected_filtered_sensor_body_count
            ),
            "expected_filtered_target_count": self.expected_filtered_target_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> CollisionPolicySpec:
        data = _mapping(value, "fixture.collision_policy")
        _exact_keys(
            data,
            {
                "native_hand_collisions_enabled",
                "self_collisions_enabled",
                "ccd_enabled",
                "allowed_pair_id",
                "mujoco_condim",
                "pair_active_source",
                "pair_active_force_threshold_n",
                "pair_active_threshold_freeze_basis",
                "excluded_pilot_scientific_evidence",
                "raw_pair_force_cross_simulator_comparable",
                "expected_enabled_collision_shape_count",
                "expected_filtered_sensor_body_count",
                "expected_filtered_target_count",
            },
            "fixture.collision_policy",
        )
        return cls(
            native_hand_collisions_enabled=_boolean(
                data["native_hand_collisions_enabled"],
                "fixture.collision_policy.native_hand_collisions_enabled",
            ),
            self_collisions_enabled=_boolean(
                data["self_collisions_enabled"],
                "fixture.collision_policy.self_collisions_enabled",
            ),
            ccd_enabled=_boolean(
                data["ccd_enabled"], "fixture.collision_policy.ccd_enabled"
            ),
            allowed_pair_id=_string(
                data["allowed_pair_id"], "fixture.collision_policy.allowed_pair_id"
            ),
            mujoco_condim=_integer(
                data["mujoco_condim"], "fixture.collision_policy.mujoco_condim"
            ),
            pair_active_source=_string(
                data["pair_active_source"],
                "fixture.collision_policy.pair_active_source",
            ),
            pair_active_force_threshold_n=_number(
                data["pair_active_force_threshold_n"],
                "fixture.collision_policy.pair_active_force_threshold_n",
            ),
            pair_active_threshold_freeze_basis=_string(
                data["pair_active_threshold_freeze_basis"],
                "fixture.collision_policy.pair_active_threshold_freeze_basis",
            ),
            excluded_pilot_scientific_evidence=_boolean(
                data["excluded_pilot_scientific_evidence"],
                "fixture.collision_policy.excluded_pilot_scientific_evidence",
            ),
            raw_pair_force_cross_simulator_comparable=_boolean(
                data["raw_pair_force_cross_simulator_comparable"],
                "fixture.collision_policy.raw_pair_force_cross_simulator_comparable",
            ),
            expected_enabled_collision_shape_count=_integer(
                data["expected_enabled_collision_shape_count"],
                "fixture.collision_policy.expected_enabled_collision_shape_count",
            ),
            expected_filtered_sensor_body_count=_integer(
                data["expected_filtered_sensor_body_count"],
                "fixture.collision_policy.expected_filtered_sensor_body_count",
            ),
            expected_filtered_target_count=_integer(
                data["expected_filtered_target_count"],
                "fixture.collision_policy.expected_filtered_target_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class SyntheticFixtureSpec:
    profile_id: str
    probe: SphereProbeSpec
    target: StaticBoxSpec
    material: ContactMaterialSpec
    collision_policy: CollisionPolicySpec

    def __post_init__(self) -> None:
        _require_exact(self.profile_id, "synthetic_sphere_box_v1", "fixture.profile_id")
        if not isinstance(self.probe, SphereProbeSpec):
            raise ValueError("fixture.probe must be SphereProbeSpec")
        if not isinstance(self.target, StaticBoxSpec):
            raise ValueError("fixture.target must be StaticBoxSpec")
        if not isinstance(self.material, ContactMaterialSpec):
            raise ValueError("fixture.material must be ContactMaterialSpec")
        if not isinstance(self.collision_policy, CollisionPolicySpec):
            raise ValueError("fixture.collision_policy must be CollisionPolicySpec")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "probe": self.probe.to_dict(),
            "target": self.target.to_dict(),
            "material": self.material.to_dict(),
            "collision_policy": self.collision_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> SyntheticFixtureSpec:
        data = _mapping(value, "fixture")
        _exact_keys(
            data,
            {"profile_id", "probe", "target", "material", "collision_policy"},
            "fixture",
        )
        return cls(
            profile_id=_identifier(data["profile_id"], "fixture.profile_id"),
            probe=SphereProbeSpec.from_dict(data["probe"]),
            target=StaticBoxSpec.from_dict(data["target"]),
            material=ContactMaterialSpec.from_dict(data["material"]),
            collision_policy=CollisionPolicySpec.from_dict(data["collision_policy"]),
        )


@dataclass(frozen=True, slots=True)
class IndexTargetSpec:
    index_MCP_FE: float
    index_MCP_AA: float
    index_PIP: float
    index_DIP: float

    def __post_init__(self) -> None:
        frozen = {
            "index_MCP_FE": 0.60,
            "index_MCP_AA": 0.0,
            "index_PIP": 0.90,
            "index_DIP": 0.60,
        }
        for name, expected in frozen.items():
            _require_exact(
                _number(getattr(self, name), f"scenario.driven_joint_targets_rad.{name}"),
                expected,
                f"scenario.driven_joint_targets_rad.{name}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "index_MCP_FE": self.index_MCP_FE,
            "index_MCP_AA": self.index_MCP_AA,
            "index_PIP": self.index_PIP,
            "index_DIP": self.index_DIP,
        }

    @classmethod
    def from_dict(cls, value: object) -> IndexTargetSpec:
        data = _mapping(value, "scenario.driven_joint_targets_rad")
        fields = {"index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP"}
        _exact_keys(data, fields, "scenario.driven_joint_targets_rad")
        return cls(
            index_MCP_FE=_number(
                data["index_MCP_FE"],
                "scenario.driven_joint_targets_rad.index_MCP_FE",
            ),
            index_MCP_AA=_number(
                data["index_MCP_AA"],
                "scenario.driven_joint_targets_rad.index_MCP_AA",
            ),
            index_PIP=_number(
                data["index_PIP"], "scenario.driven_joint_targets_rad.index_PIP"
            ),
            index_DIP=_number(
                data["index_DIP"], "scenario.driven_joint_targets_rad.index_DIP"
            ),
        )


@dataclass(frozen=True, slots=True)
class PhaseScheduleSpec:
    settle_end_s: float
    press_end_s: float
    hold_end_s: float
    retract_end_s: float
    recovery_end_s: float

    def __post_init__(self) -> None:
        expected = {
            "settle_end_s": 0.5,
            "press_end_s": 3.0,
            "hold_end_s": 4.0,
            "retract_end_s": 6.5,
            "recovery_end_s": 7.0,
        }
        for name, frozen in expected.items():
            _require_exact(
                _number(getattr(self, name), f"scenario.schedule.{name}"),
                frozen,
                f"scenario.schedule.{name}",
            )
        values = tuple(getattr(self, name) for name in expected)
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError("scenario schedule boundaries must be strictly increasing")

    def to_dict(self) -> dict[str, object]:
        return {
            "settle_end_s": self.settle_end_s,
            "press_end_s": self.press_end_s,
            "hold_end_s": self.hold_end_s,
            "retract_end_s": self.retract_end_s,
            "recovery_end_s": self.recovery_end_s,
        }

    @classmethod
    def from_dict(cls, value: object) -> PhaseScheduleSpec:
        data = _mapping(value, "scenario.schedule")
        fields = {
            "settle_end_s",
            "press_end_s",
            "hold_end_s",
            "retract_end_s",
            "recovery_end_s",
        }
        _exact_keys(data, fields, "scenario.schedule")
        return cls(
            **{
                name: _number(data[name], f"scenario.schedule.{name}")
                for name in fields
            }
        )


@dataclass(frozen=True, slots=True)
class ContactScenarioSpec:
    scenario_id: str
    kind: ContactScenarioKind
    duration_s: float
    dt_s: float
    gravity_m_s2: tuple[float, float, float]
    initial_position_rad: float
    driven_joint_targets_rad: IndexTargetSpec
    schedule: PhaseScheduleSpec
    smoothstep_profile: SmoothstepProfile

    def __post_init__(self) -> None:
        _require_exact(self.scenario_id, CONTACT_C0_SCENARIO_ID, "scenario.scenario_id")
        _require_exact(
            self.kind,
            ContactScenarioKind.SMOOTH_PRESS_HOLD_RELEASE,
            "scenario.kind",
        )
        _require_exact(_number(self.duration_s, "scenario.duration_s"), 7.0, "scenario.duration_s")
        _require_exact(_number(self.dt_s, "scenario.dt_s"), 0.002, "scenario.dt_s")
        _require_exact(self.gravity_m_s2, (0.0, 0.0, 0.0), "scenario.gravity_m_s2")
        _require_exact(
            _number(self.initial_position_rad, "scenario.initial_position_rad"),
            0.0,
            "scenario.initial_position_rad",
        )
        if not isinstance(self.driven_joint_targets_rad, IndexTargetSpec):
            raise ValueError("scenario.driven_joint_targets_rad must be IndexTargetSpec")
        if not isinstance(self.schedule, PhaseScheduleSpec):
            raise ValueError("scenario.schedule must be PhaseScheduleSpec")
        _require_exact(
            self.smoothstep_profile,
            SmoothstepProfile.QUINTIC,
            "scenario.smoothstep_profile",
        )
        ratio = self.duration_s / self.dt_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("scenario duration must be an integer multiple of dt_s")
        for name, boundary in self.schedule.to_dict().items():
            boundary_ratio = float(boundary) / self.dt_s
            if not math.isclose(
                boundary_ratio, round(boundary_ratio), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"scenario.schedule.{name} must align to base dt")

    @property
    def steps(self) -> int:
        return round(self.duration_s / self.dt_s)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "kind": self.kind.value,
            "duration_s": self.duration_s,
            "dt_s": self.dt_s,
            "gravity_m_s2": list(self.gravity_m_s2),
            "initial_position_rad": self.initial_position_rad,
            "driven_joint_targets_rad": self.driven_joint_targets_rad.to_dict(),
            "schedule": self.schedule.to_dict(),
            "smoothstep_profile": self.smoothstep_profile.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactScenarioSpec:
        data = _mapping(value, "scenario")
        _exact_keys(
            data,
            {
                "scenario_id",
                "kind",
                "duration_s",
                "dt_s",
                "gravity_m_s2",
                "initial_position_rad",
                "driven_joint_targets_rad",
                "schedule",
                "smoothstep_profile",
            },
            "scenario",
        )
        return cls(
            scenario_id=_identifier(data["scenario_id"], "scenario.scenario_id"),
            kind=_enum(data["kind"], ContactScenarioKind, "scenario.kind"),
            duration_s=_number(data["duration_s"], "scenario.duration_s"),
            dt_s=_number(data["dt_s"], "scenario.dt_s"),
            gravity_m_s2=_vector3(data["gravity_m_s2"], "scenario.gravity_m_s2"),
            initial_position_rad=_number(
                data["initial_position_rad"], "scenario.initial_position_rad"
            ),
            driven_joint_targets_rad=IndexTargetSpec.from_dict(
                data["driven_joint_targets_rad"]
            ),
            schedule=PhaseScheduleSpec.from_dict(data["schedule"]),
            smoothstep_profile=_enum(
                data["smoothstep_profile"],
                SmoothstepProfile,
                "scenario.smoothstep_profile",
            ),
        )


@dataclass(frozen=True, slots=True)
class ContactConditionSpec:
    condition_id: ContactCondition
    pair_collision_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, ContactCondition):
            raise ValueError("condition.condition_id must be a ContactCondition")
        expected = self.condition_id is ContactCondition.CONTACT
        _require_exact(
            _boolean(self.pair_collision_enabled, "condition.pair_collision_enabled"),
            expected,
            "condition.pair_collision_enabled",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id.value,
            "pair_collision_enabled": self.pair_collision_enabled,
        }

    @classmethod
    def from_dict(cls, value: object, *, context: str) -> ContactConditionSpec:
        data = _mapping(value, context)
        _exact_keys(data, {"condition_id", "pair_collision_enabled"}, context)
        return cls(
            condition_id=_enum(
                data["condition_id"], ContactCondition, f"{context}.condition_id"
            ),
            pair_collision_enabled=_boolean(
                data["pair_collision_enabled"], f"{context}.pair_collision_enabled"
            ),
        )


@dataclass(frozen=True, slots=True)
class ContactRunPolicy:
    repeat_count: int
    dt_halving_scenario_ids: tuple[str, ...]
    debounce_window_s: float
    minimum_completion_fraction: float
    fresh_process_per_case: bool

    def __post_init__(self) -> None:
        _require_exact(_integer(self.repeat_count, "run_policy.repeat_count"), 2, "run_policy.repeat_count")
        _require_exact(
            self.dt_halving_scenario_ids,
            (CONTACT_C0_SCENARIO_ID,),
            "run_policy.dt_halving_scenario_ids",
        )
        _require_exact(
            _number(self.debounce_window_s, "run_policy.debounce_window_s"),
            0.004,
            "run_policy.debounce_window_s",
        )
        _require_exact(
            _number(
                self.minimum_completion_fraction,
                "run_policy.minimum_completion_fraction",
            ),
            1.0,
            "run_policy.minimum_completion_fraction",
        )
        _require_exact(
            _boolean(
                self.fresh_process_per_case, "run_policy.fresh_process_per_case"
            ),
            True,
            "run_policy.fresh_process_per_case",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat_count": self.repeat_count,
            "dt_halving_scenario_ids": list(self.dt_halving_scenario_ids),
            "debounce_window_s": self.debounce_window_s,
            "minimum_completion_fraction": self.minimum_completion_fraction,
            "fresh_process_per_case": self.fresh_process_per_case,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactRunPolicy:
        data = _mapping(value, "run_policy")
        _exact_keys(
            data,
            {
                "repeat_count",
                "dt_halving_scenario_ids",
                "debounce_window_s",
                "minimum_completion_fraction",
                "fresh_process_per_case",
            },
            "run_policy",
        )
        return cls(
            repeat_count=_integer(data["repeat_count"], "run_policy.repeat_count"),
            dt_halving_scenario_ids=_string_tuple(
                data["dt_halving_scenario_ids"],
                "run_policy.dt_halving_scenario_ids",
            ),
            debounce_window_s=_number(
                data["debounce_window_s"], "run_policy.debounce_window_s"
            ),
            minimum_completion_fraction=_number(
                data["minimum_completion_fraction"],
                "run_policy.minimum_completion_fraction",
            ),
            fresh_process_per_case=_boolean(
                data["fresh_process_per_case"],
                "run_policy.fresh_process_per_case",
            ),
        )


@dataclass(frozen=True, slots=True)
class AdmissionThresholds:
    fixture_geometry_max_abs_error_m: float
    material_zero_max_abs_error: float
    initial_gap_min_m: float
    sham_hold_mean_gap_max_m: float
    recovery_gap_min_m: float
    contact_hold_duty_min: float

    _FROZEN: ClassVar[dict[str, float | int]] = {
        "fixture_geometry_max_abs_error_m": 1e-6,
        "material_zero_max_abs_error": 1e-9,
        "initial_gap_min_m": 0.050,
        "sham_hold_mean_gap_max_m": -0.005,
        "recovery_gap_min_m": 0.050,
        "contact_hold_duty_min": 0.95,
    }

    def __post_init__(self) -> None:
        for name, frozen in self._FROZEN.items():
            value = getattr(self, name)
            parsed: float | int
            if isinstance(frozen, int):
                parsed = _integer(value, f"thresholds.admission.{name}")
            else:
                parsed = _number(value, f"thresholds.admission.{name}")
            _require_exact(parsed, frozen, f"thresholds.admission.{name}")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._FROZEN}

    @classmethod
    def from_dict(cls, value: object) -> AdmissionThresholds:
        data = _mapping(value, "thresholds.admission")
        _exact_keys(data, set(cls._FROZEN), "thresholds.admission")
        kwargs: dict[str, float | int] = {}
        for name, frozen in cls._FROZEN.items():
            kwargs[name] = (
                _integer(data[name], f"thresholds.admission.{name}")
                if isinstance(frozen, int)
                else _number(data[name], f"thresholds.admission.{name}")
            )
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RepeatabilityThresholds:
    joint_or_effect_max_abs_rad: float
    probe_gap_or_blocked_max_abs_m: float
    orientation_or_effect_max_abs_rad: float
    pair_active_exact: bool
    event_interval_exact: bool

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "joint_or_effect_max_abs_rad": 1e-9,
            "probe_gap_or_blocked_max_abs_m": 1e-9,
            "orientation_or_effect_max_abs_rad": 1e-9,
            "pair_active_exact": True,
            "event_interval_exact": True,
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            parsed = (
                _boolean(value, f"thresholds.repeatability.{name}")
                if isinstance(frozen, bool)
                else _number(value, f"thresholds.repeatability.{name}")
            )
            _require_exact(parsed, frozen, f"thresholds.repeatability.{name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_or_effect_max_abs_rad": self.joint_or_effect_max_abs_rad,
            "probe_gap_or_blocked_max_abs_m": self.probe_gap_or_blocked_max_abs_m,
            "orientation_or_effect_max_abs_rad": self.orientation_or_effect_max_abs_rad,
            "pair_active_exact": self.pair_active_exact,
            "event_interval_exact": self.event_interval_exact,
        }

    @classmethod
    def from_dict(cls, value: object) -> RepeatabilityThresholds:
        data = _mapping(value, "thresholds.repeatability")
        fields = {
            "joint_or_effect_max_abs_rad",
            "probe_gap_or_blocked_max_abs_m",
            "orientation_or_effect_max_abs_rad",
            "pair_active_exact",
            "event_interval_exact",
        }
        _exact_keys(data, fields, "thresholds.repeatability")
        return cls(
            joint_or_effect_max_abs_rad=_number(
                data["joint_or_effect_max_abs_rad"],
                "thresholds.repeatability.joint_or_effect_max_abs_rad",
            ),
            probe_gap_or_blocked_max_abs_m=_number(
                data["probe_gap_or_blocked_max_abs_m"],
                "thresholds.repeatability.probe_gap_or_blocked_max_abs_m",
            ),
            orientation_or_effect_max_abs_rad=_number(
                data["orientation_or_effect_max_abs_rad"],
                "thresholds.repeatability.orientation_or_effect_max_abs_rad",
            ),
            pair_active_exact=_boolean(
                data["pair_active_exact"],
                "thresholds.repeatability.pair_active_exact",
            ),
            event_interval_exact=_boolean(
                data["event_interval_exact"],
                "thresholds.repeatability.event_interval_exact",
            ),
        )


@dataclass(frozen=True, slots=True)
class DtHalvingThresholds:
    joint_or_effect_max_abs_rad: float
    probe_gap_or_blocked_max_abs_m: float
    orientation_or_effect_max_abs_rad: float
    event_midpoint_max_abs_s: float
    hold_duty_max_abs: float

    _FROZEN: ClassVar[dict[str, float]] = {
        "joint_or_effect_max_abs_rad": 0.01,
        "probe_gap_or_blocked_max_abs_m": 0.001,
        "orientation_or_effect_max_abs_rad": 0.02,
        "event_midpoint_max_abs_s": 0.004,
        "hold_duty_max_abs": 0.01,
    }

    def __post_init__(self) -> None:
        for name, frozen in self._FROZEN.items():
            _require_exact(
                _number(getattr(self, name), f"thresholds.dt_halving.{name}"),
                frozen,
                f"thresholds.dt_halving.{name}",
            )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._FROZEN}

    @classmethod
    def from_dict(cls, value: object) -> DtHalvingThresholds:
        data = _mapping(value, "thresholds.dt_halving")
        _exact_keys(data, set(cls._FROZEN), "thresholds.dt_halving")
        return cls(
            **{
                name: _number(data[name], f"thresholds.dt_halving.{name}")
                for name in cls._FROZEN
            }
        )


@dataclass(frozen=True, slots=True)
class CrossSimulatorThresholds:
    event_midpoint_max_abs_s: float
    hold_duty_max_abs: float
    onset_gap_max_abs_m: float
    steady_hold_mean_gap_max_abs_m: float
    joint_effect_max_abs_rad: float
    blocked_travel_max_abs_m: float
    orientation_effect_max_abs_rad: float

    _FROZEN: ClassVar[dict[str, float]] = {
        "event_midpoint_max_abs_s": 0.010,
        "hold_duty_max_abs": 0.02,
        "onset_gap_max_abs_m": 0.001,
        "steady_hold_mean_gap_max_abs_m": 0.001,
        "joint_effect_max_abs_rad": 0.02,
        "blocked_travel_max_abs_m": 0.002,
        "orientation_effect_max_abs_rad": 0.05,
    }

    def __post_init__(self) -> None:
        for name, frozen in self._FROZEN.items():
            _require_exact(
                _number(getattr(self, name), f"thresholds.cross_simulator.{name}"),
                frozen,
                f"thresholds.cross_simulator.{name}",
            )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._FROZEN}

    @classmethod
    def from_dict(cls, value: object) -> CrossSimulatorThresholds:
        data = _mapping(value, "thresholds.cross_simulator")
        _exact_keys(data, set(cls._FROZEN), "thresholds.cross_simulator")
        return cls(
            **{
                name: _number(data[name], f"thresholds.cross_simulator.{name}")
                for name in cls._FROZEN
            }
        )


@dataclass(frozen=True, slots=True)
class ContactThresholds:
    admission: AdmissionThresholds
    repeatability: RepeatabilityThresholds
    dt_halving: DtHalvingThresholds
    cross_simulator: CrossSimulatorThresholds

    def __post_init__(self) -> None:
        expected = (
            ("admission", self.admission, AdmissionThresholds),
            ("repeatability", self.repeatability, RepeatabilityThresholds),
            ("dt_halving", self.dt_halving, DtHalvingThresholds),
            ("cross_simulator", self.cross_simulator, CrossSimulatorThresholds),
        )
        for name, value, value_type in expected:
            if not isinstance(value, value_type):
                raise ValueError(f"thresholds.{name} must be {value_type.__name__}")

    def to_dict(self) -> dict[str, object]:
        return {
            "admission": self.admission.to_dict(),
            "repeatability": self.repeatability.to_dict(),
            "dt_halving": self.dt_halving.to_dict(),
            "cross_simulator": self.cross_simulator.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactThresholds:
        data = _mapping(value, "thresholds")
        _exact_keys(
            data,
            {"admission", "repeatability", "dt_halving", "cross_simulator"},
            "thresholds",
        )
        return cls(
            admission=AdmissionThresholds.from_dict(data["admission"]),
            repeatability=RepeatabilityThresholds.from_dict(data["repeatability"]),
            dt_halving=DtHalvingThresholds.from_dict(data["dt_halving"]),
            cross_simulator=CrossSimulatorThresholds.from_dict(
                data["cross_simulator"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ClaimBoundary:
    synthetic_fixture_only: bool
    native_fingertip_geometry_validated: bool
    full_isaac_sim: bool
    dual_hand: bool
    floating_base: bool
    hardware: bool
    sim2real: bool

    def __post_init__(self) -> None:
        expected = {
            "synthetic_fixture_only": True,
            "native_fingertip_geometry_validated": False,
            "full_isaac_sim": False,
            "dual_hand": False,
            "floating_base": False,
            "hardware": False,
            "sim2real": False,
        }
        for name, frozen in expected.items():
            _require_exact(
                _boolean(getattr(self, name), f"claim_boundary.{name}"),
                frozen,
                f"claim_boundary.{name}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "synthetic_fixture_only": self.synthetic_fixture_only,
            "native_fingertip_geometry_validated": self.native_fingertip_geometry_validated,
            "full_isaac_sim": self.full_isaac_sim,
            "dual_hand": self.dual_hand,
            "floating_base": self.floating_base,
            "hardware": self.hardware,
            "sim2real": self.sim2real,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClaimBoundary:
        data = _mapping(value, "claim_boundary")
        fields = {
            "synthetic_fixture_only",
            "native_fingertip_geometry_validated",
            "full_isaac_sim",
            "dual_hand",
            "floating_base",
            "hardware",
            "sim2real",
        }
        _exact_keys(data, fields, "claim_boundary")
        return cls(
            **{
                name: _boolean(data[name], f"claim_boundary.{name}")
                for name in fields
            }
        )


@dataclass(frozen=True, slots=True)
class ContactManifest:
    schema_version: int
    manifest_id: str
    provenance: AssetProvenance
    simulators: tuple[Simulator, ...]
    hands: tuple[HandSpec, ...]
    fixture: SyntheticFixtureSpec
    scenario: ContactScenarioSpec
    conditions: tuple[ContactConditionSpec, ...]
    run_policy: ContactRunPolicy
    thresholds: ContactThresholds
    claim_boundary: ClaimBoundary

    def __post_init__(self) -> None:
        _require_exact(
            _integer(self.schema_version, "manifest.schema_version"),
            CONTACT_C0_MANIFEST_SCHEMA_VERSION,
            "manifest.schema_version",
        )
        _require_exact(self.manifest_id, CONTACT_C0_MANIFEST_ID, "manifest.manifest_id")
        if not isinstance(self.provenance, AssetProvenance):
            raise ValueError("manifest.provenance must be AssetProvenance")
        _require_exact(
            self.simulators,
            (Simulator.MUJOCO, Simulator.OVPHYSX),
            "manifest.simulators",
        )
        if (
            not isinstance(self.hands, tuple)
            or any(not isinstance(hand, HandSpec) for hand in self.hands)
            or tuple(hand.side for hand in self.hands)
            != (HandSide.LEFT, HandSide.RIGHT)
        ):
            raise ValueError("manifest.hands must contain canonical left then right hands")
        if not isinstance(self.fixture, SyntheticFixtureSpec):
            raise ValueError("manifest.fixture must be SyntheticFixtureSpec")
        if not isinstance(self.scenario, ContactScenarioSpec):
            raise ValueError("manifest.scenario must be ContactScenarioSpec")
        expected_conditions = (
            ContactConditionSpec(ContactCondition.CONTACT, True),
            ContactConditionSpec(ContactCondition.SHAM, False),
        )
        _require_exact(self.conditions, expected_conditions, "manifest.conditions")
        if not isinstance(self.run_policy, ContactRunPolicy):
            raise ValueError("manifest.run_policy must be ContactRunPolicy")
        if not isinstance(self.thresholds, ContactThresholds):
            raise ValueError("manifest.thresholds must be ContactThresholds")
        if not isinstance(self.claim_boundary, ClaimBoundary):
            raise ValueError("manifest.claim_boundary must be ClaimBoundary")
        for hand in self.hands:
            probe_name = self.fixture.probe.parent_frame_name(hand.side)
            if probe_name not in hand.distal_frame_names:
                raise ValueError(
                    f"manifest fixture probe frame {probe_name!r} is not mapped"
                )
        if self.expected_joint_mapping_count != 44:
            raise ValueError("C0 must retain complete 44/44 joint mapping")
        if self.expected_distal_frame_mapping_count != 10:
            raise ValueError("C0 must retain complete 10/10 distal-frame mapping")
        if self.expected_case_count != 32:
            raise ValueError("C0 frozen matrix must contain exactly 32 cases")

    @property
    def expected_joint_mapping_count(self) -> int:
        return sum(len(hand.joint_names) for hand in self.hands)

    @property
    def expected_distal_frame_mapping_count(self) -> int:
        return sum(len(hand.distal_frame_names) for hand in self.hands)

    @property
    def expected_case_count(self) -> int:
        return (
            len(self.simulators)
            * len(self.hands)
            * len(self.conditions)
            * 2
            * self.run_policy.repeat_count
        )

    def hand(self, side: HandSide) -> HandSpec:
        if not isinstance(side, HandSide):
            raise ValueError("side must be a HandSide")
        for hand in self.hands:
            if hand.side is side:
                return hand
        raise KeyError(side.value)

    def condition(self, condition: ContactCondition) -> ContactConditionSpec:
        if not isinstance(condition, ContactCondition):
            raise ValueError("condition must be a ContactCondition")
        for item in self.conditions:
            if item.condition_id is condition:
                return item
        raise KeyError(condition.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "provenance": self.provenance.to_dict(),
            "simulators": [item.value for item in self.simulators],
            "hands": [hand.to_dict() for hand in self.hands],
            "fixture": self.fixture.to_dict(),
            "scenario": self.scenario.to_dict(),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "run_policy": self.run_policy.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "claim_boundary": self.claim_boundary.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactManifest:
        data = _mapping(value, "manifest")
        _exact_keys(
            data,
            {
                "schema_version",
                "manifest_id",
                "provenance",
                "simulators",
                "hands",
                "fixture",
                "scenario",
                "conditions",
                "run_policy",
                "thresholds",
                "claim_boundary",
            },
            "manifest",
        )
        hands_value = data["hands"]
        conditions_value = data["conditions"]
        if not isinstance(hands_value, list):
            raise ValueError("manifest.hands must be an array")
        if not isinstance(conditions_value, list):
            raise ValueError("manifest.conditions must be an array")
        return cls(
            schema_version=_integer(data["schema_version"], "manifest.schema_version"),
            manifest_id=_identifier(data["manifest_id"], "manifest.manifest_id"),
            provenance=AssetProvenance.from_dict(data["provenance"]),
            simulators=_enum_tuple(data["simulators"], Simulator, "manifest.simulators"),
            hands=tuple(
                HandSpec.from_dict(item, context=f"manifest.hands[{index}]")
                for index, item in enumerate(hands_value)
            ),
            fixture=SyntheticFixtureSpec.from_dict(data["fixture"]),
            scenario=ContactScenarioSpec.from_dict(data["scenario"]),
            conditions=tuple(
                ContactConditionSpec.from_dict(
                    item, context=f"manifest.conditions[{index}]"
                )
                for index, item in enumerate(conditions_value)
            ),
            run_policy=ContactRunPolicy.from_dict(data["run_policy"]),
            thresholds=ContactThresholds.from_dict(data["thresholds"]),
            claim_boundary=ClaimBoundary.from_dict(data["claim_boundary"]),
        )


__all__ = [
    "AdmissionThresholds",
    "ClaimBoundary",
    "CollisionPolicySpec",
    "CONTACT_C0_MANIFEST_ID",
    "CONTACT_C0_MANIFEST_SCHEMA_VERSION",
    "CONTACT_C0_PAIR_ID",
    "CONTACT_C0_RUN_SCHEMA_VERSION",
    "CONTACT_C0_SCENARIO_ID",
    "ContactCondition",
    "ContactConditionSpec",
    "ContactManifest",
    "ContactMaterialSpec",
    "ContactRunPolicy",
    "ContactScenarioKind",
    "ContactScenarioSpec",
    "ContactThresholds",
    "CrossSimulatorThresholds",
    "DtHalvingThresholds",
    "IndexTargetSpec",
    "PhaseScheduleSpec",
    "RepeatabilityThresholds",
    "SmoothstepProfile",
    "SphereProbeSpec",
    "StaticBoxSpec",
    "SyntheticFixtureSpec",
]
