"""Strict execution records for WaveSimParity Contact Gate C0.

The contact record contract is intentionally independent from the no-contact
parity ``TraceSample`` contract.  C0 records one synthetic sphere/box pair and
keeps backend-native contact details in a private, opaque JSON subtree.  Every
portable field used by scientific analysis is closed and finite.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    CONTACT_C0_MANIFEST_ID,
    CONTACT_C0_PAIR_ID,
    CONTACT_C0_RUN_SCHEMA_VERSION,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_CASE_ID_RE = re.compile(
    r"^(?:mujoco|ovphysx)\.(?:left|right)\.press_hold_release\."
    r"(?:contact|sham)\.(?:base|halved)\.r(?:01|02)$"
)

_RUN_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "manifest_sha256",
        "case",
        "execution",
        "mapping",
        "fixture_readback",
        "contact_observation",
        "samples",
        "provenance",
    }
)
_CASE_KEYS = frozenset(
    {
        "case_id",
        "simulator",
        "hand",
        "scenario_id",
        "condition",
        "timestep_variant",
        "repeat_index",
        "dt_s",
        "model_path",
    }
)
_MAPPING_KEYS = frozenset(
    {
        "expected_joint_count",
        "observed_joint_count",
        "expected_distal_frame_count",
        "observed_distal_frame_count",
        "probe_frame_name",
        "probe_frame_mapped",
    }
)
_FIXTURE_READBACK_KEYS = frozenset(
    {
        "performed",
        "profile_id",
        "probe_parent_frame_name",
        "probe_local_center_m",
        "probe_radius_m",
        "target_world_center_m",
        "target_half_extents_m",
        "target_top_surface_z_m",
        "static_friction",
        "dynamic_friction",
        "restitution",
        "native_hand_collisions_enabled",
        "self_collisions_enabled",
        "ccd_enabled",
        "allowed_pair_id",
        "mujoco_condim",
        "native_collision_prim_count",
        "native_collision_disabled_count",
        "enabled_collision_shape_count",
        "collision_inventory_sha256",
        "mass_properties_preserved",
    }
)
_CONTACT_OBSERVATION_KEYS = frozenset(
    {
        "performed",
        "capability",
        "selected_pair_id",
        "pair_active_source",
        "pair_active_force_threshold_n",
        "pair_active_record_count",
        "missing_pair_active_count",
        "filtered_sensor_body_count",
        "filtered_target_count",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "backend",
        "source_revision",
        "source_tree",
        "asset_commit",
        "asset_git_tree",
        "manifest_sha256",
        "fixture_overlay_sha256",
        "runtime_fingerprint_sha256",
        "fresh_process_identity_sha256",
    }
)
_EXECUTION_KEYS = frozenset(
    {"status", "message", "requested_steps", "completed_steps"}
)
_SAMPLE_KEYS = frozenset(
    {
        "step",
        "time_s",
        "joint_positions",
        "joint_velocities",
        "frame_poses",
        "position_targets",
        "probe_center_world_m",
        "signed_gap_m",
        "pair_active",
        "raw_contact_count",
        "native_contact_observation",
    }
)


class ContactRecordValidationError(ValueError):
    """A C0 record is malformed, incomplete, ambiguous, or non-finite."""


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContactRecordValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> object:
    raise ContactRecordValidationError(f"non-finite JSON number: {token}")


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ContactRecordValidationError(
            f"{context} must be an object with string keys"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ContactRecordValidationError(
            f"{context} fields are not exact: missing={missing}, unknown={unknown}"
        )


def _string(value: object, context: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContactRecordValidationError(f"{context} must be a non-empty string")
    if identifier and _IDENTIFIER_RE.fullmatch(value) is None:
        raise ContactRecordValidationError(f"{context} is not a portable identifier")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContactRecordValidationError(
            f"{context} must be an integer >= {minimum}"
        )
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContactRecordValidationError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContactRecordValidationError(f"{context} must be finite")
    return result


def _finite_vector(value: object, length: int, context: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ContactRecordValidationError(
            f"{context} must be an array of {length} numbers"
        )
    return tuple(_finite(item, f"{context}[{index}]") for index, item in enumerate(value))


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ContactRecordValidationError(f"{context} must be a boolean")
    return value


def _nullable_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _nullable_finite(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _finite(value, context)


def _nullable_vector(
    value: object, length: int, context: str
) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _finite_vector(value, length, context)


def _nullable_boolean(value: object, context: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, context)


def _nullable_hash(
    value: object, context: str, *, sha256: bool
) -> str | None:
    if value is None:
        return None
    parsed = _string(value, context)
    pattern = _SHA256_RE if sha256 else re.compile(r"^[0-9a-f]{40}$")
    if pattern.fullmatch(parsed) is None:
        length = 64 if sha256 else 40
        raise ContactRecordValidationError(
            f"{context} must be a lowercase {length}-character hexadecimal digest"
        )
    return parsed


def _finite_number_mapping(value: object, context: str) -> dict[str, float]:
    data = _mapping(value, context)
    if not data:
        raise ContactRecordValidationError(f"{context} must not be empty")
    result: dict[str, float] = {}
    for name, item in data.items():
        _string(name, f"{context} key")
        result[name] = _finite(item, f"{context}[{name!r}]")
    return result


def _frame_mapping(
    value: object, context: str
) -> dict[str, tuple[float, float, float, float, float, float, float]]:
    data = _mapping(value, context)
    if not data:
        raise ContactRecordValidationError(f"{context} must not be empty")
    result: dict[str, tuple[float, float, float, float, float, float, float]] = {}
    for name, raw_pose in data.items():
        pose = _finite_vector(raw_pose, 7, f"{context}[{name!r}]")
        quaternion_norm = math.sqrt(sum(component * component for component in pose[3:]))
        if quaternion_norm <= 0.0:
            raise ContactRecordValidationError(
                f"{context}[{name!r}] has a zero quaternion"
            )
        result[name] = pose  # type: ignore[assignment]
    return result


def _json_value(value: object, context: str) -> object:
    """Return a detached finite JSON value while preserving backend details."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContactRecordValidationError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContactRecordValidationError(f"{context} has a non-string key")
        return {
            key: _json_value(item, f"{context}.{key}")
            for key, item in value.items()
        }
    raise ContactRecordValidationError(
        f"{context} contains a non-JSON value of type {type(value).__name__}"
    )


def _validate_mapping_record(value: object, context: str) -> Mapping[str, Any]:
    data = _mapping(value, context)
    _exact_keys(data, _MAPPING_KEYS, context)
    expected_joints = _integer(
        data["expected_joint_count"], f"{context}.expected_joint_count"
    )
    observed_joints = _integer(
        data["observed_joint_count"], f"{context}.observed_joint_count"
    )
    expected_frames = _integer(
        data["expected_distal_frame_count"],
        f"{context}.expected_distal_frame_count",
    )
    observed_frames = _integer(
        data["observed_distal_frame_count"],
        f"{context}.observed_distal_frame_count",
    )
    if expected_joints != 22 or observed_joints > 22:
        raise ContactRecordValidationError(
            f"{context} joint counts are outside the frozen 22-joint contract"
        )
    if expected_frames != 5 or observed_frames > 5:
        raise ContactRecordValidationError(
            f"{context} frame counts are outside the frozen five-frame contract"
        )
    probe_name = _string(data["probe_frame_name"], f"{context}.probe_frame_name")
    if re.fullmatch(r"(?:left|right)_index_DP", probe_name) is None:
        raise ContactRecordValidationError(
            f"{context}.probe_frame_name is not a canonical index DP frame"
        )
    _boolean(data["probe_frame_mapped"], f"{context}.probe_frame_mapped")
    return data


def _validate_fixture_readback(
    value: object, context: str
) -> Mapping[str, Any]:
    data = _mapping(value, context)
    _exact_keys(data, _FIXTURE_READBACK_KEYS, context)
    _boolean(data["performed"], f"{context}.performed")
    for name in ("profile_id", "probe_parent_frame_name", "allowed_pair_id"):
        _nullable_string(data[name], f"{context}.{name}")
    for name in (
        "probe_local_center_m",
        "target_world_center_m",
        "target_half_extents_m",
    ):
        _nullable_vector(data[name], 3, f"{context}.{name}")
    for name in (
        "probe_radius_m",
        "target_top_surface_z_m",
        "static_friction",
        "dynamic_friction",
        "restitution",
    ):
        _nullable_finite(data[name], f"{context}.{name}")
    for name in (
        "native_hand_collisions_enabled",
        "self_collisions_enabled",
        "ccd_enabled",
    ):
        _nullable_boolean(data[name], f"{context}.{name}")
    condim = data["mujoco_condim"]
    if condim is not None:
        _integer(condim, f"{context}.mujoco_condim")
    for name in (
        "native_collision_prim_count",
        "native_collision_disabled_count",
        "enabled_collision_shape_count",
    ):
        raw_count = data[name]
        if raw_count is not None:
            _integer(raw_count, f"{context}.{name}")
    _nullable_hash(
        data["collision_inventory_sha256"],
        f"{context}.collision_inventory_sha256",
        sha256=True,
    )
    _nullable_boolean(
        data["mass_properties_preserved"],
        f"{context}.mass_properties_preserved",
    )
    return data


def _validate_contact_observation(
    value: object, context: str
) -> Mapping[str, Any]:
    data = _mapping(value, context)
    _exact_keys(data, _CONTACT_OBSERVATION_KEYS, context)
    _boolean(data["performed"], f"{context}.performed")
    capability = _string(data["capability"], f"{context}.capability")
    if capability not in {"direct_filtered_pair_force_threshold", "unavailable"}:
        raise ContactRecordValidationError(
            f"{context}.capability is not a frozen C0 capability"
        )
    for name in ("selected_pair_id", "pair_active_source"):
        _nullable_string(data[name], f"{context}.{name}")
    threshold = _nullable_finite(
        data["pair_active_force_threshold_n"],
        f"{context}.pair_active_force_threshold_n",
    )
    if threshold is not None and threshold < 0.0:
        raise ContactRecordValidationError(
            f"{context}.pair_active_force_threshold_n must be non-negative"
        )
    _integer(
        data["pair_active_record_count"],
        f"{context}.pair_active_record_count",
    )
    _integer(
        data["missing_pair_active_count"],
        f"{context}.missing_pair_active_count",
    )
    _integer(
        data["filtered_sensor_body_count"],
        f"{context}.filtered_sensor_body_count",
    )
    _integer(
        data["filtered_target_count"],
        f"{context}.filtered_target_count",
    )
    return data


def _validate_provenance(value: object, context: str) -> Mapping[str, Any]:
    data = _mapping(value, context)
    _exact_keys(data, _PROVENANCE_KEYS, context)
    backend = _string(data["backend"], f"{context}.backend")
    if backend not in {"mujoco", "ovphysx"}:
        raise ContactRecordValidationError(f"{context}.backend is unsupported")
    for name in ("source_revision", "source_tree", "asset_commit", "asset_git_tree"):
        _nullable_hash(data[name], f"{context}.{name}", sha256=False)
    for name in (
        "manifest_sha256",
        "fixture_overlay_sha256",
        "runtime_fingerprint_sha256",
        "fresh_process_identity_sha256",
    ):
        _nullable_hash(data[name], f"{context}.{name}", sha256=True)
    return data


@dataclass(frozen=True, slots=True)
class ContactCaseRecord:
    """Portable scalar identity for one of the frozen 32 C0 cases."""

    case_id: str
    simulator: str
    hand: str
    scenario_id: str
    condition: str
    timestep_variant: str
    repeat_index: int
    dt_s: float
    model_path: str

    def __post_init__(self) -> None:
        if _CASE_ID_RE.fullmatch(self.case_id) is None:
            raise ContactRecordValidationError("case.case_id is not a frozen C0 ID")
        if self.simulator not in {"mujoco", "ovphysx"}:
            raise ContactRecordValidationError("case.simulator is unsupported")
        if self.hand not in {"left", "right"}:
            raise ContactRecordValidationError("case.hand is unsupported")
        if self.scenario_id != "press_hold_release":
            raise ContactRecordValidationError("case.scenario_id is not frozen")
        if self.condition not in {"contact", "sham"}:
            raise ContactRecordValidationError("case.condition is unsupported")
        if self.timestep_variant not in {"base", "halved"}:
            raise ContactRecordValidationError("case.timestep_variant is unsupported")
        if self.repeat_index not in {1, 2}:
            raise ContactRecordValidationError("case.repeat_index must be 1 or 2")
        expected_dt = 0.002 if self.timestep_variant == "base" else 0.001
        if not math.isclose(
            _finite(self.dt_s, "case.dt_s"),
            expected_dt,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ContactRecordValidationError("case.dt_s contradicts its variant")
        expected_id = (
            f"{self.simulator}.{self.hand}.{self.scenario_id}.{self.condition}."
            f"{self.timestep_variant}.r{self.repeat_index:02d}"
        )
        if self.case_id != expected_id:
            raise ContactRecordValidationError(
                "case.case_id contradicts its component fields"
            )
        if (
            not isinstance(self.model_path, str)
            or not self.model_path
            or self.model_path.startswith("/")
            or "\\" in self.model_path
            or any(part == ".." for part in self.model_path.split("/"))
            or re.fullmatch(r".+\.(?:xml|usda)", self.model_path) is None
        ):
            raise ContactRecordValidationError(
                "case.model_path must be a relative traversal-free .xml or .usda path"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "simulator": self.simulator,
            "hand": self.hand,
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "timestep_variant": self.timestep_variant,
            "repeat_index": self.repeat_index,
            "dt_s": self.dt_s,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactCaseRecord:
        data = _mapping(value, "case")
        _exact_keys(data, _CASE_KEYS, "case")
        return cls(
            case_id=_string(data["case_id"], "case.case_id"),
            simulator=_string(data["simulator"], "case.simulator"),
            hand=_string(data["hand"], "case.hand"),
            scenario_id=_string(data["scenario_id"], "case.scenario_id"),
            condition=_string(data["condition"], "case.condition"),
            timestep_variant=_string(
                data["timestep_variant"], "case.timestep_variant"
            ),
            repeat_index=_integer(
                data["repeat_index"], "case.repeat_index", minimum=1
            ),
            dt_s=_finite(data["dt_s"], "case.dt_s"),
            model_path=_string(data["model_path"], "case.model_path"),
        )


@dataclass(frozen=True, slots=True)
class ContactExecutionRecord:
    status: str
    message: str | None
    requested_steps: int
    completed_steps: int

    def __post_init__(self) -> None:
        if self.status not in {"completed", "error"}:
            raise ContactRecordValidationError(
                "execution.status must be 'completed' or 'error'"
            )
        requested = _integer(
            self.requested_steps, "execution.requested_steps", minimum=1
        )
        completed = _integer(self.completed_steps, "execution.completed_steps")
        if completed > requested:
            raise ContactRecordValidationError(
                "execution.completed_steps exceeds requested_steps"
            )
        if self.message is not None and (
            not isinstance(self.message, str) or not self.message.strip()
        ):
            raise ContactRecordValidationError(
                "execution.message must be null or a non-empty string"
            )
        if self.status == "completed":
            if completed != requested:
                raise ContactRecordValidationError(
                    "completed execution did not finish every requested step"
                )
            if self.message is not None:
                raise ContactRecordValidationError(
                    "completed execution.message must be null"
                )
        elif self.message is None:
            raise ContactRecordValidationError(
                "error execution must carry a non-empty message"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "message": self.message,
            "requested_steps": self.requested_steps,
            "completed_steps": self.completed_steps,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactExecutionRecord:
        data = _mapping(value, "execution")
        _exact_keys(data, _EXECUTION_KEYS, "execution")
        message = data["message"]
        if message is not None and not isinstance(message, str):
            raise ContactRecordValidationError(
                "execution.message must be null or a string"
            )
        return cls(
            status=_string(data["status"], "execution.status"),
            message=message,
            requested_steps=_integer(
                data["requested_steps"], "execution.requested_steps", minimum=1
            ),
            completed_steps=_integer(
                data["completed_steps"], "execution.completed_steps"
            ),
        )


@dataclass(frozen=True, slots=True)
class ContactTraceSample:
    """One post-step C0 observation.

    ``pair_active`` corresponds to the interval ending at this sample.  Sample
    zero therefore must be inactive.  ``signed_gap_m`` uses positive separation
    and negative penetration.  Backend-native details are descriptive only.
    """

    step: int
    time_s: float
    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]
    frame_poses: Mapping[
        str, tuple[float, float, float, float, float, float, float]
    ]
    position_targets: Mapping[str, float]
    probe_center_world_m: tuple[float, float, float]
    signed_gap_m: float
    pair_active: bool
    raw_contact_count: int | None
    native_contact_observation: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        step = _integer(self.step, "sample.step")
        if step > 7000:
            raise ContactRecordValidationError("sample.step exceeds the frozen maximum")
        time_s = _finite(self.time_s, "sample.time_s")
        if not 0.0 <= time_s <= 7.0:
            raise ContactRecordValidationError(
                "sample.time_s must lie within the frozen scenario duration"
            )
        positions = _finite_number_mapping(
            self.joint_positions, "sample.joint_positions"
        )
        velocities = _finite_number_mapping(
            self.joint_velocities, "sample.joint_velocities"
        )
        targets = _finite_number_mapping(
            self.position_targets, "sample.position_targets"
        )
        if set(positions) != set(velocities) or set(positions) != set(targets):
            raise ContactRecordValidationError(
                "sample joint positions, velocities, and targets must cover the same names"
            )
        if len(positions) != 22:
            raise ContactRecordValidationError(
                "sample joint mappings must contain exactly 22 entries"
            )
        frames = _frame_mapping(self.frame_poses, "sample.frame_poses")
        if len(frames) != 5:
            raise ContactRecordValidationError(
                "sample.frame_poses must contain exactly five distal frames"
            )
        if (
            not isinstance(self.probe_center_world_m, tuple)
            or len(self.probe_center_world_m) != 3
        ):
            raise ContactRecordValidationError(
                "sample.probe_center_world_m must be a tuple of three finite numbers"
            )
        for index, value in enumerate(self.probe_center_world_m):
            _finite(value, f"sample.probe_center_world_m[{index}]")
        _finite(self.signed_gap_m, "sample.signed_gap_m")
        if not isinstance(self.pair_active, bool):
            raise ContactRecordValidationError("sample.pair_active must be a boolean")
        if self.step == 0 and self.pair_active:
            raise ContactRecordValidationError("sample zero cannot report an active pair")
        if self.raw_contact_count is not None:
            _integer(self.raw_contact_count, "sample.raw_contact_count")
        if self.native_contact_observation is not None:
            native = _mapping(
                self.native_contact_observation,
                "sample.native_contact_observation",
            )
            _exact_keys(
                native,
                frozenset({"selected_pair_force_norm_n"}),
                "sample.native_contact_observation",
            )
            force_norm = _finite(
                native["selected_pair_force_norm_n"],
                "sample.native_contact_observation.selected_pair_force_norm_n",
            )
            if force_norm < 0.0:
                raise ContactRecordValidationError(
                    "sample native selected-pair force norm must be non-negative"
                )
            if self.step > 0 and self.pair_active != (force_norm > 1e-4):
                raise ContactRecordValidationError(
                    "sample.pair_active contradicts the frozen selected-pair force threshold"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "time_s": self.time_s,
            "joint_positions": dict(self.joint_positions),
            "joint_velocities": dict(self.joint_velocities),
            "frame_poses": {
                name: list(pose) for name, pose in self.frame_poses.items()
            },
            "position_targets": dict(self.position_targets),
            "probe_center_world_m": list(self.probe_center_world_m),
            "signed_gap_m": self.signed_gap_m,
            "pair_active": self.pair_active,
            "raw_contact_count": self.raw_contact_count,
            "native_contact_observation": (
                dict(self.native_contact_observation)
                if self.native_contact_observation is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: object, *, context: str = "sample") -> ContactTraceSample:
        data = _mapping(value, context)
        _exact_keys(data, _SAMPLE_KEYS, context)
        raw_gap = data["signed_gap_m"]
        raw_count = data["raw_contact_count"]
        raw_native = data["native_contact_observation"]
        if not isinstance(data["pair_active"], bool):
            raise ContactRecordValidationError(f"{context}.pair_active must be boolean")
        if raw_native is not None:
            raw_native = _mapping(raw_native, f"{context}.native_contact_observation")
        return cls(
            step=_integer(data["step"], f"{context}.step"),
            time_s=_finite(data["time_s"], f"{context}.time_s"),
            joint_positions=_finite_number_mapping(
                data["joint_positions"], f"{context}.joint_positions"
            ),
            joint_velocities=_finite_number_mapping(
                data["joint_velocities"], f"{context}.joint_velocities"
            ),
            frame_poses=_frame_mapping(
                data["frame_poses"], f"{context}.frame_poses"
            ),
            position_targets=_finite_number_mapping(
                data["position_targets"], f"{context}.position_targets"
            ),
            probe_center_world_m=_finite_vector(
                data["probe_center_world_m"],
                3,
                f"{context}.probe_center_world_m",
            ),  # type: ignore[arg-type]
            signed_gap_m=_finite(raw_gap, f"{context}.signed_gap_m"),
            pair_active=data["pair_active"],
            raw_contact_count=(
                None
                if raw_count is None
                else _integer(raw_count, f"{context}.raw_contact_count")
            ),
            native_contact_observation=(
                None
                if raw_native is None
                else _json_value(raw_native, f"{context}.native_contact_observation")
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ContactRun:
    """One strictly parsed C0 case payload."""

    schema_version: int
    manifest_id: str
    manifest_sha256: str
    case: ContactCaseRecord
    execution: ContactExecutionRecord
    mapping: Mapping[str, Any]
    fixture_readback: Mapping[str, Any]
    contact_observation: Mapping[str, Any]
    samples: tuple[ContactTraceSample, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != CONTACT_C0_RUN_SCHEMA_VERSION:
            raise ContactRecordValidationError("run.schema_version is unsupported")
        if self.manifest_id != CONTACT_C0_MANIFEST_ID:
            raise ContactRecordValidationError("run.manifest_id is not Contact C0")
        if _SHA256_RE.fullmatch(self.manifest_sha256) is None:
            raise ContactRecordValidationError(
                "run.manifest_sha256 must be lowercase SHA-256"
            )
        if not isinstance(self.case, ContactCaseRecord):
            raise ContactRecordValidationError("run.case must be ContactCaseRecord")
        if not isinstance(self.execution, ContactExecutionRecord):
            raise ContactRecordValidationError(
                "run.execution must be ContactExecutionRecord"
            )
        mapping = _validate_mapping_record(self.mapping, "run.mapping")
        fixture = _validate_fixture_readback(
            self.fixture_readback, "run.fixture_readback"
        )
        observation = _validate_contact_observation(
            self.contact_observation, "run.contact_observation"
        )
        provenance = _validate_provenance(self.provenance, "run.provenance")
        for name, value in (
            ("mapping", mapping),
            ("fixture_readback", fixture),
            ("contact_observation", observation),
            ("provenance", provenance),
        ):
            _json_value(value, f"run.{name}")
        if not isinstance(self.samples, tuple) or any(
            not isinstance(sample, ContactTraceSample) for sample in self.samples
        ):
            raise ContactRecordValidationError(
                "run.samples must be a tuple of ContactTraceSample"
            )
        expected_steps = round(7.0 / self.case.dt_s)
        if self.execution.requested_steps != expected_steps:
            raise ContactRecordValidationError(
                "execution.requested_steps contradicts case dt and frozen duration"
            )
        if self.execution.status == "completed":
            if len(self.samples) != expected_steps + 1:
                raise ContactRecordValidationError(
                    "completed run must contain N+1 samples"
                )
            expected_names: set[str] | None = None
            for index, sample in enumerate(self.samples):
                if sample.step != index or not math.isclose(
                    sample.time_s,
                    index * self.case.dt_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ContactRecordValidationError(
                        "completed run sample axis is not canonical"
                    )
                names = set(sample.joint_positions)
                if expected_names is None:
                    expected_names = names
                elif names != expected_names:
                    raise ContactRecordValidationError(
                        "completed run joint coverage changes across samples"
                    )
            expected_probe_frame = f"{self.case.hand}_index_DP"
            if (
                mapping["observed_joint_count"] != 22
                or mapping["observed_distal_frame_count"] != 5
                or mapping["probe_frame_name"] != expected_probe_frame
                or mapping["probe_frame_mapped"] is not True
            ):
                raise ContactRecordValidationError(
                    "completed run mapping is incomplete or contradicts its hand"
                )

            def require_close(field: str, expected: float) -> None:
                raw_value = fixture[field]
                if raw_value is None or not math.isclose(
                    float(raw_value), expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ContactRecordValidationError(
                        f"completed fixture readback {field!r} is not frozen"
                    )

            def require_vector(field: str, expected: tuple[float, ...]) -> None:
                raw_value = fixture[field]
                if not isinstance(raw_value, (list, tuple)) or len(raw_value) != len(expected):
                    raise ContactRecordValidationError(
                        f"completed fixture readback {field!r} is missing"
                    )
                if any(
                    not math.isclose(
                        float(actual), target, rel_tol=0.0, abs_tol=1e-12
                    )
                    for actual, target in zip(raw_value, expected)
                ):
                    raise ContactRecordValidationError(
                        f"completed fixture readback {field!r} is not frozen"
                    )

            if (
                fixture["performed"] is not True
                or fixture["profile_id"] != "synthetic_sphere_box_v1"
                or fixture["probe_parent_frame_name"] != expected_probe_frame
                or fixture["native_hand_collisions_enabled"] is not False
                or fixture["self_collisions_enabled"] is not False
                or fixture["ccd_enabled"] is not False
                or fixture["allowed_pair_id"] != CONTACT_C0_PAIR_ID
                or fixture["mujoco_condim"] != 1
                or fixture["native_collision_prim_count"] is None
                or fixture["native_collision_prim_count"] < 1
                or fixture["native_collision_disabled_count"]
                != fixture["native_collision_prim_count"]
                or fixture["enabled_collision_shape_count"] != 2
                or fixture["collision_inventory_sha256"] is None
                or fixture["mass_properties_preserved"] is not True
            ):
                raise ContactRecordValidationError(
                    "completed fixture readback is missing or contradicts the frozen profile"
                )
            require_vector("probe_local_center_m", (0.025, 0.0, 0.0))
            require_vector("target_world_center_m", (0.0, 0.0, 0.120))
            require_vector("target_half_extents_m", (0.150, 0.150, 0.010))
            require_close("probe_radius_m", 0.005)
            require_close("target_top_surface_z_m", 0.130)
            require_close("static_friction", 0.0)
            require_close("dynamic_friction", 0.0)
            require_close("restitution", 0.0)

            if (
                observation["performed"] is not True
                or observation["capability"]
                != "direct_filtered_pair_force_threshold"
                or observation["selected_pair_id"] != CONTACT_C0_PAIR_ID
                or observation["pair_active_source"]
                != "direct_filtered_selected_pair_force_norm_n"
                or observation["pair_active_force_threshold_n"] != 1e-4
                or observation["pair_active_record_count"] != len(self.samples)
                or observation["missing_pair_active_count"] != 0
                or observation["filtered_sensor_body_count"] != 1
                or observation["filtered_target_count"] != 1
            ):
                raise ContactRecordValidationError(
                    "completed run lacks the frozen selected-pair observation capability"
                )
            if (
                provenance["backend"] != self.case.simulator
                or provenance["manifest_sha256"] != self.manifest_sha256
                or any(
                    provenance[name] is None
                    for name in _PROVENANCE_KEYS
                    if name != "backend"
                )
            ):
                raise ContactRecordValidationError(
                    "completed run provenance is incomplete or contradictory"
                )
        elif len(self.samples) > self.execution.completed_steps + 1:
            raise ContactRecordValidationError(
                "error run contains more samples than completed physics permits"
            )

    @property
    def completed(self) -> bool:
        return self.execution.status == "completed"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "case": self.case.to_dict(),
            "execution": self.execution.to_dict(),
            "mapping": dict(self.mapping),
            "fixture_readback": dict(self.fixture_readback),
            "contact_observation": dict(self.contact_observation),
            "samples": [sample.to_dict() for sample in self.samples],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactRun:
        data = _mapping(value, "run")
        _exact_keys(data, _RUN_KEYS, "run")
        raw_samples = data["samples"]
        if not isinstance(raw_samples, list):
            raise ContactRecordValidationError("run.samples must be an array")
        return cls(
            schema_version=_integer(data["schema_version"], "run.schema_version"),
            manifest_id=_string(
                data["manifest_id"], "run.manifest_id", identifier=True
            ),
            manifest_sha256=_string(
                data["manifest_sha256"], "run.manifest_sha256"
            ),
            case=ContactCaseRecord.from_dict(data["case"]),
            execution=ContactExecutionRecord.from_dict(data["execution"]),
            mapping=_json_value(
                _validate_mapping_record(data["mapping"], "run.mapping"),
                "run.mapping",
            ),  # type: ignore[arg-type]
            fixture_readback=_json_value(
                _validate_fixture_readback(
                    data["fixture_readback"], "run.fixture_readback"
                ),
                "run.fixture_readback",
            ),  # type: ignore[arg-type]
            contact_observation=_json_value(
                _validate_contact_observation(
                    data["contact_observation"], "run.contact_observation"
                ),
                "run.contact_observation",
            ),  # type: ignore[arg-type]
            samples=tuple(
                ContactTraceSample.from_dict(item, context=f"run.samples[{index}]")
                for index, item in enumerate(raw_samples)
            ),
            provenance=_json_value(
                _validate_provenance(data["provenance"], "run.provenance"),
                "run.provenance",
            ),  # type: ignore[arg-type]
        )


def loads_contact_run(text: str) -> ContactRun:
    """Parse one strict finite C0 run from JSON text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ContactRecordValidationError(f"invalid contact run JSON: {exc}") from exc
    return ContactRun.from_dict(value)


def load_contact_run(path: str | Path) -> ContactRun:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContactRecordValidationError(
            "contact run path must be a regular non-symlink file"
        )
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContactRecordValidationError(f"cannot read contact run: {exc}") from exc
    return loads_contact_run(text)


def canonical_contact_run_json(run: ContactRun) -> str:
    if not isinstance(run, ContactRun):
        raise TypeError("run must be ContactRun")
    return json.dumps(
        run.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pair_active_bitset(samples: Sequence[ContactTraceSample]) -> tuple[bool, ...]:
    """Return the exact, step-ordered intended-pair activity sequence."""

    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("samples must be a sequence")
    result: list[bool] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, ContactTraceSample):
            raise TypeError(f"samples[{index}] must be ContactTraceSample")
        result.append(sample.pair_active)
    return tuple(result)


__all__ = [
    "ContactCaseRecord",
    "ContactExecutionRecord",
    "ContactRecordValidationError",
    "ContactRun",
    "ContactTraceSample",
    "canonical_contact_run_json",
    "load_contact_run",
    "loads_contact_run",
    "pair_active_bitset",
]
