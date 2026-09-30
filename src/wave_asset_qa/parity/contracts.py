"""Strict, simulator-neutral contracts for WaveSimParity manifests and results.

The contracts deliberately use only the Python standard library.  JSON Schema
files are published for external consumers, while these dataclasses perform
the runtime validation used by the local runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, TypeVar


RESULT_SCHEMA_VERSION = 1
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = (1, 2)
# Backwards-compatible public name retained for the v1 result contract.
SCHEMA_VERSION = RESULT_SCHEMA_VERSION
JOINT_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
DISTAL_FRAME_SUFFIXES = (
    "index_DP",
    "middle_DP",
    "pinky_DP",
    "ring_DP",
    "thumb_DP",
)

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Simulator(str, Enum):
    """Supported simulation backends for the MVP."""

    MUJOCO = "mujoco"
    OVPHYSX = "ovphysx"


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Mounting(str, Enum):
    FIXED_BASE = "fixed_base"


class ControlMode(str, Enum):
    POSITION = "position"


class ScenarioKind(str, Enum):
    ZERO_HOLD = "zero_hold"
    SMALL_STEP = "small_step"
    GRAVITY_SETTLING = "gravity_settling"
    OFFSET_SINE = "offset_sine"
    OFFSET_LINEAR_CHIRP = "offset_linear_chirp"


class ExecutionStatus(str, Enum):
    """Whether the requested simulator work itself completed."""

    COMPLETED = "completed"
    ERROR = "error"


class ComparisonStatus(str, Enum):
    """Scientific interpretation of two successfully collected traces."""

    WITHIN_TOLERANCE = "within_tolerance"
    DIVERGENT = "divergent"
    INCONCLUSIVE = "inconclusive"


EnumT = TypeVar("EnumT", bound=Enum)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _exact_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _string(value: object, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{context} must not be empty")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


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


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must be an array of three numbers")
    converted = tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(value))
    return converted  # type: ignore[return-value]


def _validate_identifier(value: str, context: str) -> None:
    _string(value, context)
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{context} must match {_IDENTIFIER_RE.pattern}")


def _validate_sha256(value: str, context: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase 64-character SHA-256")


def _validate_relative_path(value: str, context: str, suffix: str) -> None:
    _string(value, context)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{context} must be a normalized relative POSIX path")
    if path.suffix != suffix:
        raise ValueError(f"{context} must end with {suffix}")


def _validate_unique(values: tuple[str, ...], context: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{context} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class AssetProvenance:
    repository: str
    commit: str
    asset_git_tree: str
    canonical_lf_asset_tree_sha256: str
    asset_root: str

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not self.repository.startswith("https://"):
            raise ValueError("provenance.repository must be an HTTPS URL")
        if not isinstance(self.commit, str) or not _COMMIT_RE.fullmatch(self.commit):
            raise ValueError("provenance.commit must be a lowercase 40-character Git commit")
        if not isinstance(self.asset_git_tree, str) or not _COMMIT_RE.fullmatch(
            self.asset_git_tree
        ):
            raise ValueError(
                "provenance.asset_git_tree must be a lowercase 40-character Git tree"
            )
        _validate_sha256(
            self.canonical_lf_asset_tree_sha256,
            "provenance.canonical_lf_asset_tree_sha256",
        )
        _validate_relative_path(self.asset_root + "/placeholder.usd", "provenance.asset_root", ".usd")

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "commit": self.commit,
            "asset_git_tree": self.asset_git_tree,
            "canonical_lf_asset_tree_sha256": self.canonical_lf_asset_tree_sha256,
            "asset_root": self.asset_root,
        }

    @classmethod
    def from_dict(cls, value: object) -> AssetProvenance:
        data = _mapping(value, "provenance")
        _exact_keys(
            data,
            {
                "repository",
                "commit",
                "asset_git_tree",
                "canonical_lf_asset_tree_sha256",
                "asset_root",
            },
            "provenance",
        )
        return cls(
            repository=_string(data["repository"], "provenance.repository"),
            commit=_string(data["commit"], "provenance.commit"),
            asset_git_tree=_string(
                data["asset_git_tree"], "provenance.asset_git_tree"
            ),
            canonical_lf_asset_tree_sha256=_string(
                data["canonical_lf_asset_tree_sha256"],
                "provenance.canonical_lf_asset_tree_sha256",
            ),
            asset_root=_string(data["asset_root"], "provenance.asset_root"),
        )


@dataclass(frozen=True, slots=True)
class ModelPaths:
    mujoco: str
    ovphysx: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.mujoco, "model_paths.mujoco", ".xml")
        _validate_relative_path(self.ovphysx, "model_paths.ovphysx", ".usda")

    def for_simulator(self, simulator: Simulator) -> str:
        if simulator is Simulator.MUJOCO:
            return self.mujoco
        if simulator is Simulator.OVPHYSX:
            return self.ovphysx
        raise ValueError(f"unsupported simulator: {simulator!r}")

    def to_dict(self) -> dict[str, object]:
        return {"mujoco": self.mujoco, "ovphysx": self.ovphysx}

    @classmethod
    def from_dict(cls, value: object) -> ModelPaths:
        data = _mapping(value, "model_paths")
        _exact_keys(data, {"mujoco", "ovphysx"}, "model_paths")
        return cls(
            mujoco=_string(data["mujoco"], "model_paths.mujoco"),
            ovphysx=_string(data["ovphysx"], "model_paths.ovphysx"),
        )


@dataclass(frozen=True, slots=True)
class HandSpec:
    side: HandSide
    model_name: str
    mounting: Mounting
    control_mode: ControlMode
    model_paths: ModelPaths
    joint_names: tuple[str, ...]
    distal_frame_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.side, HandSide):
            raise ValueError("hand.side must be a HandSide")
        if not isinstance(self.mounting, Mounting):
            raise ValueError("hand.mounting must be a Mounting")
        if not isinstance(self.control_mode, ControlMode):
            raise ValueError("hand.control_mode must be a ControlMode")
        if not isinstance(self.model_paths, ModelPaths):
            raise ValueError("hand.model_paths must be ModelPaths")
        expected_name = f"{self.side.value}_sharpa_wave"
        if self.model_name != expected_name:
            raise ValueError(f"hand.model_name must be {expected_name!r}")
        expected_joints = tuple(f"{self.side.value}_{suffix}" for suffix in JOINT_SUFFIXES)
        if self.joint_names != expected_joints:
            raise ValueError(
                f"hand.joint_names must contain the 22 canonical {self.side.value} joints in MJCF order"
            )
        expected_frames = tuple(
            f"{self.side.value}_{suffix}" for suffix in DISTAL_FRAME_SUFFIXES
        )
        if self.distal_frame_names != expected_frames:
            raise ValueError(
                f"hand.distal_frame_names must contain the five canonical {self.side.value} distal frames"
            )
        _validate_unique(self.joint_names, "hand.joint_names")
        _validate_unique(self.distal_frame_names, "hand.distal_frame_names")
        if PurePosixPath(self.model_paths.mujoco).name != f"{self.model_name}.xml":
            raise ValueError("hand.model_paths.mujoco filename must match hand.model_name")
        if PurePosixPath(self.model_paths.ovphysx).name != f"{self.model_name}.usda":
            raise ValueError("hand.model_paths.ovphysx filename must match hand.model_name")

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side.value,
            "model_name": self.model_name,
            "mounting": self.mounting.value,
            "control_mode": self.control_mode.value,
            "model_paths": self.model_paths.to_dict(),
            "joint_names": list(self.joint_names),
            "distal_frame_names": list(self.distal_frame_names),
        }

    @classmethod
    def from_dict(cls, value: object, *, context: str = "hand") -> HandSpec:
        data = _mapping(value, context)
        _exact_keys(
            data,
            {
                "side",
                "model_name",
                "mounting",
                "control_mode",
                "model_paths",
                "joint_names",
                "distal_frame_names",
            },
            context,
        )
        return cls(
            side=_enum(data["side"], HandSide, f"{context}.side"),
            model_name=_string(data["model_name"], f"{context}.model_name"),
            mounting=_enum(data["mounting"], Mounting, f"{context}.mounting"),
            control_mode=_enum(
                data["control_mode"], ControlMode, f"{context}.control_mode"
            ),
            model_paths=ModelPaths.from_dict(data["model_paths"]),
            joint_names=_string_tuple(data["joint_names"], f"{context}.joint_names"),
            distal_frame_names=_string_tuple(
                data["distal_frame_names"], f"{context}.distal_frame_names"
            ),
        )


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scenario_id: str
    kind: ScenarioKind
    duration_s: float
    dt_s: float
    gravity_m_s2: tuple[float, float, float]
    initial_position_rad: float
    target_position_rad: float
    step_start_s: float | None
    expect_no_contacts: bool

    def __post_init__(self) -> None:
        _validate_identifier(self.scenario_id, "scenario.scenario_id")
        if not isinstance(self.kind, ScenarioKind):
            raise ValueError("scenario.kind must be a ScenarioKind")
        if self.kind not in {
            ScenarioKind.ZERO_HOLD,
            ScenarioKind.SMALL_STEP,
            ScenarioKind.GRAVITY_SETTLING,
        }:
            raise ValueError("v1 scenario.kind must be a Gate 0 scenario kind")
        duration = _number(self.duration_s, "scenario.duration_s")
        dt = _number(self.dt_s, "scenario.dt_s")
        initial = _number(self.initial_position_rad, "scenario.initial_position_rad")
        target = _number(self.target_position_rad, "scenario.target_position_rad")
        if duration <= 0.0 or dt <= 0.0:
            raise ValueError("scenario duration_s and dt_s must be positive")
        ratio = duration / dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("scenario.duration_s must be an integer multiple of scenario.dt_s")
        if (
            not isinstance(self.gravity_m_s2, tuple)
            or len(self.gravity_m_s2) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in self.gravity_m_s2
            )
            or any(not math.isfinite(float(item)) for item in self.gravity_m_s2)
        ):
            raise ValueError("scenario.gravity_m_s2 must contain three finite numbers")
        if not isinstance(self.expect_no_contacts, bool):
            raise ValueError("scenario.expect_no_contacts must be a boolean")
        if initial != 0.0:
            raise ValueError("scenario.initial_position_rad must be zero for Gate 0")

        zero_gravity = all(float(item) == 0.0 for item in self.gravity_m_s2)
        if self.kind is ScenarioKind.SMALL_STEP:
            if target <= 0.0:
                raise ValueError("small_step target_position_rad must be positive")
            if self.step_start_s is None:
                raise ValueError("small_step requires step_start_s")
            start = _number(self.step_start_s, "scenario.step_start_s")
            if not 0.0 < start < duration:
                raise ValueError("small_step step_start_s must be inside the scenario duration")
            if not math.isclose(start / dt, round(start / dt), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("small_step step_start_s must align to scenario.dt_s")
            if not zero_gravity:
                raise ValueError("small_step must use zero gravity in Gate 0")
        else:
            if self.step_start_s is not None:
                raise ValueError(f"{self.kind.value} step_start_s must be null")
            if target != 0.0:
                raise ValueError(f"{self.kind.value} target_position_rad must be zero")

        if self.kind is ScenarioKind.ZERO_HOLD and not zero_gravity:
            raise ValueError("zero_hold must use zero gravity in Gate 0")
        if self.kind is ScenarioKind.GRAVITY_SETTLING and zero_gravity:
            raise ValueError("gravity_settling must use non-zero gravity")

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
            "target_position_rad": self.target_position_rad,
            "step_start_s": self.step_start_s,
            "expect_no_contacts": self.expect_no_contacts,
        }

    @classmethod
    def from_dict(cls, value: object, *, context: str = "scenario") -> ScenarioSpec:
        data = _mapping(value, context)
        _exact_keys(
            data,
            {
                "scenario_id",
                "kind",
                "duration_s",
                "dt_s",
                "gravity_m_s2",
                "initial_position_rad",
                "target_position_rad",
                "step_start_s",
                "expect_no_contacts",
            },
            context,
        )
        return cls(
            scenario_id=_string(data["scenario_id"], f"{context}.scenario_id"),
            kind=_enum(data["kind"], ScenarioKind, f"{context}.kind"),
            duration_s=_number(data["duration_s"], f"{context}.duration_s"),
            dt_s=_number(data["dt_s"], f"{context}.dt_s"),
            gravity_m_s2=_vector3(data["gravity_m_s2"], f"{context}.gravity_m_s2"),
            initial_position_rad=_number(
                data["initial_position_rad"], f"{context}.initial_position_rad"
            ),
            target_position_rad=_number(
                data["target_position_rad"], f"{context}.target_position_rad"
            ),
            step_start_s=(
                None
                if data["step_start_s"] is None
                else _number(data["step_start_s"], f"{context}.step_start_s")
            ),
            expect_no_contacts=_boolean(
                data["expect_no_contacts"], f"{context}.expect_no_contacts"
            ),
        )


@dataclass(frozen=True, slots=True)
class WaveformScenarioSpec:
    """Canonical v2 all-joint, no-contact position-command waveform."""

    scenario_id: str
    kind: ScenarioKind
    duration_s: float
    dt_s: float
    gravity_m_s2: tuple[float, float, float]
    initial_position_rad: float
    command_offset_rad: float
    command_amplitude_rad: float
    frequency_start_hz: float
    frequency_end_hz: float
    expect_no_contacts: bool

    def __post_init__(self) -> None:
        _validate_identifier(self.scenario_id, "waveform_scenario.scenario_id")
        if self.kind not in {
            ScenarioKind.OFFSET_SINE,
            ScenarioKind.OFFSET_LINEAR_CHIRP,
        }:
            raise ValueError(
                "waveform_scenario.kind must be offset_sine or offset_linear_chirp"
            )
        duration = _number(self.duration_s, "waveform_scenario.duration_s")
        dt = _number(self.dt_s, "waveform_scenario.dt_s")
        initial = _number(
            self.initial_position_rad, "waveform_scenario.initial_position_rad"
        )
        offset = _number(
            self.command_offset_rad, "waveform_scenario.command_offset_rad"
        )
        amplitude = _number(
            self.command_amplitude_rad, "waveform_scenario.command_amplitude_rad"
        )
        start_hz = _number(
            self.frequency_start_hz, "waveform_scenario.frequency_start_hz"
        )
        end_hz = _number(
            self.frequency_end_hz, "waveform_scenario.frequency_end_hz"
        )
        if duration <= 0.0 or dt <= 0.0:
            raise ValueError("waveform scenario duration_s and dt_s must be positive")
        ratio = duration / dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "waveform_scenario.duration_s must be an integer multiple of dt_s"
            )
        if (
            not isinstance(self.gravity_m_s2, tuple)
            or len(self.gravity_m_s2) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in self.gravity_m_s2
            )
            or any(not math.isfinite(float(item)) for item in self.gravity_m_s2)
        ):
            raise ValueError(
                "waveform_scenario.gravity_m_s2 must contain three finite numbers"
            )
        if any(float(item) != 0.0 for item in self.gravity_m_s2):
            raise ValueError("waveform scenarios must use zero gravity")
        if not isinstance(self.expect_no_contacts, bool):
            raise ValueError("waveform_scenario.expect_no_contacts must be a boolean")
        if not self.expect_no_contacts:
            raise ValueError("waveform scenarios must declare expect_no_contacts=true")
        if initial != 0.0:
            raise ValueError("waveform_scenario.initial_position_rad must be zero")
        if offset <= 0.0 or amplitude <= 0.0:
            raise ValueError("waveform command offset and amplitude must be positive")
        if offset != amplitude:
            raise ValueError(
                "waveform command_offset_rad must equal command_amplitude_rad"
            )
        lower_target = offset - amplitude
        upper_target = offset + amplitude
        if lower_target < -1e-12 or upper_target > 0.05 + 1e-12:
            raise ValueError("waveform command envelope must remain within [0, 0.05] rad")
        if start_hz <= 0.0 or end_hz <= 0.0:
            raise ValueError("waveform frequencies must be positive")
        if self.kind is ScenarioKind.OFFSET_SINE and end_hz != start_hz:
            raise ValueError("offset_sine must use one constant frequency")
        if (
            self.kind is ScenarioKind.OFFSET_LINEAR_CHIRP
            and end_hz <= start_hz
        ):
            raise ValueError(
                "offset_linear_chirp frequency_end_hz must exceed frequency_start_hz"
            )
        cycles = start_hz * duration + 0.5 * (end_hz - start_hz) * duration
        if not math.isclose(cycles, round(cycles), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "waveform phase must complete an integer number of cycles"
            )

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
            "command_offset_rad": self.command_offset_rad,
            "command_amplitude_rad": self.command_amplitude_rad,
            "frequency_start_hz": self.frequency_start_hz,
            "frequency_end_hz": self.frequency_end_hz,
            "expect_no_contacts": self.expect_no_contacts,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        context: str = "waveform_scenario",
    ) -> WaveformScenarioSpec:
        data = _mapping(value, context)
        _exact_keys(
            data,
            {
                "scenario_id",
                "kind",
                "duration_s",
                "dt_s",
                "gravity_m_s2",
                "initial_position_rad",
                "command_offset_rad",
                "command_amplitude_rad",
                "frequency_start_hz",
                "frequency_end_hz",
                "expect_no_contacts",
            },
            context,
        )
        return cls(
            scenario_id=_string(data["scenario_id"], f"{context}.scenario_id"),
            kind=_enum(data["kind"], ScenarioKind, f"{context}.kind"),
            duration_s=_number(data["duration_s"], f"{context}.duration_s"),
            dt_s=_number(data["dt_s"], f"{context}.dt_s"),
            gravity_m_s2=_vector3(
                data["gravity_m_s2"], f"{context}.gravity_m_s2"
            ),
            initial_position_rad=_number(
                data["initial_position_rad"], f"{context}.initial_position_rad"
            ),
            command_offset_rad=_number(
                data["command_offset_rad"], f"{context}.command_offset_rad"
            ),
            command_amplitude_rad=_number(
                data["command_amplitude_rad"],
                f"{context}.command_amplitude_rad",
            ),
            frequency_start_hz=_number(
                data["frequency_start_hz"], f"{context}.frequency_start_hz"
            ),
            frequency_end_hz=_number(
                data["frequency_end_hz"], f"{context}.frequency_end_hz"
            ),
            expect_no_contacts=_boolean(
                data["expect_no_contacts"], f"{context}.expect_no_contacts"
            ),
        )


CanonicalScenarioSpec = ScenarioSpec | WaveformScenarioSpec


@dataclass(frozen=True, slots=True)
class RunPolicy:
    repeat_count: int
    dt_halving_scenario_ids: tuple[str, ...]
    minimum_completion_fraction: float

    def __post_init__(self) -> None:
        if isinstance(self.repeat_count, bool) or not isinstance(self.repeat_count, int):
            raise ValueError("run_policy.repeat_count must be an integer")
        if self.repeat_count < 2:
            raise ValueError("run_policy.repeat_count must be at least two")
        if not isinstance(self.dt_halving_scenario_ids, tuple):
            raise ValueError("run_policy.dt_halving_scenario_ids must be a tuple")
        for index, scenario_id in enumerate(self.dt_halving_scenario_ids):
            _validate_identifier(
                scenario_id, f"run_policy.dt_halving_scenario_ids[{index}]"
            )
        if not self.dt_halving_scenario_ids:
            raise ValueError("run_policy.dt_halving_scenario_ids must not be empty")
        _validate_unique(
            self.dt_halving_scenario_ids, "run_policy.dt_halving_scenario_ids"
        )
        completion = _number(
            self.minimum_completion_fraction,
            "run_policy.minimum_completion_fraction",
        )
        if not 0.0 < completion <= 1.0:
            raise ValueError(
                "run_policy.minimum_completion_fraction must be in (0, 1]"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat_count": self.repeat_count,
            "dt_halving_scenario_ids": list(self.dt_halving_scenario_ids),
            "minimum_completion_fraction": self.minimum_completion_fraction,
        }

    @classmethod
    def from_dict(cls, value: object) -> RunPolicy:
        data = _mapping(value, "run_policy")
        _exact_keys(
            data,
            {
                "repeat_count",
                "dt_halving_scenario_ids",
                "minimum_completion_fraction",
            },
            "run_policy",
        )
        return cls(
            repeat_count=_integer(data["repeat_count"], "run_policy.repeat_count"),
            dt_halving_scenario_ids=_string_tuple(
                data["dt_halving_scenario_ids"],
                "run_policy.dt_halving_scenario_ids",
            ),
            minimum_completion_fraction=_number(
                data["minimum_completion_fraction"],
                "run_policy.minimum_completion_fraction",
            ),
        )


@dataclass(frozen=True, slots=True)
class ParityManifest:
    schema_version: int
    manifest_id: str
    provenance: AssetProvenance
    simulators: tuple[Simulator, ...]
    hands: tuple[HandSpec, ...]
    scenarios: tuple[CanonicalScenarioSpec, ...]
    run_policy: RunPolicy

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                "manifest.schema_version must be one of "
                f"{SUPPORTED_MANIFEST_SCHEMA_VERSIONS}"
            )
        _validate_identifier(self.manifest_id, "manifest_id")
        if not isinstance(self.provenance, AssetProvenance):
            raise ValueError("provenance must be AssetProvenance")
        if self.simulators != (Simulator.MUJOCO, Simulator.OVPHYSX):
            raise ValueError("simulators must be ['mujoco', 'ovphysx'] in canonical order")
        if (
            not isinstance(self.hands, tuple)
            or any(not isinstance(hand, HandSpec) for hand in self.hands)
            or tuple(hand.side for hand in self.hands) != (HandSide.LEFT, HandSide.RIGHT)
        ):
            raise ValueError("hands must contain left then right")
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise ValueError("scenarios must not be empty")
        scenario_type = (
            ScenarioSpec if self.schema_version == 1 else WaveformScenarioSpec
        )
        if any(not isinstance(scenario, scenario_type) for scenario in self.scenarios):
            raise ValueError(
                f"manifest schema v{self.schema_version} contains an invalid scenario type"
            )
        scenario_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        _validate_unique(scenario_ids, "scenarios[].scenario_id")
        kinds = tuple(scenario.kind for scenario in self.scenarios)
        if self.schema_version == 1:
            required_kinds = {
                ScenarioKind.ZERO_HOLD,
                ScenarioKind.SMALL_STEP,
                ScenarioKind.GRAVITY_SETTLING,
            }
            if set(kinds) != required_kinds or len(kinds) != len(required_kinds):
                raise ValueError(
                    "scenarios must contain exactly one zero_hold, small_step, and gravity_settling"
                )
        else:
            required_order = (
                ScenarioKind.OFFSET_SINE,
                ScenarioKind.OFFSET_LINEAR_CHIRP,
            )
            if kinds != required_order:
                raise ValueError(
                    "v2 scenarios must contain offset_sine then offset_linear_chirp"
                )
            required_ids = tuple(kind.value for kind in required_order)
            if scenario_ids != required_ids:
                raise ValueError(
                    "v2 scenario IDs must match the canonical waveform kind order"
                )
            waveform_parameters = tuple(
                (
                    scenario.duration_s,
                    scenario.dt_s,
                    scenario.command_offset_rad,
                    scenario.command_amplitude_rad,
                    scenario.frequency_start_hz,
                    scenario.frequency_end_hz,
                )
                for scenario in self.scenarios
                if isinstance(scenario, WaveformScenarioSpec)
            )
            if waveform_parameters != (
                (4.0, 0.002, 0.025, 0.025, 1.0, 1.0),
                (4.0, 0.002, 0.025, 0.025, 0.5, 4.0),
            ):
                raise ValueError("v2 waveform parameters must match the frozen T1 profile")
        if not isinstance(self.run_policy, RunPolicy):
            raise ValueError("run_policy must be RunPolicy")
        if self.schema_version == 1 and self.run_policy.minimum_completion_fraction != 1.0:
            raise ValueError(
                "v1 run_policy.minimum_completion_fraction must be 1.0 for Gate 0"
            )
        if self.schema_version == 2:
            if self.run_policy.repeat_count != 2:
                raise ValueError("v2 run_policy.repeat_count must be 2")
            if self.run_policy.dt_halving_scenario_ids != scenario_ids:
                raise ValueError(
                    "v2 dt_halving_scenario_ids must contain both waveform scenarios in order"
                )
            if self.run_policy.minimum_completion_fraction != 0.99:
                raise ValueError(
                    "v2 run_policy.minimum_completion_fraction must be 0.99"
                )
        unknown_dt_scenarios = set(self.run_policy.dt_halving_scenario_ids) - set(scenario_ids)
        if unknown_dt_scenarios:
            raise ValueError(
                "run_policy.dt_halving_scenario_ids references unknown scenario(s): "
                + ", ".join(sorted(unknown_dt_scenarios))
            )

    @property
    def expected_joint_mapping_count(self) -> int:
        return sum(len(hand.joint_names) for hand in self.hands)

    @property
    def expected_distal_frame_mapping_count(self) -> int:
        return sum(len(hand.distal_frame_names) for hand in self.hands)

    def hand(self, side: HandSide) -> HandSpec:
        for hand in self.hands:
            if hand.side is side:
                return hand
        raise KeyError(side.value)

    def scenario(self, scenario_id: str) -> CanonicalScenarioSpec:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(scenario_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "provenance": self.provenance.to_dict(),
            "simulators": [item.value for item in self.simulators],
            "hands": [hand.to_dict() for hand in self.hands],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "run_policy": self.run_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ParityManifest:
        data = _mapping(value, "manifest")
        _exact_keys(
            data,
            {
                "schema_version",
                "manifest_id",
                "provenance",
                "simulators",
                "hands",
                "scenarios",
                "run_policy",
            },
            "manifest",
        )
        hands_value = data["hands"]
        scenarios_value = data["scenarios"]
        if not isinstance(hands_value, list):
            raise ValueError("manifest.hands must be an array")
        if not isinstance(scenarios_value, list):
            raise ValueError("manifest.scenarios must be an array")
        schema_version = _integer(data["schema_version"], "manifest.schema_version")
        if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                "manifest.schema_version must be one of "
                f"{SUPPORTED_MANIFEST_SCHEMA_VERSIONS}"
            )
        scenario_factory = (
            ScenarioSpec.from_dict
            if schema_version == 1
            else WaveformScenarioSpec.from_dict
        )
        return cls(
            schema_version=schema_version,
            manifest_id=_string(data["manifest_id"], "manifest.manifest_id"),
            provenance=AssetProvenance.from_dict(data["provenance"]),
            simulators=_enum_tuple(data["simulators"], Simulator, "manifest.simulators"),
            hands=tuple(
                HandSpec.from_dict(item, context=f"manifest.hands[{index}]")
                for index, item in enumerate(hands_value)
            ),
            scenarios=tuple(
                scenario_factory(item, context=f"manifest.scenarios[{index}]")
                for index, item in enumerate(scenarios_value)
            ),
            run_policy=RunPolicy.from_dict(data["run_policy"]),
        )


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    simulator: Simulator
    execution_status: ExecutionStatus
    completed_repeats: int
    requested_repeats: int
    finite: bool
    bundle_root_sha256: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.simulator, Simulator):
            raise ValueError("execution.simulator must be a Simulator")
        if not isinstance(self.execution_status, ExecutionStatus):
            raise ValueError("execution.execution_status must be an ExecutionStatus")
        for name, value in (
            ("completed_repeats", self.completed_repeats),
            ("requested_repeats", self.requested_repeats),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"execution.{name} must be an integer")
        if self.requested_repeats < 1:
            raise ValueError("execution.requested_repeats must be positive")
        if not 0 <= self.completed_repeats <= self.requested_repeats:
            raise ValueError("execution.completed_repeats must be within the requested range")
        if not isinstance(self.finite, bool):
            raise ValueError("execution.finite must be a boolean")
        if self.bundle_root_sha256 is not None:
            _validate_sha256(self.bundle_root_sha256, "execution.bundle_root_sha256")
        if self.message is not None:
            _string(self.message, "execution.message")
        if self.execution_status is ExecutionStatus.COMPLETED:
            if self.completed_repeats != self.requested_repeats:
                raise ValueError("completed execution must finish every requested repeat")
            if not self.finite:
                raise ValueError("completed execution must report finite=true")
        elif self.message is None:
            raise ValueError("error execution requires a message")

    def to_dict(self) -> dict[str, object]:
        return {
            "simulator": self.simulator.value,
            "execution_status": self.execution_status.value,
            "completed_repeats": self.completed_repeats,
            "requested_repeats": self.requested_repeats,
            "finite": self.finite,
            "bundle_root_sha256": self.bundle_root_sha256,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: object, *, context: str = "execution") -> ExecutionRecord:
        data = _mapping(value, context)
        _exact_keys(
            data,
            {
                "simulator",
                "execution_status",
                "completed_repeats",
                "requested_repeats",
                "finite",
                "bundle_root_sha256",
                "message",
            },
            context,
        )
        return cls(
            simulator=_enum(data["simulator"], Simulator, f"{context}.simulator"),
            execution_status=_enum(
                data["execution_status"], ExecutionStatus, f"{context}.execution_status"
            ),
            completed_repeats=_integer(
                data["completed_repeats"], f"{context}.completed_repeats"
            ),
            requested_repeats=_integer(
                data["requested_repeats"], f"{context}.requested_repeats"
            ),
            finite=_boolean(data["finite"], f"{context}.finite"),
            bundle_root_sha256=_optional_string(
                data["bundle_root_sha256"], f"{context}.bundle_root_sha256"
            ),
            message=_optional_string(data["message"], f"{context}.message"),
        )


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    comparison_status: ComparisonStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.comparison_status, ComparisonStatus):
            raise ValueError("comparison.comparison_status must be a ComparisonStatus")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("comparison.metrics must be an object")
        for name, value in self.metrics.items():
            _validate_identifier(name, "comparison.metrics key")
            _number(value, f"comparison.metrics.{name}")
        if self.message is not None:
            _string(self.message, "comparison.message")
        if self.comparison_status is ComparisonStatus.INCONCLUSIVE and self.message is None:
            raise ValueError("inconclusive comparison requires a message")

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_status": self.comparison_status.value,
            "metrics": {name: float(self.metrics[name]) for name in sorted(self.metrics)},
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: object) -> ComparisonRecord:
        data = _mapping(value, "comparison")
        _exact_keys(data, {"comparison_status", "metrics", "message"}, "comparison")
        metrics_data = _mapping(data["metrics"], "comparison.metrics")
        metrics = {
            name: _number(metric, f"comparison.metrics.{name}")
            for name, metric in metrics_data.items()
        }
        return cls(
            comparison_status=_enum(
                data["comparison_status"],
                ComparisonStatus,
                "comparison.comparison_status",
            ),
            metrics=metrics,
            message=_optional_string(data["message"], "comparison.message"),
        )


@dataclass(frozen=True, slots=True)
class ParityResult:
    schema_version: int
    manifest_id: str
    manifest_sha256: str
    hand: HandSide
    scenario_id: str
    executions: tuple[ExecutionRecord, ...]
    comparison: ComparisonRecord

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {RESULT_SCHEMA_VERSION}")
        _validate_identifier(self.manifest_id, "result.manifest_id")
        _validate_sha256(self.manifest_sha256, "result.manifest_sha256")
        if not isinstance(self.hand, HandSide):
            raise ValueError("result.hand must be a HandSide")
        _validate_identifier(self.scenario_id, "result.scenario_id")
        if (
            not isinstance(self.executions, tuple)
            or any(not isinstance(item, ExecutionRecord) for item in self.executions)
            or tuple(item.simulator for item in self.executions)
            != (Simulator.MUJOCO, Simulator.OVPHYSX)
        ):
            raise ValueError("result.executions must contain MuJoCo then OVPhysX")
        if not isinstance(self.comparison, ComparisonRecord):
            raise ValueError("result.comparison must be ComparisonRecord")
        if any(
            execution.execution_status is ExecutionStatus.ERROR
            for execution in self.executions
        ) and self.comparison.comparison_status is not ComparisonStatus.INCONCLUSIVE:
            raise ValueError("an execution error requires an inconclusive comparison")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "hand": self.hand.value,
            "scenario_id": self.scenario_id,
            "executions": [item.to_dict() for item in self.executions],
            "comparison": self.comparison.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ParityResult:
        data = _mapping(value, "result")
        _exact_keys(
            data,
            {
                "schema_version",
                "manifest_id",
                "manifest_sha256",
                "hand",
                "scenario_id",
                "executions",
                "comparison",
            },
            "result",
        )
        executions_value = data["executions"]
        if not isinstance(executions_value, list):
            raise ValueError("result.executions must be an array")
        return cls(
            schema_version=_integer(data["schema_version"], "result.schema_version"),
            manifest_id=_string(data["manifest_id"], "result.manifest_id"),
            manifest_sha256=_string(data["manifest_sha256"], "result.manifest_sha256"),
            hand=_enum(data["hand"], HandSide, "result.hand"),
            scenario_id=_string(data["scenario_id"], "result.scenario_id"),
            executions=tuple(
                ExecutionRecord.from_dict(item, context=f"result.executions[{index}]")
                for index, item in enumerate(executions_value)
            ),
            comparison=ComparisonRecord.from_dict(data["comparison"]),
        )


__all__ = [
    "AssetProvenance",
    "CanonicalScenarioSpec",
    "ComparisonRecord",
    "ComparisonStatus",
    "ControlMode",
    "DISTAL_FRAME_SUFFIXES",
    "ExecutionRecord",
    "ExecutionStatus",
    "HandSide",
    "HandSpec",
    "JOINT_SUFFIXES",
    "ModelPaths",
    "Mounting",
    "ParityManifest",
    "ParityResult",
    "RESULT_SCHEMA_VERSION",
    "RunPolicy",
    "SCHEMA_VERSION",
    "ScenarioKind",
    "ScenarioSpec",
    "Simulator",
    "SUPPORTED_MANIFEST_SCHEMA_VERSIONS",
    "WaveformScenarioSpec",
]
