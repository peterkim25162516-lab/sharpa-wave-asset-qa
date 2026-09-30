"""Local numerical comparison for WaveSimParity Gate 0 result traces.

Execution validation and scientific comparison are intentionally separate.
Malformed, incomplete, or non-finite simulator output is an execution error
and makes parity inconclusive; it is never promoted to a divergence finding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Iterable, Mapping, Sequence

from wave_asset_qa.adapters.base import AdapterRunResult, TraceSample

from .contracts import (
    ComparisonRecord,
    ComparisonStatus,
    ExecutionRecord,
    ExecutionStatus,
    HandSide,
    HandSpec,
    ParityManifest,
    ParityResult,
    Simulator,
)
from .mapping import (
    MAPPING_SCHEMA_VERSION,
    FrameMapping,
    JointMapping,
    MappingValidationError,
    validate_mapping,
)
from .scenarios import (
    ScenarioCase,
    TimestepVariant,
    canonical_position_targets,
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    manifest_sha256,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are data-integrity guardrails, not parity-performance thresholds.  A
# trace outside them is treated as invalid/inconclusive even when both
# simulators happen to fail in the same way.
MAX_ABS_CANONICAL_JOINT_POSITION_RAD = math.tau
MAX_ABS_CANONICAL_POSITION_TARGET_RAD = math.tau
MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S = 1_000.0
MAX_FRAME_ORIGIN_DISTANCE_M = 10.0
UNIT_QUATERNION_NORM_TOLERANCE = 1e-3
SMALL_STEP_MIN_OBSERVED_RESPONSE_RAD = 1e-5


@dataclass(frozen=True, slots=True)
class ComparisonThresholds:
    """Explicit diagnostic thresholds used by Gate 0.

    These defaults are initial cross-simulator QA thresholds, not hardware or
    product-performance specifications.  Every value is included in reports.
    """

    minimum_completion_fraction: float = 1.0
    crosssim_joint_max_abs_rad: float = 0.02
    crosssim_frame_position_max_m: float = 0.002
    crosssim_frame_orientation_max_rad: float = 0.05
    repeat_joint_max_abs_rad: float = 1e-9
    repeat_frame_position_max_m: float = 1e-9
    repeat_frame_orientation_max_rad: float = 1e-9
    dt_halving_joint_max_abs_rad: float = 0.01
    dt_halving_frame_position_max_m: float = 0.001
    dt_halving_frame_orientation_max_rad: float = 0.02

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"threshold {name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"threshold {name} must be finite")
            if name == "minimum_completion_fraction":
                if not 0.0 < float(value) <= 1.0:
                    raise ValueError("minimum_completion_fraction must be in (0, 1]")
            elif float(value) < 0.0:
                raise ValueError(f"threshold {name} must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_completion_fraction": self.minimum_completion_fraction,
            "crosssim_joint_max_abs_rad": self.crosssim_joint_max_abs_rad,
            "crosssim_frame_position_max_m": self.crosssim_frame_position_max_m,
            "crosssim_frame_orientation_max_rad": self.crosssim_frame_orientation_max_rad,
            "repeat_joint_max_abs_rad": self.repeat_joint_max_abs_rad,
            "repeat_frame_position_max_m": self.repeat_frame_position_max_m,
            "repeat_frame_orientation_max_rad": self.repeat_frame_orientation_max_rad,
            "dt_halving_joint_max_abs_rad": self.dt_halving_joint_max_abs_rad,
            "dt_halving_frame_position_max_m": self.dt_halving_frame_position_max_m,
            "dt_halving_frame_orientation_max_rad": self.dt_halving_frame_orientation_max_rad,
        }


@dataclass(frozen=True, slots=True)
class CollectedRun:
    """One expected run case paired with its backend result bundle payload."""

    case: ScenarioCase
    result: AdapterRunResult
    bundle_root_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case, ScenarioCase):
            raise ValueError("collected run case must be ScenarioCase")
        if not isinstance(self.result, AdapterRunResult):
            raise ValueError("collected run result must be AdapterRunResult")
        if self.bundle_root_sha256 is not None and not _SHA256_RE.fullmatch(
            self.bundle_root_sha256
        ):
            raise ValueError("bundle_root_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class Gate0Comparison:
    """Self-contained local comparison summary for reporting and JSON export."""

    manifest_id: str
    manifest_sha256: str
    upstream_repository: str
    upstream_commit: str
    asset_git_tree: str
    canonical_lf_asset_tree_sha256: str
    expected_joint_mapping_count: int
    expected_distal_frame_mapping_count: int
    observed_joint_mapping_count: int
    observed_distal_frame_mapping_count: int
    expected_run_count: int
    received_run_count: int
    completed_run_count: int
    completion_fraction: float
    step_completion_fraction: float
    nonfinite_run_count: int
    execution_status: ExecutionStatus
    comparison_status: ComparisonStatus
    thresholds: ComparisonThresholds
    results: tuple[ParityResult, ...]
    observations: tuple[str, ...] = ()

    @property
    def completion_target_met(self) -> bool:
        return self.completion_fraction >= self.thresholds.minimum_completion_fraction

    @property
    def finite(self) -> bool:
        return self.nonfinite_run_count == 0

    @property
    def mapping_validation_complete(self) -> bool:
        return (
            self.execution_status is ExecutionStatus.COMPLETED
            and self.observed_joint_mapping_count
            == self.expected_joint_mapping_count
            and self.observed_distal_frame_mapping_count
            == self.expected_distal_frame_mapping_count
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "metadata": {
                "manifest_id": self.manifest_id,
                "manifest_sha256": self.manifest_sha256,
                "upstream_repository": self.upstream_repository,
                "upstream_commit": self.upstream_commit,
                "asset_git_tree": self.asset_git_tree,
                "canonical_lf_asset_tree_sha256": (
                    self.canonical_lf_asset_tree_sha256
                ),
                "backend_pair": ["mujoco", "kit-less ovphysx"],
            },
            "summary": {
                "execution_status": self.execution_status.value,
                "comparison_status": self.comparison_status.value,
                "expected_joint_mapping_count": self.expected_joint_mapping_count,
                "expected_distal_frame_mapping_count": (
                    self.expected_distal_frame_mapping_count
                ),
                "observed_joint_mapping_count": self.observed_joint_mapping_count,
                "observed_distal_frame_mapping_count": (
                    self.observed_distal_frame_mapping_count
                ),
                "mapping_validation_complete": self.mapping_validation_complete,
                "expected_run_count": self.expected_run_count,
                "received_run_count": self.received_run_count,
                "completed_run_count": self.completed_run_count,
                "completion_fraction": self.completion_fraction,
                "step_completion_fraction": self.step_completion_fraction,
                "completion_target_met": self.completion_target_met,
                "nonfinite_run_count": self.nonfinite_run_count,
                "finite": self.finite,
            },
            "thresholds": self.thresholds.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "observations": list(self.observations),
        }


@dataclass(frozen=True, slots=True)
class _RunAssessment:
    collected: CollectedRun
    expected_steps: int
    finite: bool
    errors: tuple[str, ...]
    joint_mapping: tuple[JointMapping, ...] = ()
    frame_mapping: tuple[FrameMapping, ...] = ()
    backend_joint_names: tuple[str, ...] = ()
    backend_frame_names: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        return not self.errors


class _ComparisonUnavailable(ValueError):
    pass


def _all_finite(values: Iterable[float]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _sample_is_finite(sample: TraceSample) -> bool:
    try:
        if not _all_finite((sample.time_s, *sample.qpos, *sample.qvel)):
            return False
        for values in (
            sample.joint_positions.values(),
            sample.position_targets.values(),
        ):
            if not _all_finite(values):
                return False
        for pose in sample.frame_poses.values():
            if not _all_finite(pose):
                return False
        return True
    except (AttributeError, TypeError):
        return False


def _provenance_name_order(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise MappingValidationError(f"{context} must be an array")
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise MappingValidationError(f"{context} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise MappingValidationError(f"{context} must not contain duplicates")
    return names


def _mapping_records(
    value: object,
    record_type: type[JointMapping] | type[FrameMapping],
    context: str,
) -> tuple[JointMapping, ...] | tuple[FrameMapping, ...]:
    if not isinstance(value, (list, tuple)):
        raise MappingValidationError(f"{context} must be an array")
    try:
        return tuple(record_type.from_dict(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise MappingValidationError(f"invalid {context}: {exc}") from exc


def _validated_mapping_provenance(
    collected: CollectedRun,
    manifest: ParityManifest,
) -> tuple[
    tuple[JointMapping, ...],
    tuple[FrameMapping, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    case = collected.case
    hand = manifest.hand(case.hand)
    provenance = collected.result.provenance
    ov_target_readback_values: tuple[float, ...] = ()
    if not isinstance(provenance, Mapping):
        raise MappingValidationError("result.provenance must be an object")

    expected_manifest_sha = manifest_sha256(manifest)
    if provenance.get("manifest_sha256") != expected_manifest_sha:
        raise MappingValidationError(
            "provenance.manifest_sha256 does not match the canonical manifest"
        )
    session_id = provenance.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise MappingValidationError("provenance.session_id must be a non-empty string")
    source_revision = provenance.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise MappingValidationError(
            "provenance.source_revision must be a lowercase 40-character code commit"
        )
    if provenance.get("asset_commit") != manifest.provenance.commit:
        raise MappingValidationError(
            "provenance.asset_commit does not match the pinned upstream asset commit"
        )
    if provenance.get("asset_git_tree") != manifest.provenance.asset_git_tree:
        raise MappingValidationError(
            "provenance.asset_git_tree does not match the pinned asset Git tree"
        )
    if (
        provenance.get("asset_tree_sha256")
        != manifest.provenance.canonical_lf_asset_tree_sha256
    ):
        raise MappingValidationError(
            "provenance.asset_tree_sha256 does not match the canonical LF asset tree"
        )
    if manifest.schema_version == 2:
        scenario = manifest.scenario(case.scenario_id)
        expected_steps = round(scenario.duration_s / case.dt_s)
        expected_schedule_digest = canonical_target_sequence_sha256(
            scenario,
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=True,
        )
        common_target_evidence = {
            "target_sequence_digest_schema_version": 1,
            "target_sequence_digest_encoding": "utf8_json_lines_float_hex_v1",
            "target_sequence_digest_projection": "ieee754_binary32_roundtrip",
            "target_sequence_canonical_joint_names": list(hand.joint_names),
            "scheduled_target_sequence_semantics": (
                "q[0..N]; target q[k] is recorded at t_k and applies to "
                "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
            ),
            "scheduled_target_sequence_count": expected_steps + 1,
            "scheduled_target_sequence_sha256": expected_schedule_digest,
        }
        for field, expected in common_target_evidence.items():
            if provenance.get(field) != expected:
                raise MappingValidationError(
                    f"provenance target-sequence field {field!r} is invalid"
                )
    if case.simulator is Simulator.OVPHYSX:
        scenario = manifest.scenario(case.scenario_id)
        simulation_configuration = provenance.get("simulation_configuration")
        if not isinstance(simulation_configuration, Mapping):
            raise MappingValidationError(
                "OVPhysX provenance.simulation_configuration must be an object"
            )
        expected_configuration_keys = {
            "verified",
            "requested_dt_s",
            "cfg_dt_s",
            "backend_dt_s",
            "requested_gravity_m_s2",
            "cfg_gravity_m_s2",
            "physics_scene_gravity_m_s2",
            "physics_prim_path",
        }
        if set(simulation_configuration) != expected_configuration_keys:
            raise MappingValidationError(
                "OVPhysX simulation_configuration has invalid fields"
            )
        if simulation_configuration.get("verified") is not True:
            raise MappingValidationError("OVPhysX simulation configuration is not verified")
        for name in ("requested_dt_s", "cfg_dt_s", "backend_dt_s"):
            value = simulation_configuration.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(float(value), case.dt_s, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise MappingValidationError(f"OVPhysX {name} does not match the case")
        for name, tolerance in (
            ("requested_gravity_m_s2", 1e-12),
            ("cfg_gravity_m_s2", 1e-6),
            ("physics_scene_gravity_m_s2", 1e-5),
        ):
            values = simulation_configuration.get(name)
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
                or any(
                    not math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=tolerance
                    )
                    for actual, expected in zip(values, scenario.gravity_m_s2)
                )
            ):
                raise MappingValidationError(f"OVPhysX {name} does not match the scenario")
        if simulation_configuration.get("physics_prim_path") != "/physicsScene":
            raise MappingValidationError("OVPhysX used an unexpected PhysicsScene path")
        actuation_contract_version = provenance.get("actuation_contract_version")
        if (
            isinstance(actuation_contract_version, bool)
            or not isinstance(actuation_contract_version, int)
            or actuation_contract_version != 2
        ):
            raise MappingValidationError(
                "OVPhysX actuation_contract_version must be 2"
            )
        if provenance.get("actuator_model") != "IdealPDActuator":
            raise MappingValidationError(
                "OVPhysX actuator_model must be IdealPDActuator"
            )
        if provenance.get("control_path") != "explicit_pd_effort":
            raise MappingValidationError(
                "OVPhysX control_path must be explicit_pd_effort"
            )
        for name, strictly_positive in (
            ("controller_dof_stiffness", True),
            ("controller_dof_damping", False),
            ("controller_dof_effort_limit", True),
            ("controller_dof_effort_limit_sim", True),
        ):
            values = provenance.get(name)
            if (
                not isinstance(values, list)
                or len(values) != len(hand.joint_names)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise MappingValidationError(f"OVPhysX provenance.{name} is invalid")
            if any(
                float(value) <= 0.0 if strictly_positive else float(value) < 0.0
                for value in values
            ):
                qualifier = "positive" if strictly_positive else "non-negative"
                raise MappingValidationError(
                    f"OVPhysX provenance.{name} must be {qualifier}"
                )
        if provenance.get("controller_parameter_source") != "ideal_pd_actuator_tensor":
            raise MappingValidationError("OVPhysX controller parameter source is invalid")
        for name in ("backend_dof_stiffness", "backend_dof_damping"):
            values = provenance.get(name)
            if (
                not isinstance(values, list)
                or len(values) != len(hand.joint_names)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or abs(float(value)) > 1e-8
                    for value in values
                )
            ):
                raise MappingValidationError(
                    f"OVPhysX provenance.{name} must be zeroed PhysX drive readbacks"
                )
        if provenance.get("backend_dof_drive_readback_source") != "root_view_cpu_numpy_binding":
            raise MappingValidationError("OVPhysX drive readback source is invalid")
        if provenance.get("position_target_readback_verified") is not True:
            raise MappingValidationError("OVPhysX position-target binding is not verified")
        if (
            provenance.get("position_target_readback_source")
            != "articulation_data_joint_pos_target_torch"
        ):
            raise MappingValidationError("OVPhysX target readback source is invalid")
        if provenance.get("zero_velocity_target_verified") is not True:
            raise MappingValidationError("OVPhysX zero velocity target is not verified")
        if provenance.get("zero_feedforward_effort_target_verified") is not True:
            raise MappingValidationError(
                "OVPhysX zero feedforward effort target is not verified"
            )
        expected_steps = round(scenario.duration_s / case.dt_s)
        readback_count = provenance.get("position_target_readback_count")
        if (
            isinstance(readback_count, bool)
            or not isinstance(readback_count, int)
            or readback_count < expected_steps + 2
        ):
            raise MappingValidationError("OVPhysX target readback coverage is incomplete")
        readback_error = provenance.get("position_target_readback_max_abs_error_rad")
        if (
            isinstance(readback_error, bool)
            or not isinstance(readback_error, (int, float))
            or not math.isfinite(float(readback_error))
            or float(readback_error) > 1e-6
        ):
            raise MappingValidationError("OVPhysX target readback error is invalid")
        if manifest.schema_version == 2:
            canonical_readback_error = provenance.get(
                "position_target_canonical_readback_max_abs_error_rad"
            )
            if (
                isinstance(canonical_readback_error, bool)
                or not isinstance(canonical_readback_error, (int, float))
                or not math.isfinite(float(canonical_readback_error))
                or float(canonical_readback_error) < 0.0
                or float(canonical_readback_error) > 1e-6
            ):
                raise MappingValidationError(
                    "OVPhysX canonical target readback error is invalid"
                )
        raw_readback_values = provenance.get("position_target_readback_values_rad")
        if (
            not isinstance(raw_readback_values, list)
            or len(raw_readback_values) != len(hand.joint_names)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in raw_readback_values
            )
        ):
            raise MappingValidationError("OVPhysX final target readback is invalid")
        ov_target_readback_values = tuple(float(value) for value in raw_readback_values)
        expected_nonzero_target = scenario.kind.value in {
            "small_step",
            "offset_sine",
            "offset_linear_chirp",
        }
        if provenance.get("position_target_nonzero_readback_observed") is not (
            expected_nonzero_target
        ):
            raise MappingValidationError(
                "OVPhysX nonzero target readback does not match the scenario"
            )
        if manifest.schema_version == 2:
            sequence_metadata = {
                "target_sequence_digest_schema_version": 1,
                "target_sequence_digest_encoding": "utf8_json_lines_float_hex_v1",
                "target_sequence_digest_projection": (
                    "ieee754_binary32_roundtrip"
                ),
                "target_sequence_canonical_joint_names": list(hand.joint_names),
                "scheduled_target_sequence_semantics": (
                    "q[0..N]; target q[k] is recorded at t_k and applies to "
                    "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
                ),
                "requested_target_sequence_semantics": (
                    "q[0..N] passed to set_joint_position_target_index"
                ),
                "immediate_target_readback_sequence_semantics": (
                    "q[0..N] read immediately from joint_pos_target after each request"
                ),
                "pre_step_applied_target_readback_sequence_semantics": (
                    "q[0..N-1] read after write_data_to_sim and before each physics step"
                ),
            }
            for field, expected in sequence_metadata.items():
                if provenance.get(field) != expected:
                    raise MappingValidationError(
                        f"OVPhysX target-sequence metadata {field!r} is invalid"
                    )
            sequence_counts = {
                "scheduled_target_sequence_count": expected_steps + 1,
                "requested_target_sequence_count": expected_steps + 1,
                "immediate_target_readback_sequence_count": expected_steps + 1,
                "pre_step_applied_target_readback_sequence_count": expected_steps,
            }
            for field, expected in sequence_counts.items():
                observed = provenance.get(field)
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or observed != expected
                ):
                    raise MappingValidationError(
                        f"OVPhysX target-sequence count {field!r} is incomplete"
                    )
            expected_full_digest = canonical_target_sequence_sha256(
                scenario,
                hand.joint_names,
                dt_s=case.dt_s,
                include_terminal=True,
            )
            expected_prefix_digest = canonical_target_sequence_sha256(
                scenario,
                hand.joint_names,
                dt_s=case.dt_s,
                include_terminal=False,
            )
            sequence_digests = {
                "scheduled_target_sequence_sha256": expected_full_digest,
                "requested_target_sequence_sha256": expected_full_digest,
                "immediate_target_readback_sequence_sha256": expected_full_digest,
                "pre_step_applied_target_readback_sequence_sha256": (
                    expected_prefix_digest
                ),
            }
            for field, expected in sequence_digests.items():
                if provenance.get(field) != expected:
                    raise MappingValidationError(
                        f"OVPhysX target-sequence digest {field!r} does not "
                        "match the canonical schedule"
                    )
        for name in ("computed_effort_peak_abs_nm", "applied_effort_peak_abs_nm"):
            values = provenance.get(name)
            if (
                not isinstance(values, list)
                or len(values) != len(hand.joint_names)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in values
                )
            ):
                raise MappingValidationError(f"OVPhysX provenance.{name} is invalid")
        effort_observation_count = provenance.get("effort_observation_count")
        if (
            isinstance(effort_observation_count, bool)
            or not isinstance(effort_observation_count, int)
            or effort_observation_count < expected_steps + 1
        ):
            raise MappingValidationError("OVPhysX effort observation coverage is incomplete")
        for name, tolerance in (
            ("effort_formula_max_abs_error_nm", 1e-5),
            ("effort_clip_max_abs_error_nm", 1e-6),
        ):
            value = provenance.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or float(value) > tolerance
            ):
                raise MappingValidationError(f"OVPhysX provenance.{name} is invalid")
        effort_clip_count = provenance.get("effort_clip_count")
        if (
            isinstance(effort_clip_count, bool)
            or not isinstance(effort_clip_count, int)
            or effort_clip_count < 0
        ):
            raise MappingValidationError("OVPhysX effort_clip_count is invalid")
        if (
            provenance.get("effort_command_source")
            != "articulation_data_computed_and_applied_torque_torch"
        ):
            raise MappingValidationError("OVPhysX effort command source is invalid")
    version = provenance.get("mapping_schema_version")
    if isinstance(version, bool) or version != MAPPING_SCHEMA_VERSION:
        raise MappingValidationError(
            f"provenance.mapping_schema_version must be {MAPPING_SCHEMA_VERSION}"
        )

    backend_joint_names = _provenance_name_order(
        provenance.get("backend_joint_names"), "provenance.backend_joint_names"
    )
    backend_frame_names = _provenance_name_order(
        provenance.get("backend_frame_names"), "provenance.backend_frame_names"
    )
    joint_mapping = _mapping_records(
        provenance.get("joint_mapping"), JointMapping, "provenance.joint_mapping"
    )
    frame_mapping = _mapping_records(
        provenance.get("frame_mapping"), FrameMapping, "provenance.frame_mapping"
    )
    assert all(isinstance(item, JointMapping) for item in joint_mapping)
    assert all(isinstance(item, FrameMapping) for item in frame_mapping)
    typed_joints = tuple(item for item in joint_mapping if isinstance(item, JointMapping))
    typed_frames = tuple(item for item in frame_mapping if isinstance(item, FrameMapping))

    for entry in (*typed_joints, *typed_frames):
        if entry.backend != case.simulator.value:
            raise MappingValidationError(
                f"mapping backend {entry.backend!r} does not match {case.simulator.value!r}"
            )
        if entry.scope != case.hand.value:
            raise MappingValidationError(
                f"mapping scope {entry.scope!r} does not match hand {case.hand.value!r}"
            )
    if any(entry.unit != "rad" for entry in typed_joints):
        raise MappingValidationError("Gate 0 joint mappings must use radians")
    if any(entry.unit != "xyz_m_qwxyz" for entry in typed_frames):
        raise MappingValidationError(
            "Gate 0 frame mappings must use the xyz_m_qwxyz pose convention"
        )

    validate_mapping(
        typed_joints,
        typed_frames,
        expected_joint_count=len(hand.joint_names),
        expected_frame_count=len(hand.distal_frame_names),
        expected_joint_ids=hand.joint_names,
        expected_frame_ids=hand.distal_frame_names,
        expected_backends=(case.simulator.value,),
    )
    for entry in typed_joints:
        if entry.index >= len(backend_joint_names):
            raise MappingValidationError(
                f"joint mapping index {entry.index} is outside backend_joint_names"
            )
        if backend_joint_names[entry.index] != entry.backend_name:
            raise MappingValidationError(
                f"joint mapping index/name mismatch for {entry.canonical_id!r}"
            )
    for entry in typed_frames:
        if entry.index >= len(backend_frame_names):
            raise MappingValidationError(
                f"frame mapping index {entry.index} is outside backend_frame_names"
            )
        if backend_frame_names[entry.index] != entry.backend_name:
            raise MappingValidationError(
                f"frame mapping index/name mismatch for {entry.canonical_id!r}"
            )
    if len(backend_joint_names) != len(hand.joint_names):
        raise MappingValidationError("backend_joint_names must contain 22 entries")
    if len(backend_frame_names) < len(hand.distal_frame_names):
        raise MappingValidationError(
            "backend_frame_names must contain the complete backend body index domain"
        )
    if case.simulator is Simulator.OVPHYSX:
        final_targets = canonical_position_targets(
            manifest.scenario(case.scenario_id),
            hand.joint_names,
            step_index=round(
                manifest.scenario(case.scenario_id).duration_s / case.dt_s
            ),
            dt_s=case.dt_s,
        )
        for entry in typed_joints:
            expected_backend_target = (
                final_targets[entry.canonical_id] - entry.offset
            ) / entry.sign
            if not math.isclose(
                ov_target_readback_values[entry.index],
                expected_backend_target,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise MappingValidationError(
                    "OVPhysX final target readback does not match the canonical scenario"
                )
    return typed_joints, typed_frames, backend_joint_names, backend_frame_names


def _assess_run(
    collected: CollectedRun,
    manifest: ParityManifest,
) -> _RunAssessment:
    case = collected.case
    run = collected.result
    hand = manifest.hand(case.hand)
    scenario = manifest.scenario(case.scenario_id)
    expected_steps = round(scenario.duration_s / case.dt_s)
    errors: list[str] = []
    joint_mapping: tuple[JointMapping, ...] = ()
    frame_mapping: tuple[FrameMapping, ...] = ()
    backend_joint_names: tuple[str, ...] = ()
    backend_frame_names: tuple[str, ...] = ()

    try:
        (
            joint_mapping,
            frame_mapping,
            backend_joint_names,
            backend_frame_names,
        ) = _validated_mapping_provenance(collected, manifest)
    except (MappingValidationError, TypeError, ValueError) as exc:
        errors.append(f"mapping/provenance validation failed: {exc}")

    if run.backend != case.simulator.value:
        errors.append(f"backend {run.backend!r} does not match case {case.simulator.value!r}")
    if run.scenario_id != case.scenario_id:
        errors.append(
            f"scenario {run.scenario_id!r} does not match case {case.scenario_id!r}"
        )
    valid_dt = (
        not isinstance(run.dt, bool)
        and isinstance(run.dt, (int, float))
        and math.isfinite(float(run.dt))
    )
    if not valid_dt or not math.isclose(
        float(run.dt) if valid_dt else math.nan,
        case.dt_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(f"dt {run.dt!r} does not match expected {case.dt_s!r}")
    if run.status != "completed":
        errors.append(f"adapter execution_status={run.status}: {run.message}")
    if (
        isinstance(run.requested_steps, bool)
        or not isinstance(run.requested_steps, int)
        or run.requested_steps != expected_steps
    ):
        errors.append(
            f"requested_steps={run.requested_steps}, expected {expected_steps}"
        )
    if (
        isinstance(run.completed_steps, bool)
        or not isinstance(run.completed_steps, int)
        or run.completed_steps != expected_steps
    ):
        errors.append(
            f"completed_steps={run.completed_steps}, expected {expected_steps}"
        )
    if run.joint_names != hand.joint_names:
        errors.append("joint mapping/order does not match the 22 canonical joints")
    if run.frame_names != hand.distal_frame_names:
        errors.append("frame mapping/order does not match the five canonical distal frames")
    if len(run.samples) != expected_steps + 1:
        errors.append(
            f"sample_count={len(run.samples)}, expected {expected_steps + 1} including t=0"
        )

    finite = True
    expected_joint_set = set(hand.joint_names)
    expected_frame_set = set(hand.distal_frame_names)
    previous_time = -math.inf
    for index, sample in enumerate(run.samples):
        if not isinstance(sample, TraceSample):
            finite = False
            errors.append(f"sample {index} is not a TraceSample")
            continue
        sample_finite = _sample_is_finite(sample)
        if not sample_finite:
            finite = False
            errors.append(f"sample {index} contains NaN or Inf")
        if sample.step != index:
            errors.append(f"sample {index} has non-canonical step={sample.step}")
        expected_time = index * case.dt_s
        valid_time = (
            not isinstance(sample.time_s, bool)
            and isinstance(sample.time_s, (int, float))
            and math.isfinite(float(sample.time_s))
        )
        if not valid_time or not math.isclose(
            float(sample.time_s) if valid_time else math.nan,
            expected_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(
                f"sample {index} time_s={sample.time_s!r}, expected {expected_time!r}"
            )
        if valid_time and float(sample.time_s) <= previous_time and index > 0:
            errors.append(f"sample {index} time is not strictly increasing")
        if valid_time:
            previous_time = float(sample.time_s)
        if not isinstance(sample.joint_positions, Mapping) or set(
            sample.joint_positions
        ) != expected_joint_set:
            errors.append(f"sample {index} does not contain all canonical joint positions")
        if not isinstance(sample.position_targets, Mapping) or set(
            sample.position_targets
        ) != expected_joint_set:
            errors.append(f"sample {index} does not contain all canonical position targets")
        else:
            expected_targets = canonical_position_targets(
                scenario,
                hand.joint_names,
                step_index=index,
                dt_s=case.dt_s,
            )
            mismatched_targets = [
                name
                for name in hand.joint_names
                if not math.isclose(
                    float(sample.position_targets[name]),
                    expected_targets[name],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            if mismatched_targets:
                errors.append(
                    f"sample {index} canonical target trajectory mismatch: "
                    + ", ".join(mismatched_targets[:5])
                )
        if not isinstance(sample.frame_poses, Mapping) or set(sample.frame_poses) != expected_frame_set:
            errors.append(f"sample {index} does not contain all canonical distal frame poses")
        if (
            not isinstance(sample.qpos, tuple)
            or not isinstance(sample.qvel, tuple)
            or len(sample.qpos) != len(hand.joint_names)
            or len(sample.qvel) != len(hand.joint_names)
        ):
            errors.append(f"sample {index} does not contain 22 qpos/qvel values")
        if isinstance(sample.qvel, tuple) and any(
            abs(float(value)) > MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S
            for value in sample.qvel
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            errors.append(
                f"sample {index} exceeds the joint-velocity integrity bound"
            )
        if isinstance(sample.joint_positions, Mapping):
            if any(
                abs(float(value)) > MAX_ABS_CANONICAL_JOINT_POSITION_RAD
                for value in sample.joint_positions.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ):
                errors.append(
                    f"sample {index} exceeds the canonical joint-position integrity bound"
                )
            if index == 0:
                wrong_initial = [
                    name
                    for name in hand.joint_names
                    if name in sample.joint_positions
                    and not math.isclose(
                        float(sample.joint_positions[name]),
                        scenario.initial_position_rad,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ]
                if wrong_initial:
                    errors.append(
                        "sample 0 does not match the canonical zero initial state"
                    )
        if isinstance(sample.position_targets, Mapping) and any(
            abs(float(value)) > MAX_ABS_CANONICAL_POSITION_TARGET_RAD
            for value in sample.position_targets.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            errors.append(
                f"sample {index} exceeds the canonical position-target integrity bound"
            )
        if (
            isinstance(sample.qpos, tuple)
            and len(sample.qpos) == len(backend_joint_names)
            and isinstance(sample.joint_positions, Mapping)
        ):
            for entry in joint_mapping:
                canonical_value = (
                    entry.sign * float(sample.qpos[entry.index]) + entry.offset
                )
                observed = sample.joint_positions.get(entry.canonical_id)
                if observed is None or not math.isclose(
                    float(observed),
                    canonical_value,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    errors.append(
                        f"sample {index} raw qpos does not map to canonical joint "
                        f"{entry.canonical_id!r}"
                    )
                    break
        frame_items = (
            sample.frame_poses.items()
            if isinstance(sample.frame_poses, Mapping)
            else ()
        )
        for frame_name, pose in frame_items:
            if (
                not isinstance(pose, Sequence)
                or isinstance(pose, (str, bytes))
                or len(pose) != 7
            ):
                errors.append(f"sample {index} frame {frame_name!r} pose must have 7 values")
            elif _all_finite(pose) and math.sqrt(
                sum(float(value) * float(value) for value in pose[3:])
            ) <= 0.0:
                errors.append(f"sample {index} frame {frame_name!r} quaternion is zero")
            elif _all_finite(pose):
                frame_distance = math.sqrt(
                    sum(float(value) * float(value) for value in pose[:3])
                )
                quaternion_norm = math.sqrt(
                    sum(float(value) * float(value) for value in pose[3:])
                )
                if frame_distance > MAX_FRAME_ORIGIN_DISTANCE_M:
                    errors.append(
                        f"sample {index} frame {frame_name!r} exceeds the frame-position integrity bound"
                    )
                if (
                    abs(quaternion_norm - 1.0)
                    > UNIT_QUATERNION_NORM_TOLERANCE
                ):
                    errors.append(
                        f"sample {index} frame {frame_name!r} quaternion is not unit length"
                    )
        contact_count = sample.contact_count
        if case.simulator is Simulator.MUJOCO:
            if isinstance(contact_count, bool) or not isinstance(contact_count, int):
                errors.append(
                    f"sample {index} MuJoCo contact_count must be an explicit integer"
                )
            elif contact_count < 0:
                errors.append(
                    f"sample {index} MuJoCo contact_count must be non-negative"
                )
            elif scenario.expect_no_contacts and contact_count != 0:
                errors.append(
                    f"sample {index} observed {contact_count} unexpected contact(s)"
                )
        elif manifest.schema_version >= 2:
            if contact_count is not None:
                errors.append(
                    f"sample {index} OVPhysX contact_count must be null because "
                    "contact observation is unavailable in the v2 protocol"
                )
        elif contact_count is not None:
            if isinstance(contact_count, bool) or not isinstance(contact_count, int):
                errors.append(
                    f"sample {index} OVPhysX contact_count must be null or an integer"
                )
            elif contact_count < 0:
                errors.append(
                    f"sample {index} OVPhysX contact_count must be non-negative"
                )
            elif scenario.expect_no_contacts and contact_count != 0:
                errors.append(
                    f"sample {index} observed {contact_count} unexpected contact(s)"
                )

    if scenario.kind.value == "small_step" and run.samples:
        first = run.samples[0]
        response = 0.0
        if isinstance(first, TraceSample) and isinstance(first.joint_positions, Mapping):
            for sample in run.samples[1:]:
                if not isinstance(sample, TraceSample) or not isinstance(
                    sample.joint_positions, Mapping
                ):
                    continue
                for name in hand.joint_names:
                    if name in first.joint_positions and name in sample.joint_positions:
                        response = max(
                            response,
                            abs(
                                float(sample.joint_positions[name])
                                - float(first.joint_positions[name])
                            ),
                        )
        if response <= SMALL_STEP_MIN_OBSERVED_RESPONSE_RAD:
            errors.append(
                "small_step produced no observable canonical joint response: "
                f"max={response:.6g} <= {SMALL_STEP_MIN_OBSERVED_RESPONSE_RAD:.6g} rad"
            )

    # Keep diagnostics compact when one structural issue repeats at every sample.
    errors = list(dict.fromkeys(errors))
    if len(errors) > 8:
        omitted = len(errors) - 8
        errors = [*errors[:8], f"{omitted} additional validation error(s) omitted"]
    return _RunAssessment(
        collected=collected,
        expected_steps=expected_steps,
        finite=finite,
        errors=tuple(errors),
        joint_mapping=joint_mapping,
        frame_mapping=frame_mapping,
        backend_joint_names=backend_joint_names,
        backend_frame_names=backend_frame_names,
    )


def _mapping_fingerprint(assessment: _RunAssessment) -> tuple[object, ...]:
    def record(entry: JointMapping | FrameMapping) -> tuple[object, ...]:
        return (
            entry.canonical_id,
            entry.backend,
            entry.scope,
            entry.backend_name,
            entry.index,
            float(entry.sign),
            float(entry.offset),
            entry.unit,
        )

    return (
        assessment.backend_joint_names,
        assessment.backend_frame_names,
        tuple(sorted(record(entry) for entry in assessment.joint_mapping)),
        tuple(sorted(record(entry) for entry in assessment.frame_mapping)),
    )


def _enforce_run_consistency(
    assessments: Mapping[str, _RunAssessment],
    manifest: ParityManifest,
) -> dict[str, _RunAssessment]:
    """Reject mixed sessions and mapping drift across repeated cases.

    Manifest v1 campaigns launch both simulators in one local session, so the
    historical contract requires one session identifier globally.  Waveform
    T1 (manifest v2) deliberately launches MuJoCo locally and OVPhysX on a
    remote host; each backend must still be internally consistent, but the two
    backend session identifiers are expected to differ.
    """

    result = dict(assessments)
    session_scopes = (
        ((None, tuple(result)),)
        if manifest.schema_version == 1
        else tuple(
            (
                simulator,
                tuple(
                    case_id
                    for case_id, assessment in result.items()
                    if assessment.collected.case.simulator is simulator
                ),
            )
            for simulator in manifest.simulators
        )
    )
    for simulator, case_ids in session_scopes:
        session_ids = {
            result[case_id].collected.result.provenance.get("session_id")
            for case_id in case_ids
            if isinstance(result[case_id].collected.result.provenance, Mapping)
            and isinstance(
                result[case_id].collected.result.provenance.get("session_id"), str
            )
            and result[case_id].collected.result.provenance.get("session_id")
        }
        if len(session_ids) <= 1:
            continue
        message = "received runs mix multiple session_id values"
        if simulator is not None:
            message += f" within the {simulator.value} backend"
        for case_id in case_ids:
            assessment = result[case_id]
            result[case_id] = replace(
                assessment,
                errors=tuple(dict.fromkeys((*assessment.errors, message))),
            )

    source_revisions = {
        assessment.collected.result.provenance.get("source_revision")
        for assessment in result.values()
        if isinstance(assessment.collected.result.provenance, Mapping)
        and isinstance(
            assessment.collected.result.provenance.get("source_revision"), str
        )
        and re.fullmatch(
            r"[0-9a-f]{40}",
            assessment.collected.result.provenance.get("source_revision", ""),
        )
    }
    if len(source_revisions) > 1:
        for case_id, assessment in tuple(result.items()):
            result[case_id] = replace(
                assessment,
                errors=tuple(
                    dict.fromkeys(
                        (
                            *assessment.errors,
                            "received runs mix multiple source_revision values",
                        )
                    )
                ),
            )

    fingerprints: dict[tuple[Simulator, HandSide], tuple[object, ...]] = {}
    for case_id, assessment in tuple(result.items()):
        if not assessment.joint_mapping or not assessment.frame_mapping:
            continue
        key = (assessment.collected.case.simulator, assessment.collected.case.hand)
        fingerprint = _mapping_fingerprint(assessment)
        reference = fingerprints.setdefault(key, fingerprint)
        if fingerprint != reference:
            result[case_id] = replace(
                assessment,
                errors=tuple(
                    dict.fromkeys(
                        (
                            *assessment.errors,
                            "mapping/order differs across runs for the same backend and hand",
                        )
                    )
                ),
            )
    return result


def _observed_mapping_counts(
    manifest: ParityManifest,
    assessments: Mapping[str, _RunAssessment],
) -> tuple[int, int]:
    joint_ids: dict[Simulator, set[tuple[str, str]]] = {
        simulator: set() for simulator in manifest.simulators
    }
    frame_ids: dict[Simulator, set[tuple[str, str]]] = {
        simulator: set() for simulator in manifest.simulators
    }
    for assessment in assessments.values():
        if not assessment.completed:
            continue
        simulator = assessment.collected.case.simulator
        joint_ids[simulator].update(
            (entry.scope, entry.canonical_id) for entry in assessment.joint_mapping
        )
        frame_ids[simulator].update(
            (entry.scope, entry.canonical_id) for entry in assessment.frame_mapping
        )
    paired_joints = set.intersection(
        *(joint_ids[simulator] for simulator in manifest.simulators)
    )
    paired_frames = set.intersection(
        *(frame_ids[simulator] for simulator in manifest.simulators)
    )
    return len(paired_joints), len(paired_frames)


def _execution_record(
    simulator: Simulator,
    expected_cases: Sequence[ScenarioCase],
    assessments: Mapping[str, _RunAssessment],
    repeat_count: int,
) -> ExecutionRecord:
    base_cases = [
        case
        for case in expected_cases
        if case.simulator is simulator and case.timestep_variant is TimestepVariant.BASE
    ]
    all_cases = [case for case in expected_cases if case.simulator is simulator]
    valid_base_repeats = sum(
        1
        for case in base_cases
        if case.case_id in assessments and assessments[case.case_id].completed
    )
    failures: list[str] = []
    finite = True
    hashes: set[str] = set()
    for case in all_cases:
        assessment = assessments.get(case.case_id)
        if assessment is None:
            failures.append(f"missing {case.case_id}")
            continue
        finite = finite and assessment.finite
        if assessment.collected.bundle_root_sha256 is not None:
            hashes.add(assessment.collected.bundle_root_sha256)
        if assessment.errors:
            failures.append(f"{case.case_id}: {'; '.join(assessment.errors)}")

    # The result contract counts canonical base repeats, not extra dt-halving runs.
    completed_repeats = min(valid_base_repeats, repeat_count)
    if failures:
        message = " | ".join(failures)
        return ExecutionRecord(
            simulator=simulator,
            execution_status=ExecutionStatus.ERROR,
            completed_repeats=completed_repeats,
            requested_repeats=repeat_count,
            finite=finite,
            bundle_root_sha256=next(iter(hashes)) if len(hashes) == 1 else None,
            message=message,
        )
    return ExecutionRecord(
        simulator=simulator,
        execution_status=ExecutionStatus.COMPLETED,
        completed_repeats=repeat_count,
        requested_repeats=repeat_count,
        finite=True,
        bundle_root_sha256=next(iter(hashes)) if len(hashes) == 1 else None,
    )


def _quaternion_angle(pose_a: Sequence[float], pose_b: Sequence[float]) -> float:
    qa = tuple(float(value) for value in pose_a[3:7])
    qb = tuple(float(value) for value in pose_b[3:7])
    norm_a = math.sqrt(sum(value * value for value in qa))
    norm_b = math.sqrt(sum(value * value for value in qb))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise _ComparisonUnavailable("cannot compare a zero quaternion")
    unit_a = tuple(value / norm_a for value in qa)
    unit_b = tuple(value / norm_b for value in qb)
    if sum(a * b for a, b in zip(unit_a, unit_b)) < 0.0:
        unit_b = tuple(-value for value in unit_b)
    difference_norm = math.sqrt(
        sum((a - b) * (a - b) for a, b in zip(unit_a, unit_b))
    )
    sum_norm = math.sqrt(
        sum((a + b) * (a + b) for a, b in zip(unit_a, unit_b))
    )
    # This atan2 form is equivalent to 2*acos(|dot|), but remains stable at
    # zero angle and returns exactly zero for identical repeated samples.
    return 4.0 * math.atan2(difference_norm, sum_norm)


def trace_delta(
    reference: AdapterRunResult,
    candidate: AdapterRunResult,
    hand: HandSpec,
) -> tuple[float, float, float]:
    """Return canonical joint, distal-frame position, and orientation maxima.

    Samples are paired by integer integration-step ratios.  Timestamps remain
    an integrity check, but they are deliberately not the join key: a base-dt
    sample at step ``k`` pairs with a halved-dt sample at step ``2*k``.  This
    avoids making binary floating-point timestamp formatting part of the
    scientific alignment contract.
    """
    if not math.isfinite(reference.dt) or reference.dt <= 0.0:
        raise _ComparisonUnavailable("reference trace has an invalid timestep")
    if not math.isfinite(candidate.dt) or candidate.dt <= 0.0:
        raise _ComparisonUnavailable("candidate trace has an invalid timestep")
    step_ratio = reference.dt / candidate.dt
    if not math.isfinite(step_ratio) or step_ratio <= 0.0:
        raise _ComparisonUnavailable("trace timestep ratio is invalid")

    candidate_by_step: dict[int, TraceSample] = {}
    for sample in candidate.samples:
        if sample.step in candidate_by_step:
            raise _ComparisonUnavailable("candidate trace has duplicate sample steps")
        candidate_by_step[sample.step] = sample

    joint_max = 0.0
    position_max = 0.0
    orientation_max = 0.0
    for reference_sample in reference.samples:
        candidate_step_float = reference_sample.step * step_ratio
        candidate_step = round(candidate_step_float)
        if not math.isclose(
            candidate_step_float,
            candidate_step,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise _ComparisonUnavailable(
                "trace timestep ratio does not map common times to integer steps"
            )
        candidate_sample = candidate_by_step.get(candidate_step)
        if candidate_sample is None:
            raise _ComparisonUnavailable(
                f"trace has no sample at canonical step {candidate_step}"
            )
        expected_time = reference_sample.step * reference.dt
        candidate_time = candidate_sample.step * candidate.dt
        if not math.isclose(expected_time, candidate_time, rel_tol=0.0, abs_tol=1e-12):
            raise _ComparisonUnavailable(
                "integer-aligned samples do not represent the same physical time"
            )
        for name in hand.joint_names:
            joint_max = max(
                joint_max,
                abs(
                    float(reference_sample.joint_positions[name])
                    - float(candidate_sample.joint_positions[name])
                ),
            )
        for name in hand.distal_frame_names:
            pose_a = reference_sample.frame_poses[name]
            pose_b = candidate_sample.frame_poses[name]
            position_max = max(
                position_max,
                math.sqrt(
                    sum((float(a) - float(b)) ** 2 for a, b in zip(pose_a[:3], pose_b[:3]))
                ),
            )
            orientation_max = max(
                orientation_max,
                _quaternion_angle(pose_a, pose_b),
            )
    return joint_max, position_max, orientation_max


def _max_delta(
    current: tuple[float, float, float],
    update: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(max(a, b) for a, b in zip(current, update))  # type: ignore[return-value]


def _comparison_record(
    manifest: ParityManifest,
    hand: HandSpec,
    scenario_id: str,
    expected_cases: Sequence[ScenarioCase],
    assessments: Mapping[str, _RunAssessment],
    executions: tuple[ExecutionRecord, ExecutionRecord],
    thresholds: ComparisonThresholds,
) -> ComparisonRecord:
    if any(item.execution_status is ExecutionStatus.ERROR for item in executions):
        detail = "; ".join(
            f"{item.simulator.value}: {item.message}"
            for item in executions
            if item.execution_status is ExecutionStatus.ERROR
        )
        return ComparisonRecord(
            comparison_status=ComparisonStatus.INCONCLUSIVE,
            message="Comparison not attempted because execution was incomplete or invalid. "
            + detail,
        )

    def run(
        simulator: Simulator,
        variant: TimestepVariant,
        repeat_index: int,
    ) -> AdapterRunResult:
        matching = [
            case
            for case in expected_cases
            if case.simulator is simulator
            and case.timestep_variant is variant
            and case.repeat_index == repeat_index
        ]
        if len(matching) != 1:
            raise _ComparisonUnavailable("expected run matrix is ambiguous")
        return assessments[matching[0].case_id].collected.result

    crosssim = (0.0, 0.0, 0.0)
    repeat = (0.0, 0.0, 0.0)
    dt_halving = (0.0, 0.0, 0.0)
    try:
        variants = [TimestepVariant.BASE]
        if scenario_id in manifest.run_policy.dt_halving_scenario_ids:
            variants.append(TimestepVariant.HALVED)
        # Gate 0 historically assigns the cross-simulator label on the base
        # grid.  Trajectory T1 preregisters both grids as active scientific
        # cases, so its cross-simulator maximum must cover every variant and
        # repeat rather than silently ignoring the halved-dt observations.
        crosssim_variants = (
            variants
            if manifest.schema_version == 2
            else [TimestepVariant.BASE]
        )
        for variant in crosssim_variants:
            for repeat_index in range(1, manifest.run_policy.repeat_count + 1):
                crosssim = _max_delta(
                    crosssim,
                    trace_delta(
                        run(Simulator.MUJOCO, variant, repeat_index),
                        run(Simulator.OVPHYSX, variant, repeat_index),
                        hand,
                    ),
                )
        for simulator in manifest.simulators:
            for variant in variants:
                reference = run(simulator, variant, 1)
                for repeat_index in range(2, manifest.run_policy.repeat_count + 1):
                    repeat = _max_delta(
                        repeat,
                        trace_delta(
                            reference,
                            run(simulator, variant, repeat_index),
                            hand,
                        ),
                    )
            if TimestepVariant.HALVED in variants:
                for repeat_index in range(1, manifest.run_policy.repeat_count + 1):
                    dt_halving = _max_delta(
                        dt_halving,
                        trace_delta(
                            run(simulator, TimestepVariant.BASE, repeat_index),
                            run(simulator, TimestepVariant.HALVED, repeat_index),
                            hand,
                        ),
                    )
    except (KeyError, _ComparisonUnavailable) as exc:
        return ComparisonRecord(
            comparison_status=ComparisonStatus.INCONCLUSIVE,
            message=f"Comparison traces could not be aligned: {exc}",
        )

    metrics = {
        "crosssim_joint_max_abs_rad": crosssim[0],
        "crosssim_frame_position_max_m": crosssim[1],
        "crosssim_frame_orientation_max_rad": crosssim[2],
        "repeat_joint_max_abs_rad": repeat[0],
        "repeat_frame_position_max_m": repeat[1],
        "repeat_frame_orientation_max_rad": repeat[2],
        "dt_halving_joint_max_abs_rad": dt_halving[0],
        "dt_halving_frame_position_max_m": dt_halving[1],
        "dt_halving_frame_orientation_max_rad": dt_halving[2],
    }
    limits = thresholds.to_dict()
    exceeded = [
        name
        for name, value in metrics.items()
        if name != "minimum_completion_fraction" and value > limits[name]
    ]
    if manifest.schema_version == 2:
        prerequisite_exceeded = [
            name
            for name in exceeded
            if name.startswith("repeat_") or name.startswith("dt_halving_")
        ]
        if prerequisite_exceeded:
            detail = ", ".join(
                f"{name}={metrics[name]:.6g} > {limits[name]:.6g}"
                for name in prerequisite_exceeded
            )
            return ComparisonRecord(
                comparison_status=ComparisonStatus.INCONCLUSIVE,
                metrics=metrics,
                message=(
                    "Trajectory T1 repeatability or dt-halving prerequisite failed: "
                    f"{detail}. Cross-simulator status was not assigned."
                ),
            )
        exceeded = [name for name in exceeded if name.startswith("crosssim_")]
    if exceeded:
        detail = ", ".join(
            f"{name}={metrics[name]:.6g} > {limits[name]:.6g}" for name in exceeded
        )
        return ComparisonRecord(
            comparison_status=ComparisonStatus.DIVERGENT,
            metrics=metrics,
            message=(
                "Observed threshold exceedance under the pinned simulation-only setup: "
                f"{detail}. This is a cross-simulator observation, not an official bug finding."
            ),
        )
    return ComparisonRecord(
        comparison_status=ComparisonStatus.WITHIN_TOLERANCE,
        metrics=metrics,
        message=(
            "All measured trajectory deltas are within the reported diagnostic thresholds."
            if manifest.schema_version == 2
            else "All measured Gate 0 deltas are within the reported diagnostic thresholds."
        ),
    )


def compare_gate0_runs(
    manifest: ParityManifest,
    collected_runs: Iterable[CollectedRun],
    *,
    thresholds: ComparisonThresholds | None = None,
) -> Gate0Comparison:
    """Validate expected executions and compare complete finite trace groups."""

    if not isinstance(manifest, ParityManifest):
        raise TypeError("manifest must be a ParityManifest")
    limits = thresholds or ComparisonThresholds()
    if (
        limits.minimum_completion_fraction
        != manifest.run_policy.minimum_completion_fraction
    ):
        raise ValueError(
            "comparison completion threshold must match the canonical manifest"
        )
    expected_cases = expand_scenario_cases(manifest)
    expected_by_id = {case.case_id: case for case in expected_cases}
    collected_by_id: dict[str, CollectedRun] = {}
    for collected in collected_runs:
        if not isinstance(collected, CollectedRun):
            raise ValueError("collected_runs must contain CollectedRun values")
        case_id = collected.case.case_id
        if case_id not in expected_by_id:
            raise ValueError(f"unexpected Gate 0 case: {case_id}")
        if collected.case != expected_by_id[case_id]:
            raise ValueError(f"case fields do not match the canonical manifest: {case_id}")
        if case_id in collected_by_id:
            raise ValueError(f"duplicate Gate 0 case: {case_id}")
        collected_by_id[case_id] = collected

    assessments = _enforce_run_consistency(
        {
            case_id: _assess_run(collected, manifest)
            for case_id, collected in collected_by_id.items()
        },
        manifest,
    )
    result_records: list[ParityResult] = []
    observations: list[str] = []
    manifest_digest = manifest_sha256(manifest)
    for hand in manifest.hands:
        for scenario in manifest.scenarios:
            group_cases = [
                case
                for case in expected_cases
                if case.hand is hand.side and case.scenario_id == scenario.scenario_id
            ]
            executions = tuple(
                _execution_record(
                    simulator,
                    group_cases,
                    assessments,
                    manifest.run_policy.repeat_count,
                )
                for simulator in manifest.simulators
            )
            comparison = _comparison_record(
                manifest,
                hand,
                scenario.scenario_id,
                group_cases,
                assessments,
                executions,  # type: ignore[arg-type]
                limits,
            )
            result = ParityResult(
                schema_version=1,
                manifest_id=manifest.manifest_id,
                manifest_sha256=manifest_digest,
                hand=hand.side,
                scenario_id=scenario.scenario_id,
                executions=executions,
                comparison=comparison,
            )
            result_records.append(result)
            if comparison.comparison_status is not ComparisonStatus.WITHIN_TOLERANCE:
                assert comparison.message is not None
                observations.append(
                    f"{hand.side.value}/{scenario.scenario_id}: {comparison.message}"
                )

    completed_count = sum(assessment.completed for assessment in assessments.values())
    expected_count = len(expected_cases)
    completion_fraction = completed_count / expected_count if expected_count else 0.0
    total_expected_steps = sum(
        round(manifest.scenario(case.scenario_id).duration_s / case.dt_s)
        for case in expected_cases
    )
    completed_steps = 0
    for case in expected_cases:
        assessment = assessments.get(case.case_id)
        if assessment is None:
            continue
        raw_completed = assessment.collected.result.completed_steps
        if isinstance(raw_completed, int) and not isinstance(raw_completed, bool):
            completed_steps += min(max(raw_completed, 0), assessment.expected_steps)
    step_completion_fraction = (
        completed_steps / total_expected_steps if total_expected_steps else 0.0
    )
    nonfinite_count = sum(not assessment.finite for assessment in assessments.values())
    observed_joint_mapping_count, observed_frame_mapping_count = (
        _observed_mapping_counts(manifest, assessments)
    )
    execution_status = (
        ExecutionStatus.COMPLETED
        if completed_count == expected_count
        else ExecutionStatus.ERROR
    )
    comparison_statuses = [
        result.comparison.comparison_status for result in result_records
    ]
    if ComparisonStatus.INCONCLUSIVE in comparison_statuses:
        comparison_status = ComparisonStatus.INCONCLUSIVE
    elif ComparisonStatus.DIVERGENT in comparison_statuses:
        comparison_status = ComparisonStatus.DIVERGENT
    else:
        comparison_status = ComparisonStatus.WITHIN_TOLERANCE

    if completion_fraction < limits.minimum_completion_fraction:
        observations.append(
            "Scenario completion target was not met: "
            f"{completion_fraction:.2%} < {limits.minimum_completion_fraction:.2%}."
        )
    if nonfinite_count:
        observations.append(
            f"Detected non-finite values in {nonfinite_count} received run(s); "
            "affected comparisons are inconclusive."
        )

    return Gate0Comparison(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest_digest,
        upstream_repository=manifest.provenance.repository,
        upstream_commit=manifest.provenance.commit,
        asset_git_tree=manifest.provenance.asset_git_tree,
        canonical_lf_asset_tree_sha256=(
            manifest.provenance.canonical_lf_asset_tree_sha256
        ),
        expected_joint_mapping_count=manifest.expected_joint_mapping_count,
        expected_distal_frame_mapping_count=(
            manifest.expected_distal_frame_mapping_count
        ),
        observed_joint_mapping_count=observed_joint_mapping_count,
        observed_distal_frame_mapping_count=observed_frame_mapping_count,
        expected_run_count=expected_count,
        received_run_count=len(collected_by_id),
        completed_run_count=completed_count,
        completion_fraction=completion_fraction,
        step_completion_fraction=step_completion_fraction,
        nonfinite_run_count=nonfinite_count,
        execution_status=execution_status,
        comparison_status=comparison_status,
        thresholds=limits,
        results=tuple(result_records),
        observations=tuple(observations),
    )


__all__ = [
    "CollectedRun",
    "ComparisonThresholds",
    "Gate0Comparison",
    "MAX_ABS_BACKEND_JOINT_VELOCITY_RAD_S",
    "MAX_ABS_CANONICAL_JOINT_POSITION_RAD",
    "MAX_ABS_CANONICAL_POSITION_TARGET_RAD",
    "MAX_FRAME_ORIGIN_DISTANCE_M",
    "SMALL_STEP_MIN_OBSERVED_RESPONSE_RAD",
    "UNIT_QUATERNION_NORM_TOLERANCE",
    "compare_gate0_runs",
    "trace_delta",
]
