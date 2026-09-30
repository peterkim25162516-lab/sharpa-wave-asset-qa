"""Deterministic, publication-safe reporting for WaveSimParity Trajectory T1.

The report layer intentionally keeps evidence validity separate from the
scientific cross-simulator label.  It accepts an already-computed T1 comparison,
descriptive metrics, and deliberately sanitized provenance; it never consumes
raw worker evidence or host-specific execution metadata.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from .compare import Gate0Comparison


PUBLIC_SUMMARY_SCHEMA_VERSION = 2
T1_MANIFEST_ID = "wavesimparity-waveform-t1"

_EXPECTED_HANDS = ("left", "right")
_EXPECTED_SCENARIOS = ("offset_sine", "offset_linear_chirp")
_EXPECTED_BACKENDS = ("mujoco", "ovphysx")
_EXPECTED_OVPHYSX_EFFORT_CASE_COUNT = 16
_EXPECTED_OVPHYSX_JOINTS_PER_OBSERVATION = 22
# Eight base-dt cases observe 2,001 controller states each and eight halved-dt
# cases observe 4,001 each: 8 * 2,001 + 8 * 4,001 = 48,016.
_EXPECTED_OVPHYSX_EFFORT_OBSERVATION_COUNT = 48_016
_STATUS_VALUES = ("within_tolerance", "divergent", "inconclusive")
_CROSSSIM_METRIC_KEYS = (
    "crosssim_joint_max_abs_rad",
    "crosssim_frame_position_max_m",
    "crosssim_frame_orientation_max_rad",
)
_PREREQUISITE_METRIC_KEYS = (
    "repeat_joint_max_abs_rad",
    "repeat_frame_position_max_m",
    "repeat_frame_orientation_max_rad",
    "dt_halving_joint_max_abs_rad",
    "dt_halving_frame_position_max_m",
    "dt_halving_frame_orientation_max_rad",
)
_METRIC_KEYS = (*_CROSSSIM_METRIC_KEYS, *_PREREQUISITE_METRIC_KEYS)
_FROZEN_THRESHOLDS: dict[str, float] = {
    "minimum_completion_fraction": 0.99,
    "crosssim_joint_max_abs_rad": 0.02,
    "crosssim_frame_position_max_m": 0.002,
    "crosssim_frame_orientation_max_rad": 0.05,
    "repeat_joint_max_abs_rad": 1e-9,
    "repeat_frame_position_max_m": 1e-9,
    "repeat_frame_orientation_max_rad": 1e-9,
    "dt_halving_joint_max_abs_rad": 0.01,
    "dt_halving_frame_position_max_m": 0.001,
    "dt_halving_frame_orientation_max_rad": 0.02,
}
_FORBIDDEN_KEY_PARTS = (
    "session_id",
    "run_id",
    "worker_pid",
    "launcher_pid",
    "transport_pid",
    "process_id",
    "pgid",
    "process_group",
    "fresh_process_id",
    "gpu_uuid",
    "gpu_index",
    "selected_gpu",
    "private_plan",
    "ownership_token",
    "hostname",
    "absolute_path",
)
_FORBIDDEN_DESCRIPTIVE_DECISIONS = {
    "validation_status",
    "scientific_label",
    "comparison_status",
    "formal_gate0",
    "pass_ready",
}
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?:^|[\s\"'(<])(?:[A-Za-z]:[\\/]|\\\\)")
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_POSIX_RE = re.compile(
    r"(?:^|[\s\"'(<])/(?:data|home|mnt|tmp|var|users)(?:/|\b)",
    re.IGNORECASE,
)


class WaveformPublicReportError(ValueError):
    """A public report input is contradictory, non-finite, or not sanitized."""


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise WaveformPublicReportError(f"{context} must be an object with string keys")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise WaveformPublicReportError(f"{context} must be an array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise WaveformPublicReportError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WaveformPublicReportError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WaveformPublicReportError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise WaveformPublicReportError(f"{context} must be a finite number")
    return number


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise WaveformPublicReportError(f"{context} must be a boolean")
    return value


def _curate_ovphysx_effort_saturation(
    sanitized_provenance: Mapping[str, Any],
) -> dict[str, object]:
    """Derive the public saturation statement from frozen numeric evidence."""

    observation = _mapping(
        sanitized_provenance.get("ovphysx_effort_clipping_observation"),
        "sanitized_provenance.ovphysx_effort_clipping_observation",
    )
    expected_fields = {
        "schema_version",
        "backend_case_count",
        "joint_count_per_observation",
        "effort_observation_count",
        "effort_clip_count",
    }
    if set(observation) != expected_fields:
        raise WaveformPublicReportError(
            "OVPhysX effort-clipping observation fields are not exact"
        )
    schema_version = _integer(
        observation.get("schema_version"),
        "OVPhysX effort-clipping schema_version",
        minimum=1,
    )
    case_count = _integer(
        observation.get("backend_case_count"),
        "OVPhysX effort-clipping backend_case_count",
        minimum=1,
    )
    joint_count = _integer(
        observation.get("joint_count_per_observation"),
        "OVPhysX effort-clipping joint_count_per_observation",
        minimum=1,
    )
    observation_count = _integer(
        observation.get("effort_observation_count"),
        "OVPhysX effort_observation_count",
        minimum=1,
    )
    clip_count = _integer(
        observation.get("effort_clip_count"),
        "OVPhysX effort_clip_count",
    )
    if (
        schema_version != 1
        or case_count != _EXPECTED_OVPHYSX_EFFORT_CASE_COUNT
        or joint_count != _EXPECTED_OVPHYSX_JOINTS_PER_OBSERVATION
        or observation_count != _EXPECTED_OVPHYSX_EFFORT_OBSERVATION_COUNT
    ):
        raise WaveformPublicReportError(
            "OVPhysX effort-clipping observation does not cover the frozen 16-case matrix"
        )
    joint_effort_value_count = observation_count * joint_count
    if clip_count > joint_effort_value_count:
        raise WaveformPublicReportError(
            "OVPhysX effort_clip_count exceeds observed joint-effort values"
        )
    saturated = clip_count > 0
    return {
        "backend": "kit-less ovphysx",
        "backend_case_count": case_count,
        "effort_observation_count": observation_count,
        "joint_effort_value_count": joint_effort_value_count,
        "observed_effort_clip_count": clip_count,
        "saturation_observed": saturated,
        "response_regime": (
            "SATURATED_NONLINEAR"
            if saturated
            else "NO_EFFORT_CLIPPING_OBSERVED"
        ),
        "linear_transfer_function_claimed": False,
    }


def _validate_public_string(value: str, context: str) -> None:
    lowered = value.lower()
    forbidden_phrases = (
        "session_id",
        "run_id",
        "private_plan",
        "private plan",
        "gpu_uuid",
        "gpu uuid",
        "worker_pid",
        "worker pid",
    )
    if (
        value.startswith("/")
        or _WINDOWS_ABSOLUTE_RE.search(value) is not None
        or _PRIVATE_POSIX_RE.search(value) is not None
        or _GPU_UUID_RE.search(value) is not None
        or any(phrase in lowered for phrase in forbidden_phrases)
    ):
        raise WaveformPublicReportError(
            f"{context} contains a private path or execution identifier"
        )


def _public_key_is_forbidden(key: str) -> bool:
    lowered = key.lower()
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    return any(part in lowered for part in _FORBIDDEN_KEY_PARTS) or any(
        token in compact
        for token in ("launcherpid", "transportpid", "pgid", "processgroup")
    )


def _clone_public(value: object, context: str) -> object:
    """Return a key-sorted JSON value after public-safety and finite checks."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise WaveformPublicReportError(f"{context} contains a non-string key")
        result: dict[str, object] = {}
        for key in sorted(value):
            if _public_key_is_forbidden(key):
                raise WaveformPublicReportError(
                    f"{context}.{key} is forbidden in public output"
                )
            result[key] = _clone_public(value[key], f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _clone_public(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WaveformPublicReportError(f"{context} contains NaN or Infinity")
        return value
    if isinstance(value, str):
        _validate_public_string(value, context)
        return value
    raise WaveformPublicReportError(
        f"{context} contains a non-JSON value of type {type(value).__name__}"
    )


def _validate_descriptive_keys(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_DESCRIPTIVE_DECISIONS:
                raise WaveformPublicReportError(
                    f"{context}.{key} must not carry a scientific decision"
                )
            _validate_descriptive_keys(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_descriptive_keys(child, f"{context}[{index}]")


def _comparison_payload(value: Gate0Comparison | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Gate0Comparison):
        return value.to_dict()
    return _mapping(value, "comparison")


def _curate_thresholds(value: object) -> dict[str, float]:
    thresholds = _mapping(value, "comparison.thresholds")
    if set(thresholds) != set(_FROZEN_THRESHOLDS):
        raise WaveformPublicReportError(
            "comparison.thresholds do not exactly match the frozen T1 fields"
        )
    curated: dict[str, float] = {}
    for key, expected in _FROZEN_THRESHOLDS.items():
        observed = _finite(thresholds[key], f"comparison.thresholds.{key}")
        if observed != expected:
            raise WaveformPublicReportError(
                f"comparison.thresholds.{key} drifted from the frozen T1 value"
            )
        curated[key] = observed
    return curated


def _derive_cell_status(
    executions: Mapping[str, Mapping[str, object]],
    metrics: Mapping[str, float | None],
    thresholds: Mapping[str, float],
) -> str:
    """Derive the only publication-safe cell label from frozen gates."""

    if any(
        execution["execution_status"] != "COMPLETED" or not execution["finite"]
        for execution in executions.values()
    ):
        return "inconclusive"
    if any(metrics[key] is None for key in _METRIC_KEYS):
        return "inconclusive"

    def exceeds_threshold(key: str) -> bool:
        observed = metrics[key]
        assert observed is not None
        return observed > thresholds[key]

    if any(exceeds_threshold(key) for key in _PREREQUISITE_METRIC_KEYS):
        return "inconclusive"
    if any(exceeds_threshold(key) for key in _CROSSSIM_METRIC_KEYS):
        return "divergent"
    return "within_tolerance"


def _curate_results(
    value: object,
    thresholds: Mapping[str, float],
) -> tuple[list[dict[str, object]], str, bool]:
    raw_results = _sequence(value, "comparison.results")
    expected_cells = {
        (hand, scenario) for hand in _EXPECTED_HANDS for scenario in _EXPECTED_SCENARIOS
    }
    by_cell: dict[tuple[str, str], dict[str, object]] = {}
    all_executions_completed = True
    for index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, f"comparison.results[{index}]")
        hand = _string(result.get("hand"), f"comparison.results[{index}].hand")
        scenario = _string(
            result.get("scenario_id"),
            f"comparison.results[{index}].scenario_id",
        )
        cell = (hand, scenario)
        if cell not in expected_cells:
            raise WaveformPublicReportError(f"unexpected T1 result cell: {cell!r}")
        if cell in by_cell:
            raise WaveformPublicReportError(f"duplicate T1 result cell: {cell!r}")

        raw_executions = _sequence(
            result.get("executions"), f"comparison.results[{index}].executions"
        )
        if len(raw_executions) != 2:
            raise WaveformPublicReportError("each T1 result must contain two executions")
        executions: dict[str, dict[str, object]] = {}
        for execution_index, raw_execution in enumerate(raw_executions):
            execution = _mapping(
                raw_execution,
                f"comparison.results[{index}].executions[{execution_index}]",
            )
            backend = _string(
                execution.get("simulator"),
                f"comparison.results[{index}].executions[{execution_index}].simulator",
            )
            if backend not in _EXPECTED_BACKENDS or backend in executions:
                raise WaveformPublicReportError(
                    "T1 executions must contain MuJoCo and OVPhysX exactly once"
                )
            execution_status = _string(
                execution.get("execution_status"),
                f"comparison.results[{index}].executions[{execution_index}].execution_status",
            )
            if execution_status not in {"completed", "error"}:
                raise WaveformPublicReportError("unsupported execution status")
            completed_repeats = _integer(
                execution.get("completed_repeats"),
                "execution.completed_repeats",
            )
            requested_repeats = _integer(
                execution.get("requested_repeats"),
                "execution.requested_repeats",
                minimum=1,
            )
            finite = _boolean(execution.get("finite"), "execution.finite")
            if requested_repeats != 2:
                raise WaveformPublicReportError(
                    "each T1 backend execution must request exactly two repeats"
                )
            if completed_repeats > requested_repeats:
                raise WaveformPublicReportError(
                    "completed repeats cannot exceed requested repeats"
                )
            if execution_status == "completed" and (
                completed_repeats != 2 or not finite
            ):
                raise WaveformPublicReportError(
                    "a completed execution must finish exactly 2/2 repeats "
                    "with finite state"
                )
            all_executions_completed &= execution_status == "completed"
            executions[backend] = {
                "execution_status": execution_status.upper(),
                "completed_repeats": completed_repeats,
                "requested_repeats": requested_repeats,
                "finite": finite,
            }
        if tuple(executions) != _EXPECTED_BACKENDS:
            raise WaveformPublicReportError(
                "T1 executions must be ordered MuJoCo then OVPhysX"
            )

        comparison = _mapping(
            result.get("comparison"), f"comparison.results[{index}].comparison"
        )
        status = _string(
            comparison.get("comparison_status"),
            f"comparison.results[{index}].comparison.comparison_status",
        )
        if status not in _STATUS_VALUES:
            raise WaveformPublicReportError("unsupported scientific comparison status")
        raw_metrics = _mapping(
            comparison.get("metrics"),
            f"comparison.results[{index}].comparison.metrics",
        )
        unknown_metrics = set(raw_metrics) - set(_METRIC_KEYS)
        if unknown_metrics:
            raise WaveformPublicReportError(
                "comparison metrics contain unknown fields: "
                + ", ".join(sorted(unknown_metrics))
            )
        if status != "inconclusive" and set(raw_metrics) != set(_METRIC_KEYS):
            raise WaveformPublicReportError(
                "a conclusive T1 cell must contain every frozen comparison metric"
            )
        metrics: dict[str, float | None] = {}
        for key in _METRIC_KEYS:
            metrics[key] = (
                _finite(raw_metrics[key], f"result metrics.{key}")
                if key in raw_metrics
                else None
            )
            if metrics[key] is not None and metrics[key] < 0.0:
                raise WaveformPublicReportError(
                    f"result metrics.{key} must be non-negative"
                )
        derived_status = _derive_cell_status(executions, metrics, thresholds)
        if status != derived_status:
            raise WaveformPublicReportError(
                "cell comparison status contradicts the frozen T1 gates: "
                f"reported={status}, derived={derived_status}"
            )
        by_cell[cell] = {
            "hand": hand,
            "scenario_id": scenario,
            "executions": executions,
            "comparison_status": derived_status.upper(),
            "metrics": metrics,
        }

    if set(by_cell) != expected_cells:
        missing = sorted(expected_cells - set(by_cell))
        raise WaveformPublicReportError(f"T1 comparison is missing result cells: {missing}")
    ordered = [
        by_cell[(hand, scenario)]
        for hand in _EXPECTED_HANDS
        for scenario in _EXPECTED_SCENARIOS
    ]
    statuses = [str(item["comparison_status"]).lower() for item in ordered]
    if "inconclusive" in statuses:
        overall = "inconclusive"
    elif "divergent" in statuses:
        overall = "divergent"
    else:
        overall = "within_tolerance"
    return ordered, overall, all_executions_completed


def build_public_summary(
    comparison: Gate0Comparison | Mapping[str, Any],
    descriptive_metrics: Mapping[str, Any],
    sanitized_provenance: Mapping[str, Any],
) -> dict[str, object]:
    """Build the deterministic, publication-safe T1 summary payload."""

    data = _comparison_payload(comparison)
    metadata = _mapping(data.get("metadata"), "comparison.metadata")
    if metadata.get("manifest_id") != T1_MANIFEST_ID:
        raise WaveformPublicReportError(
            f"comparison manifest_id must be {T1_MANIFEST_ID!r}"
        )
    if metadata.get("backend_pair") != ["mujoco", "kit-less ovphysx"]:
        raise WaveformPublicReportError(
            "comparison metadata must identify MuJoCo and kit-less OVPhysX"
        )
    manifest_metadata = {
        key: _clone_public(metadata[key], f"comparison.metadata.{key}")
        for key in (
            "manifest_id",
            "manifest_sha256",
            "upstream_repository",
            "upstream_commit",
            "asset_git_tree",
            "canonical_lf_asset_tree_sha256",
        )
        if key in metadata
    }
    if set(manifest_metadata) != {
        "manifest_id",
        "manifest_sha256",
        "upstream_repository",
        "upstream_commit",
        "asset_git_tree",
        "canonical_lf_asset_tree_sha256",
    }:
        raise WaveformPublicReportError("comparison metadata is incomplete")
    if (
        not isinstance(manifest_metadata["manifest_sha256"], str)
        or _SHA256_RE.fullmatch(manifest_metadata["manifest_sha256"]) is None
        or not isinstance(manifest_metadata["upstream_commit"], str)
        or _GIT_OID_RE.fullmatch(manifest_metadata["upstream_commit"]) is None
        or not isinstance(manifest_metadata["asset_git_tree"], str)
        or _GIT_OID_RE.fullmatch(manifest_metadata["asset_git_tree"]) is None
        or not isinstance(
            manifest_metadata["canonical_lf_asset_tree_sha256"], str
        )
        or _SHA256_RE.fullmatch(
            manifest_metadata["canonical_lf_asset_tree_sha256"]
        )
        is None
    ):
        raise WaveformPublicReportError(
            "comparison metadata contains an invalid revision or SHA-256"
        )

    thresholds = _curate_thresholds(data.get("thresholds"))
    scenario_results, derived_science, all_cell_executions_completed = _curate_results(
        data.get("results"), thresholds
    )
    raw_summary = _mapping(data.get("summary"), "comparison.summary")
    execution_status = _string(
        raw_summary.get("execution_status"), "comparison.summary.execution_status"
    )
    if execution_status not in {"completed", "error"}:
        raise WaveformPublicReportError("unsupported aggregate execution status")
    comparison_status = _string(
        raw_summary.get("comparison_status"),
        "comparison.summary.comparison_status",
    )
    if comparison_status != derived_science:
        raise WaveformPublicReportError(
            "aggregate comparison status contradicts the four T1 result cells"
        )
    if (execution_status == "completed") != all_cell_executions_completed:
        raise WaveformPublicReportError(
            "aggregate execution status contradicts the T1 result cells"
        )

    expected_runs = _integer(
        raw_summary.get("expected_run_count"), "summary.expected_run_count", minimum=1
    )
    received_runs = _integer(
        raw_summary.get("received_run_count"), "summary.received_run_count"
    )
    completed_runs = _integer(
        raw_summary.get("completed_run_count"), "summary.completed_run_count"
    )
    if expected_runs != 32 or completed_runs > received_runs or received_runs > expected_runs:
        raise WaveformPublicReportError("T1 run counts are inconsistent with the 32-case matrix")
    completion_fraction = _finite(
        raw_summary.get("completion_fraction"), "summary.completion_fraction"
    )
    step_completion_fraction = _finite(
        raw_summary.get("step_completion_fraction"), "summary.step_completion_fraction"
    )
    if not 0.0 <= completion_fraction <= 1.0 or not 0.0 <= step_completion_fraction <= 1.0:
        raise WaveformPublicReportError("completion fractions must be in [0, 1]")
    if completion_fraction != completed_runs / expected_runs:
        raise WaveformPublicReportError("completion fraction contradicts the case counts")
    completion_target_met = _boolean(
        raw_summary.get("completion_target_met"), "summary.completion_target_met"
    )
    if completion_target_met != (
        completion_fraction >= thresholds["minimum_completion_fraction"]
    ):
        raise WaveformPublicReportError("completion target flag is inconsistent")

    expected_joints = _integer(
        raw_summary.get("expected_joint_mapping_count"),
        "summary.expected_joint_mapping_count",
    )
    observed_joints = _integer(
        raw_summary.get("observed_joint_mapping_count"),
        "summary.observed_joint_mapping_count",
    )
    expected_frames = _integer(
        raw_summary.get("expected_distal_frame_mapping_count"),
        "summary.expected_distal_frame_mapping_count",
    )
    observed_frames = _integer(
        raw_summary.get("observed_distal_frame_mapping_count"),
        "summary.observed_distal_frame_mapping_count",
    )
    if expected_joints != 44 or expected_frames != 10:
        raise WaveformPublicReportError("T1 expected mapping must be 44 joints and 10 DP frames")
    mapping_complete = _boolean(
        raw_summary.get("mapping_validation_complete"),
        "summary.mapping_validation_complete",
    )
    expected_mapping_complete = (
        execution_status == "completed"
        and observed_joints == expected_joints
        and observed_frames == expected_frames
    )
    if mapping_complete != expected_mapping_complete:
        raise WaveformPublicReportError("mapping completeness flag is inconsistent")
    nonfinite_runs = _integer(
        raw_summary.get("nonfinite_run_count"), "summary.nonfinite_run_count"
    )
    finite = _boolean(raw_summary.get("finite"), "summary.finite")
    if finite != (nonfinite_runs == 0):
        raise WaveformPublicReportError("finite flag contradicts nonfinite run count")

    valid = (
        execution_status == "completed"
        and expected_runs == received_runs == completed_runs == 32
        and completion_fraction == 1.0
        and step_completion_fraction == 1.0
        and completion_target_met
        and mapping_complete
        and observed_joints == 44
        and observed_frames == 10
        and finite
        and derived_science != "inconclusive"
    )
    validation_status = "VALID" if valid else "INVALID"
    scientific_label = derived_science.upper() if valid else "INCONCLUSIVE"

    metrics_input = _mapping(descriptive_metrics, "descriptive_metrics")
    _validate_descriptive_keys(metrics_input, "descriptive_metrics")
    curated_metrics = _clone_public(metrics_input, "descriptive_metrics")
    assert isinstance(curated_metrics, dict)
    if valid and not curated_metrics:
        raise WaveformPublicReportError(
            "a valid T1 summary requires the preregistered descriptive metrics"
        )
    provenance_input = _mapping(sanitized_provenance, "sanitized_provenance")
    ovphysx_effort_saturation = _curate_ovphysx_effort_saturation(
        provenance_input
    )
    curated_provenance = _clone_public(provenance_input, "sanitized_provenance")
    assert isinstance(curated_provenance, dict)
    if not curated_provenance:
        raise WaveformPublicReportError("sanitized provenance must not be empty")

    summary: dict[str, object] = {
        "schema_version": PUBLIC_SUMMARY_SCHEMA_VERSION,
        "protocol_id": "wavesimparity-waveform-t1-v1",
        "validation_status": validation_status,
        "scientific_label": scientific_label,
        "formal_gate0": {
            "comparison_status": "DIVERGENT",
            "pass_ready": False,
            "unchanged": True,
        },
        "execution": {
            "execution_status": execution_status.upper(),
            "expected_case_count": expected_runs,
            "received_case_count": received_runs,
            "completed_case_count": completed_runs,
            "completion_fraction": completion_fraction,
            "step_completion_fraction": step_completion_fraction,
            "nonfinite_run_count": nonfinite_runs,
        },
        "mapping": {
            "joint_mapping": f"{observed_joints}/{expected_joints}",
            "dp_frame_mapping": f"{observed_frames}/{expected_frames}",
            "complete": mapping_complete,
        },
        "thresholds": thresholds,
        "scenario_results": scenario_results,
        "descriptive_metrics": curated_metrics,
        "ovphysx_effort_saturation": ovphysx_effort_saturation,
        "provenance": {
            "manifest": manifest_metadata,
            "sanitized_execution": curated_provenance,
        },
        "scope": {
            "backend_pair": ["mujoco", "kit-less ovphysx"],
            "kit_less_ovphysx": True,
            "full_isaac_sim": False,
            "kit": False,
            "renderer": False,
            "camera": False,
            "fixed_base": True,
            "separate_left_and_right_hands": True,
            "dual_hand_model": False,
            "position_control": True,
            "contacts_validated": False,
            "dp_frame_semantics": (
                "distal_phalanx_link_origins_not_physical_fingertip_contact_surfaces"
            ),
        },
        "claim_boundary": {
            "official_bug_claimed": False,
            "hardware_claimed": False,
            "sim2real_claimed": False,
            "cross_engine_parameter_equivalence_claimed": False,
            "physical_fingertip_surface_claimed": False,
        },
    }
    # This is both a final finite check and a guarantee that downstream writers
    # can use strict JSON without silently emitting NaN/Infinity.
    json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return summary


def canonical_summary_json(summary: Mapping[str, Any]) -> str:
    """Serialize a built public summary deterministically with one final newline."""

    curated = _clone_public(_mapping(summary, "summary"), "summary")
    return (
        json.dumps(
            curated,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _format_float(value: object) -> str:
    if value is None:
        return "—"
    number = _finite(value, "report metric")
    return "0" if number == 0.0 else f"{number:.6g}"


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(item) for item in row) + " |"
        for row in rows
    )
    return lines


def render_public_report(summary: Mapping[str, Any]) -> str:
    """Render deterministic Markdown from a validated public summary."""

    data = _clone_public(_mapping(summary, "summary"), "summary")
    assert isinstance(data, dict)
    # Requiring a canonical rebuild shape prevents arbitrary mappings from
    # being presented as a T1 public report.
    required = {
        "schema_version",
        "protocol_id",
        "validation_status",
        "scientific_label",
        "formal_gate0",
        "execution",
        "mapping",
        "thresholds",
        "scenario_results",
        "descriptive_metrics",
        "ovphysx_effort_saturation",
        "provenance",
        "scope",
        "claim_boundary",
    }
    if set(data) != required:
        raise WaveformPublicReportError("summary does not have the canonical T1 public shape")
    if data.get("schema_version") != PUBLIC_SUMMARY_SCHEMA_VERSION:
        raise WaveformPublicReportError("summary schema_version is not canonical")
    validation_status = _string(data["validation_status"], "summary.validation_status")
    scientific_label = _string(data["scientific_label"], "summary.scientific_label")
    if validation_status not in {"VALID", "INVALID"}:
        raise WaveformPublicReportError("unsupported validation status")
    if scientific_label not in {"WITHIN_TOLERANCE", "DIVERGENT", "INCONCLUSIVE"}:
        raise WaveformPublicReportError("unsupported scientific label")
    if validation_status == "INVALID" and scientific_label != "INCONCLUSIVE":
        raise WaveformPublicReportError("an invalid summary must be scientifically inconclusive")
    if validation_status == "VALID" and scientific_label == "INCONCLUSIVE":
        raise WaveformPublicReportError("a valid summary must have a conclusive science label")
    formal = _mapping(data["formal_gate0"], "summary.formal_gate0")
    if formal != {
        "comparison_status": "DIVERGENT",
        "pass_ready": False,
        "unchanged": True,
    }:
        raise WaveformPublicReportError("formal Gate 0 boundary is not frozen")
    execution = _mapping(data["execution"], "summary.execution")
    mapping = _mapping(data["mapping"], "summary.mapping")
    thresholds = _mapping(data["thresholds"], "summary.thresholds")
    scenario_results = _sequence(data["scenario_results"], "summary.scenario_results")
    provenance = _mapping(data["provenance"], "summary.provenance")
    if set(provenance) != {"manifest", "sanitized_execution"}:
        raise WaveformPublicReportError("summary provenance fields are not exact")
    sanitized_execution = _mapping(
        provenance.get("sanitized_execution"),
        "summary.provenance.sanitized_execution",
    )
    effort_saturation = _mapping(
        data["ovphysx_effort_saturation"],
        "summary.ovphysx_effort_saturation",
    )
    if effort_saturation != _curate_ovphysx_effort_saturation(
        sanitized_execution
    ):
        raise WaveformPublicReportError(
            "OVPhysX effort-saturation statement contradicts its numeric evidence"
        )
    scope = _mapping(data["scope"], "summary.scope")
    expected_scope = {
        "backend_pair": ["mujoco", "kit-less ovphysx"],
        "kit_less_ovphysx": True,
        "full_isaac_sim": False,
        "kit": False,
        "renderer": False,
        "camera": False,
        "fixed_base": True,
        "separate_left_and_right_hands": True,
        "dual_hand_model": False,
        "position_control": True,
        "contacts_validated": False,
        "dp_frame_semantics": (
            "distal_phalanx_link_origins_not_physical_fingertip_contact_surfaces"
        ),
    }
    if scope != expected_scope:
        raise WaveformPublicReportError("T1 public scope boundary is inconsistent")
    claim_boundary = _mapping(data["claim_boundary"], "summary.claim_boundary")
    if claim_boundary != {
        "official_bug_claimed": False,
        "hardware_claimed": False,
        "sim2real_claimed": False,
        "cross_engine_parameter_equivalence_claimed": False,
        "physical_fingertip_surface_claimed": False,
    }:
        raise WaveformPublicReportError("T1 public claim boundary is inconsistent")

    lines = [
        "# WaveSimParity Trajectory T1 report",
        "",
        "> Unofficial, simulation-only MuJoCo ↔ kit-less OVPhysX comparison.",
        "> This is not full Isaac Sim, hardware validation, or Sim2Real evidence.",
        "",
        "## Decision",
        "",
        f"- Validation status: **{validation_status}**",
        f"- Scientific comparison: **{scientific_label}**",
        "- Formal Gate 0: **DIVERGENT**, `pass_ready=false` (unchanged)",
        "",
        "`VALID` describes evidence integrity and prerequisite gates. "
        "`WITHIN_TOLERANCE`, `DIVERGENT`, and `INCONCLUSIVE` describe the "
        "separate cross-simulator interpretation.",
        "",
        "## Execution and mapping",
        "",
    ]
    lines.extend(
        _table(
            ("Check", "Observed"),
            (
                ("Execution", execution.get("execution_status")),
                (
                    "Completed cases",
                    f"{execution.get('completed_case_count')}/{execution.get('expected_case_count')}",
                ),
                ("Step completion", _format_float(execution.get("step_completion_fraction"))),
                ("Non-finite runs", execution.get("nonfinite_run_count")),
                ("Canonical joints", mapping.get("joint_mapping")),
                ("Canonical DP frames", mapping.get("dp_frame_mapping")),
            ),
        )
    )

    observed_effort_clips = _integer(
        effort_saturation.get("observed_effort_clip_count"),
        "summary OVPhysX observed_effort_clip_count",
    )
    effort_observations = _integer(
        effort_saturation.get("effort_observation_count"),
        "summary OVPhysX effort_observation_count",
        minimum=1,
    )
    joint_effort_values = _integer(
        effort_saturation.get("joint_effort_value_count"),
        "summary OVPhysX joint_effort_value_count",
        minimum=1,
    )
    lines.extend(
        [
            "",
            "## OVPhysX effort clipping",
            "",
            f"- Controller-state observations: **{effort_observations}** across 16 OVPhysX cases.",
            f"- Observed joint-effort values: **{joint_effort_values}**.",
            f"- Observed effort clip count: **{observed_effort_clips}**.",
        ]
    )
    if observed_effort_clips > 0:
        lines.append(
            "- Interpretation: **SATURATED / NONLINEAR**. Effort clipping was "
            "observed, so this response must not be described as a linear "
            "transfer function."
        )
    else:
        lines.append(
            "- Interpretation: **NO EFFORT CLIPPING OBSERVED** in the frozen "
            "samples. This count of zero does not establish globally linear behavior."
        )

    lines.extend(["", "## Scenario comparisons", ""])
    rows: list[tuple[object, ...]] = []
    for raw_result in scenario_results:
        result = _mapping(raw_result, "summary.scenario_results[]")
        executions = _mapping(result.get("executions"), "scenario executions")
        mujoco = _mapping(executions.get("mujoco"), "scenario MuJoCo execution")
        ovphysx = _mapping(executions.get("ovphysx"), "scenario OVPhysX execution")
        metrics = _mapping(result.get("metrics"), "scenario metrics")
        rows.append(
            (
                result.get("hand"),
                result.get("scenario_id"),
                mujoco.get("execution_status"),
                ovphysx.get("execution_status"),
                result.get("comparison_status"),
                _format_float(metrics.get("crosssim_joint_max_abs_rad")),
                _format_float(metrics.get("crosssim_frame_position_max_m")),
                _format_float(metrics.get("crosssim_frame_orientation_max_rad")),
            )
        )
    lines.extend(
        _table(
            (
                "Hand",
                "Scenario",
                "MuJoCo",
                "OVPhysX",
                "Comparison",
                "Joint max (rad)",
                "DP position max (m)",
                "DP orientation max (rad)",
            ),
            rows,
        )
    )

    lines.extend(["", "## Frozen thresholds", ""])
    lines.extend(
        _table(
            ("Metric", "Threshold"),
            tuple(
                (name, _format_float(thresholds[name]))
                for name in sorted(thresholds)
            ),
        )
    )

    descriptive_json = json.dumps(
        data["descriptive_metrics"],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    provenance_json = json.dumps(
        data["provenance"],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    lines.extend(
        [
            "",
            "## Preregistered descriptive metrics",
            "",
            "These metrics are descriptive only and add no pass/fail threshold.",
            "",
            "```json",
            descriptive_json,
            "```",
            "",
            "## Sanitized provenance",
            "",
            "```json",
            provenance_json,
            "```",
            "",
            "## Scope and claim boundary",
            "",
            "- `DIVERGENT` is a reproducible observation under the pinned setup, not by itself an official bug finding.",
            "- OVPhysX is kit-less and headless here. No Kit application, renderer, camera, or full Isaac Sim workflow is exercised.",
            "- This fixed-base simulation result establishes no hardware behavior, safety, controller quality, or Sim2Real claim.",
            "- The reported `*_DP` frames are distal-phalanx link origins, not physical fingertip contact surfaces.",
            "- Contact parity, dual-hand models, wrist/flange variants, and floating bases remain outside T1.",
            "",
        ]
    )
    report = "\n".join(lines)
    _validate_public_string(report, "report")
    return report


def build_public_artifacts(
    comparison: Gate0Comparison | Mapping[str, Any],
    descriptive_metrics: Mapping[str, Any],
    sanitized_provenance: Mapping[str, Any],
) -> tuple[dict[str, object], str]:
    """Return the public ``summary.json`` payload and ``report.md`` text."""

    summary = build_public_summary(
        comparison, descriptive_metrics, sanitized_provenance
    )
    return summary, render_public_report(summary)


__all__ = [
    "PUBLIC_SUMMARY_SCHEMA_VERSION",
    "T1_MANIFEST_ID",
    "WaveformPublicReportError",
    "build_public_artifacts",
    "build_public_summary",
    "canonical_summary_json",
    "render_public_report",
]
