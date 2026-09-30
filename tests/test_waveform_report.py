from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from wave_asset_qa.parity.waveform_report import (
    WaveformPublicReportError,
    build_public_artifacts,
    build_public_summary,
    canonical_summary_json,
    render_public_report,
)


FROZEN_THRESHOLDS = {
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


def _metrics(joint: float, position: float, orientation: float) -> dict[str, float]:
    return {
        "crosssim_joint_max_abs_rad": joint,
        "crosssim_frame_position_max_m": position,
        "crosssim_frame_orientation_max_rad": orientation,
        "repeat_joint_max_abs_rad": 0.0,
        "repeat_frame_position_max_m": 0.0,
        "repeat_frame_orientation_max_rad": 0.0,
        "dt_halving_joint_max_abs_rad": 0.001,
        "dt_halving_frame_position_max_m": 0.0001,
        "dt_halving_frame_orientation_max_rad": 0.002,
    }


def _result(hand: str, scenario: str, status: str) -> dict[str, object]:
    conclusive = status != "inconclusive"
    return {
        "schema_version": 1,
        "manifest_id": "wavesimparity-waveform-t1",
        "manifest_sha256": "a" * 64,
        "hand": hand,
        "scenario_id": scenario,
        "executions": [
            {
                "simulator": backend,
                "execution_status": "completed" if conclusive else "error",
                "completed_repeats": 2 if conclusive else 0,
                "requested_repeats": 2,
                "finite": conclusive,
                "bundle_root_sha256": None,
                "message": None if conclusive else "sanitized execution failure",
            }
            for backend in ("mujoco", "ovphysx")
        ],
        "comparison": {
            "comparison_status": status,
            "metrics": (
                _metrics(0.021 if status == "divergent" else 0.01, 0.001, 0.01)
                if conclusive
                else {}
            ),
            "message": "sanitized observation",
        },
    }


def _comparison(status: str = "divergent") -> dict[str, object]:
    results = [
        _result("left", "offset_sine", status),
        _result("left", "offset_linear_chirp", status),
        _result("right", "offset_sine", status),
        _result("right", "offset_linear_chirp", status),
    ]
    conclusive = status != "inconclusive"
    return {
        "schema_version": 1,
        "metadata": {
            "manifest_id": "wavesimparity-waveform-t1",
            "manifest_sha256": "a" * 64,
            "upstream_repository": "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git",
            "upstream_commit": "b" * 40,
            "asset_git_tree": "c" * 40,
            "canonical_lf_asset_tree_sha256": "d" * 64,
            "backend_pair": ["mujoco", "kit-less ovphysx"],
        },
        "summary": {
            "execution_status": "completed" if conclusive else "error",
            "comparison_status": status,
            "expected_joint_mapping_count": 44,
            "expected_distal_frame_mapping_count": 10,
            "observed_joint_mapping_count": 44 if conclusive else 0,
            "observed_distal_frame_mapping_count": 10 if conclusive else 0,
            "mapping_validation_complete": conclusive,
            "expected_run_count": 32,
            "received_run_count": 32,
            "completed_run_count": 32 if conclusive else 0,
            "completion_fraction": 1.0 if conclusive else 0.0,
            "step_completion_fraction": 1.0 if conclusive else 0.0,
            "completion_target_met": conclusive,
            "nonfinite_run_count": 0,
            "finite": True,
        },
        "thresholds": dict(FROZEN_THRESHOLDS),
        "results": results,
        "observations": [],
    }


def _descriptive_metrics() -> dict[str, object]:
    return {
        "sine": {
            "phase_lag_rad": None,
            "joint_rmse_rad": 0.003,
            "phase_unavailable_below_rad": 1e-6,
        },
        "chirp": {
            "bands_hz": [
                [0.5, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
            "joint_rmse_rad": 0.004,
        },
    }


def _provenance() -> dict[str, object]:
    return {
        "execution_revision": "e" * 40,
        "execution_tree": "f" * 40,
        "source_archive_sha256": "1" * 64,
        "immutable_snapshot_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "ovphysx_effort_clipping_observation": {
            "schema_version": 1,
            "backend_case_count": 16,
            "joint_count_per_observation": 22,
            "effort_observation_count": 48_016,
            "effort_clip_count": 0,
        },
    }


def test_valid_divergent_public_artifacts_separate_the_two_decisions() -> None:
    summary, report = build_public_artifacts(
        _comparison(), _descriptive_metrics(), _provenance()
    )

    assert summary["validation_status"] == "VALID"
    assert summary["scientific_label"] == "DIVERGENT"
    assert summary["formal_gate0"] == {
        "comparison_status": "DIVERGENT",
        "pass_ready": False,
        "unchanged": True,
    }
    assert summary["scope"]["full_isaac_sim"] is False
    assert summary["scope"]["dp_frame_semantics"].endswith(
        "not_physical_fingertip_contact_surfaces"
    )
    assert summary["ovphysx_effort_saturation"] == {
        "backend": "kit-less ovphysx",
        "backend_case_count": 16,
        "effort_observation_count": 48_016,
        "joint_effort_value_count": 1_056_352,
        "observed_effort_clip_count": 0,
        "saturation_observed": False,
        "response_regime": "NO_EFFORT_CLIPPING_OBSERVED",
        "linear_transfer_function_claimed": False,
    }
    assert "Validation status: **VALID**" in report
    assert "Scientific comparison: **DIVERGENT**" in report
    assert "Formal Gate 0: **DIVERGENT**, `pass_ready=false` (unchanged)" in report
    assert "not full Isaac Sim" in report
    assert "hardware validation" in report
    assert "Sim2Real" in report
    assert "distal-phalanx link origins" in report
    assert "Observed effort clip count: **0**" in report
    assert "NO EFFORT CLIPPING OBSERVED" in report
    assert "does not establish globally linear behavior" in report
    json.dumps(summary, sort_keys=True, allow_nan=False)
    public_text = canonical_summary_json(summary) + report
    for forbidden in (
        "session_id",
        "worker_pid",
        "gpu_uuid",
        "private_plan",
        "GPU-12345678",
        "C:\\",
        "/data/home/",
    ):
        assert forbidden not in public_text


def test_invalid_evidence_forces_science_to_inconclusive() -> None:
    summary = build_public_summary(
        _comparison("inconclusive"), {}, _provenance()
    )

    assert summary["validation_status"] == "INVALID"
    assert summary["scientific_label"] == "INCONCLUSIVE"
    assert summary["formal_gate0"]["comparison_status"] == "DIVERGENT"
    report = render_public_report(summary)
    assert "Validation status: **INVALID**" in report
    assert "Scientific comparison: **INCONCLUSIVE**" in report


def test_public_outputs_are_deterministic_under_input_mapping_and_result_order() -> None:
    first_comparison = _comparison("within_tolerance")
    second_comparison = deepcopy(first_comparison)
    second_comparison["results"] = list(reversed(second_comparison["results"]))
    first_metrics = _descriptive_metrics()
    second_metrics = {
        key: first_metrics[key] for key in reversed(tuple(first_metrics))
    }
    first_provenance = _provenance()
    second_provenance = {
        key: first_provenance[key] for key in reversed(tuple(first_provenance))
    }

    first_summary, first_report = build_public_artifacts(
        first_comparison, first_metrics, first_provenance
    )
    second_summary, second_report = build_public_artifacts(
        second_comparison, second_metrics, second_provenance
    )

    assert first_summary == second_summary
    assert canonical_summary_json(first_summary) == canonical_summary_json(second_summary)
    assert canonical_summary_json(first_summary).endswith("\n")
    assert first_report == second_report
    assert first_summary["validation_status"] == "VALID"
    assert first_summary["scientific_label"] == "WITHIN_TOLERANCE"


@pytest.mark.parametrize(
    ("where", "key", "value"),
    (
        ("provenance", "session_id", "hidden"),
        ("provenance", "worker_pid", 1234),
        ("provenance", "launcher_pid", 1234),
        ("provenance", "transport-pid", 1234),
        ("provenance", "workerPgid", 1234),
        ("provenance", "process_group", 1234),
        ("provenance", "process-group", 1234),
        ("provenance", "ProcessGroup", 1234),
        (
            "provenance",
            "gpu_uuid",
            "GPU-0311660a-44a0-594f-9e46-2d209c8dc9e2",
        ),
        ("provenance", "private_plan", "secret"),
        ("provenance", "evidence", "C:\\private\\result.json"),
        ("metrics", "raw", "/data/home/person/result.json"),
        ("metrics", "bad_number", math.nan),
    ),
)
def test_private_or_nonfinite_values_are_rejected(
    where: str, key: str, value: object
) -> None:
    metrics = _descriptive_metrics()
    provenance = _provenance()
    target = provenance if where == "provenance" else metrics
    target[key] = value

    with pytest.raises(WaveformPublicReportError):
        build_public_summary(_comparison(), metrics, provenance)


@pytest.mark.parametrize(
    "reserved",
    (
        "validation_status",
        "scientific_label",
        "comparison_status",
        "formal_gate0",
        "pass_ready",
    ),
)
def test_descriptive_metrics_cannot_smuggle_a_decision(reserved: str) -> None:
    metrics = _descriptive_metrics()
    metrics[reserved] = "PASS"

    with pytest.raises(WaveformPublicReportError, match="scientific decision"):
        build_public_summary(_comparison(), metrics, _provenance())


def test_effort_clip_count_derives_saturated_nonlinear_disclosure() -> None:
    provenance = _provenance()
    observation = provenance["ovphysx_effort_clipping_observation"]
    assert isinstance(observation, dict)
    observation["effort_clip_count"] = 37

    summary, report = build_public_artifacts(
        _comparison(), _descriptive_metrics(), provenance
    )

    assert summary["ovphysx_effort_saturation"]["observed_effort_clip_count"] == 37
    assert summary["ovphysx_effort_saturation"]["saturation_observed"] is True
    assert (
        summary["ovphysx_effort_saturation"]["response_regime"]
        == "SATURATED_NONLINEAR"
    )
    assert summary["ovphysx_effort_saturation"][
        "linear_transfer_function_claimed"
    ] is False
    assert "Observed effort clip count: **37**" in report
    assert "SATURATED / NONLINEAR" in report
    assert "must not be described as a linear transfer function" in report


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "must be an object"),
        ("boolean_count", "must be an integer"),
        ("extra_conclusion", "fields are not exact"),
        ("wrong_observations", "frozen 16-case matrix"),
        ("impossible_count", "exceeds observed"),
    ),
)
def test_effort_clipping_input_cannot_self_report_or_omit_evidence(
    mutation: str,
    match: str,
) -> None:
    provenance = _provenance()
    observation = provenance["ovphysx_effort_clipping_observation"]
    assert isinstance(observation, dict)
    if mutation == "missing":
        del provenance["ovphysx_effort_clipping_observation"]
    elif mutation == "boolean_count":
        observation["effort_clip_count"] = True
    elif mutation == "extra_conclusion":
        observation["saturation_observed"] = False
    elif mutation == "wrong_observations":
        observation["effort_observation_count"] = 48_015
    else:
        observation["effort_clip_count"] = 1_056_353

    with pytest.raises(WaveformPublicReportError, match=match):
        build_public_summary(_comparison(), _descriptive_metrics(), provenance)


def test_render_rejects_forged_effort_saturation_interpretation() -> None:
    summary = build_public_summary(
        _comparison(), _descriptive_metrics(), _provenance()
    )
    summary["ovphysx_effort_saturation"]["response_regime"] = (
        "SATURATED_NONLINEAR"
    )

    with pytest.raises(WaveformPublicReportError, match="contradicts"):
        render_public_report(summary)


def test_threshold_or_aggregate_status_drift_is_rejected() -> None:
    threshold_drift = _comparison()
    threshold_drift["thresholds"]["crosssim_joint_max_abs_rad"] = 0.03
    with pytest.raises(WaveformPublicReportError, match="drifted"):
        build_public_summary(threshold_drift, _descriptive_metrics(), _provenance())

    status_drift = _comparison()
    status_drift["summary"]["comparison_status"] = "within_tolerance"
    with pytest.raises(WaveformPublicReportError, match="contradicts"):
        build_public_summary(status_drift, _descriptive_metrics(), _provenance())


def test_noncanonical_one_of_one_repeats_are_rejected() -> None:
    comparison = _comparison()
    execution = comparison["results"][0]["executions"][0]
    execution["requested_repeats"] = 1
    execution["completed_repeats"] = 1

    with pytest.raises(WaveformPublicReportError, match="request exactly two repeats"):
        build_public_summary(comparison, _descriptive_metrics(), _provenance())


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("repeat_joint_max_abs_rad", 2e-9),
        ("dt_halving_joint_max_abs_rad", 0.02),
    ),
)
def test_prerequisite_failure_cannot_self_report_divergent(
    metric: str, value: float
) -> None:
    comparison = _comparison("divergent")
    comparison["results"][0]["comparison"]["metrics"][metric] = value

    with pytest.raises(WaveformPublicReportError, match="derived=inconclusive"):
        build_public_summary(comparison, _descriptive_metrics(), _provenance())


def test_crosssim_threshold_failure_cannot_self_report_within_tolerance() -> None:
    comparison = _comparison("within_tolerance")
    comparison["results"][0]["comparison"]["metrics"][
        "crosssim_joint_max_abs_rad"
    ] = 0.03

    with pytest.raises(WaveformPublicReportError, match="derived=divergent"):
        build_public_summary(comparison, _descriptive_metrics(), _provenance())


def test_within_threshold_metrics_cannot_self_report_divergent() -> None:
    comparison = _comparison("divergent")
    comparison["results"][0]["comparison"]["metrics"][
        "crosssim_joint_max_abs_rad"
    ] = 0.01

    with pytest.raises(WaveformPublicReportError, match="derived=within_tolerance"):
        build_public_summary(comparison, _descriptive_metrics(), _provenance())


def test_prerequisite_failure_takes_priority_over_crosssim_divergence() -> None:
    comparison = _comparison("divergent")
    first_cell = comparison["results"][0]["comparison"]
    first_cell["metrics"]["repeat_joint_max_abs_rad"] = 2e-9
    first_cell["metrics"]["crosssim_joint_max_abs_rad"] = 0.03
    first_cell["comparison_status"] = "inconclusive"
    comparison["summary"]["comparison_status"] = "inconclusive"

    summary = build_public_summary(
        comparison, _descriptive_metrics(), _provenance()
    )

    assert summary["validation_status"] == "INVALID"
    assert summary["scientific_label"] == "INCONCLUSIVE"
    assert summary["scenario_results"][0]["comparison_status"] == "INCONCLUSIVE"


def test_completed_two_of_two_with_missing_metric_is_inconclusive() -> None:
    comparison = _comparison("within_tolerance")
    first_cell = comparison["results"][0]["comparison"]
    del first_cell["metrics"]["crosssim_joint_max_abs_rad"]
    first_cell["comparison_status"] = "inconclusive"
    comparison["summary"]["comparison_status"] = "inconclusive"

    summary = build_public_summary(
        comparison, _descriptive_metrics(), _provenance()
    )

    assert summary["validation_status"] == "INVALID"
    assert summary["scientific_label"] == "INCONCLUSIVE"
    assert (
        summary["scenario_results"][0]["metrics"][
            "crosssim_joint_max_abs_rad"
        ]
        is None
    )


@pytest.mark.parametrize("metric", tuple(FROZEN_THRESHOLDS)[1:])
def test_frozen_metric_equal_to_threshold_is_not_a_failure(metric: str) -> None:
    comparison = _comparison("within_tolerance")
    comparison["results"][0]["comparison"]["metrics"][metric] = (
        FROZEN_THRESHOLDS[metric]
    )

    summary = build_public_summary(
        comparison, _descriptive_metrics(), _provenance()
    )

    assert summary["scientific_label"] == "WITHIN_TOLERANCE"


def test_render_rejects_scope_or_claim_boundary_mutation() -> None:
    summary = build_public_summary(
        _comparison(), _descriptive_metrics(), _provenance()
    )
    bad_scope = deepcopy(summary)
    bad_scope["scope"]["full_isaac_sim"] = True
    with pytest.raises(WaveformPublicReportError, match="scope boundary"):
        render_public_report(bad_scope)

    bad_claim = deepcopy(summary)
    bad_claim["claim_boundary"]["hardware_claimed"] = True
    with pytest.raises(WaveformPublicReportError, match="claim boundary"):
        render_public_report(bad_claim)
