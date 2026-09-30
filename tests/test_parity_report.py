from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from wave_asset_qa.parity.report import render_markdown, write_reports


REPORT = {
    "schema_version": 1,
    "metadata": {
        "manifest_id": "wavesimparity-gate0",
        "manifest_sha256": "a" * 64,
        "upstream_repository": "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git",
        "upstream_commit": "6eea427eb24189519f32b9f21674cd534d3f973c",
        "asset_git_tree": "bb00a9d5527b8a76de576ce876ebece67d8ffde1",
        "canonical_lf_asset_tree_sha256": "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad",
        "backend_pair": ["mujoco", "kit-less ovphysx"],
    },
    "summary": {
        "execution_status": "completed",
        "comparison_status": "divergent",
        "expected_joint_mapping_count": 44,
        "expected_distal_frame_mapping_count": 10,
        "observed_joint_mapping_count": 44,
        "observed_distal_frame_mapping_count": 10,
        "mapping_validation_complete": True,
        "expected_run_count": 32,
        "received_run_count": 32,
        "completed_run_count": 32,
        "completion_fraction": 1.0,
        "step_completion_fraction": 1.0,
        "completion_target_met": True,
        "nonfinite_run_count": 0,
        "finite": True,
    },
    "thresholds": {
        "minimum_completion_fraction": 1.0,
        "crosssim_joint_max_abs_rad": 0.02,
        "crosssim_frame_position_max_m": 0.002,
        "crosssim_frame_orientation_max_rad": 0.05,
        "repeat_joint_max_abs_rad": 1e-9,
        "repeat_frame_position_max_m": 1e-9,
        "repeat_frame_orientation_max_rad": 1e-9,
        "dt_halving_joint_max_abs_rad": 0.01,
        "dt_halving_frame_position_max_m": 0.001,
        "dt_halving_frame_orientation_max_rad": 0.02,
    },
    "results": [
        {
            "schema_version": 1,
            "manifest_id": "wavesimparity-gate0",
            "manifest_sha256": "a" * 64,
            "hand": "left",
            "scenario_id": "small_step",
            "executions": [
                {"simulator": "mujoco", "execution_status": "completed"},
                {"simulator": "ovphysx", "execution_status": "completed"},
            ],
            "comparison": {
                "comparison_status": "divergent",
                "metrics": {
                    "crosssim_joint_max_abs_rad": 0.03,
                    "crosssim_frame_position_max_m": 0.0001,
                    "repeat_joint_max_abs_rad": 0.0,
                    "repeat_frame_position_max_m": 0.0,
                    "repeat_frame_orientation_max_rad": 0.0,
                    "dt_halving_joint_max_abs_rad": 0.001,
                    "dt_halving_frame_position_max_m": 0.0001,
                    "dt_halving_frame_orientation_max_rad": 0.001,
                },
                "message": "Observed threshold exceedance; this is not an official bug finding.",
            },
        }
    ],
    "observations": [
        "left/small_step: observed cross-simulator threshold exceedance."
    ],
}


def test_markdown_states_scope_status_boundary_and_acceptance_checks() -> None:
    rendered = render_markdown(REPORT)

    assert rendered.startswith("# WaveSimParity Gate 0 report")
    assert "Unofficial, simulation-only" in rendered
    assert "MuJoCo ↔ kit-less OVPhysX" in rendered
    assert "not a full Isaac Sim" in rendered
    assert "Execution status:** `COMPLETED`" in rendered
    assert "Comparison status:** `DIVERGENT`" in rendered
    assert "44/44" in rendered
    assert "10/10" in rendered
    assert "100.00%" in rendered
    assert "not by itself an official" in rendered
    assert "no hardware behavior" in rendered


def test_report_preserves_execution_and_comparison_as_separate_columns() -> None:
    rendered = render_markdown(REPORT)

    assert "MuJoCo execution" in rendered
    assert "OVPhysX execution" in rendered
    assert "Comparison" in rendered
    assert "Frame orientation max (rad)" in rendered
    assert "small_step" in rendered
    assert "divergent" in rendered
    assert "not an official bug finding" in rendered


def test_write_reports_emits_stable_json_and_markdown(tmp_path: Path) -> None:
    json_path, markdown_path = write_reports(REPORT, tmp_path / "gate0")

    assert json.loads(json_path.read_text(encoding="utf-8")) == REPORT
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert markdown_path.read_text(encoding="utf-8").endswith("\n")
    assert "kit-less OVPhysX" in markdown_path.read_text(encoding="utf-8")


def test_incomplete_execution_cannot_show_mapping_or_finite_pass() -> None:
    incomplete = deepcopy(REPORT)
    incomplete["summary"].update(
        {
            "execution_status": "error",
            "comparison_status": "inconclusive",
            "received_run_count": 31,
            "completed_run_count": 31,
            "completion_fraction": 31 / 32,
            "step_completion_fraction": 31 / 32,
            "completion_target_met": False,
            "mapping_validation_complete": False,
        }
    )

    rendered = render_markdown(incomplete)
    mapping_line = next(
        line for line in rendered.splitlines() if "Canonical joint mapping" in line
    )
    finite_line = next(line for line in rendered.splitlines() if "Finite traces" in line)

    assert mapping_line.endswith("INCONCLUSIVE |")
    assert finite_line.endswith("INCONCLUSIVE |")
