from copy import deepcopy
import json
from pathlib import Path

import pytest

from wave_asset_qa.parity.contracts import (
    ComparisonRecord,
    ComparisonStatus,
    ExecutionRecord,
    ExecutionStatus,
    HandSide,
    ParityManifest,
    ParityResult,
    ScenarioKind,
    Simulator,
)
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    TimestepVariant,
    canonical_initial_positions,
    canonical_manifest_json,
    canonical_position_targets,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "gate0.json"


def _payload() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_gate0_manifest_is_strict_and_round_trips_stably() -> None:
    payload = _payload()
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest.to_dict() == payload
    assert ParityManifest.from_dict(manifest.to_dict()) == manifest
    assert canonical_manifest_json(manifest) == canonical_manifest_json(
        ParityManifest.from_dict(payload)
    )
    assert len(manifest_sha256(manifest)) == 64


def test_gate0_manifest_pins_scope_and_canonical_mapping_order() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest.provenance.commit == "6eea427eb24189519f32b9f21674cd534d3f973c"
    assert manifest.provenance.asset_git_tree == (
        "bb00a9d5527b8a76de576ce876ebece67d8ffde1"
    )
    assert manifest.provenance.canonical_lf_asset_tree_sha256 == (
        "b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad"
    )
    assert manifest.expected_joint_mapping_count == 44
    assert manifest.expected_distal_frame_mapping_count == 10
    assert manifest.hands[0].joint_names[:5] == (
        "left_thumb_CMC_FE",
        "left_thumb_CMC_AA",
        "left_thumb_MCP_FE",
        "left_thumb_MCP_AA",
        "left_thumb_IP",
    )
    assert manifest.hands[0].distal_frame_names == (
        "left_index_DP",
        "left_middle_DP",
        "left_pinky_DP",
        "left_ring_DP",
        "left_thumb_DP",
    )
    assert all(hand.mounting.value == "fixed_base" for hand in manifest.hands)
    assert all(hand.control_mode.value == "position" for hand in manifest.hands)
    assert tuple(hand.model_paths.ovphysx for hand in manifest.hands) == (
        "wave_01/left_sharpa_wave/left_sharpa_wave.usda",
        "wave_01/right_sharpa_wave/right_sharpa_wave.usda",
    )
    assert all(scenario.initial_position_rad == 0.0 for scenario in manifest.scenarios)


def test_gate0_manifest_includes_required_scenarios_repeat_and_dt_halving() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    assert tuple(scenario.kind for scenario in manifest.scenarios) == (
        ScenarioKind.ZERO_HOLD,
        ScenarioKind.SMALL_STEP,
        ScenarioKind.GRAVITY_SETTLING,
    )
    assert manifest.run_policy.repeat_count == 2
    assert manifest.run_policy.dt_halving_scenario_ids == ("small_step",)
    assert manifest.run_policy.minimum_completion_fraction == 1.0


def test_run_matrix_is_complete_and_deterministic() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    cases = expand_scenario_cases(manifest)

    # Per simulator and hand: 3 base scenarios * 2 repeats plus 2 halved-dt repeats.
    assert len(cases) == 2 * 2 * 8
    assert len({case.case_id for case in cases}) == len(cases)
    assert cases[0].case_id == "mujoco.left.zero_hold.base.r01"
    assert cases[-1].case_id == "ovphysx.right.gravity_settling.base.r02"
    halved = [case for case in cases if case.timestep_variant is TimestepVariant.HALVED]
    assert len(halved) == 2 * 2 * 2
    assert all(case.scenario_id == "small_step" and case.dt_s == 0.001 for case in halved)
    assert all(case.model_path.endswith((".xml", ".usda")) for case in cases)
    assert ScenarioCase.from_dict(cases[0].to_dict()) == cases[0]


def test_canonical_initial_state_and_target_trajectory_cover_base_and_halved_dt() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    hand = manifest.hands[0]
    scenario = manifest.scenario("small_step")

    assert set(canonical_initial_positions(scenario, hand.joint_names).values()) == {
        0.0
    }
    assert set(
        canonical_position_targets(
            scenario, hand.joint_names, step_index=49, dt_s=0.002
        ).values()
    ) == {0.0}
    assert set(
        canonical_position_targets(
            scenario, hand.joint_names, step_index=50, dt_s=0.002
        ).values()
    ) == {0.05}
    assert set(
        canonical_position_targets(
            scenario, hand.joint_names, step_index=99, dt_s=0.001
        ).values()
    ) == {0.0}
    assert set(
        canonical_position_targets(
            scenario, hand.joint_names, step_index=100, dt_s=0.001
        ).values()
    ) == {0.05}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update({"surprise": True}), "unknown field"),
        (lambda data: data["hands"][0]["joint_names"].pop(), "22 canonical"),
        (
            lambda data: data["hands"][0]["distal_frame_names"].reverse(),
            "five canonical",
        ),
        (lambda data: data["scenarios"].pop(), "exactly one"),
        (
            lambda data: data["run_policy"].update(
                {"dt_halving_scenario_ids": ["missing"]}
            ),
            "unknown scenario",
        ),
        (
            lambda data: data["scenarios"][1].update({"step_start_s": 0.1001}),
            "align",
        ),
        (
            lambda data: data["scenarios"][0].update(
                {"initial_position_rad": 0.01}
            ),
            "initial_position_rad must be zero",
        ),
    ],
)
def test_manifest_rejects_invalid_or_ambiguous_inputs(mutate, match: str) -> None:
    payload = deepcopy(_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        ParityManifest.from_dict(payload)


def test_execution_and_comparison_statuses_are_separate_contracts() -> None:
    complete = ExecutionRecord(
        simulator=Simulator.MUJOCO,
        execution_status=ExecutionStatus.COMPLETED,
        completed_repeats=2,
        requested_repeats=2,
        finite=True,
        bundle_root_sha256="a" * 64,
    )
    remote_error = ExecutionRecord(
        simulator=Simulator.OVPHYSX,
        execution_status=ExecutionStatus.ERROR,
        completed_repeats=1,
        requested_repeats=2,
        finite=False,
        message="runtime stopped before the second repeat",
    )
    comparison = ComparisonRecord(
        comparison_status=ComparisonStatus.INCONCLUSIVE,
        message="both simulator traces are required",
    )
    result = ParityResult(
        schema_version=1,
        manifest_id="wavesimparity-gate0",
        manifest_sha256="b" * 64,
        hand=HandSide.LEFT,
        scenario_id="small_step",
        executions=(complete, remote_error),
        comparison=comparison,
    )

    assert result.to_dict()["executions"][0]["execution_status"] == "completed"
    assert result.to_dict()["comparison"]["comparison_status"] == "inconclusive"
    assert ParityResult.from_dict(result.to_dict()) == result

    with pytest.raises(ValueError, match="inconclusive"):
        ParityResult(
            schema_version=1,
            manifest_id="wavesimparity-gate0",
            manifest_sha256="b" * 64,
            hand=HandSide.LEFT,
            scenario_id="small_step",
            executions=(complete, remote_error),
            comparison=ComparisonRecord(
                comparison_status=ComparisonStatus.DIVERGENT,
                message="difference observed",
            ),
        )


def test_published_schemas_are_valid_json_and_closed_at_root() -> None:
    for name in ("parity-manifest.schema.json", "parity-result.schema.json"):
        schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
