from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from wave_asset_qa.contact.scenarios import (
    ContactCase,
    analytic_signed_gap_m,
    canonical_contact_position_targets,
    canonical_contact_scale,
    debounce_interval_count,
    expand_contact_cases,
    half_open_sample_indices,
    load_contact_manifest,
    quintic_smoothstep,
    recovery_sample_indices,
    steady_hold_sample_indices,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"


@pytest.mark.parametrize(
    ("time_s", "expected"),
    [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.75, 0.5),
        (3.0, 1.0),
        (3.5, 1.0),
        (4.0, 1.0),
        (5.25, 0.5),
        (6.5, 0.0),
        (7.0, 0.0),
    ],
)
def test_frozen_seven_second_scale(time_s: float, expected: float) -> None:
    scenario = load_contact_manifest(MANIFEST_PATH).scenario
    assert canonical_contact_scale(scenario, time_s) == pytest.approx(
        expected, abs=1e-15
    )


def test_quintic_profile_is_bounded_monotonic_and_endpoint_smooth() -> None:
    values = [quintic_smoothstep(index / 1000.0) for index in range(1001)]

    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(left <= right for left, right in zip(values, values[1:]))
    epsilon = 1e-5
    assert quintic_smoothstep(epsilon) / epsilon < 1e-7
    assert (1.0 - quintic_smoothstep(1.0 - epsilon)) / epsilon < 1e-7


def test_targets_cover_all_joints_and_are_dt_invariant_at_common_times() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    scenario = manifest.scenario
    joints = manifest.hands[0].joint_names

    for base_step in range(scenario.steps + 1):
        base = canonical_contact_position_targets(
            scenario, joints, step_index=base_step, dt_s=0.002
        )
        halved = canonical_contact_position_targets(
            scenario, joints, step_index=2 * base_step, dt_s=0.001
        )
        assert base == halved
        assert tuple(base) == joints
        assert len(base) == 22

    initial = canonical_contact_position_targets(
        scenario, joints, step_index=0, dt_s=0.002
    )
    hold = canonical_contact_position_targets(
        scenario, joints, step_index=1500, dt_s=0.002
    )
    terminal = canonical_contact_position_targets(
        scenario, joints, step_index=3500, dt_s=0.002
    )
    assert set(initial.values()) == {0.0}
    assert set(terminal.values()) == {0.0}
    assert hold["left_index_MCP_FE"] == 0.60
    assert hold["left_index_MCP_AA"] == 0.0
    assert hold["left_index_PIP"] == 0.90
    assert hold["left_index_DIP"] == 0.60
    assert sum(value != 0.0 for value in hold.values()) == 3


def test_right_hand_targets_use_right_canonical_names() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    targets = canonical_contact_position_targets(
        manifest.scenario,
        manifest.hands[1].joint_names,
        step_index=1500,
    )
    assert targets["right_index_MCP_FE"] == 0.60
    assert targets["right_index_PIP"] == 0.90
    assert targets["right_index_DIP"] == 0.60
    assert all(not name.startswith("left_") for name in targets)


def test_gap_semantics_are_backend_independent() -> None:
    fixture = load_contact_manifest(MANIFEST_PATH).fixture

    assert analytic_signed_gap_m((0.0, 0.0, 0.200), fixture) == pytest.approx(
        0.065
    )
    assert analytic_signed_gap_m((0.0, 0.0, 0.135), fixture) == pytest.approx(
        0.0, abs=1e-15
    )
    assert analytic_signed_gap_m((0.0, 0.0, 0.125), fixture) == pytest.approx(
        -0.010
    )


def test_debounce_and_metric_windows_are_physical_time_invariant() -> None:
    scenario = load_contact_manifest(MANIFEST_PATH).scenario

    assert debounce_interval_count(0.002) == 2
    assert debounce_interval_count(0.001) == 4
    assert tuple(steady_hold_sample_indices(scenario, dt_s=0.002)) == tuple(
        range(1625, 1875)
    )
    assert tuple(steady_hold_sample_indices(scenario, dt_s=0.001)) == tuple(
        range(3250, 3750)
    )
    assert tuple(recovery_sample_indices(scenario, dt_s=0.002)) == tuple(
        range(3375, 3500)
    )
    assert tuple(recovery_sample_indices(scenario, dt_s=0.001)) == tuple(
        range(6750, 7000)
    )


def test_contact_case_round_trip_and_identity_are_fail_closed() -> None:
    case = expand_contact_cases(load_contact_manifest(MANIFEST_PATH))[0]
    payload = case.to_dict()

    assert ContactCase.from_dict(payload) == case
    bad_id = deepcopy(payload)
    bad_id["case_id"] = "mujoco.left.press_hold_release.contact.base.r02"
    with pytest.raises(ValueError, match="does not match"):
        ContactCase.from_dict(bad_id)
    extra = deepcopy(payload)
    extra["unknown"] = True
    with pytest.raises(ValueError, match="unknown unknown"):
        ContactCase.from_dict(extra)


@pytest.mark.parametrize(
    "operation",
    [
        lambda scenario, joints: canonical_contact_scale(scenario, -0.001),
        lambda scenario, joints: canonical_contact_scale(scenario, 7.001),
        lambda scenario, joints: canonical_contact_position_targets(
            scenario, joints, step_index=-1
        ),
        lambda scenario, joints: canonical_contact_position_targets(
            scenario, joints, step_index=3501
        ),
        lambda scenario, joints: canonical_contact_position_targets(
            scenario, joints[:-1], step_index=0
        ),
        lambda scenario, joints: debounce_interval_count(0.003),
        lambda scenario, joints: half_open_sample_indices(
            scenario, start_s=3.251, end_s=3.75
        ),
    ],
)
def test_scenario_helpers_reject_ambiguous_or_out_of_contract_inputs(operation) -> None:  # type: ignore[no-untyped-def]
    manifest = load_contact_manifest(MANIFEST_PATH)
    with pytest.raises(ValueError):
        operation(manifest.scenario, manifest.hands[0].joint_names)
