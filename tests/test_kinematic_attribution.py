from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest

from wave_asset_qa.parity.kinematic_attribution import (
    CLOSES,
    DOES_NOT_CLOSE,
    DOMINANT_POSITIVE_CLOSER,
    KinematicAttributionError,
    MATERIAL_ADVERSE,
    MIXED,
    NON_DOMINANT,
    analyze_kinematic_attribution,
    evaluate_closure,
)


def _scalar_rows(*values: float) -> np.ndarray:
    result = np.zeros((len(values), 3), dtype=float)
    result[:, 0] = values
    return result


def _additive_coalitions(
    raw: np.ndarray,
    player_fractions: tuple[float, float, float, float, float],
) -> np.ndarray:
    result = np.zeros((32, *raw.shape), dtype=float)
    for mask in range(32):
        fraction = sum(
            player_fractions[player]
            for player in range(5)
            if mask & (1 << player)
        )
        result[mask] = fraction * raw
    return result


def _vector_additive_coalitions(
    player_displacements: np.ndarray,
) -> np.ndarray:
    assert player_displacements.shape[0] == 5
    result = np.zeros((32, *player_displacements.shape[1:]), dtype=float)
    for mask in range(32):
        for player in range(5):
            if mask & (1 << player):
                result[mask] += player_displacements[player]
    return result


def test_closure_uses_window_and_earliest_raw_peak_rules() -> None:
    raw = _scalar_rows(2.0, 2.0, 1.0)
    closes = evaluate_closure(raw, 0.95 * raw)

    assert closes.rho_window == pytest.approx(0.05)
    assert closes.rho_peak == pytest.approx(0.05)
    assert closes.peak_index == 0
    assert closes.status == CLOSES
    assert closes.explained_energy_fraction == pytest.approx(0.9975)

    does_not_close = evaluate_closure(raw, 0.40 * raw)
    assert does_not_close.rho_window == pytest.approx(0.60)
    assert does_not_close.rho_peak == pytest.approx(0.60)
    assert does_not_close.status == DOES_NOT_CLOSE


def test_closure_is_mixed_when_window_and_peak_rules_disagree() -> None:
    raw = _scalar_rows(1.0, 1.0)
    fk = _scalar_rows(0.95, 0.20)

    result = evaluate_closure(raw, fk)

    assert result.rho_window > 0.50
    assert result.rho_peak == pytest.approx(0.05)
    assert result.status == MIXED


def test_closure_thresholds_are_inclusive() -> None:
    raw = _scalar_rows(1.0)

    assert evaluate_closure(raw, 0.90 * raw).status == CLOSES
    assert evaluate_closure(raw, 0.50 * raw).status == DOES_NOT_CLOSE

    boundary_raw = _scalar_rows(1.0083245936759398e-9)
    direct = evaluate_closure(boundary_raw, 0.9 * boundary_raw)
    assert direct.rho_window > 0.10
    assert direct.status == MIXED


def test_exact_shapley_is_symmetric_and_efficient() -> None:
    raw = _scalar_rows(1.0, 0.5)
    coalitions = _additive_coalitions(raw, (0.2,) * 5)

    result = analyze_kinematic_attribution(raw, raw, coalitions)

    assert result.closure.status == CLOSES
    assert result.full_coalition_value == pytest.approx(1.0)
    assert result.efficiency_error <= 1.0e-12
    assert result.vector_efficiency_max_error_m <= 1.0e-12
    assert sum(player.normalized_attribution for player in result.players) == (
        pytest.approx(1.0)
    )
    for player in result.players:
        assert player.normalized_attribution == pytest.approx(0.2)
        assert player.positive_share == pytest.approx(0.2)
        assert player.label == NON_DOMINANT


def test_vector_shapley_preserves_player_order_and_per_sample_efficiency() -> None:
    player_displacements = np.asarray(
        [
            [[0.10, 0.00, 0.00], [0.01, 0.00, 0.00]],
            [[0.00, 0.20, 0.00], [0.00, 0.02, 0.00]],
            [[0.00, 0.00, 0.30], [0.00, 0.00, 0.03]],
            [[0.04, 0.05, 0.00], [0.004, 0.005, 0.00]],
            [[-0.01, 0.00, 0.02], [-0.001, 0.00, 0.002]],
        ],
        dtype=float,
    )
    coalitions = _vector_additive_coalitions(player_displacements)
    fk = np.sum(player_displacements, axis=0)
    raw = fk.copy()

    result = analyze_kinematic_attribution(raw, fk, coalitions)

    observed = np.asarray(result.vector_attributions_m)
    assert observed.shape == (5, 2, 3)
    assert observed == pytest.approx(player_displacements)
    assert np.sum(observed, axis=0) == pytest.approx(
        coalitions[-1] - coalitions[0]
    )
    assert result.vector_efficiency_max_error_m <= 1.0e-12


def test_exact_shapley_factorial_weights_cover_non_additive_unanimity_game() -> None:
    raw = _scalar_rows(1.0)
    coalitions = np.zeros((32, 1, 3), dtype=float)
    for mask in range(32):
        if mask & 0b00011 == 0b00011:
            coalitions[mask] = 0.5 * raw

    result = analyze_kinematic_attribution(raw, 0.5 * raw, coalitions)

    assert result.players[0].normalized_attribution == pytest.approx(0.375)
    assert result.players[1].normalized_attribution == pytest.approx(0.375)
    assert all(
        player.normalized_attribution == pytest.approx(0.0)
        for player in result.players[2:]
    )
    assert result.efficiency_error <= 1.0e-12


def test_vector_shapley_factorial_weights_cover_non_additive_unanimity_game() -> None:
    endpoint = np.asarray(
        [[0.1, -0.2, 0.3], [-0.04, 0.05, 0.06]],
        dtype=float,
    )
    coalitions = np.zeros((32, 2, 3), dtype=float)
    for mask in range(32):
        if mask & 0b01010 == 0b01010:
            coalitions[mask] = endpoint

    result = analyze_kinematic_attribution(endpoint, endpoint, coalitions)
    observed = np.asarray(result.vector_attributions_m)

    assert observed[1] == pytest.approx(0.5 * endpoint)
    assert observed[3] == pytest.approx(0.5 * endpoint)
    assert observed[[0, 2, 4]] == pytest.approx(0.0)
    assert result.vector_efficiency_max_error_m <= 1.0e-12


def test_attribution_result_is_finite_json_payload_after_asdict() -> None:
    raw = _scalar_rows(1.0, 0.5)
    coalitions = _additive_coalitions(raw, (0.2,) * 5)
    result = analyze_kinematic_attribution(raw, raw, coalitions)

    encoded = json.dumps(asdict(result), allow_nan=False, sort_keys=True)

    assert '"vector_attributions_m"' in encoded
    assert '"vector_efficiency_max_error_m"' in encoded


def test_exact_shapley_labels_dominant_positive_closer() -> None:
    raw = _scalar_rows(1.0, 0.5)
    coalitions = _additive_coalitions(raw, (1.0, 0.0, 0.0, 0.0, 0.0))

    result = analyze_kinematic_attribution(raw, raw, coalitions)

    assert result.players[0].normalized_attribution == pytest.approx(1.0)
    assert result.players[0].positive_share == pytest.approx(1.0)
    assert result.players[0].label == DOMINANT_POSITIVE_CLOSER
    assert all(player.label == NON_DOMINANT for player in result.players[1:])


def test_exact_shapley_preserves_signed_adverse_attribution() -> None:
    raw = _scalar_rows(1.0, 0.5)
    coalitions = _additive_coalitions(raw, (1.0, -0.2, 0.0, 0.0, 0.0))
    fk = 0.8 * raw

    result = analyze_kinematic_attribution(raw, fk, coalitions)

    assert result.full_coalition_value == pytest.approx(0.96)
    assert result.players[0].normalized_attribution == pytest.approx(1.2)
    assert result.players[0].label == DOMINANT_POSITIVE_CLOSER
    assert result.players[1].normalized_attribution == pytest.approx(-0.24)
    assert result.players[1].label == MATERIAL_ADVERSE
    assert sum(player.normalized_attribution for player in result.players) == (
        pytest.approx(result.closure.explained_energy_fraction)
    )


@pytest.mark.parametrize(
    ("raw", "fk", "match"),
    [
        (np.zeros((0, 3)), np.zeros((0, 3)), "shape"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "shape"),
        (np.zeros((2, 3)), np.zeros((3, 3)), "identical shapes"),
        (
            np.asarray([[np.nan, 0.0, 0.0]]),
            np.zeros((1, 3)),
            "finite",
        ),
        (np.ones((1, 3), dtype=bool), np.ones((1, 3)), "booleans"),
        (np.zeros((2, 3)), np.zeros((2, 3)), "greater than"),
    ],
)
def test_closure_rejects_invalid_arrays(
    raw: np.ndarray,
    fk: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(KinematicAttributionError, match=match):
        evaluate_closure(raw, fk)


def test_closure_wraps_ragged_input_as_attribution_error() -> None:
    with pytest.raises(KinematicAttributionError, match="rectangular"):
        evaluate_closure([[1.0, 0.0, 0.0], [1.0]], np.ones((2, 3)))


def test_closure_rejects_boolean_hidden_in_mixed_python_input() -> None:
    with pytest.raises(KinematicAttributionError, match="booleans"):
        evaluate_closure([[True, 0.0, 0.0]], np.ones((1, 3)))


def test_attribution_rejects_bad_coalition_shape_and_nonfinite_values() -> None:
    raw = _scalar_rows(1.0)
    with pytest.raises(KinematicAttributionError, match="shape"):
        analyze_kinematic_attribution(raw, raw, np.zeros((31, 1, 3)))

    coalitions = _additive_coalitions(raw, (0.2,) * 5)
    coalitions[7, 0, 0] = np.inf
    with pytest.raises(KinematicAttributionError, match="finite"):
        analyze_kinematic_attribution(raw, raw, coalitions)


def test_attribution_rejects_empty_and_full_endpoint_mismatches() -> None:
    raw = _scalar_rows(1.0)
    coalitions = _additive_coalitions(raw, (0.2,) * 5)
    coalitions[0, 0, 0] = 2.0e-12
    with pytest.raises(KinematicAttributionError, match="empty coalition"):
        analyze_kinematic_attribution(raw, raw, coalitions)

    coalitions = _additive_coalitions(raw, (0.2,) * 5)
    coalitions[-1, 0, 0] += 2.0e-10
    with pytest.raises(KinematicAttributionError, match="full coalition"):
        analyze_kinematic_attribution(raw, raw, coalitions)


def test_attribution_fails_closed_when_tolerated_endpoint_breaks_efficiency() -> None:
    raw = _scalar_rows(1.0e-6)
    coalitions = _additive_coalitions(raw, (0.2,) * 5)
    coalitions[-1, 0, 0] += 2.0e-12

    with pytest.raises(
        KinematicAttributionError,
        match="full coalition closure value",
    ):
        analyze_kinematic_attribution(raw, raw, coalitions)
