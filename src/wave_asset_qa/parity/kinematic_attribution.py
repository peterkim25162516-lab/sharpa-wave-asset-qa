"""Fail-closed mathematics for a bounded common-kinematics attribution.

The caller is responsible for producing Cartesian displacement arrays from
frozen traces and one common FK implementation.  This module deliberately
does not load assets, simulator results, or provenance.  It only evaluates the
pre-registered closure ratios and a five-player exact Shapley decomposition.

Coalitions use integer bit-mask order: row ``mask`` contains the displacement
obtained after replacing every player whose bit is set.  With five players the
required coalition array therefore has shape ``(32, sample_count, 3)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, hypot, isfinite
from typing import Final

import numpy as np


PLAYER_COUNT: Final = 5
COALITION_COUNT: Final = 1 << PLAYER_COUNT
RAW_NORM_MIN_M: Final = 1.0e-12
EMPTY_COALITION_ATOL_M: Final = 1.0e-12
FULL_COALITION_ATOL_M: Final = 1.0e-10
SHAPLEY_EFFICIENCY_ATOL: Final = 1.0e-12
VECTOR_EFFICIENCY_ATOL_M: Final = 1.0e-12

CLOSE_RHO_MAX: Final = 0.10
DOES_NOT_CLOSE_RHO_MIN: Final = 0.50
DOMINANT_MIN_NORMALIZED_ATTRIBUTION: Final = 0.25
DOMINANT_MIN_POSITIVE_SHARE: Final = 0.50
MATERIAL_ADVERSE_MAX_NORMALIZED_ATTRIBUTION: Final = -0.10

CLOSES: Final = "CLOSES"
MIXED: Final = "MIXED"
DOES_NOT_CLOSE: Final = "DOES_NOT_CLOSE"

DOMINANT_POSITIVE_CLOSER: Final = "DOMINANT_POSITIVE_CLOSER"
MATERIAL_ADVERSE: Final = "MATERIAL_ADVERSE"
NON_DOMINANT: Final = "NON_DOMINANT"


class KinematicAttributionError(ValueError):
    """Raised when attribution inputs or derived metrics fail validation."""


@dataclass(frozen=True, slots=True)
class ClosureResult:
    """Window and raw-peak closure ratios for one trace comparison."""

    rho_window: float
    rho_peak: float
    peak_index: int
    status: str
    raw_norm_m: float
    residual_norm_m: float
    raw_peak_norm_m: float
    residual_peak_norm_m: float
    explained_energy_fraction: float


@dataclass(frozen=True, slots=True)
class PlayerAttribution:
    """One player's signed exact-Shapley share of raw squared-gap closure."""

    player_index: int
    energy_attribution_m2: float
    normalized_attribution: float
    positive_share: float
    label: str


@dataclass(frozen=True, slots=True)
class KinematicAttributionResult:
    """Combined closure decision and exact five-player attribution."""

    closure: ClosureResult
    players: tuple[PlayerAttribution, ...]
    raw_energy_m2: float
    empty_coalition_value: float
    full_coalition_value: float
    efficiency_error: float
    vector_attributions_m: tuple[
        tuple[tuple[float, float, float], ...], ...
    ]
    vector_efficiency_max_error_m: float


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype.kind == "O":
            return any(_contains_boolean(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _displacements(value: object, *, name: str) -> np.ndarray:
    if _contains_boolean(value):
        raise KinematicAttributionError(f"{name} must not contain booleans")
    try:
        raw = np.asarray(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise KinematicAttributionError(
            f"{name} must be a rectangular real numeric array"
        ) from exc
    if raw.dtype.kind == "b":
        raise KinematicAttributionError(f"{name} must not contain booleans")
    if raw.dtype.kind not in "iuf":
        raise KinematicAttributionError(f"{name} must be a real numeric array")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise KinematicAttributionError(
            f"{name} must be convertible to finite float64 values"
        ) from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 3:
        raise KinematicAttributionError(
            f"{name} must have shape (sample_count, 3) with sample_count > 0"
        )
    if not np.all(np.isfinite(array)):
        raise KinematicAttributionError(f"{name} must contain only finite values")
    return array


def _coalitions(value: object, *, sample_count: int) -> np.ndarray:
    if _contains_boolean(value):
        raise KinematicAttributionError(
            "coalition_displacements must not contain booleans"
        )
    try:
        raw = np.asarray(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise KinematicAttributionError(
            "coalition_displacements must be a rectangular real numeric array"
        ) from exc
    if raw.dtype.kind == "b":
        raise KinematicAttributionError(
            "coalition_displacements must not contain booleans"
        )
    if raw.dtype.kind not in "iuf":
        raise KinematicAttributionError(
            "coalition_displacements must be a real numeric array"
        )
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise KinematicAttributionError(
            "coalition_displacements must be convertible to finite float64 values"
        ) from exc
    expected = (COALITION_COUNT, sample_count, 3)
    if array.shape != expected:
        raise KinematicAttributionError(
            f"coalition_displacements must have shape {expected}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise KinematicAttributionError(
            "coalition_displacements must contain only finite values"
        )
    return array


def _norm(values: np.ndarray, *, name: str) -> float:
    result = hypot(*(float(value) for value in values.ravel()))
    if not isfinite(result):
        raise KinematicAttributionError(f"derived {name} is not finite")
    return result


def _row_norms(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(
        [hypot(*(float(value) for value in row)) for row in values],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise KinematicAttributionError(f"derived {name} are not finite")
    return result


def _square(value: float, *, name: str) -> float:
    result = value * value
    if not isfinite(result):
        raise KinematicAttributionError(f"derived {name} is not finite")
    return result


def _closure_arrays(
    raw_displacements: object,
    fk_displacements: object,
) -> tuple[np.ndarray, np.ndarray, ClosureResult]:
    raw = _displacements(raw_displacements, name="raw_displacements")
    fk = _displacements(fk_displacements, name="fk_displacements")
    if fk.shape != raw.shape:
        raise KinematicAttributionError(
            "raw_displacements and fk_displacements must have identical shapes"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        residual = raw - fk
    if not np.all(np.isfinite(residual)):
        raise KinematicAttributionError(
            "raw minus FK displacement produced non-finite values"
        )
    raw_norm = _norm(raw, name="raw window norm")
    if raw_norm <= RAW_NORM_MIN_M:
        raise KinematicAttributionError(
            f"raw window norm must be greater than {RAW_NORM_MIN_M:g} m"
        )
    residual_norm = _norm(residual, name="closure residual window norm")
    rho_window = residual_norm / raw_norm

    row_norms = _row_norms(raw, name="raw row norms")
    peak_index = int(np.argmax(row_norms))
    raw_peak_norm = float(row_norms[peak_index])
    if raw_peak_norm <= RAW_NORM_MIN_M:
        raise KinematicAttributionError(
            f"raw peak norm must be greater than {RAW_NORM_MIN_M:g} m"
        )
    residual_peak_norm = _norm(
        residual[peak_index], name="closure residual raw-peak norm"
    )
    rho_peak = residual_peak_norm / raw_peak_norm

    if not all(isfinite(value) for value in (rho_window, rho_peak)):
        raise KinematicAttributionError("derived closure ratio is not finite")
    if rho_window <= CLOSE_RHO_MAX and rho_peak <= CLOSE_RHO_MAX:
        status = CLOSES
    elif (
        rho_window >= DOES_NOT_CLOSE_RHO_MIN
        and rho_peak >= DOES_NOT_CLOSE_RHO_MIN
    ):
        status = DOES_NOT_CLOSE
    else:
        status = MIXED

    explained_energy_fraction = 1.0 - rho_window * rho_window
    if not isfinite(explained_energy_fraction):
        raise KinematicAttributionError(
            "derived explained energy fraction is not finite"
        )
    return raw, fk, ClosureResult(
        rho_window=rho_window,
        rho_peak=rho_peak,
        peak_index=peak_index,
        status=status,
        raw_norm_m=raw_norm,
        residual_norm_m=residual_norm,
        raw_peak_norm_m=raw_peak_norm,
        residual_peak_norm_m=residual_peak_norm,
        explained_energy_fraction=explained_energy_fraction,
    )


def evaluate_closure(
    raw_displacements: object,
    fk_displacements: object,
) -> ClosureResult:
    """Evaluate the fixed two-threshold closure rule.

    Rows are ordered samples and columns are XYZ in metres.  The raw peak is
    the earliest row attaining the largest Euclidean raw gap.
    """

    return _closure_arrays(raw_displacements, fk_displacements)[2]


def _normalized_coalition_values(
    raw: np.ndarray,
    coalitions: np.ndarray,
    *,
    raw_energy: float,
) -> np.ndarray:
    values = np.empty(COALITION_COUNT, dtype=np.float64)
    for mask in range(COALITION_COUNT):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = raw - coalitions[mask]
        if not np.all(np.isfinite(residual)):
            raise KinematicAttributionError(
                f"raw minus coalition {mask} displacement produced non-finite values"
            )
        residual_norm = _norm(
            residual,
            name=f"coalition {mask} residual norm",
        )
        residual_energy = _square(
            residual_norm,
            name=f"coalition {mask} residual energy",
        )
        values[mask] = 1.0 - residual_energy / raw_energy
    if not np.all(np.isfinite(values)):
        raise KinematicAttributionError(
            "derived normalized coalition values are not finite"
        )
    return values


def analyze_kinematic_attribution(
    raw_displacements: object,
    fk_displacements: object,
    coalition_displacements: object,
) -> KinematicAttributionResult:
    """Evaluate closure and an exact five-player Shapley attribution.

    ``raw_displacements`` and ``fk_displacements`` have shape ``(N, 3)``.
    ``coalition_displacements`` has shape ``(32, N, 3)`` in bit-mask order.
    Normalized Shapley values are signed fractional contributions to the raw
    squared-gap energy closed by common FK; multiply by 100 for percentage
    points.
    """

    raw, fk, closure = _closure_arrays(raw_displacements, fk_displacements)
    coalitions = _coalitions(
        coalition_displacements,
        sample_count=raw.shape[0],
    )

    empty_error = float(
        np.max(_row_norms(coalitions[0], name="empty coalition row norms"))
    )
    if not isfinite(empty_error) or empty_error > EMPTY_COALITION_ATOL_M:
        raise KinematicAttributionError(
            "empty coalition displacement exceeds the fixed absolute tolerance"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        endpoint_residual = coalitions[-1] - fk
    if not np.all(np.isfinite(endpoint_residual)):
        raise KinematicAttributionError(
            "full coalition minus FK displacement produced non-finite values"
        )
    endpoint_error = float(
        np.max(_row_norms(endpoint_residual, name="full coalition endpoint row norms"))
    )
    if not isfinite(endpoint_error) or endpoint_error > FULL_COALITION_ATOL_M:
        raise KinematicAttributionError(
            "full coalition displacement does not reproduce FK displacement"
        )

    raw_energy = _square(closure.raw_norm_m, name="raw window energy")
    values = _normalized_coalition_values(
        raw,
        coalitions,
        raw_energy=raw_energy,
    )
    if abs(float(values[0])) > SHAPLEY_EFFICIENCY_ATOL:
        raise KinematicAttributionError(
            "empty coalition has non-zero normalized closure value"
        )
    if (
        abs(float(values[-1]) - closure.explained_energy_fraction)
        > SHAPLEY_EFFICIENCY_ATOL
    ):
        raise KinematicAttributionError(
            "full coalition closure value differs from the FK closure metric"
        )

    normalized = np.zeros(PLAYER_COUNT, dtype=np.float64)
    vector_attributions = np.zeros(
        (PLAYER_COUNT, raw.shape[0], 3),
        dtype=np.float64,
    )
    denominator = factorial(PLAYER_COUNT)
    for player in range(PLAYER_COUNT):
        bit = 1 << player
        for mask in range(COALITION_COUNT):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = (
                factorial(size)
                * factorial(PLAYER_COUNT - size - 1)
                / denominator
            )
            normalized[player] += weight * (
                values[mask | bit] - values[mask]
            )
            vector_attributions[player] += weight * (
                coalitions[mask | bit] - coalitions[mask]
            )
    if not np.all(np.isfinite(normalized)):
        raise KinematicAttributionError(
            "derived normalized Shapley attribution is not finite"
        )
    if not np.all(np.isfinite(vector_attributions)):
        raise KinematicAttributionError(
            "derived vector Shapley attribution is not finite"
        )

    efficiency_target = float(values[-1] - values[0])
    efficiency_error = abs(
        float(np.sum(normalized)) - efficiency_target
    )
    if not isfinite(efficiency_error) or efficiency_error > SHAPLEY_EFFICIENCY_ATOL:
        raise KinematicAttributionError(
            "exact Shapley attribution failed its efficiency identity"
        )
    registered_identity_error = abs(
        float(np.sum(normalized)) - closure.explained_energy_fraction
    )
    if (
        not isfinite(registered_identity_error)
        or registered_identity_error > SHAPLEY_EFFICIENCY_ATOL
    ):
        raise KinematicAttributionError(
            "exact Shapley attribution does not sum to the registered closure identity"
        )

    vector_target = coalitions[-1] - coalitions[0]
    vector_efficiency_residual = (
        np.sum(vector_attributions, axis=0) - vector_target
    )
    if not np.all(np.isfinite(vector_efficiency_residual)):
        raise KinematicAttributionError(
            "derived vector Shapley efficiency residual is not finite"
        )
    vector_efficiency_max_error = float(
        np.max(
            _row_norms(
                vector_efficiency_residual,
                name="vector Shapley efficiency row norms",
            )
        )
    )
    if vector_efficiency_max_error > VECTOR_EFFICIENCY_ATOL_M:
        raise KinematicAttributionError(
            "vector Shapley attribution failed its per-sample efficiency identity"
        )

    positive_total = float(np.sum(np.maximum(normalized, 0.0)))
    if not isfinite(positive_total):
        raise KinematicAttributionError(
            "derived positive Shapley attribution total is not finite"
        )
    players: list[PlayerAttribution] = []
    for index, attribution in enumerate(normalized):
        normalized_attribution = float(attribution)
        positive_share = (
            max(normalized_attribution, 0.0) / positive_total
            if positive_total > 0.0
            else 0.0
        )
        if (
            normalized_attribution >= DOMINANT_MIN_NORMALIZED_ATTRIBUTION
            and positive_share >= DOMINANT_MIN_POSITIVE_SHARE
        ):
            label = DOMINANT_POSITIVE_CLOSER
        elif (
            normalized_attribution
            <= MATERIAL_ADVERSE_MAX_NORMALIZED_ATTRIBUTION
        ):
            label = MATERIAL_ADVERSE
        else:
            label = NON_DOMINANT
        energy_attribution = normalized_attribution * raw_energy
        if not all(
            isfinite(value)
            for value in (positive_share, energy_attribution)
        ):
            raise KinematicAttributionError(
                f"derived player {index} attribution is not finite"
            )
        players.append(
            PlayerAttribution(
                player_index=index,
                energy_attribution_m2=energy_attribution,
                normalized_attribution=normalized_attribution,
                positive_share=positive_share,
                label=label,
            )
        )

    return KinematicAttributionResult(
        closure=closure,
        players=tuple(players),
        raw_energy_m2=raw_energy,
        empty_coalition_value=float(values[0]),
        full_coalition_value=float(values[-1]),
        efficiency_error=efficiency_error,
        vector_attributions_m=tuple(
            tuple(
                tuple(float(coordinate) for coordinate in sample)
                for sample in player
            )
            for player in vector_attributions
        ),
        vector_efficiency_max_error_m=vector_efficiency_max_error,
    )


__all__ = [
    "CLOSE_RHO_MAX",
    "CLOSES",
    "DOES_NOT_CLOSE_RHO_MIN",
    "DOES_NOT_CLOSE",
    "DOMINANT_MIN_NORMALIZED_ATTRIBUTION",
    "DOMINANT_MIN_POSITIVE_SHARE",
    "DOMINANT_POSITIVE_CLOSER",
    "EMPTY_COALITION_ATOL_M",
    "FULL_COALITION_ATOL_M",
    "KinematicAttributionError",
    "KinematicAttributionResult",
    "MATERIAL_ADVERSE_MAX_NORMALIZED_ATTRIBUTION",
    "MATERIAL_ADVERSE",
    "MIXED",
    "NON_DOMINANT",
    "RAW_NORM_MIN_M",
    "SHAPLEY_EFFICIENCY_ATOL",
    "VECTOR_EFFICIENCY_ATOL_M",
    "ClosureResult",
    "PlayerAttribution",
    "analyze_kinematic_attribution",
    "evaluate_closure",
]
