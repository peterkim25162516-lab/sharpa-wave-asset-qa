"""Canonical-to-backend joint and frame mappings for simulator parity.

Mappings deliberately retain the coordinate conversion next to the backend
name and index.  This keeps result bundles interpretable without requiring the
adapter that originally produced them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeVar


class MappingValidationError(ValueError):
    """Raised when a canonical mapping is incomplete or ambiguous."""


MAPPING_SCHEMA_VERSION = 1
_RECORD_KEYS = frozenset(
    {
        "canonical_id",
        "backend",
        "scope",
        "backend_name",
        "index",
        "sign",
        "offset",
        "unit",
    }
)


def _validate_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MappingValidationError(f"{field} must be a non-empty, trimmed string")
    if any(ord(character) < 32 for character in value):
        raise MappingValidationError(f"{field} must not contain control characters")


def _validate_record(record: object) -> None:
    for field in ("canonical_id", "backend", "scope", "backend_name", "unit"):
        _validate_text(getattr(record, field), field)

    index = getattr(record, "index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise MappingValidationError("index must be a non-negative integer")

    sign = getattr(record, "sign")
    if isinstance(sign, bool) or not isinstance(sign, (int, float)):
        raise MappingValidationError("sign must be either -1 or 1")
    if not math.isfinite(float(sign)) or float(sign) not in {-1.0, 1.0}:
        raise MappingValidationError("sign must be either -1 or 1")

    offset = getattr(record, "offset")
    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
        raise MappingValidationError("offset must be a finite number")
    if not math.isfinite(float(offset)):
        raise MappingValidationError("offset must be a finite number")


def _record_to_dict(record: object) -> dict[str, object]:
    return {
        "canonical_id": getattr(record, "canonical_id"),
        "backend": getattr(record, "backend"),
        "scope": getattr(record, "scope"),
        "backend_name": getattr(record, "backend_name"),
        "index": getattr(record, "index"),
        "sign": float(getattr(record, "sign")),
        "offset": float(getattr(record, "offset")),
        "unit": getattr(record, "unit"),
    }


_MappingRecord = TypeVar("_MappingRecord", bound="JointMapping | FrameMapping")


def _record_from_dict(
    cls: type[_MappingRecord], data: Mapping[str, Any]
) -> _MappingRecord:
    if not isinstance(data, Mapping):
        raise MappingValidationError("mapping record must be an object")
    keys = set(data)
    missing = sorted(_RECORD_KEYS - keys)
    unknown = sorted(keys - _RECORD_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise MappingValidationError("invalid mapping record (" + "; ".join(details) + ")")
    return cls(
        canonical_id=data["canonical_id"],
        backend=data["backend"],
        scope=data["scope"],
        backend_name=data["backend_name"],
        index=data["index"],
        sign=data["sign"],
        offset=data["offset"],
        unit=data["unit"],
    )


@dataclass(frozen=True, slots=True)
class JointMapping:
    """One canonical joint binding in one simulator backend.

    Canonical position is ``sign * backend_position + offset`` and is expressed
    in ``unit``.  Gate 0 uses radians, while retaining the unit in every record
    makes that convention independently auditable.
    """

    canonical_id: str
    backend: str
    scope: str
    backend_name: str
    index: int
    sign: float
    offset: float
    unit: str

    def __post_init__(self) -> None:
        _validate_record(self)

    def to_dict(self) -> dict[str, object]:
        return _record_to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JointMapping":
        return _record_from_dict(cls, data)


@dataclass(frozen=True, slots=True)
class FrameMapping:
    """One canonical frame binding in one simulator backend.

    ``sign``, ``offset`` and ``unit`` are retained explicitly for the sampled
    scalar frame coordinate represented by this record.  Gate 0 frame identity
    mappings normally use ``sign=1`` and ``offset=0``.
    """

    canonical_id: str
    backend: str
    scope: str
    backend_name: str
    index: int
    sign: float
    offset: float
    unit: str

    def __post_init__(self) -> None:
        _validate_record(self)

    def to_dict(self) -> dict[str, object]:
        return _record_to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameMapping":
        return _record_from_dict(cls, data)


def _validate_expected_count(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MappingValidationError(f"{name} must be a non-negative integer")


def _validated_ids(values: Sequence[str], label: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise MappingValidationError(f"{label} must be a sequence of identifiers")
    validated: list[str] = []
    for value in values:
        _validate_text(value, label)
        validated.append(value)
    if len(validated) != len(set(validated)):
        raise MappingValidationError(f"{label} contains duplicate identifiers")
    return frozenset(validated)


def _normalize_expected_ids(
    expected: Sequence[str] | Mapping[str, Sequence[str]] | None,
    label: str,
) -> tuple[frozenset[str] | None, dict[str, frozenset[str]]]:
    if expected is None:
        return None, {}
    if isinstance(expected, Mapping):
        by_backend: dict[str, frozenset[str]] = {}
        for backend, identifiers in expected.items():
            _validate_text(backend, f"{label} backend")
            by_backend[backend] = _validated_ids(identifiers, f"{label}[{backend}]")
        return None, by_backend
    return _validated_ids(expected, label), {}


def _validate_entries(
    entries: Sequence[JointMapping] | Sequence[FrameMapping],
    entry_type: type[JointMapping] | type[FrameMapping],
    kind: str,
) -> dict[tuple[str, str], list[JointMapping | FrameMapping]]:
    if isinstance(entries, (str, bytes)):
        raise MappingValidationError(f"{kind} mappings must be a sequence")

    grouped: dict[tuple[str, str], list[JointMapping | FrameMapping]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, entry_type):
            raise MappingValidationError(
                f"{kind} mapping at position {position} must be {entry_type.__name__}"
            )
        grouped.setdefault((entry.backend, entry.scope), []).append(entry)

    for (backend, scope), backend_entries in sorted(grouped.items()):
        canonical: dict[str, int] = {}
        names: dict[str, str] = {}
        indices: dict[int, str] = {}
        for position, entry in enumerate(backend_entries):
            if entry.canonical_id in canonical:
                raise MappingValidationError(
                    f"duplicate {kind} canonical_id {entry.canonical_id!r} for "
                    f"backend {backend!r}, scope {scope!r}"
                )
            canonical[entry.canonical_id] = position
            if entry.backend_name in names:
                raise MappingValidationError(
                    f"duplicate {kind} backend_name {entry.backend_name!r} for "
                    f"backend {backend!r}, scope {scope!r}"
                )
            names[entry.backend_name] = entry.canonical_id
            if entry.index in indices:
                raise MappingValidationError(
                    f"{kind} index conflict for backend {backend!r}, scope {scope!r}: "
                    f"index {entry.index} maps both {indices[entry.index]!r} "
                    f"and {entry.canonical_id!r}"
                )
            indices[entry.index] = entry.canonical_id
    return grouped


def _check_coverage(
    *,
    kind: str,
    grouped: Mapping[str, Sequence[JointMapping | FrameMapping]],
    backends: Sequence[str],
    expected_count: int | None,
    common_expected_ids: frozenset[str] | None,
    expected_ids_by_backend: Mapping[str, frozenset[str]],
) -> None:
    observed_sets: dict[str, frozenset[str]] = {}
    for backend in backends:
        observed = frozenset(entry.canonical_id for entry in grouped.get(backend, ()))
        observed_sets[backend] = observed
        expected = expected_ids_by_backend.get(backend, common_expected_ids)

        if expected is not None:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            if missing or unexpected:
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if unexpected:
                    details.append("unexpected " + ", ".join(unexpected))
                raise MappingValidationError(
                    f"{kind} coverage mismatch for backend {backend!r}: " + "; ".join(details)
                )
        if expected_count is not None and len(observed) != expected_count:
            raise MappingValidationError(
                f"{kind} count mismatch for backend {backend!r}: "
                f"expected {expected_count}, found {len(observed)}"
            )

    if len(observed_sets) > 1 and common_expected_ids is None and not expected_ids_by_backend:
        reference_backend = sorted(observed_sets)[0]
        reference = observed_sets[reference_backend]
        for backend in sorted(observed_sets)[1:]:
            observed = observed_sets[backend]
            if observed != reference:
                missing = sorted(reference - observed)
                extra = sorted(observed - reference)
                raise MappingValidationError(
                    f"{kind} canonical coverage differs between backends "
                    f"{reference_backend!r} and {backend!r}: missing={missing}, extra={extra}"
                )

    units: dict[str, str] = {}
    for entries in grouped.values():
        for entry in entries:
            previous = units.setdefault(entry.canonical_id, entry.unit)
            if previous != entry.unit:
                raise MappingValidationError(
                    f"{kind} unit mismatch for canonical_id {entry.canonical_id!r}: "
                    f"{previous!r} versus {entry.unit!r}"
                )


def validate_mapping(
    joints: Sequence[JointMapping],
    frames: Sequence[FrameMapping],
    *,
    expected_joint_count: int | None = None,
    expected_frame_count: int | None = None,
    expected_joint_ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    expected_frame_ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    expected_backends: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate coverage and uniqueness, returning a JSON-safe summary.

    Expected counts are intentionally supplied by the caller rather than being
    tied to the current 44-joint/10-frame Gate 0 scope.  Expected identifiers
    can be a common sequence or a backend-to-sequence mapping.
    """

    _validate_expected_count(expected_joint_count, "expected_joint_count")
    _validate_expected_count(expected_frame_count, "expected_frame_count")
    common_joint_ids, joint_ids_by_backend = _normalize_expected_ids(
        expected_joint_ids, "expected_joint_ids"
    )
    common_frame_ids, frame_ids_by_backend = _normalize_expected_ids(
        expected_frame_ids, "expected_frame_ids"
    )

    if common_joint_ids is not None and expected_joint_count is not None:
        if len(common_joint_ids) != expected_joint_count:
            raise MappingValidationError(
                "expected_joint_count does not match expected_joint_ids length"
            )
    if common_frame_ids is not None and expected_frame_count is not None:
        if len(common_frame_ids) != expected_frame_count:
            raise MappingValidationError(
                "expected_frame_count does not match expected_frame_ids length"
            )

    joint_groups = _validate_entries(joints, JointMapping, "joint")
    frame_groups = _validate_entries(frames, FrameMapping, "frame")
    joint_groups_by_backend: dict[str, list[JointMapping | FrameMapping]] = {}
    frame_groups_by_backend: dict[str, list[JointMapping | FrameMapping]] = {}
    for (backend, _scope), entries in joint_groups.items():
        joint_groups_by_backend.setdefault(backend, []).extend(entries)
    for (backend, _scope), entries in frame_groups.items():
        frame_groups_by_backend.setdefault(backend, []).extend(entries)
    observed_backends = set(joint_groups_by_backend) | set(frame_groups_by_backend)
    observed_backends.update(joint_ids_by_backend)
    observed_backends.update(frame_ids_by_backend)

    if expected_backends is not None:
        expected_backend_set = _validated_ids(expected_backends, "expected_backends")
        missing_backends = sorted(expected_backend_set - observed_backends)
        unexpected_backends = sorted(observed_backends - expected_backend_set)
        if missing_backends or unexpected_backends:
            raise MappingValidationError(
                "backend coverage mismatch: "
                f"missing={missing_backends}, unexpected={unexpected_backends}"
            )
        observed_backends = set(expected_backend_set)

    backends = sorted(observed_backends)
    if not backends and any(
        value not in (None, 0) for value in (expected_joint_count, expected_frame_count)
    ):
        raise MappingValidationError("mapping has no backends")

    _check_coverage(
        kind="joint",
        grouped=joint_groups_by_backend,
        backends=backends,
        expected_count=expected_joint_count,
        common_expected_ids=common_joint_ids,
        expected_ids_by_backend=joint_ids_by_backend,
    )
    _check_coverage(
        kind="frame",
        grouped=frame_groups_by_backend,
        backends=backends,
        expected_count=expected_frame_count,
        common_expected_ids=common_frame_ids,
        expected_ids_by_backend=frame_ids_by_backend,
    )

    joint_ids = sorted({entry.canonical_id for entry in joints})
    frame_ids = sorted({entry.canonical_id for entry in frames})
    by_backend: dict[str, dict[str, Any]] = {}
    for backend in backends:
        backend_joint_ids = sorted(
            entry.canonical_id for entry in joint_groups_by_backend.get(backend, ())
        )
        backend_frame_ids = sorted(
            entry.canonical_id for entry in frame_groups_by_backend.get(backend, ())
        )
        scopes = sorted(
            {
                scope
                for candidate_backend, scope in (*joint_groups, *frame_groups)
                if candidate_backend == backend
            }
        )
        by_backend[backend] = {
            "joint_count": len(backend_joint_ids),
            "frame_count": len(backend_frame_ids),
            "joint_canonical_ids": backend_joint_ids,
            "frame_canonical_ids": backend_frame_ids,
            "by_scope": {
                scope: {
                    "joint_count": len(joint_groups.get((backend, scope), ())),
                    "frame_count": len(frame_groups.get((backend, scope), ())),
                }
                for scope in scopes
            },
        }

    return {
        "joint_count": len(joint_ids),
        "frame_count": len(frame_ids),
        "joint_canonical_ids": joint_ids,
        "frame_canonical_ids": frame_ids,
        "by_backend": by_backend,
    }


__all__ = [
    "FrameMapping",
    "JointMapping",
    "MAPPING_SCHEMA_VERSION",
    "MappingValidationError",
    "validate_mapping",
]
