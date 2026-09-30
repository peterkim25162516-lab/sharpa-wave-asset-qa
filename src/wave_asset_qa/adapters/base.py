"""Backend-neutral contracts for WaveSimParity simulation adapters.

The adapter boundary intentionally stays smaller than either simulator API.
Adapters own simulator-specific objects and expose only lifecycle operations,
position targets, finite-state stepping, and portable trace samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class AdapterLifecycle(str, Enum):
    """Observable adapter lifecycle.

    ``CREATED`` and ``CLOSED`` may transition to ``READY`` through ``open``.
    ``READY`` may transition to ``RUNNING`` for a scenario and back to
    ``READY`` when it completes.  An unrecoverable backend error moves the
    adapter to ``FAILED``; callers must then ``close`` it before reuse.
    """

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    CLOSED = "closed"


class AdapterError(RuntimeError):
    """Base class for adapter configuration and execution failures."""


class AdapterLifecycleError(AdapterError):
    """Raised when an operation is invalid for the current lifecycle state."""


class AdapterCapabilityError(AdapterError):
    """Raised when the installed backend cannot safely perform an operation.

    Capability errors are not physics results and must not be reported as a
    cross-simulator divergence.  ``capability`` is stable enough for bundles
    and automation to classify the failure without parsing ``message``.
    """

    def __init__(
        self,
        backend: str,
        capability: str,
        message: str,
        *,
        hint: str | None = None,
    ) -> None:
        self.backend = backend
        self.capability = capability
        self.detail = message
        self.hint = hint
        rendered = f"{backend} capability '{capability}' is unavailable: {message}"
        if hint:
            rendered += f" Hint: {hint}"
        super().__init__(rendered)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "backend": self.backend,
            "capability": self.capability,
            "message": self.detail,
            "hint": self.hint,
        }


@dataclass(frozen=True, slots=True)
class TraceSample:
    """One portable state observation.

    Frame poses use ``(x, y, z, qw, qx, qy, qz)`` in the simulator world
    frame.  ``qpos`` and ``qvel`` preserve backend-native generalized-state
    order, while ``joint_positions`` provides the canonical named view.
    """

    step: int
    time_s: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    joint_positions: Mapping[str, float]
    frame_poses: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    position_targets: Mapping[str, float] = field(default_factory=dict)
    contact_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "time_s": self.time_s,
            "qpos": list(self.qpos),
            "qvel": list(self.qvel),
            "joint_positions": dict(self.joint_positions),
            "frame_poses": {name: list(pose) for name, pose in self.frame_poses.items()},
            "position_targets": dict(self.position_targets),
            "contact_count": self.contact_count,
        }


@dataclass(frozen=True, slots=True)
class AdapterRunResult:
    """Raw result from one backend execution.

    ``status`` is strictly an execution status (``completed`` or ``error``).
    Cross-simulator labels such as ``within_tolerance`` and ``divergent`` are
    deliberately absent and belong to the comparison layer.
    """

    backend: str
    scenario_id: str
    status: str
    message: str
    dt: float
    requested_steps: int
    completed_steps: int
    joint_names: tuple[str, ...] = ()
    frame_names: tuple[str, ...] = ()
    samples: tuple[TraceSample, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "error"}:
            raise ValueError("adapter run status must be 'completed' or 'error'")

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.completed_steps == self.requested_steps

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "scenario_id": self.scenario_id,
            "status": self.status,
            "message": self.message,
            "dt": self.dt,
            "requested_steps": self.requested_steps,
            "completed_steps": self.completed_steps,
            "joint_names": list(self.joint_names),
            "frame_names": list(self.frame_names),
            "samples": [sample.to_dict() for sample in self.samples],
            "provenance": dict(self.provenance),
            "error": dict(self.error) if self.error is not None else None,
        }


@dataclass(frozen=True, slots=True)
class AdapterProbeResult:
    """Structured environment/capability probe output."""

    backend: str
    mode: str
    status: str
    message: str
    capabilities: Mapping[str, bool] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable", "capability_error", "error"}:
            raise ValueError("unsupported adapter probe status")

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "status": self.status,
            "message": self.message,
            "capabilities": dict(self.capabilities),
            "provenance": dict(self.provenance),
            "error": dict(self.error) if self.error is not None else None,
        }


@runtime_checkable
class SimulationAdapter(Protocol):
    """Minimal lifecycle required by the parity runner."""

    @property
    def backend_name(self) -> str: ...

    @property
    def lifecycle(self) -> AdapterLifecycle: ...

    def open(self, manifest: object, *, dt_override: float | None = None) -> None: ...

    def reset(self) -> None: ...

    def set_position_targets(self, targets: Mapping[str, float]) -> None: ...

    def step(self) -> None: ...

    def sample(
        self,
        *,
        step: int,
        joint_names: tuple[str, ...] = (),
        frame_names: tuple[str, ...] = (),
    ) -> TraceSample: ...

    def run_scenario(
        self,
        manifest: object,
        scenario: object,
        *,
        dt_override: float | None = None,
    ) -> AdapterRunResult: ...

    def close(self) -> None: ...


__all__ = [
    "AdapterCapabilityError",
    "AdapterError",
    "AdapterLifecycle",
    "AdapterLifecycleError",
    "AdapterProbeResult",
    "AdapterRunResult",
    "SimulationAdapter",
    "TraceSample",
]
