"""Optional headless MuJoCo smoke and determinism checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True, slots=True)
class MuJoCoSmokeResult:
    source_path: str
    status: str
    message: str
    mujoco_version: str | None = None
    nq: int | None = None
    nv: int | None = None
    nu: int | None = None
    nbody: int | None = None
    njnt: int | None = None
    ngeom: int | None = None
    steps: int = 0
    simulated_seconds: float | None = None
    wall_seconds: float | None = None
    realtime_factor: float | None = None
    max_state_repeat_delta: float | None = None
    repeat_tolerance: float | None = None
    max_contacts: int | None = None
    minimum_contact_distance: float | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rollout(model: object, mujoco: object, steps: int) -> tuple[np.ndarray, int, float | None, float]:
    data = mujoco.MjData(model)
    maximum_contacts = 0
    minimum_distance: float | None = None
    started = perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, int(data.ncon))
        if data.ncon:
            current = min(float(data.contact[index].dist) for index in range(data.ncon))
            minimum_distance = current if minimum_distance is None else min(minimum_distance, current)
    elapsed = perf_counter() - started
    state = np.concatenate((np.asarray(data.qpos), np.asarray(data.qvel), np.asarray(data.act)))
    return state.copy(), maximum_contacts, minimum_distance, elapsed


def run_mujoco_smoke(
    model_path: str | Path,
    *,
    steps: int = 50,
    repeat_tolerance: float = 0.0,
) -> MuJoCoSmokeResult:
    """Load an MJCF model, step it twice and check finite deterministic state.

    This is deliberately a small CPU regression check.  It does not validate
    controller quality, contact realism, hardware behavior or Sim2Real.
    """

    source = Path(model_path).expanduser().resolve()
    if steps < 1:
        raise ValueError("steps must be at least one")
    if repeat_tolerance < 0.0:
        raise ValueError("repeat_tolerance must be non-negative")

    try:
        import mujoco
    except ImportError:
        return MuJoCoSmokeResult(
            source_path=str(source),
            status="skip",
            message="MuJoCo is not installed; install the 'simulation' extra.",
        )

    try:
        model = mujoco.MjModel.from_xml_path(str(source))
        first, contacts_a, distance_a, elapsed_a = _rollout(model, mujoco, steps)
        second, contacts_b, distance_b, elapsed_b = _rollout(model, mujoco, steps)
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise RuntimeError("non-finite qpos/qvel/act state after stepping")
        repeat_delta = float(np.max(np.abs(first - second))) if first.size else 0.0
        repeat_passed = repeat_delta <= repeat_tolerance
        simulated_seconds = float(model.opt.timestep) * steps
        elapsed = (elapsed_a + elapsed_b) * 0.5
        distances = [value for value in (distance_a, distance_b) if value is not None]
        return MuJoCoSmokeResult(
            source_path=str(source),
            status="pass" if repeat_passed else "fail",
            message=(
                "Model loaded and repeated headless rollouts remained finite and within the repeat tolerance."
                if repeat_passed
                else f"Repeated rollouts differ by {repeat_delta:.6g}, above tolerance {repeat_tolerance:.6g}."
            ),
            mujoco_version=getattr(mujoco, "__version__", None),
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
            nbody=int(model.nbody),
            njnt=int(model.njnt),
            ngeom=int(model.ngeom),
            steps=steps,
            simulated_seconds=simulated_seconds,
            wall_seconds=elapsed,
            realtime_factor=(simulated_seconds / elapsed) if elapsed > 0.0 else None,
            max_state_repeat_delta=repeat_delta,
            repeat_tolerance=repeat_tolerance,
            max_contacts=max(contacts_a, contacts_b),
            minimum_contact_distance=min(distances) if distances else None,
        )
    except Exception as error:  # MuJoCo exposes several parser/runtime exception types.
        return MuJoCoSmokeResult(
            source_path=str(source),
            status="fail",
            message=f"{type(error).__name__}: {error}",
            mujoco_version=getattr(mujoco, "__version__", None),
            steps=steps,
        )
