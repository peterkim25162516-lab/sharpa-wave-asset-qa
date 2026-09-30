"""Canonical command generation and deterministic matrix expansion for C0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from wave_asset_qa.parity.contracts import HandSide, JOINT_SUFFIXES, Simulator

from .contracts import (
    ContactCondition,
    ContactManifest,
    ContactScenarioSpec,
    SyntheticFixtureSpec,
)


class ContactTimestepVariant(str, Enum):
    BASE = "base"
    HALVED = "halved"


@dataclass(frozen=True, slots=True)
class ContactCase:
    simulator: Simulator
    hand: HandSide
    scenario_id: str
    condition: ContactCondition
    timestep_variant: ContactTimestepVariant
    repeat_index: int
    dt_s: float
    model_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.simulator, Simulator):
            raise ValueError("case.simulator must be a Simulator")
        if not isinstance(self.hand, HandSide):
            raise ValueError("case.hand must be a HandSide")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("case.scenario_id must be a non-empty string")
        if not isinstance(self.condition, ContactCondition):
            raise ValueError("case.condition must be a ContactCondition")
        if not isinstance(self.timestep_variant, ContactTimestepVariant):
            raise ValueError(
                "case.timestep_variant must be a ContactTimestepVariant"
            )
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 1
        ):
            raise ValueError("case.repeat_index must be a positive integer")
        if (
            isinstance(self.dt_s, bool)
            or not isinstance(self.dt_s, (int, float))
            or not math.isfinite(float(self.dt_s))
            or float(self.dt_s) <= 0.0
        ):
            raise ValueError("case.dt_s must be a positive finite number")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("case.model_path must be a non-empty string")

    @property
    def pair_collision_enabled(self) -> bool:
        return self.condition is ContactCondition.CONTACT

    @property
    def case_id(self) -> str:
        return (
            f"{self.simulator.value}.{self.hand.value}.{self.scenario_id}."
            f"{self.condition.value}.{self.timestep_variant.value}."
            f"r{self.repeat_index:02d}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "simulator": self.simulator.value,
            "hand": self.hand.value,
            "scenario_id": self.scenario_id,
            "condition": self.condition.value,
            "timestep_variant": self.timestep_variant.value,
            "repeat_index": self.repeat_index,
            "dt_s": self.dt_s,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> ContactCase:
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise ValueError("case must be an object with string keys")
        required = {
            "case_id",
            "simulator",
            "hand",
            "scenario_id",
            "condition",
            "timestep_variant",
            "repeat_index",
            "dt_s",
            "model_path",
        }
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("invalid case fields: " + "; ".join(details))
        try:
            simulator = Simulator(value["simulator"])
            hand = HandSide(value["hand"])
            condition = ContactCondition(value["condition"])
            variant = ContactTimestepVariant(value["timestep_variant"])
        except (TypeError, ValueError) as exc:
            raise ValueError("case contains an unsupported enum value") from exc
        for name in ("case_id", "scenario_id", "model_path"):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError(f"case.{name} must be a non-empty string")
        repeat_index = value["repeat_index"]
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            raise ValueError("case.repeat_index must be an integer")
        dt_s = value["dt_s"]
        if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)):
            raise ValueError("case.dt_s must be a number")
        case = cls(
            simulator=simulator,
            hand=hand,
            scenario_id=value["scenario_id"],
            condition=condition,
            timestep_variant=variant,
            repeat_index=repeat_index,
            dt_s=float(dt_s),
            model_path=value["model_path"],
        )
        if value["case_id"] != case.case_id:
            raise ValueError("case.case_id does not match its component fields")
        return case


def canonical_contact_manifest_json(manifest: ContactManifest) -> str:
    """Return canonical UTF-8-safe JSON text for semantic identity hashing."""

    if not isinstance(manifest, ContactManifest):
        raise TypeError("manifest must be a ContactManifest")
    return json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def contact_manifest_sha256(manifest: ContactManifest) -> str:
    return hashlib.sha256(
        canonical_contact_manifest_json(manifest).encode("utf-8")
    ).hexdigest()


def load_contact_manifest(path: str | Path) -> ContactManifest:
    """Load a JSON manifest with exact-key and frozen-profile validation."""

    source = Path(path)
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON contact manifest: {source}: {exc}") from exc
    return ContactManifest.from_dict(value)


def quintic_smoothstep(value: float) -> float:
    """Evaluate ``10u^3 - 15u^4 + 6u^5`` on the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("smoothstep input must be a number")
    u = float(value)
    if not math.isfinite(u) or not 0.0 <= u <= 1.0:
        raise ValueError("smoothstep input must be finite and within [0, 1]")
    if u == 0.0 or u == 1.0:
        return u
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def canonical_contact_scale(scenario: ContactScenarioSpec, time_s: float) -> float:
    """Return the frozen press/hold/retract scalar at physical time ``time_s``."""

    if not isinstance(scenario, ContactScenarioSpec):
        raise TypeError("scenario must be a ContactScenarioSpec")
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
        raise ValueError("time_s must be a number")
    time_value = float(time_s)
    if (
        not math.isfinite(time_value)
        or time_value < 0.0
        or time_value > scenario.duration_s
    ):
        raise ValueError("time_s must be finite and within the scenario duration")
    schedule = scenario.schedule
    if time_value <= schedule.settle_end_s:
        return 0.0
    if time_value < schedule.press_end_s:
        u = (time_value - schedule.settle_end_s) / (
            schedule.press_end_s - schedule.settle_end_s
        )
        return quintic_smoothstep(u)
    if time_value <= schedule.hold_end_s:
        return 1.0
    if time_value < schedule.retract_end_s:
        u = (time_value - schedule.hold_end_s) / (
            schedule.retract_end_s - schedule.hold_end_s
        )
        return 1.0 - quintic_smoothstep(u)
    return 0.0


def _canonical_joint_side(joint_names: Sequence[str]) -> HandSide:
    names = tuple(joint_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("joint_names must not contain duplicates")
    for side in HandSide:
        expected = tuple(f"{side.value}_{suffix}" for suffix in JOINT_SUFFIXES)
        if names == expected:
            return side
    raise ValueError(
        "joint_names must be the 22 canonical joints for exactly one hand in MJCF order"
    )


def canonical_contact_initial_positions(
    scenario: ContactScenarioSpec,
    joint_names: Sequence[str],
) -> dict[str, float]:
    if not isinstance(scenario, ContactScenarioSpec):
        raise TypeError("scenario must be a ContactScenarioSpec")
    _canonical_joint_side(joint_names)
    return {name: float(scenario.initial_position_rad) for name in joint_names}


def canonical_contact_position_targets(
    scenario: ContactScenarioSpec,
    joint_names: Sequence[str],
    *,
    step_index: int,
    dt_s: float | None = None,
) -> dict[str, float]:
    """Return the complete 22-joint target for recorded sample ``step_index``.

    ``target[k]`` becomes active over ``[t_k, t_(k+1))``.  The state recorded
    at ``t_k`` is the observation after the preceding interval and has not yet
    advanced under this newly recorded target.
    """

    if not isinstance(scenario, ContactScenarioSpec):
        raise TypeError("scenario must be a ContactScenarioSpec")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    effective_dt = scenario.dt_s if dt_s is None else dt_s
    if (
        isinstance(effective_dt, bool)
        or not isinstance(effective_dt, (int, float))
        or not math.isfinite(float(effective_dt))
        or float(effective_dt) <= 0.0
    ):
        raise ValueError("dt_s must be a positive finite number")
    effective_dt = float(effective_dt)
    duration_ratio = scenario.duration_s / effective_dt
    if not math.isclose(
        duration_ratio, round(duration_ratio), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("scenario duration must be an integer multiple of dt_s")
    step_count = round(duration_ratio)
    if step_index > step_count:
        raise ValueError("step_index exceeds the canonical scenario duration")
    side = _canonical_joint_side(joint_names)
    time_s = (
        scenario.duration_s
        if step_index == step_count
        else step_index * effective_dt
    )
    scale = canonical_contact_scale(scenario, time_s)
    targets = canonical_contact_initial_positions(scenario, joint_names)
    frozen = scenario.driven_joint_targets_rad.to_dict()
    for suffix, target in frozen.items():
        targets[f"{side.value}_{suffix}"] = scale * float(target)
    return targets


def analytic_signed_gap_m(
    probe_center_world_m: Sequence[float],
    fixture: SyntheticFixtureSpec,
) -> float:
    """Compute the portable sphere-to-top-plane signed gap.

    Positive means analytic separation, zero means tangency, and negative
    means analytic overlap.  No backend-native contact distance is used.
    """

    if not isinstance(fixture, SyntheticFixtureSpec):
        raise TypeError("fixture must be a SyntheticFixtureSpec")
    if (
        not isinstance(probe_center_world_m, Sequence)
        or isinstance(probe_center_world_m, (str, bytes))
        or len(probe_center_world_m) != 3
    ):
        raise ValueError("probe_center_world_m must contain three numbers")
    center: list[float] = []
    for index, item in enumerate(probe_center_world_m):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"probe_center_world_m[{index}] must be a finite number"
            )
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(
                f"probe_center_world_m[{index}] must be a finite number"
            )
        center.append(numeric)
    return (
        center[2]
        - fixture.target.top_surface_z_m
        - fixture.probe.radius_m
    )


def debounce_interval_count(dt_s: float, debounce_window_s: float = 0.004) -> int:
    """Return consecutive physics intervals required by the 4 ms debounce."""

    for value, name in ((dt_s, "dt_s"), (debounce_window_s, "debounce_window_s")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be a positive finite number")
    ratio = float(debounce_window_s) / float(dt_s)
    rounded = round(ratio)
    if not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("debounce_window_s must be an integer multiple of dt_s")
    return rounded


def half_open_sample_indices(
    scenario: ContactScenarioSpec,
    *,
    start_s: float,
    end_s: float,
    dt_s: float | None = None,
) -> range:
    """Return sample indices whose timestamps lie in ``[start_s, end_s)``."""

    if not isinstance(scenario, ContactScenarioSpec):
        raise TypeError("scenario must be a ContactScenarioSpec")
    effective_dt = scenario.dt_s if dt_s is None else float(dt_s)
    if not math.isfinite(effective_dt) or effective_dt <= 0.0:
        raise ValueError("dt_s must be a positive finite number")
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise ValueError("window bounds must be numbers")
    start = float(start_s)
    end = float(end_s)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("window bounds must be finite")
    if not 0.0 <= start < end <= scenario.duration_s:
        raise ValueError("window must lie within the scenario duration")
    ratios = (start / effective_dt, end / effective_dt)
    if any(
        not math.isclose(item, round(item), rel_tol=0.0, abs_tol=1e-9)
        for item in ratios
    ):
        raise ValueError("window bounds must align to dt_s")
    return range(round(ratios[0]), round(ratios[1]))


def steady_hold_sample_indices(
    scenario: ContactScenarioSpec,
    *,
    dt_s: float | None = None,
) -> range:
    return half_open_sample_indices(
        scenario, start_s=3.25, end_s=3.75, dt_s=dt_s
    )


def recovery_sample_indices(
    scenario: ContactScenarioSpec,
    *,
    dt_s: float | None = None,
) -> range:
    return half_open_sample_indices(
        scenario, start_s=6.75, end_s=7.0, dt_s=dt_s
    )


def expand_contact_cases(manifest: ContactManifest) -> tuple[ContactCase, ...]:
    """Expand the frozen 32-case matrix in stable canonical order."""

    if not isinstance(manifest, ContactManifest):
        raise TypeError("manifest must be a ContactManifest")
    cases: list[ContactCase] = []
    for simulator in manifest.simulators:
        for hand in manifest.hands:
            model_path = hand.model_paths.for_simulator(simulator)
            for condition in manifest.conditions:
                for variant in (
                    ContactTimestepVariant.BASE,
                    ContactTimestepVariant.HALVED,
                ):
                    dt_s = (
                        manifest.scenario.dt_s
                        if variant is ContactTimestepVariant.BASE
                        else manifest.scenario.dt_s / 2.0
                    )
                    for repeat_index in range(
                        1, manifest.run_policy.repeat_count + 1
                    ):
                        cases.append(
                            ContactCase(
                                simulator=simulator,
                                hand=hand.side,
                                scenario_id=manifest.scenario.scenario_id,
                                condition=condition.condition_id,
                                timestep_variant=variant,
                                repeat_index=repeat_index,
                                dt_s=dt_s,
                                model_path=model_path,
                            )
                        )
    result = tuple(cases)
    if len(result) != manifest.expected_case_count or len(
        {case.case_id for case in result}
    ) != manifest.expected_case_count:
        raise RuntimeError("expanded contact matrix is incomplete or contains duplicate IDs")
    return result


# Short aliases are intentionally provided for orchestration code while the
# contact-qualified names remain the unambiguous public API.
canonical_manifest_json = canonical_contact_manifest_json
canonical_position_targets = canonical_contact_position_targets
load_manifest = load_contact_manifest
manifest_sha256 = contact_manifest_sha256


__all__ = [
    "ContactCase",
    "ContactTimestepVariant",
    "analytic_signed_gap_m",
    "canonical_contact_initial_positions",
    "canonical_contact_manifest_json",
    "canonical_contact_position_targets",
    "canonical_contact_scale",
    "canonical_manifest_json",
    "canonical_position_targets",
    "contact_manifest_sha256",
    "debounce_interval_count",
    "expand_contact_cases",
    "half_open_sample_indices",
    "load_contact_manifest",
    "load_manifest",
    "manifest_sha256",
    "quintic_smoothstep",
    "recovery_sample_indices",
    "steady_hold_sample_indices",
]
