"""Load canonical manifests and expand them into deterministic run cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

from .contracts import (
    CanonicalScenarioSpec,
    HandSide,
    ParityManifest,
    ScenarioKind,
    ScenarioSpec,
    Simulator,
    WaveformScenarioSpec,
)


TARGET_SEQUENCE_DIGEST_SCHEMA_VERSION = 1


class TimestepVariant(str, Enum):
    BASE = "base"
    HALVED = "halved"


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    simulator: Simulator
    hand: HandSide
    scenario_id: str
    repeat_index: int
    timestep_variant: TimestepVariant
    dt_s: float
    model_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.simulator, Simulator):
            raise ValueError("case.simulator must be a Simulator")
        if not isinstance(self.hand, HandSide):
            raise ValueError("case.hand must be a HandSide")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("case.scenario_id must be a non-empty string")
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 1
        ):
            raise ValueError("case.repeat_index must be a positive integer")
        if not isinstance(self.timestep_variant, TimestepVariant):
            raise ValueError("case.timestep_variant must be a TimestepVariant")
        if (
            isinstance(self.dt_s, bool)
            or not isinstance(self.dt_s, (int, float))
            or not math.isfinite(float(self.dt_s))
            or self.dt_s <= 0.0
        ):
            raise ValueError("case.dt_s must be a positive finite number")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("case.model_path must be a non-empty string")

    @property
    def case_id(self) -> str:
        return (
            f"{self.simulator.value}.{self.hand.value}.{self.scenario_id}."
            f"{self.timestep_variant.value}.r{self.repeat_index:02d}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "simulator": self.simulator.value,
            "hand": self.hand.value,
            "scenario_id": self.scenario_id,
            "repeat_index": self.repeat_index,
            "timestep_variant": self.timestep_variant.value,
            "dt_s": self.dt_s,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> ScenarioCase:
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ValueError("case must be an object with string keys")
        required = {
            "case_id",
            "simulator",
            "hand",
            "scenario_id",
            "repeat_index",
            "timestep_variant",
            "dt_s",
            "model_path",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            raise ValueError("invalid case fields: " + "; ".join(detail))
        try:
            simulator = Simulator(value["simulator"])
            hand = HandSide(value["hand"])
            timestep_variant = TimestepVariant(value["timestep_variant"])
        except (TypeError, ValueError) as exc:
            raise ValueError("case contains an unsupported enum value") from exc
        for field_name in ("case_id", "scenario_id", "model_path"):
            if not isinstance(value[field_name], str) or not value[field_name]:
                raise ValueError(f"case.{field_name} must be a non-empty string")
        repeat_index = value["repeat_index"]
        dt_s = value["dt_s"]
        case = cls(
            simulator=simulator,
            hand=hand,
            scenario_id=value["scenario_id"],
            repeat_index=(
                repeat_index
                if isinstance(repeat_index, int) and not isinstance(repeat_index, bool)
                else 0
            ),
            timestep_variant=timestep_variant,
            dt_s=(
                float(dt_s)
                if isinstance(dt_s, (int, float)) and not isinstance(dt_s, bool)
                else math.nan
            ),
            model_path=value["model_path"],
        )
        if value["case_id"] != case.case_id:
            raise ValueError("case.case_id does not match its component fields")
        return case


def canonical_manifest_json(manifest: ParityManifest) -> str:
    """Serialize *manifest* with stable ordering for hashing and transport."""

    if not isinstance(manifest, ParityManifest):
        raise TypeError("manifest must be a ParityManifest")
    return json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def manifest_sha256(manifest: ParityManifest) -> str:
    return hashlib.sha256(canonical_manifest_json(manifest).encode("utf-8")).hexdigest()


def canonical_initial_positions(
    scenario: CanonicalScenarioSpec,
    joint_names: Sequence[str],
) -> dict[str, float]:
    """Return the manifest-owned canonical state applied before the t=0 sample."""

    if not isinstance(scenario, (ScenarioSpec, WaveformScenarioSpec)):
        raise TypeError("scenario must be a canonical scenario spec")
    names = tuple(joint_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("joint_names must not contain duplicates")
    return {name: float(scenario.initial_position_rad) for name in names}


def canonical_position_targets(
    scenario: CanonicalScenarioSpec,
    joint_names: Sequence[str],
    *,
    step_index: int,
    dt_s: float | None = None,
) -> dict[str, float]:
    """Return the exact all-joint position target for one recorded sample.

    ``step_index`` is the sample index: zero is the explicitly initialized
    state, and a small-step target is visible on the sample whose timestamp is
    exactly ``step_start_s``.  Passing the case timestep keeps the same
    physical onset in the dt-halved runs.
    """

    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    effective_dt = scenario.dt_s if dt_s is None else float(dt_s)
    if not math.isfinite(effective_dt) or effective_dt <= 0.0:
        raise ValueError("dt_s must be a positive finite number")
    duration_ratio = scenario.duration_s / effective_dt
    if not math.isclose(
        duration_ratio, round(duration_ratio), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("scenario duration must be an integer multiple of dt_s")
    if step_index > round(duration_ratio):
        raise ValueError("step_index exceeds the canonical scenario duration")

    targets = canonical_initial_positions(scenario, joint_names)
    if scenario.kind is ScenarioKind.SMALL_STEP:
        assert isinstance(scenario, ScenarioSpec)
        assert scenario.step_start_s is not None
        onset_ratio = scenario.step_start_s / effective_dt
        if not math.isclose(onset_ratio, round(onset_ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("small_step onset must be an integer multiple of dt_s")
        if step_index >= round(onset_ratio):
            commanded = scenario.initial_position_rad + scenario.target_position_rad
            targets = {name: float(commanded) for name in targets}
    elif isinstance(scenario, WaveformScenarioSpec):
        step_count = round(duration_ratio)
        # The protocol defines the command on physical time, not a normalized
        # sample index.  Pin the terminal timestamp to the declared duration
        # only to avoid a multiplication roundoff beyond the closed interval.
        time_s = (
            scenario.duration_s
            if step_index == step_count
            else step_index * effective_dt
        )
        frequency_slope_hz_s = (
            scenario.frequency_end_hz - scenario.frequency_start_hz
        ) / scenario.duration_s
        cycles = (
            scenario.frequency_start_hz * time_s
            + 0.5 * frequency_slope_hz_s * time_s * time_s
        )
        commanded = (
            scenario.command_offset_rad
            - scenario.command_amplitude_rad * math.cos(math.tau * cycles)
        )
        targets = {name: float(commanded) for name in targets}
    return targets


class CanonicalTargetSequenceDigest:
    """Incrementally hash canonical target vectors without retaining the sequence.

    The byte stream is UTF-8 canonical JSON Lines.  Its header binds schema v1
    and canonical joint order, each record binds its zero-based step and the
    binary32-rounded values rendered with ``float.hex``, and the non-destructive
    final trailer binds the record count.
    """

    __slots__ = ("_count", "_hasher", "_joint_names")

    def __init__(self, joint_names: Sequence[str]) -> None:
        names = tuple(joint_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("joint_names must contain non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("joint_names must not contain duplicates")
        self._joint_names = names
        self._count = 0
        self._hasher = hashlib.sha256()
        self._update(
            {
                "joint_names": list(names),
                "projection": "ieee754_binary32_roundtrip",
                "schema_version": TARGET_SEQUENCE_DIGEST_SCHEMA_VERSION,
            }
        )

    @staticmethod
    def _encoded_line(payload: Mapping[str, object]) -> bytes:
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def _update(self, payload: Mapping[str, object]) -> None:
        self._hasher.update(self._encoded_line(payload))

    @property
    def count(self) -> int:
        return self._count

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def append(self, targets: Mapping[str, float]) -> None:
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping")
        if set(targets) != set(self._joint_names):
            raise ValueError("targets must exactly cover the canonical joint names")
        values_hex: list[str] = []
        for name in self._joint_names:
            value = targets[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"target for {name!r} must be a finite number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"target for {name!r} must be a finite number")
            try:
                projected = struct.unpack("!f", struct.pack("!f", numeric))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(
                    f"target for {name!r} is outside finite binary32 range"
                ) from exc
            if not math.isfinite(projected):
                raise ValueError(
                    f"target for {name!r} is outside finite binary32 range"
                )
            values_hex.append(float(projected).hex())
        self._update(
            {
                "step_index": self._count,
                "values_hex": values_hex,
            }
        )
        self._count += 1

    def copy(self) -> CanonicalTargetSequenceDigest:
        clone = object.__new__(type(self))
        clone._joint_names = self._joint_names
        clone._count = self._count
        clone._hasher = self._hasher.copy()
        return clone

    def digest(self) -> bytes:
        final = self._hasher.copy()
        final.update(self._encoded_line({"record_count": self._count}))
        return final.digest()

    def hexdigest(self) -> str:
        final = self._hasher.copy()
        final.update(self._encoded_line({"record_count": self._count}))
        return final.hexdigest()


def canonical_target_sequence_sha256(
    scenario: CanonicalScenarioSpec,
    joint_names: Sequence[str],
    *,
    dt_s: float | None = None,
    include_terminal: bool = True,
) -> str:
    """Hash q[0..N] or the pre-step prefix q[0..N-1] for one case."""

    if not isinstance(include_terminal, bool):
        raise TypeError("include_terminal must be a boolean")
    effective_dt = scenario.dt_s if dt_s is None else float(dt_s)
    if not math.isfinite(effective_dt) or effective_dt <= 0.0:
        raise ValueError("dt_s must be a positive finite number")
    ratio = scenario.duration_s / effective_dt
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("scenario duration must be an integer multiple of dt_s")
    step_count = round(ratio)
    digest = CanonicalTargetSequenceDigest(joint_names)
    stop = step_count + 1 if include_terminal else step_count
    for step_index in range(stop):
        digest.append(
            canonical_position_targets(
                scenario,
                joint_names,
                step_index=step_index,
                dt_s=effective_dt,
            )
        )
    return digest.hexdigest()


def load_manifest(path: str | Path) -> ParityManifest:
    """Read and strictly validate a parity manifest from JSON."""

    source = Path(path)
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest: {source}: {exc}") from exc
    return ParityManifest.from_dict(value)


def expand_scenario_cases(manifest: ParityManifest) -> tuple[ScenarioCase, ...]:
    """Return the complete run matrix in stable simulator/hand/scenario order."""

    cases: list[ScenarioCase] = []
    halved_ids = set(manifest.run_policy.dt_halving_scenario_ids)
    for simulator in manifest.simulators:
        for hand in manifest.hands:
            model_path = hand.model_paths.for_simulator(simulator)
            for scenario in manifest.scenarios:
                variants = [TimestepVariant.BASE]
                if scenario.scenario_id in halved_ids:
                    variants.append(TimestepVariant.HALVED)
                for variant in variants:
                    dt_s = (
                        scenario.dt_s
                        if variant is TimestepVariant.BASE
                        else scenario.dt_s / 2.0
                    )
                    for repeat_index in range(1, manifest.run_policy.repeat_count + 1):
                        cases.append(
                            ScenarioCase(
                                simulator=simulator,
                                hand=hand.side,
                                scenario_id=scenario.scenario_id,
                                repeat_index=repeat_index,
                                timestep_variant=variant,
                                dt_s=dt_s,
                                model_path=model_path,
                            )
                        )
    return tuple(cases)


__all__ = [
    "ScenarioCase",
    "CanonicalTargetSequenceDigest",
    "TARGET_SEQUENCE_DIGEST_SCHEMA_VERSION",
    "TimestepVariant",
    "canonical_initial_positions",
    "canonical_manifest_json",
    "canonical_position_targets",
    "canonical_target_sequence_sha256",
    "expand_scenario_cases",
    "load_manifest",
    "manifest_sha256",
]
