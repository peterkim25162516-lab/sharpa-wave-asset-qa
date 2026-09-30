"""Deterministic local execution and JSON I/O for WaveSimParity cases.

The runner deliberately executes one case per adapter instance.  Callers that
need process isolation (notably kit-less OVPhysX) can select a single case ID
and launch this module in a fresh process without changing the result format.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import uuid

from wave_asset_qa.adapters.base import AdapterRunResult, SimulationAdapter, TraceSample
from wave_asset_qa.adapters.mujoco import MuJoCoAdapter

from .bundle import write_json_atomic
from .compare import CollectedRun
from .contracts import ParityManifest, Simulator
from .scenarios import (
    ScenarioCase,
    canonical_manifest_json,
    canonical_position_targets,
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


COLLECTED_RUN_SCHEMA_VERSION = 1
RUN_FILE_SUFFIX = ".run.json"
WORKER_RESULT_SUFFIX = ".worker.result.json"
WORKER_STDOUT_SUFFIX = ".worker.stdout.log"
WORKER_STDERR_SUFFIX = ".worker.stderr.log"
WORKER_EXITCODE_SUFFIX = ".worker.exitcode.txt"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAPPING_KEYS = frozenset(
    {"canonical_id", "backend", "scope", "backend_name", "index", "sign", "offset", "unit"}
)
_TRACE_KEYS = frozenset(
    {
        "step",
        "time_s",
        "qpos",
        "qvel",
        "joint_positions",
        "frame_poses",
        "position_targets",
        "contact_count",
    }
)
_RESULT_KEYS = frozenset(
    {
        "backend",
        "scenario_id",
        "status",
        "message",
        "dt",
        "requested_steps",
        "completed_steps",
        "joint_names",
        "frame_names",
        "samples",
        "provenance",
        "error",
    }
)
_COLLECTED_KEYS = frozenset(
    {"schema_version", "case", "result", "bundle_root_sha256"}
)


class RunPayloadValidationError(ValueError):
    """Raised when a persisted run is not strict, finite, portable JSON."""


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunPayloadValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> object:
    raise RunPayloadValidationError(f"non-finite JSON number is forbidden: {token}")


def _strict_json_load(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunPayloadValidationError(f"cannot read run JSON {path}: {exc}") from exc


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RunPayloadValidationError(f"{context} must be an object with string keys")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RunPayloadValidationError(
            f"invalid {context} fields: missing={missing}, unknown={unknown}"
        )


def _string(value: object, context: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        qualifier = "a string" if empty else "a non-empty string"
        raise RunPayloadValidationError(f"{context} must be {qualifier}")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RunPayloadValidationError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunPayloadValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise RunPayloadValidationError(f"{context} must be a finite number >= {minimum}")
    return number


def _name_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RunPayloadValidationError(f"{context} must be an array")
    names = tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(value))
    if len(names) != len(set(names)):
        raise RunPayloadValidationError(f"{context} must not contain duplicates")
    return names


def _finite_tuple(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise RunPayloadValidationError(f"{context} must be an array")
    return tuple(_finite(item, f"{context}[{index}]") for index, item in enumerate(value))


def _number_mapping(value: object, context: str) -> dict[str, float]:
    data = _object(value, context)
    return {
        _string(name, f"{context} key"): _finite(number, f"{context}.{name}")
        for name, number in data.items()
    }


def _json_value(value: object, context: str) -> object:
    """Validate and copy one arbitrary JSON value while rejecting NaN/Inf."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunPayloadValidationError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RunPayloadValidationError(f"{context} object keys must be strings")
        return {key: _json_value(item, f"{context}.{key}") for key, item in value.items()}
    raise RunPayloadValidationError(
        f"{context} contains a non-JSON value of type {type(value).__name__}"
    )


def trace_sample_from_dict(value: object, *, context: str = "sample") -> TraceSample:
    """Strictly reconstruct one finite :class:`TraceSample`."""

    data = _object(value, context)
    _exact_keys(data, _TRACE_KEYS, context)
    frame_data = _object(data["frame_poses"], f"{context}.frame_poses")
    frame_poses: dict[str, tuple[float, ...]] = {}
    for name, raw_pose in frame_data.items():
        pose = _finite_tuple(raw_pose, f"{context}.frame_poses.{name}")
        if len(pose) != 7:
            raise RunPayloadValidationError(
                f"{context}.frame_poses.{name} must contain seven values"
            )
        frame_poses[_string(name, f"{context}.frame_poses key")] = pose

    raw_contacts = data["contact_count"]
    contact_count = (
        None
        if raw_contacts is None
        else _integer(raw_contacts, f"{context}.contact_count")
    )
    return TraceSample(
        step=_integer(data["step"], f"{context}.step"),
        time_s=_finite(data["time_s"], f"{context}.time_s", minimum=0.0),
        qpos=_finite_tuple(data["qpos"], f"{context}.qpos"),
        qvel=_finite_tuple(data["qvel"], f"{context}.qvel"),
        joint_positions=_number_mapping(
            data["joint_positions"], f"{context}.joint_positions"
        ),
        frame_poses=frame_poses,
        position_targets=_number_mapping(
            data["position_targets"], f"{context}.position_targets"
        ),
        contact_count=contact_count,
    )


def adapter_run_result_from_dict(value: object) -> AdapterRunResult:
    """Strictly reconstruct one finite :class:`AdapterRunResult`."""

    data = _object(value, "result")
    _exact_keys(data, _RESULT_KEYS, "result")
    status = _string(data["status"], "result.status")
    if status not in {"completed", "error"}:
        raise RunPayloadValidationError("result.status must be 'completed' or 'error'")
    requested_steps = _integer(data["requested_steps"], "result.requested_steps")
    completed_steps = _integer(data["completed_steps"], "result.completed_steps")
    if completed_steps > requested_steps:
        raise RunPayloadValidationError("result.completed_steps exceeds requested_steps")
    if status == "completed" and completed_steps != requested_steps:
        raise RunPayloadValidationError(
            "completed result must complete every requested step"
        )
    joint_names = _name_tuple(data["joint_names"], "result.joint_names")
    frame_names = _name_tuple(data["frame_names"], "result.frame_names")
    raw_samples = data["samples"]
    if not isinstance(raw_samples, list):
        raise RunPayloadValidationError("result.samples must be an array")
    samples = tuple(
        trace_sample_from_dict(item, context=f"result.samples[{index}]")
        for index, item in enumerate(raw_samples)
    )
    for index, sample in enumerate(samples):
        if sample.step != index:
            raise RunPayloadValidationError(
                f"result.samples[{index}].step must equal its array index"
            )
        if index and sample.time_s <= samples[index - 1].time_s:
            raise RunPayloadValidationError("result sample times must be strictly increasing")

    provenance = _json_value(data["provenance"], "result.provenance")
    if not isinstance(provenance, dict):
        raise RunPayloadValidationError("result.provenance must be an object")
    raw_error = data["error"]
    error: dict[str, object] | None
    if raw_error is None:
        error = None
    else:
        validated_error = _json_value(raw_error, "result.error")
        if not isinstance(validated_error, dict):
            raise RunPayloadValidationError("result.error must be an object or null")
        error = validated_error
    if status == "error" and error is None:
        raise RunPayloadValidationError("error result must include result.error")
    if status == "completed" and error is not None:
        raise RunPayloadValidationError("completed result must set result.error to null")

    return AdapterRunResult(
        backend=_string(data["backend"], "result.backend"),
        scenario_id=_string(data["scenario_id"], "result.scenario_id"),
        status=status,
        message=_string(data["message"], "result.message", empty=True),
        dt=_finite(data["dt"], "result.dt", minimum=0.0),
        requested_steps=requested_steps,
        completed_steps=completed_steps,
        joint_names=joint_names,
        frame_names=frame_names,
        samples=samples,
        provenance=provenance,
        error=error,
    )


def collected_run_to_dict(collected: CollectedRun) -> dict[str, object]:
    if not isinstance(collected, CollectedRun):
        raise TypeError("collected must be a CollectedRun")
    return {
        "schema_version": COLLECTED_RUN_SCHEMA_VERSION,
        "case": collected.case.to_dict(),
        "result": collected.result.to_dict(),
        "bundle_root_sha256": collected.bundle_root_sha256,
    }


def collected_run_from_dict(value: object) -> CollectedRun:
    """Strictly reconstruct one collected run, including its canonical case."""

    data = _object(value, "collected_run")
    _exact_keys(data, _COLLECTED_KEYS, "collected_run")
    version = data["schema_version"]
    if isinstance(version, bool) or version != COLLECTED_RUN_SCHEMA_VERSION:
        raise RunPayloadValidationError(
            f"collected_run.schema_version must be {COLLECTED_RUN_SCHEMA_VERSION}"
        )
    try:
        case = ScenarioCase.from_dict(data["case"])
    except (TypeError, ValueError) as exc:
        raise RunPayloadValidationError(f"invalid collected_run.case: {exc}") from exc
    result = adapter_run_result_from_dict(data["result"])
    if result.backend != case.simulator.value:
        raise RunPayloadValidationError("result.backend does not match case.simulator")
    if result.scenario_id != case.scenario_id:
        raise RunPayloadValidationError("result.scenario_id does not match case.scenario_id")
    if not math.isclose(result.dt, case.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise RunPayloadValidationError("result.dt does not match case.dt_s")
    digest = data["bundle_root_sha256"]
    if digest is not None and (
        not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
    ):
        raise RunPayloadValidationError(
            "collected_run.bundle_root_sha256 must be null or a lowercase SHA-256"
        )
    return CollectedRun(case=case, result=result, bundle_root_sha256=digest)


def _safe_output_root(output_dir: str | Path) -> Path:
    raw = Path(output_dir).expanduser()
    if raw.exists() and raw.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {raw}")
    raw.mkdir(parents=True, exist_ok=True)
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"output path is not a directory: {root}")
    return root


def _run_path(root: Path, case: ScenarioCase) -> Path:
    destination = (root / f"{case.case_id}{RUN_FILE_SUFFIX}").resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:  # defense in depth for future case-ID formats
        raise ValueError(f"case output escapes the result directory: {case.case_id}") from exc
    return destination


def _write_text_exclusive(path: Path, content: str) -> Path:
    """Create one durable text artifact without overwriting prior evidence."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return destination


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _asset_tree_sha256(root: Path) -> str:
    """Hash one materialized asset subtree using the pinned v0.1 algorithm."""

    digest = sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _within(path: str | Path, root: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the approved root: {resolved}") from exc
    return resolved


def _validated_executable_entrypoint(path: str | Path, label: str) -> Path:
    """Validate an executable target without erasing a virtualenv symlink.

    Resolving ``env/bin/python`` and then executing the resolved target bypasses
    ``pyvenv.cfg`` and silently drops the environment's site-packages.  Keep
    the absolute entrypoint for execution while separately resolving it for
    existence/type validation.
    """

    entrypoint = Path(os.path.abspath(Path(path).expanduser()))
    resolved = entrypoint.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must resolve to an existing file")
    return entrypoint


def _safe_remote_output_root(output_dir: str | Path, approved_root: Path) -> Path:
    """Create an OVPhysX output directory only after proving its confinement.

    ``_safe_output_root`` is intentionally convenient for local execution and
    creates missing parents.  The remote path has a stricter contract: a bad
    argument must not create even an empty directory outside the approved
    ``results`` subtree.
    """

    results_path = approved_root / "results"
    if results_path.exists() and results_path.is_symlink():
        raise ValueError("approved results directory must not be a symbolic link")
    results_path.mkdir(exist_ok=True)
    results_root = results_path.resolve(strict=True)
    try:
        results_root.relative_to(approved_root)
    except ValueError as exc:
        raise ValueError("approved results directory escapes approved_root") from exc

    raw = Path(output_dir).expanduser()
    if raw.exists() and raw.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {raw}")
    candidate = raw.resolve(strict=False)
    try:
        candidate.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"output_dir escapes the approved results root: {candidate}") from exc

    raw.mkdir(parents=True, exist_ok=True)
    resolved = raw.resolve(strict=True)
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:  # protect against a parent changing during mkdir
        raise ValueError(f"output_dir escapes the approved results root: {resolved}") from exc
    if not resolved.is_dir():
        raise ValueError(f"output_dir is not a directory: {resolved}")
    return resolved


def _default_ovphysx_worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "probe_ovphysx_runtime.py"


def _worker_artifacts(root: Path, case: ScenarioCase) -> dict[str, Path]:
    return {
        "run": _run_path(root, case),
        "result": (root / f"{case.case_id}{WORKER_RESULT_SUFFIX}").resolve(),
        "stdout": (root / f"{case.case_id}{WORKER_STDOUT_SUFFIX}").resolve(),
        "stderr": (root / f"{case.case_id}{WORKER_STDERR_SUFFIX}").resolve(),
        "exitcode": (root / f"{case.case_id}{WORKER_EXITCODE_SUFFIX}").resolve(),
    }


def write_collected_run(output_dir: str | Path, collected: CollectedRun) -> Path:
    """Atomically write one strict run payload below ``output_dir``."""

    root = _safe_output_root(output_dir)
    # Round-trip validation catches non-finite adapter output before JSON I/O.
    validated = collected_run_from_dict(collected_run_to_dict(collected))
    return write_json_atomic(_run_path(root, validated.case), collected_run_to_dict(validated))


def load_collected_runs(path: str | Path) -> tuple[CollectedRun, ...]:
    """Load a single ``*.run.json`` file or all such files in one directory."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise RunPayloadValidationError(f"run input must not be a symbolic link: {source}")
    if source.is_file():
        paths = (source.resolve(strict=True),)
    elif source.is_dir():
        root = source.resolve(strict=True)
        paths = tuple(sorted(root.glob(f"*{RUN_FILE_SUFFIX}"), key=lambda item: item.name))
        if any(item.is_symlink() for item in paths):
            raise RunPayloadValidationError("run directory contains a symbolic-link payload")
    else:
        raise RunPayloadValidationError(f"run input does not exist: {source}")

    collected: list[CollectedRun] = []
    seen: set[str] = set()
    for item in paths:
        run = collected_run_from_dict(_strict_json_load(item))
        expected_name = f"{run.case.case_id}{RUN_FILE_SUFFIX}"
        if item.name != expected_name:
            raise RunPayloadValidationError(
                f"run filename {item.name!r} must be {expected_name!r}"
            )
        if run.case.case_id in seen:
            raise RunPayloadValidationError(f"duplicate collected case: {run.case.case_id}")
        seen.add(run.case.case_id)
        collected.append(run)
    return tuple(collected)


def _select_cases(
    manifest: ParityManifest,
    simulator: Simulator,
    case_ids: Sequence[str] | str | None,
) -> tuple[ScenarioCase, ...]:
    all_cases = expand_scenario_cases(manifest)
    backend_cases = tuple(case for case in all_cases if case.simulator is simulator)
    if case_ids is None:
        return backend_cases
    requested = (case_ids,) if isinstance(case_ids, str) else tuple(case_ids)
    if any(not isinstance(case_id, str) or not case_id for case_id in requested):
        raise ValueError("case_ids must contain non-empty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("case_ids must not contain duplicates")
    known = {case.case_id: case for case in backend_cases}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(
            f"case_ids are not canonical {simulator.value} cases: {', '.join(unknown)}"
        )
    selected = set(requested)
    return tuple(case for case in backend_cases if case.case_id in selected)


def _structured_runner_error(
    case: ScenarioCase,
    error: Exception,
    *,
    requested_steps: int,
    provenance: Mapping[str, object] | None = None,
) -> AdapterRunResult:
    return AdapterRunResult(
        backend=case.simulator.value,
        scenario_id=case.scenario_id,
        status="error",
        message=f"{type(error).__name__}: {error}",
        dt=case.dt_s,
        requested_steps=requested_steps,
        completed_steps=0,
        provenance=dict(provenance or {}),
        error={"type": type(error).__name__, "message": str(error)},
    )


def _finite_or_error(
    case: ScenarioCase,
    result: AdapterRunResult,
    *,
    requested_steps: int,
) -> AdapterRunResult:
    try:
        return adapter_run_result_from_dict(result.to_dict())
    except (TypeError, ValueError) as exc:
        return _structured_runner_error(
            case,
            RunPayloadValidationError(f"adapter returned invalid/non-finite output: {exc}"),
            requested_steps=requested_steps,
        )


def _run_cases(
    *,
    asset_root: str | Path,
    manifest: ParityManifest,
    output_dir: str | Path,
    simulator: Simulator,
    adapter_factory: Callable[..., SimulationAdapter],
    case_ids: Sequence[str] | str | None,
    run_provenance: Mapping[str, object] | None = None,
) -> tuple[CollectedRun, ...]:
    if not isinstance(manifest, ParityManifest):
        raise TypeError("manifest must be a ParityManifest")
    asset_path = Path(asset_root).expanduser().resolve(strict=True)
    if not asset_path.is_dir():
        raise ValueError(f"asset_root is not a directory: {asset_path}")
    output_root = _safe_output_root(output_dir)
    selected = _select_cases(manifest, simulator, case_ids)
    collected: list[CollectedRun] = []
    for case in selected:
        hand = manifest.hand(case.hand)
        scenario = manifest.scenario(case.scenario_id)
        requested_steps = round(scenario.duration_s / case.dt_s)
        adapter: SimulationAdapter | None = None
        result: AdapterRunResult | None = None
        try:
            adapter = adapter_factory(asset_root=asset_path)
            result = adapter.run_scenario(hand, scenario, dt_override=case.dt_s)
            if not isinstance(result, AdapterRunResult):
                raise TypeError("adapter.run_scenario did not return AdapterRunResult")
        except Exception as exc:
            result = _structured_runner_error(
                case, exc, requested_steps=requested_steps
            )
        finally:
            if adapter is not None and getattr(adapter.lifecycle, "value", None) != "closed":
                try:
                    adapter.close()
                except Exception as exc:
                    if result is None or result.status == "completed":
                        result = _structured_runner_error(
                            case, exc, requested_steps=requested_steps
                        )
        assert result is not None
        # Backend adapters may need absolute paths internally, but portable run
        # records use the manifest-relative source identity.
        provenance = dict(result.provenance)
        provenance.update(dict(run_provenance or {}))
        if "source_path" in provenance:
            provenance["source_path"] = case.model_path
        result = replace(result, provenance=provenance)
        result = _finite_or_error(
            case, result, requested_steps=requested_steps
        )
        run = CollectedRun(case=case, result=result)
        write_collected_run(output_root, run)
        collected.append(run)
    return tuple(collected)


def _validate_worker_mapping(
    raw_records: object,
    *,
    canonical_names: tuple[str, ...],
    backend_names: tuple[str, ...],
    scope: str,
    unit: str,
    label: str,
) -> None:
    if not isinstance(raw_records, list) or len(raw_records) != len(canonical_names):
        raise RunPayloadValidationError(
            f"worker {label} must contain {len(canonical_names)} records"
        )
    observed: list[str] = []
    indices: set[int] = set()
    for position, raw in enumerate(raw_records):
        record = _object(raw, f"worker {label}[{position}]")
        _exact_keys(record, _MAPPING_KEYS, f"worker {label}[{position}]")
        canonical_id = _string(
            record["canonical_id"], f"worker {label}[{position}].canonical_id"
        )
        observed.append(canonical_id)
        if record["backend"] != Simulator.OVPHYSX.value:
            raise RunPayloadValidationError(f"worker {label} backend must be ovphysx")
        if record["scope"] != scope:
            raise RunPayloadValidationError(f"worker {label} scope must be {scope!r}")
        backend_name = _string(
            record["backend_name"], f"worker {label}[{position}].backend_name"
        )
        index = _integer(record["index"], f"worker {label}[{position}].index")
        if index >= len(backend_names) or backend_names[index] != backend_name:
            raise RunPayloadValidationError(
                f"worker {label}[{position}] index/name does not match backend order"
            )
        if index in indices:
            raise RunPayloadValidationError(f"worker {label} contains duplicate indices")
        indices.add(index)
        sign = _finite(record["sign"], f"worker {label}[{position}].sign")
        if sign not in {-1.0, 1.0}:
            raise RunPayloadValidationError(f"worker {label} sign must be -1 or 1")
        _finite(record["offset"], f"worker {label}[{position}].offset")
        if record["unit"] != unit:
            raise RunPayloadValidationError(
                f"worker {label}[{position}] unit must be {unit!r}"
            )
    if tuple(observed) != canonical_names:
        raise RunPayloadValidationError(
            f"worker {label} canonical order does not match the manifest"
        )


def _validate_ovphysx_worker_result(
    result: AdapterRunResult,
    *,
    case: ScenarioCase,
    manifest: ParityManifest,
    expected_manifest_sha256: str,
    expected_manifest_file_sha256: str,
    session_id: str,
    source_revision: str,
    worker_script_sha256: str,
    asset_tree_sha256: str,
) -> AdapterRunResult:
    if result.backend != Simulator.OVPHYSX.value:
        raise RunPayloadValidationError("worker result backend must be ovphysx")
    if result.scenario_id != case.scenario_id:
        raise RunPayloadValidationError("worker result scenario_id does not match its case")
    if not math.isclose(result.dt, case.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise RunPayloadValidationError("worker result dt does not match its case")
    expected_steps = round(manifest.scenario(case.scenario_id).duration_s / case.dt_s)
    if result.requested_steps != expected_steps:
        raise RunPayloadValidationError("worker requested_steps does not match its case")

    provenance = result.provenance
    expected_provenance = {
        "case_id": case.case_id,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_file_sha256": expected_manifest_file_sha256,
        "session_id": session_id,
        "source_revision": source_revision,
        "worker_script_sha256": worker_script_sha256,
        "asset_tree_sha256": asset_tree_sha256,
        "mapping_schema_version": 1,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise RunPayloadValidationError(
                f"worker provenance {key!r} does not match the parent runner"
            )

    if result.status == "error":
        return result
    hand = manifest.hand(case.hand)
    if result.completed_steps != expected_steps:
        raise RunPayloadValidationError("completed worker did not finish every requested step")
    if result.joint_names != hand.joint_names:
        raise RunPayloadValidationError("worker canonical joint order does not match manifest")
    if result.frame_names != hand.distal_frame_names:
        raise RunPayloadValidationError("worker canonical frame order does not match manifest")
    if len(result.samples) != expected_steps + 1:
        raise RunPayloadValidationError("completed worker must return N+1 trace samples")

    backend_joint_names = _name_tuple(
        provenance.get("backend_joint_names"), "worker provenance.backend_joint_names"
    )
    backend_frame_names = _name_tuple(
        provenance.get("backend_frame_names"), "worker provenance.backend_frame_names"
    )
    if len(backend_joint_names) != 22:
        raise RunPayloadValidationError("worker backend joint order must contain 22 names")
    raw_joint_mapping = provenance.get("joint_mapping")
    _validate_worker_mapping(
        raw_joint_mapping,
        canonical_names=hand.joint_names,
        backend_names=backend_joint_names,
        scope=case.hand.value,
        unit="rad",
        label="joint_mapping",
    )
    _validate_worker_mapping(
        provenance.get("frame_mapping"),
        canonical_names=hand.distal_frame_names,
        backend_names=backend_frame_names,
        scope=case.hand.value,
        unit="xyz_m_qwxyz",
        label="frame_mapping",
    )
    simulation_configuration = _object(
        provenance.get("simulation_configuration"),
        "worker provenance.simulation_configuration",
    )
    _exact_keys(
        simulation_configuration,
        frozenset(
            {
                "verified",
                "requested_dt_s",
                "cfg_dt_s",
                "backend_dt_s",
                "requested_gravity_m_s2",
                "cfg_gravity_m_s2",
                "physics_scene_gravity_m_s2",
                "physics_prim_path",
            }
        ),
        "worker provenance.simulation_configuration",
    )
    if simulation_configuration["verified"] is not True:
        raise RunPayloadValidationError("worker simulation configuration is not verified")
    for key in ("requested_dt_s", "cfg_dt_s", "backend_dt_s"):
        if not math.isclose(
            _finite(simulation_configuration[key], f"simulation_configuration.{key}"),
            case.dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RunPayloadValidationError(f"worker {key} does not match the case timestep")
    expected_gravity = manifest.scenario(case.scenario_id).gravity_m_s2
    for key, tolerance in (
        ("requested_gravity_m_s2", 1e-12),
        ("cfg_gravity_m_s2", 1e-6),
        ("physics_scene_gravity_m_s2", 1e-5),
    ):
        observed_gravity = _finite_tuple(
            simulation_configuration[key], f"simulation_configuration.{key}"
        )
        if len(observed_gravity) != 3 or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
            for actual, expected in zip(observed_gravity, expected_gravity)
        ):
            raise RunPayloadValidationError(f"worker {key} does not match the scenario gravity")
    if simulation_configuration["physics_prim_path"] != "/physicsScene":
        raise RunPayloadValidationError("worker used an unexpected PhysicsScene path")

    if _integer(
        provenance.get("actuation_contract_version"),
        "worker provenance.actuation_contract_version",
    ) != 2:
        raise RunPayloadValidationError("worker actuation_contract_version must be 2")
    if provenance.get("actuator_model") != "IdealPDActuator":
        raise RunPayloadValidationError("worker actuator_model must be IdealPDActuator")
    if provenance.get("control_path") != "explicit_pd_effort":
        raise RunPayloadValidationError("worker control_path must be explicit_pd_effort")

    for field, strictly_positive in (
        ("controller_dof_stiffness", True),
        ("controller_dof_damping", False),
        ("controller_dof_effort_limit", True),
        ("controller_dof_effort_limit_sim", True),
    ):
        values = _finite_tuple(
            provenance.get(field), f"worker provenance.{field}"
        )
        if len(values) != 22 or any(
            value <= 0.0 if strictly_positive else value < 0.0
            for value in values
        ):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise RunPayloadValidationError(
                f"worker {field} must contain 22 finite {qualifier} values"
            )
    if provenance.get("controller_parameter_source") != "ideal_pd_actuator_tensor":
        raise RunPayloadValidationError("worker controller parameter source is invalid")

    for field in ("backend_dof_stiffness", "backend_dof_damping"):
        values = _finite_tuple(
            provenance.get(field), f"worker provenance.{field}"
        )
        if len(values) != 22 or any(abs(value) > 1e-8 for value in values):
            raise RunPayloadValidationError(
                f"worker {field} must contain 22 zeroed PhysX drive readbacks"
            )
    if provenance.get("backend_dof_drive_readback_source") != "root_view_cpu_numpy_binding":
        raise RunPayloadValidationError("worker drive readback did not use the CPU binding path")
    if provenance.get("position_target_readback_verified") is not True:
        raise RunPayloadValidationError("worker position-target binding was not verified")
    if (
        provenance.get("position_target_readback_source")
        != "articulation_data_joint_pos_target_torch"
    ):
        raise RunPayloadValidationError("worker target readback source is invalid")
    if provenance.get("zero_velocity_target_verified") is not True:
        raise RunPayloadValidationError("worker zero velocity target was not verified")
    if provenance.get("zero_feedforward_effort_target_verified") is not True:
        raise RunPayloadValidationError(
            "worker zero feedforward effort target was not verified"
        )
    readback_count = _integer(
        provenance.get("position_target_readback_count"),
        "worker provenance.position_target_readback_count",
    )
    if readback_count < expected_steps + 2:
        raise RunPayloadValidationError("worker did not read back targets throughout the run")
    readback_error = _finite(
        provenance.get("position_target_readback_max_abs_error_rad"),
        "worker provenance.position_target_readback_max_abs_error_rad",
    )
    if readback_error > 1e-6:
        raise RunPayloadValidationError("worker position-target readback error is too large")
    if manifest.schema_version == 2:
        canonical_readback_error = _finite(
            provenance.get(
                "position_target_canonical_readback_max_abs_error_rad"
            ),
            "worker provenance.position_target_canonical_readback_max_abs_error_rad",
            minimum=0.0,
        )
        if canonical_readback_error > 1e-6:
            raise RunPayloadValidationError(
                "worker canonical position-target readback error is too large"
            )
    readback_values = _finite_tuple(
        provenance.get("position_target_readback_values_rad"),
        "worker provenance.position_target_readback_values_rad",
    )
    if len(readback_values) != 22:
        raise RunPayloadValidationError("worker final target readback must contain 22 values")
    assert isinstance(raw_joint_mapping, list)
    final_targets = canonical_position_targets(
        manifest.scenario(case.scenario_id),
        hand.joint_names,
        step_index=expected_steps,
        dt_s=case.dt_s,
    )
    for position, raw_record in enumerate(raw_joint_mapping):
        record = _object(raw_record, f"worker joint_mapping[{position}]")
        canonical_id = str(record["canonical_id"])
        index = int(record["index"])
        sign = float(record["sign"])
        offset = float(record["offset"])
        expected_backend_target = (final_targets[canonical_id] - offset) / sign
        if not math.isclose(
            readback_values[index],
            expected_backend_target,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RunPayloadValidationError(
                "worker final target readback does not match the canonical scenario"
            )
    nonzero_readback = provenance.get("position_target_nonzero_readback_observed")
    expected_nonzero = manifest.scenario(case.scenario_id).kind.value in {
        "small_step",
        "offset_sine",
        "offset_linear_chirp",
    }
    if nonzero_readback is not expected_nonzero:
        raise RunPayloadValidationError(
            "worker nonzero target readback does not match the scenario"
        )

    if manifest.schema_version == 2:
        sequence_metadata = {
            "target_sequence_digest_schema_version": 1,
            "target_sequence_digest_encoding": "utf8_json_lines_float_hex_v1",
            "target_sequence_digest_projection": "ieee754_binary32_roundtrip",
            "target_sequence_canonical_joint_names": list(hand.joint_names),
            "scheduled_target_sequence_semantics": (
                "q[0..N]; target q[k] is recorded at t_k and applies to "
                "[t_k,t_{k+1}); q[N] is terminal and is not integrated"
            ),
            "requested_target_sequence_semantics": (
                "q[0..N] passed to set_joint_position_target_index"
            ),
            "immediate_target_readback_sequence_semantics": (
                "q[0..N] read immediately from joint_pos_target after each request"
            ),
            "pre_step_applied_target_readback_sequence_semantics": (
                "q[0..N-1] read after write_data_to_sim and before each physics step"
            ),
        }
        for field, expected in sequence_metadata.items():
            if provenance.get(field) != expected:
                raise RunPayloadValidationError(
                    f"worker target-sequence metadata {field!r} is invalid"
                )
        sequence_counts = {
            "scheduled_target_sequence_count": expected_steps + 1,
            "requested_target_sequence_count": expected_steps + 1,
            "immediate_target_readback_sequence_count": expected_steps + 1,
            "pre_step_applied_target_readback_sequence_count": expected_steps,
        }
        for field, expected in sequence_counts.items():
            if _integer(provenance.get(field), f"worker provenance.{field}") != expected:
                raise RunPayloadValidationError(
                    f"worker target-sequence count {field!r} is incomplete"
                )
        expected_full_digest = canonical_target_sequence_sha256(
            manifest.scenario(case.scenario_id),
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=True,
        )
        expected_prefix_digest = canonical_target_sequence_sha256(
            manifest.scenario(case.scenario_id),
            hand.joint_names,
            dt_s=case.dt_s,
            include_terminal=False,
        )
        sequence_digests = {
            "scheduled_target_sequence_sha256": expected_full_digest,
            "requested_target_sequence_sha256": expected_full_digest,
            "immediate_target_readback_sequence_sha256": expected_full_digest,
            "pre_step_applied_target_readback_sequence_sha256": (
                expected_prefix_digest
            ),
        }
        for field, expected in sequence_digests.items():
            if provenance.get(field) != expected:
                raise RunPayloadValidationError(
                    f"worker target-sequence digest {field!r} does not match "
                    "the canonical schedule"
                )

    for field in ("computed_effort_peak_abs_nm", "applied_effort_peak_abs_nm"):
        values = _finite_tuple(
            provenance.get(field), f"worker provenance.{field}"
        )
        if len(values) != 22 or any(value < 0.0 for value in values):
            raise RunPayloadValidationError(
                f"worker {field} must contain 22 finite non-negative values"
            )
    effort_observation_count = _integer(
        provenance.get("effort_observation_count"),
        "worker provenance.effort_observation_count",
    )
    if effort_observation_count < expected_steps + 1:
        raise RunPayloadValidationError("worker effort observation coverage is incomplete")
    effort_formula_error = _finite(
        provenance.get("effort_formula_max_abs_error_nm"),
        "worker provenance.effort_formula_max_abs_error_nm",
        minimum=0.0,
    )
    if effort_formula_error > 1e-5:
        raise RunPayloadValidationError("worker explicit-PD effort formula error is too large")
    effort_clip_error = _finite(
        provenance.get("effort_clip_max_abs_error_nm"),
        "worker provenance.effort_clip_max_abs_error_nm",
        minimum=0.0,
    )
    if effort_clip_error > 1e-6:
        raise RunPayloadValidationError("worker explicit-PD effort clipping error is too large")
    _integer(
        provenance.get("effort_clip_count"),
        "worker provenance.effort_clip_count",
    )
    if (
        provenance.get("effort_command_source")
        != "articulation_data_computed_and_applied_torque_torch"
    ):
        raise RunPayloadValidationError("worker effort command source is invalid")
    source_digest = provenance.get("source_sha256")
    if not isinstance(source_digest, str) or _SHA256_RE.fullmatch(source_digest) is None:
        raise RunPayloadValidationError("worker provenance source_sha256 is missing or invalid")
    if provenance.get("contact_check_performed") is not False:
        raise RunPayloadValidationError("Gate 0 worker must report contact_check_performed=false")
    if provenance.get("forbidden_modules") != []:
        raise RunPayloadValidationError("worker imported a forbidden Kit/render module")

    expected_joint_set = set(hand.joint_names)
    expected_frame_set = set(hand.distal_frame_names)
    for index, sample in enumerate(result.samples):
        if sample.step != index or not math.isclose(
            sample.time_s, index * case.dt_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RunPayloadValidationError("worker trace step/time axis is not canonical")
        if len(sample.qpos) != len(backend_joint_names) or len(sample.qvel) != len(
            backend_joint_names
        ):
            raise RunPayloadValidationError(
                "worker qpos/qvel must preserve the complete backend tensor order"
            )
        if set(sample.joint_positions) != expected_joint_set:
            raise RunPayloadValidationError("worker sample lacks canonical joint positions")
        if set(sample.frame_poses) != expected_frame_set:
            raise RunPayloadValidationError("worker sample lacks canonical frame poses")
        if set(sample.position_targets) != expected_joint_set:
            raise RunPayloadValidationError("worker sample lacks canonical position targets")
        if sample.contact_count is not None:
            raise RunPayloadValidationError(
                "OVPhysX contact_count must remain null until observation is implemented"
            )
    return result


def _completed_process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_ovphysx_cases_isolated(
    asset_root: str | Path,
    manifest: ParityManifest,
    output_dir: str | Path,
    case_ids: Sequence[str] | str | None = None,
    *,
    approved_root: str | Path,
    device: str = "cuda:0",
    worker_script: str | Path | None = None,
    worker_python: str | Path | None = None,
    worker_timeout_s: float = 300.0,
    session_id: str | None = None,
    source_revision: str | None = None,
) -> tuple[CollectedRun, ...]:
    """Launch one fresh kit-less OVPhysX worker process per canonical case."""

    approved = Path(approved_root).expanduser().resolve(strict=True)
    if not approved.is_dir():
        raise ValueError("approved_root must be an existing directory")
    asset_path = _within(asset_root, approved, "asset_root")
    if not asset_path.is_dir():
        raise ValueError("asset_root must be an existing directory")
    output_root = _safe_remote_output_root(output_dir, approved)
    script = _within(
        worker_script or _default_ovphysx_worker_script(), approved, "worker_script"
    )
    if not script.is_file():
        raise ValueError("worker_script must be an existing file")
    python = _validated_executable_entrypoint(
        worker_python or sys.executable, "worker_python"
    )
    if device != "cuda:0":
        raise ValueError("built-in OVPhysX execution requires logical device cuda:0")
    if (
        isinstance(worker_timeout_s, bool)
        or not isinstance(worker_timeout_s, (int, float))
        or not math.isfinite(float(worker_timeout_s))
        or worker_timeout_s <= 0.0
    ):
        raise ValueError("worker_timeout_s must be a positive finite number")
    selected_session = session_id or uuid.uuid4().hex
    if _SESSION_RE.fullmatch(selected_session) is None:
        raise ValueError("session_id must be a portable non-empty identifier")
    selected_revision = source_revision or os.environ.get("WAVEQA_SOURCE_TREE", "")
    if _REVISION_RE.fullmatch(selected_revision) is None:
        raise ValueError("source_revision must be a lowercase 40-digit Git commit")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise ValueError("CUDA_VISIBLE_DEVICES must expose exactly one physical GPU index")

    asset_subtree = _within(
        asset_path / manifest.provenance.asset_root,
        asset_path,
        "manifest asset subtree",
    )
    if not asset_subtree.is_dir():
        raise ValueError("manifest asset subtree must be an existing directory")
    actual_asset_tree_sha256 = _asset_tree_sha256(asset_subtree)
    if actual_asset_tree_sha256 != manifest.provenance.canonical_lf_asset_tree_sha256:
        raise RunPayloadValidationError(
            "canonical LF asset-tree SHA-256 does not match the materialized remote assets"
        )

    selected = _select_cases(manifest, Simulator.OVPHYSX, case_ids)
    artifacts = {case.case_id: _worker_artifacts(output_root, case) for case in selected}
    existing = [
        path
        for case_artifacts in artifacts.values()
        for path in case_artifacts.values()
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "OVPhysX runner refuses to overwrite existing case evidence: "
            + ", ".join(path.name for path in existing)
        )

    canonical_digest = manifest_sha256(manifest)
    input_dir = output_root / "worker-inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_snapshot = input_dir / f"{canonical_digest}.manifest.json"
    canonical_text = canonical_manifest_json(manifest)
    if manifest_snapshot.exists():
        if manifest_snapshot.read_text(encoding="utf-8") != canonical_text:
            raise RunPayloadValidationError("existing canonical manifest snapshot is corrupted")
    else:
        _write_text_exclusive(manifest_snapshot, canonical_text)

    script_digest = _sha256_path(script)
    manifest_file_digest = _sha256_path(manifest_snapshot)
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    environment["WAVEQA_SOURCE_TREE"] = selected_revision
    environment["WAVEQA_SESSION_ID"] = selected_session
    environment["PYTHONUNBUFFERED"] = "1"
    collected: list[CollectedRun] = []
    for case in selected:
        case_artifacts = artifacts[case.case_id]
        command = [
            str(python),
            str(script),
            "--approved-root",
            str(approved),
            "--asset-root",
            str(asset_path),
            "--manifest",
            str(manifest_snapshot),
            "--case-id",
            case.case_id,
            "--output",
            str(case_artifacts["result"]),
            "--device",
            device,
            "--session-id",
            selected_session,
            "--source-revision",
            selected_revision,
            "--asset-tree-sha256",
            actual_asset_tree_sha256,
        ]
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                timeout=float(worker_timeout_s),
            )
            returncode = completed.returncode
            stdout_text = completed.stdout
            stderr_text = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout_text = _completed_process_text(exc.stdout)
            stderr_text = _completed_process_text(exc.stderr)
            stderr_text += f"\nworker timed out after {float(worker_timeout_s):.6g}s\n"
        except OSError as exc:
            returncode = 127
            stdout_text = ""
            stderr_text = f"{type(exc).__name__}: {exc}\n"

        _write_text_exclusive(case_artifacts["stdout"], stdout_text)
        _write_text_exclusive(case_artifacts["stderr"], stderr_text)
        _write_text_exclusive(case_artifacts["exitcode"], f"{returncode}\n")
        process_provenance: dict[str, object] = {
            "manifest_sha256": canonical_digest,
            "session_id": selected_session,
            "source_revision": selected_revision,
            "asset_tree_sha256": actual_asset_tree_sha256,
            "mapping_schema_version": 1,
            "asset_commit": manifest.provenance.commit,
            "asset_git_tree": manifest.provenance.asset_git_tree,
            "worker_script_sha256": script_digest,
            "worker_exitcode": returncode,
            "worker_timed_out": timed_out,
            "worker_stdout_log": case_artifacts["stdout"].name,
            "worker_stdout_sha256": _sha256_path(case_artifacts["stdout"]),
            "worker_stderr_log": case_artifacts["stderr"].name,
            "worker_stderr_sha256": _sha256_path(case_artifacts["stderr"]),
            "worker_exitcode_log": case_artifacts["exitcode"].name,
            "worker_exitcode_sha256": _sha256_path(case_artifacts["exitcode"]),
        }
        expected_steps = round(manifest.scenario(case.scenario_id).duration_s / case.dt_s)
        try:
            if timed_out:
                raise TimeoutError(f"OVPhysX worker timed out after {worker_timeout_s}s")
            if returncode not in {0, 2}:
                raise RuntimeError(f"OVPhysX worker exited with code {returncode}")
            if not case_artifacts["result"].is_file():
                raise RunPayloadValidationError("OVPhysX worker did not create its result file")
            raw_result = adapter_run_result_from_dict(
                _strict_json_load(case_artifacts["result"])
            )
            raw_result = _validate_ovphysx_worker_result(
                raw_result,
                case=case,
                manifest=manifest,
                expected_manifest_sha256=canonical_digest,
                expected_manifest_file_sha256=manifest_file_digest,
                session_id=selected_session,
                source_revision=selected_revision,
                worker_script_sha256=script_digest,
                asset_tree_sha256=actual_asset_tree_sha256,
            )
            if (returncode == 0) != raw_result.completed:
                raise RunPayloadValidationError(
                    "worker exit code does not agree with its execution status"
                )
            provenance = dict(raw_result.provenance)
            provenance.update(process_provenance)
            if "source_path" in provenance:
                provenance["source_path"] = case.model_path
            result = replace(raw_result, provenance=provenance)
        except Exception as exc:
            result = _structured_runner_error(
                case,
                exc,
                requested_steps=expected_steps,
                provenance=process_provenance,
            )
        run = CollectedRun(case=case, result=result)
        write_collected_run(output_root, run)
        collected.append(run)
    return tuple(collected)


def run_mujoco_cases(
    asset_root: str | Path,
    manifest: ParityManifest | str | Path,
    output_dir: str | Path,
    case_ids: Sequence[str] | str | None = None,
    *,
    session_id: str | None = None,
    source_revision: str | None = None,
    verify_asset_tree: bool = False,
) -> tuple[CollectedRun, ...]:
    """Run selected canonical MuJoCo cases and persist one JSON file per case."""

    return run_backend_cases(
        asset_root,
        manifest,
        output_dir,
        backend=Simulator.MUJOCO,
        case_ids=case_ids,
        adapter_factory=MuJoCoAdapter,
        session_id=session_id,
        source_revision=source_revision,
        verify_asset_tree=verify_asset_tree,
    )


def run_backend_cases(
    asset_root: str | Path,
    manifest: ParityManifest | str | Path,
    output_dir: str | Path,
    *,
    backend: Simulator | str,
    case_ids: Sequence[str] | str | None = None,
    adapter_factory: Callable[..., SimulationAdapter] | None = None,
    device: str = "cuda:0",
    approved_root: str | Path | None = None,
    worker_script: str | Path | None = None,
    worker_python: str | Path | None = None,
    worker_timeout_s: float = 300.0,
    session_id: str | None = None,
    source_revision: str | None = None,
    verify_asset_tree: bool = False,
) -> tuple[CollectedRun, ...]:
    """Run one backend's canonical cases using independent adapter instances.

    Passing one ``case_id`` makes this suitable as the unit of work for a
    process-isolated launcher.  ``adapter_factory`` exists for version-pinned
    runtime bridges and tests; normal callers use the built-in adapters.
    """

    try:
        simulator = backend if isinstance(backend, Simulator) else Simulator(backend)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported parity backend: {backend!r}") from exc
    canonical = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    if simulator is Simulator.OVPHYSX and adapter_factory is None:
        if approved_root is None:
            raise ValueError("built-in OVPhysX execution requires approved_root")
        return run_ovphysx_cases_isolated(
            asset_root,
            canonical,
            output_dir,
            case_ids,
            approved_root=approved_root,
            device=device,
            worker_script=worker_script,
            worker_python=worker_python,
            worker_timeout_s=worker_timeout_s,
            session_id=session_id,
            source_revision=source_revision,
        )
    selected_session = session_id or uuid.uuid4().hex
    if _SESSION_RE.fullmatch(selected_session) is None:
        raise ValueError("session_id must be a portable non-empty identifier")
    selected_revision = source_revision or os.environ.get(
        "WAVEQA_SOURCE_TREE", "unrecorded"
    )
    asset_digest = canonical.provenance.canonical_lf_asset_tree_sha256
    asset_verification = "not_performed_test_or_fixture"
    if verify_asset_tree:
        if _REVISION_RE.fullmatch(selected_revision) is None:
            raise ValueError(
                "verified runs require source_revision as a lowercase 40- or 64-digit hex revision"
            )
        asset_path = Path(asset_root).expanduser().resolve(strict=True)
        subtree = (asset_path / canonical.provenance.asset_root).resolve(strict=True)
        try:
            subtree.relative_to(asset_path)
        except ValueError as exc:
            raise ValueError("manifest asset subtree escapes asset_root") from exc
        asset_digest = _asset_tree_sha256(subtree)
        if asset_digest != canonical.provenance.canonical_lf_asset_tree_sha256:
            raise RunPayloadValidationError(
                "canonical LF asset-tree SHA-256 does not match the materialized assets"
            )
        asset_verification = "scanned_path_size_bytes"
    run_provenance = {
        "manifest_sha256": manifest_sha256(canonical),
        "session_id": selected_session,
        "source_revision": selected_revision,
        "worker_pid": os.getpid(),
        "asset_tree_sha256": asset_digest,
        "asset_tree_verification": asset_verification,
        "asset_commit": canonical.provenance.commit,
        "asset_git_tree": canonical.provenance.asset_git_tree,
    }
    factory = adapter_factory
    if factory is None:
        factory = MuJoCoAdapter

    return _run_cases(
        asset_root=asset_root,
        manifest=canonical,
        output_dir=output_dir,
        simulator=simulator,
        adapter_factory=factory,
        case_ids=case_ids,
        run_provenance=run_provenance,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=tuple(item.value for item in Simulator),
        default=Simulator.MUJOCO.value,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Backend-visible device; the remote launcher owns physical GPU isolation.",
    )
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--approved-root",
        type=Path,
        help="Required for built-in OVPhysX; confines assets, source and results.",
    )
    parser.add_argument("--worker-script", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--session-id")
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this canonical case ID; repeat the option for more cases.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runs = run_backend_cases(
        args.asset_root,
        args.manifest,
        args.output_dir,
        backend=args.backend,
        case_ids=args.case_ids,
        device=args.device,
        approved_root=args.approved_root,
        worker_script=args.worker_script,
        worker_python=args.worker_python,
        worker_timeout_s=args.worker_timeout_s,
        session_id=args.session_id,
        source_revision=args.source_revision,
        verify_asset_tree=args.backend == Simulator.MUJOCO.value,
    )
    summary = {
        "backend": args.backend,
        "requested_case_count": len(runs),
        "completed_case_count": sum(run.result.completed for run in runs),
        "case_ids": [run.case.case_id for run in runs],
        # Preserve the useful logical directory name without disclosing a
        # host-specific path in evidence copied into a portable bundle.
        "output_dir": args.output_dir.name,
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if all(run.result.completed for run in runs) else 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "COLLECTED_RUN_SCHEMA_VERSION",
    "RUN_FILE_SUFFIX",
    "RunPayloadValidationError",
    "adapter_run_result_from_dict",
    "collected_run_from_dict",
    "collected_run_to_dict",
    "load_collected_runs",
    "main",
    "run_backend_cases",
    "run_mujoco_cases",
    "run_ovphysx_cases_isolated",
    "trace_sample_from_dict",
    "write_collected_run",
]
