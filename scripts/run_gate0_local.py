#!/usr/bin/env python3
"""Run the 16 canonical MuJoCo Gate 0 cases in fresh Python processes.

The launcher is deliberately evidence-oriented: every case gets its own
directory, process, stdout/stderr streams, exit-code record, command record,
and canonical ``*.run.json`` payload.  An existing output path is never
reused.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence

import wave_asset_qa

from wave_asset_qa.parity.contracts import ParityManifest, Simulator
from wave_asset_qa.parity.runner import RUN_FILE_SUFFIX, load_collected_runs
from wave_asset_qa.parity.scenarios import (
    ScenarioCase,
    canonical_manifest_json,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


EXPECTED_CASE_COUNT = 16
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TREE_RE = re.compile(r"^[0-9a-f]{40}$")


class LocalGate0Error(RuntimeError):
    """Raised when local execution or its evidence is not canonical."""


def _resolved_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise LocalGate0Error(f"{label} must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LocalGate0Error(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise LocalGate0Error(f"{label} must be a directory: {resolved}")
    return resolved


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise LocalGate0Error(f"{label} must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LocalGate0Error(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise LocalGate0Error(f"{label} must be a regular file: {resolved}")
    return resolved


def _create_output_root(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"output path already exists; refusing overwrite: {requested}")
    if requested.name in {"", ".", ".."}:
        raise LocalGate0Error("output path must name a new directory")
    parent = _resolved_directory(requested.parent, "output parent")
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"output path already exists; refusing overwrite: {destination}"
        )
    destination.mkdir(mode=0o700)
    return destination.resolve(strict=True)


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _write_text_exclusive(path, serialized)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalGate0Error(f"cannot verify the local Git source: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise LocalGate0Error(
            "cannot verify the local Git source"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout.strip()


def _validate_source_identity(project_root: Path, source_revision: str) -> str:
    """Bind a run label to the exact clean, tracked checkout being executed."""

    root = project_root.resolve(strict=True)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise LocalGate0Error(
            f"launcher project root is not the Git top level: {root} != {top_level}"
        )
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    if head != source_revision:
        raise LocalGate0Error(
            f"source_revision does not match local HEAD: {source_revision} != {head}"
        )
    source_tree = _git_text(
        root, "rev-parse", "--verify", f"{source_revision}^{{tree}}"
    )
    if _TREE_RE.fullmatch(source_tree) is None:
        raise LocalGate0Error("local source commit did not resolve to a Git tree")

    tracked_changes = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if tracked_changes:
        raise LocalGate0Error(
            "tracked worktree/index must be clean before a formal local run"
        )

    launcher_relative = Path(__file__).resolve(strict=True).relative_to(root).as_posix()
    _git_text(root, "ls-files", "--error-unmatch", "--", launcher_relative)
    expected_package = (root / "src" / "wave_asset_qa").resolve(strict=True)
    imported_package = Path(wave_asset_qa.__file__).resolve(strict=True).parent
    if imported_package != expected_package:
        raise LocalGate0Error(
            "wave_asset_qa was imported from a source tree other than this checkout"
        )
    return source_tree


def canonical_mujoco_cases(manifest: ParityManifest) -> tuple[ScenarioCase, ...]:
    cases = tuple(
        case
        for case in expand_scenario_cases(manifest)
        if case.simulator is Simulator.MUJOCO
    )
    ids = tuple(case.case_id for case in cases)
    if len(cases) != EXPECTED_CASE_COUNT or len(set(ids)) != EXPECTED_CASE_COUNT:
        raise LocalGate0Error(
            f"canonical manifest must expand to exactly {EXPECTED_CASE_COUNT} unique "
            f"MuJoCo cases; found {len(cases)}"
        )
    return cases


def _validate_run_provenance(
    provenance: Mapping[str, object],
    *,
    manifest: ParityManifest,
    canonical_manifest_sha256: str,
    session_id: str,
    source_revision: str,
) -> None:
    expected: dict[str, object] = {
        "manifest_sha256": canonical_manifest_sha256,
        "session_id": session_id,
        "source_revision": source_revision,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
        "asset_tree_verification": "scanned_path_size_bytes",
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "mapping_schema_version": 1,
    }
    mismatches = [
        key for key, value in expected.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise LocalGate0Error(
            "completed run has invalid provenance field(s): "
            + ", ".join(sorted(mismatches))
        )


def _validate_case_output(
    case_dir: Path,
    case: ScenarioCase,
    *,
    manifest: ParityManifest,
    canonical_manifest_sha256: str,
    session_id: str,
    source_revision: str,
) -> None:
    case_id = case.case_id
    expected_file = case_dir / f"{case_id}{RUN_FILE_SUFFIX}"
    run_files = tuple(sorted(case_dir.glob(f"*{RUN_FILE_SUFFIX}")))
    if run_files != (expected_file,):
        raise LocalGate0Error(
            f"{case_id} must produce exactly {expected_file.name}; found "
            + ", ".join(path.name for path in run_files)
        )
    runs = load_collected_runs(expected_file)
    if len(runs) != 1 or runs[0].case != case:
        raise LocalGate0Error(f"{case_id} output is not its canonical collected run")
    run = runs[0]
    if not run.result.completed:
        raise LocalGate0Error(f"{case_id} did not complete: {run.result.message}")
    if run.bundle_root_sha256 is not None:
        raise LocalGate0Error(f"{case_id} raw run unexpectedly references an old bundle")
    _validate_run_provenance(
        run.result.provenance,
        manifest=manifest,
        canonical_manifest_sha256=canonical_manifest_sha256,
        session_id=session_id,
        source_revision=source_revision,
    )


def _write_case_hashes(case_dir: Path) -> None:
    hash_file = case_dir / "evidence.sha256"
    members = tuple(
        sorted(
            (
                path
                for path in case_dir.rglob("*")
                if path.is_file() and path != hash_file
            ),
            key=lambda path: path.relative_to(case_dir).as_posix(),
        )
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(case_dir).as_posix()}"
        for path in members
    ]
    _write_text_exclusive(hash_file, "\n".join(lines) + "\n")


def run_local_gate0(
    *,
    asset_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    session_id: str,
    source_revision: str,
    python_executable: str | Path = sys.executable,
    case_timeout_s: float = 300.0,
) -> Path:
    """Execute and validate the complete 16-case local MuJoCo matrix."""

    if _SESSION_RE.fullmatch(session_id) is None:
        raise LocalGate0Error("session_id must be a portable non-empty identifier")
    if _REVISION_RE.fullmatch(source_revision) is None:
        raise LocalGate0Error(
            "source_revision must be a lowercase 40-character project code commit"
        )
    if (
        isinstance(case_timeout_s, bool)
        or not isinstance(case_timeout_s, (int, float))
        or not math.isfinite(float(case_timeout_s))
        or case_timeout_s <= 0.0
    ):
        raise LocalGate0Error("case_timeout_s must be a positive finite number")

    assets = _resolved_directory(asset_root, "asset root")
    source_manifest = _resolved_file(manifest_path, "manifest")
    python = _resolved_file(python_executable, "Python executable")
    manifest = load_manifest(source_manifest)
    cases = canonical_mujoco_cases(manifest)
    canonical_digest = manifest_sha256(manifest)
    project_root = Path(__file__).resolve(strict=True).parents[1]
    source_tree = _validate_source_identity(project_root, source_revision)
    launcher_script_sha256 = _sha256_file(Path(__file__).resolve(strict=True))
    manifest_file_sha256 = _sha256_file(source_manifest)

    def validate_static_inputs() -> None:
        observed_tree = _validate_source_identity(project_root, source_revision)
        if observed_tree != source_tree:
            raise LocalGate0Error("local Git source tree changed during the run")
        if _sha256_file(Path(__file__).resolve(strict=True)) != launcher_script_sha256:
            raise LocalGate0Error("local launcher script changed during the run")
        if _sha256_file(source_manifest) != manifest_file_sha256:
            raise LocalGate0Error("local manifest file changed during the run")

    root = _create_output_root(output_dir)
    launcher_dir = root / "launcher"
    cases_dir = root / "cases"
    launcher_dir.mkdir()
    cases_dir.mkdir()

    manifest_snapshot = launcher_dir / "gate0.manifest.json"
    _write_text_exclusive(
        manifest_snapshot, canonical_manifest_json(manifest) + "\n"
    )
    manifest_snapshot_sha256 = _sha256_file(manifest_snapshot)
    launcher_identity: dict[str, object] = {
        "schema_version": 1,
        "backend": Simulator.MUJOCO.value,
        "session_id": session_id,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "launcher_script_sha256": launcher_script_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": canonical_digest,
        "manifest_snapshot_sha256": manifest_snapshot_sha256,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "asset_tree_sha256": manifest.provenance.canonical_lf_asset_tree_sha256,
    }
    _write_json_exclusive(launcher_dir / "provenance.json", launcher_identity)
    _write_json_exclusive(
        launcher_dir / "matrix.json",
        {
            **launcher_identity,
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
        },
    )

    source_dir = project_root / "src"
    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(source_dir)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WAVEQA_SESSION_ID"] = session_id
    environment["WAVEQA_SOURCE_TREE"] = source_revision

    completed_ids: list[str] = []
    for case in cases:
        validate_static_inputs()
        case_id = case.case_id
        case_dir = cases_dir / case_id
        case_dir.mkdir()
        command = [
            str(python),
            "-P",
            "-m",
            "wave_asset_qa.parity.runner",
            "--backend",
            Simulator.MUJOCO.value,
            "--asset-root",
            str(assets),
            "--manifest",
            str(manifest_snapshot),
            "--output-dir",
            str(case_dir),
            "--session-id",
            session_id,
            "--source-revision",
            source_revision,
            "--case-id",
            case_id,
        ]
        recorded_command = [
            "<python>",
            "-P",
            "-m",
            "wave_asset_qa.parity.runner",
            "--backend",
            Simulator.MUJOCO.value,
            "--asset-root",
            "<asset-root>",
            "--manifest",
            "launcher/gate0.manifest.json",
            "--output-dir",
            f"cases/{case_id}",
            "--session-id",
            session_id,
            "--source-revision",
            source_revision,
            "--case-id",
            case_id,
        ]
        _write_json_exclusive(
            case_dir / "launcher.command.json",
            {"argv": recorded_command, "case_id": case_id, "schema_version": 1},
        )
        stdout_path = case_dir / "launcher.stdout.log"
        stderr_path = case_dir / "launcher.stderr.log"
        returncode: int
        timed_out = False
        with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
            try:
                completed = subprocess.run(
                    command,
                    shell=False,
                    check=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=environment,
                    cwd=project_root,
                    timeout=float(case_timeout_s),
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                timed_out = True
            except OSError as exc:
                returncode = 127
                stderr_handle.write(f"launcher error: {type(exc).__name__}: {exc}\n".encode())

        if timed_out:
            with stderr_path.open("ab") as stderr_handle:
                stderr_handle.write(
                    f"launcher timeout after {float(case_timeout_s):.6g}s\n".encode()
                )
        _write_text_exclusive(case_dir / "launcher.exitcode.txt", f"{returncode}\n")
        if returncode != 0:
            _write_case_hashes(case_dir)
            raise LocalGate0Error(
                f"{case_id} runner exited with code {returncode}; evidence retained at "
                f"{case_dir}"
            )
        validate_static_inputs()
        _validate_case_output(
            case_dir,
            case,
            manifest=manifest,
            canonical_manifest_sha256=canonical_digest,
            session_id=session_id,
            source_revision=source_revision,
        )
        _write_case_hashes(case_dir)
        completed_ids.append(case_id)

    observed_dirs = {
        path.name for path in cases_dir.iterdir() if path.is_dir() and not path.is_symlink()
    }
    expected_ids = {case.case_id for case in cases}
    if observed_dirs != expected_ids or len(completed_ids) != EXPECTED_CASE_COUNT:
        raise LocalGate0Error("local Gate 0 evidence is not the exact 16-case matrix")
    all_run_files = tuple(cases_dir.rglob(f"*{RUN_FILE_SUFFIX}"))
    if len(all_run_files) != EXPECTED_CASE_COUNT:
        raise LocalGate0Error("local Gate 0 evidence does not contain exactly 16 run files")

    validate_static_inputs()
    _write_json_exclusive(
        launcher_dir / "completed.json",
        {
            **launcher_identity,
            "completed_case_count": len(completed_ids),
            "case_ids": completed_ids,
        },
    )
    _write_case_hashes(launcher_dir)
    return root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--case-timeout-s", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        root = run_local_gate0(
            asset_root=args.asset_root,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            session_id=args.session_id,
            source_revision=args.source_revision,
            python_executable=args.python_executable,
            case_timeout_s=args.case_timeout_s,
        )
    except (LocalGate0Error, FileExistsError, OSError, ValueError) as exc:
        print(f"local Gate 0 failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "backend": Simulator.MUJOCO.value,
                "case_count": EXPECTED_CASE_COUNT,
                "output_dir": str(root),
                "status": "completed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
