#!/usr/bin/env python3
"""Record the real Python worker identity, then run one frozen MuJoCo case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from wave_asset_qa.parity.process_identity import (
    current_os_process_identity,
    os_process_identity_sha256,
)


IDENTITY_FILENAME = "worker-process.json"


class FreezeBMuJoCoWorkerError(RuntimeError):
    """Raised before simulation when worker evidence cannot be written exactly."""


def _run_runner(argv: list[str]) -> int:
    # Keep heavy runner/adapter imports after the identity sidecar is durable.
    # This makes the actual worker PID available to the parent before any
    # simulator construction or trajectory execution can begin.
    from wave_asset_qa.parity.runner import main as runner_main

    return runner_main(argv)


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FreezeBMuJoCoWorkerError("refusing to overwrite worker identity")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--case-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve(strict=True)
    identity_output = Path(os.path.abspath(args.identity_output.expanduser()))
    if identity_output != output_dir / IDENTITY_FILENAME:
        raise FreezeBMuJoCoWorkerError(
            "worker identity must be written to the case output directory"
        )
    identity = current_os_process_identity()
    payload: dict[str, object] = {
        "schema_version": 1,
        "worker_pid": os.getpid(),
        "os_process_identity": identity,
        "fresh_process_id": os_process_identity_sha256(identity),
    }
    _write_json_exclusive(identity_output, payload)
    return _run_runner(
        [
            "--backend",
            "mujoco",
            "--asset-root",
            str(args.asset_root),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(output_dir),
            "--session-id",
            args.session_id,
            "--source-revision",
            args.source_revision,
            "--case-id",
            args.case_id,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
