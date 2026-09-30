"""Command-line interface for Wave asset validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .audit import run_audit
from .report import write_reports


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _existing_default(relative_path: str) -> Path | None:
    candidate = _repository_root() / relative_path
    return candidate if candidate.is_file() else None


def _upstream_lock() -> dict[str, str]:
    path = _existing_default("upstream.lock.json")
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {key: str(value) for key, value in loaded.items() if isinstance(value, str)}


def _git_revision(asset_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(asset_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="waveqa",
        description="CPU-first cross-format QA for public Sharpa Wave robot assets.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit a Wave asset tree and write JSON/Markdown reports")
    audit.add_argument("asset_root", type=Path, help="directory containing URDF, MJCF and USDA entry points")
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest"),
        help="report directory (default: results/latest)",
    )
    audit.add_argument("--config", type=Path, help="QA configuration JSON")
    audit.add_argument("--known-issues", type=Path, help="known-issue registry JSON")
    audit.add_argument("--upstream-commit", help="asset revision recorded in the report")
    audit.add_argument("--no-mujoco", action="store_true", help="skip MuJoCo load/step checks")
    audit.add_argument("--simulation-steps", type=int, default=50, help="headless steps per MuJoCo repeat")
    audit.add_argument("--fk-samples", type=int, default=256, help="fixed-seed FK samples per eligible pair")
    audit.add_argument("--seed", type=int, default=20260826, help="FK sampling seed")
    audit.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 1 for warnings as well as unknown failures",
    )
    return parser


def _audit_command(args: argparse.Namespace) -> int:
    asset_root = args.asset_root.expanduser().resolve()
    config = args.config or _existing_default("configs/default.json")
    known = args.known_issues or _existing_default("configs/known_issues.json")
    detected_revision = _git_revision(asset_root)
    if args.upstream_commit and detected_revision and args.upstream_commit != detected_revision:
        raise ValueError(
            "--upstream-commit does not match the asset checkout HEAD: "
            f"expected {args.upstream_commit}, actual {detected_revision}"
        )
    revision = detected_revision or args.upstream_commit
    lock = _upstream_lock()
    expected_tree_hash = (
        lock.get("asset_tree_sha256")
        if revision == lock.get("commit")
        else None
    )

    report = run_audit(
        asset_root,
        config_path=config,
        known_issues_path=known,
        upstream_commit=revision,
        expected_asset_tree_sha256=expected_tree_hash,
        run_simulation=not args.no_mujoco,
        simulation_steps=args.simulation_steps,
        fk_samples=args.fk_samples,
        fk_seed=args.seed,
    )
    json_path, markdown_path = write_reports(report, args.output)
    summary = report["summary"]
    print(
        "Wave asset QA: "
        f"status={str(summary['status']).upper()} "
        f"pass={summary.get('pass', 0)} known={summary.get('known', 0)} "
        f"warn={summary.get('warn', 0)} fail={summary.get('fail', 0)} "
        f"skip={summary.get('skip', 0)}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    if summary.get("fail", 0):
        return 1
    if args.strict and summary.get("warn", 0):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _audit_command(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"waveqa: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
