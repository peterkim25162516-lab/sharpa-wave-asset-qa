from pathlib import Path

import wave_asset_qa.cli as cli


def _report(*, fail: int = 0, warn: int = 0) -> dict:
    return {
        "metadata": {},
        "summary": {
            "status": "fail" if fail else ("warn" if warn else "pass"),
            "pass": 1,
            "known": 0,
            "warn": warn,
            "fail": fail,
            "skip": 0,
        },
        "inventory": {},
        "comparisons": [],
        "fk_comparisons": [],
        "mujoco_smoke": [],
        "findings": [],
    }


def test_cli_writes_reports_and_honours_strict_warning_exit(tmp_path: Path, monkeypatch) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    monkeypatch.setattr(cli, "run_audit", lambda *args, **kwargs: _report(warn=1))

    output = tmp_path / "report"
    normal = cli.main(["audit", str(asset_root), "--output", str(output), "--no-mujoco"])
    strict = cli.main(
        ["audit", str(asset_root), "--output", str(output), "--no-mujoco", "--strict"]
    )

    assert normal == 0
    assert strict == 1
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()


def test_cli_returns_one_for_unknown_failure(tmp_path: Path, monkeypatch) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    monkeypatch.setattr(cli, "run_audit", lambda *args, **kwargs: _report(fail=1))

    assert cli.main(["audit", str(asset_root), "--output", str(tmp_path / "out")]) == 1


def test_cli_rejects_claimed_upstream_commit_mismatch(tmp_path: Path, monkeypatch) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    monkeypatch.setattr(cli, "_git_revision", lambda path: "actual")

    result = cli.main(
        ["audit", str(asset_root), "--upstream-commit", "claimed", "--output", str(tmp_path / "out")]
    )

    assert result == 2
