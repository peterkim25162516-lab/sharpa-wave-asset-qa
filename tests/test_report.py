import json
from pathlib import Path

from wave_asset_qa.report import render_markdown, write_reports


SAMPLE = {
    "metadata": {"upstream_commit": "abc123", "python": "3.12"},
    "summary": {"status": "known", "pass": 2, "known": 1, "warn": 0, "fail": 0, "skip": 0},
    "inventory": {"urdf": 1, "mjcf": 1, "usda": 2},
    "comparisons": [
        {
            "variant": "right_hand",
            "formats": ["urdf", "mjcf", "usda"],
            "common_joint_count": 22,
            "missing_joint_count": 0,
            "max_limit_delta_deg": 0.001,
            "status": "pass",
        }
    ],
    "mujoco_smoke": [
        {
            "model": "right.xml",
            "status": "pass",
            "njnt": 22,
            "nu": 22,
            "realtime_factor": 42.2,
            "max_state_repeat_delta": 0.0,
            "message": "ok",
        }
    ],
    "findings": [
        {"status": "known", "code": "KNOWN_TEST", "model": "dual.xml", "message": "a | b"}
    ],
}


def test_markdown_contains_summary_tables_and_escapes_pipes() -> None:
    rendered = render_markdown(SAMPLE)

    assert "Overall status:** `KNOWN`" in rendered
    assert "right_hand" in rendered
    assert "a \\| b" in rendered
    assert "not USD stage or PhysX validation" in rendered


def test_writes_sorted_json_and_markdown(tmp_path: Path) -> None:
    json_path, markdown_path = write_reports(SAMPLE, tmp_path / "output")

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded == SAMPLE
    assert markdown_path.read_text(encoding="utf-8").startswith("# Sharpa Wave Asset QA report")
    assert json_path.read_text(encoding="utf-8").index('"comparisons"') < json_path.read_text(encoding="utf-8").index('"summary"')
