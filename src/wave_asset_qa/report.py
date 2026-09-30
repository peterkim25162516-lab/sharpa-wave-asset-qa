"""Machine-readable and human-readable audit report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _escape_cell(value: object) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    if not rows:
        return ["_None._"]
    result = [
        "| " + " | ".join(_escape_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(_escape_cell(item) for item in row) + " |" for row in rows)
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a stable, compact Markdown view of an audit report mapping."""

    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    inventory = report.get("inventory", {})
    findings = report.get("findings", [])
    comparisons = report.get("comparisons", [])
    fk_comparisons = report.get("fk_comparisons", [])
    simulations = report.get("mujoco_smoke", [])

    lines = [
        "# Sharpa Wave Asset QA report",
        "",
        "> Unofficial, simulation-only report. Results are specific to the pinned asset revision and software versions below.",
        "",
        "## Summary",
        "",
        f"**Overall status:** `{summary.get('status', 'unknown').upper()}`",
        "",
    ]
    lines.extend(
        _table(
            ["PASS", "KNOWN", "WARN", "FAIL", "SKIP"],
            [[
                summary.get("pass", 0),
                summary.get("known", 0),
                summary.get("warn", 0),
                summary.get("fail", 0),
                summary.get("skip", 0),
            ]],
        )
    )
    lines.extend(["", "## Reproducibility", ""])
    lines.extend(
        _table(
            ["Field", "Value"],
            [[key, value] for key, value in sorted(metadata.items()) if value is not None],
        )
    )
    lines.extend(["", "## Inventory", ""])
    lines.extend(
        _table(
            ["Format", "Entry points"],
            [[key.upper(), value] for key, value in sorted(inventory.items())],
        )
    )

    lines.extend(["", "## Cross-format schema comparison", ""])
    lines.extend(
        _table(
            ["Variant", "Formats", "Common joints", "Missing", "Limit gaps", "Max limit delta (deg)", "Status"],
            [
                [
                    item.get("variant"),
                    ", ".join(item.get("formats", [])),
                    item.get("common_joint_count"),
                    item.get("missing_joint_count"),
                    item.get("incomplete_limit_pair_count", 0),
                    _format_float(item.get("max_limit_delta_deg")),
                    item.get("status"),
                ]
                for item in comparisons
            ],
        )
    )

    lines.extend(["", "## Sampled URDF ↔ MJCF forward kinematics", ""])
    lines.extend(
        _table(
            ["Variant", "Samples", "Joints", "Position p95 (mm)", "Position max (mm)", "Rotation max (deg)", "QA status"],
            [
                [
                    item.get("variant"),
                    item.get("samples"),
                    item.get("sampled_joint_count"),
                    _format_float(item.get("position_p95_mm")),
                    _format_float(item.get("position_max_mm")),
                    _format_float(item.get("orientation_max_deg")),
                    item.get("qa_status", item.get("status")),
                ]
                for item in fk_comparisons
            ],
        )
    )

    lines.extend(["", "## MuJoCo headless smoke checks", ""])
    lines.extend(
        _table(
            ["Model", "Status", "joints", "actuators", "RTF", "Repeat delta", "Message"],
            [
                [
                    item.get("model"),
                    item.get("status"),
                    item.get("njnt"),
                    item.get("nu"),
                    _format_float(item.get("realtime_factor")),
                    _format_float(item.get("max_state_repeat_delta")),
                    item.get("message"),
                ]
                for item in simulations
            ],
        )
    )

    lines.extend(["", "## Findings", ""])
    lines.extend(
        _table(
            ["Status", "Code", "Model", "Message", "Reference"],
            [
                [
                    item.get("status"),
                    item.get("code"),
                    item.get("model"),
                    item.get("message"),
                    item.get("reference"),
                ]
                for item in findings
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Scope limits",
            "",
            "- USDA inspection is text-level overlay validation, not USD stage or PhysX validation.",
            "- MuJoCo smoke checks validate loading, finite stepping and repeatability only.",
            "- This report does not establish controller quality, hardware behavior, Sim2Real or safety.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_float(value: object) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0.0:
        return "0"
    return f"{number:.4g}"


def write_reports(report: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write sorted JSON and Markdown reports and return their paths."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
