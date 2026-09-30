"""Stable JSON and Markdown reporting for WaveSimParity Gate 0."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .bundle import write_json_atomic
from .compare import Gate0Comparison


def _escape_cell(value: object) -> str:
    rendered = "—" if value is None else str(value)
    return rendered.replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    if not rows:
        return ["_None._"]
    lines = [
        "| " + " | ".join(_escape_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(item) for item in row) + " |" for row in rows
    )
    return lines


def _format_float(value: object) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0.0:
        return "0"
    return f"{number:.6g}"


def _payload(report: Gate0Comparison | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(report, Gate0Comparison):
        return report.to_dict()
    if not isinstance(report, Mapping):
        raise TypeError("report must be Gate0Comparison or a mapping")
    return report


def _status_label(value: object) -> str:
    return str(value or "unknown").upper()


def _metric_max(results: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values: list[float] = []
    for result in results:
        comparison = result.get("comparison", {})
        if not isinstance(comparison, Mapping):
            continue
        metrics = comparison.get("metrics", {})
        if not isinstance(metrics, Mapping) or name not in metrics:
            continue
        try:
            values.append(float(metrics[name]))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def render_markdown(report: Gate0Comparison | Mapping[str, Any]) -> str:
    """Render a concise report without overstating a divergence observation."""

    data = _payload(report)
    metadata = data.get("metadata", {})
    summary = data.get("summary", {})
    thresholds = data.get("thresholds", {})
    raw_results = data.get("results", [])
    observations = data.get("observations", [])
    if not isinstance(metadata, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("report metadata and summary must be objects")
    if not isinstance(thresholds, Mapping) or not isinstance(raw_results, list):
        raise ValueError("report thresholds must be an object and results must be an array")
    results = [item for item in raw_results if isinstance(item, Mapping)]

    execution_status = _status_label(summary.get("execution_status"))
    comparison_status = _status_label(summary.get("comparison_status"))
    completion = float(summary.get("completion_fraction", 0.0))
    completion_target = float(thresholds.get("minimum_completion_fraction", 1.0))
    nonfinite = int(summary.get("nonfinite_run_count", 0))
    expected_joint_count = int(summary.get("expected_joint_mapping_count", 0))
    expected_frame_count = int(summary.get("expected_distal_frame_mapping_count", 0))
    observed_joint_count = int(summary.get("observed_joint_mapping_count", 0))
    observed_frame_count = int(
        summary.get("observed_distal_frame_mapping_count", 0)
    )
    mapping_complete = bool(summary.get("mapping_validation_complete", False))

    repeat_metrics = {
        "joint": _metric_max(results, "repeat_joint_max_abs_rad"),
        "frame position": _metric_max(results, "repeat_frame_position_max_m"),
        "frame orientation": _metric_max(results, "repeat_frame_orientation_max_rad"),
    }
    dt_metrics = {
        "joint": _metric_max(results, "dt_halving_joint_max_abs_rad"),
        "frame position": _metric_max(results, "dt_halving_frame_position_max_m"),
        "frame orientation": _metric_max(results, "dt_halving_frame_orientation_max_rad"),
    }
    repeat_limits = {
        "joint": thresholds.get("repeat_joint_max_abs_rad"),
        "frame position": thresholds.get("repeat_frame_position_max_m"),
        "frame orientation": thresholds.get("repeat_frame_orientation_max_rad"),
    }
    dt_limits = {
        "joint": thresholds.get("dt_halving_joint_max_abs_rad"),
        "frame position": thresholds.get("dt_halving_frame_position_max_m"),
        "frame orientation": thresholds.get("dt_halving_frame_orientation_max_rad"),
    }

    def diagnostic_status(
        metrics: Mapping[str, float | None], limits: Mapping[str, object]
    ) -> str:
        if any(value is None for value in metrics.values()):
            return "INCONCLUSIVE"
        return (
            "PASS"
            if all(
                float(metrics[name]) <= float(limits[name])  # type: ignore[arg-type]
                for name in metrics
            )
            else "OBSERVED EXCEEDANCE"
        )

    lines = [
        "# WaveSimParity Gate 0 report",
        "",
        "> Unofficial, simulation-only cross-simulator report for the pinned Sharpa Wave assets.",
        "> Backend scope: MuJoCo ↔ kit-less OVPhysX; this is not a full Isaac Sim rendering workflow.",
        "",
        "## Summary",
        "",
        f"**Execution status:** `{execution_status}`  ",
        f"**Comparison status:** `{comparison_status}`",
        "",
    ]
    lines.extend(
        _table(
            ["Check", "Observed", "Target", "Status"],
            [
                [
                    "Canonical joint mapping",
                    f"{observed_joint_count}/{expected_joint_count}",
                    f"{expected_joint_count}/{expected_joint_count}",
                    (
                        "PASS"
                        if mapping_complete
                        and observed_joint_count == expected_joint_count
                        and execution_status == "COMPLETED"
                        else "INCONCLUSIVE"
                    ),
                ],
                [
                    "Canonical distal frames",
                    f"{observed_frame_count}/{expected_frame_count}",
                    f"{expected_frame_count}/{expected_frame_count}",
                    (
                        "PASS"
                        if mapping_complete
                        and observed_frame_count == expected_frame_count
                        and execution_status == "COMPLETED"
                        else "INCONCLUSIVE"
                    ),
                ],
                [
                    "Scenario completion",
                    f"{completion:.2%}",
                    f">= {completion_target:.2%}",
                    "PASS" if completion >= completion_target else "FAIL",
                ],
                [
                    "Finite traces",
                    f"{nonfinite} run(s) with NaN/Inf",
                    "0",
                    (
                        "PASS"
                        if nonfinite == 0 and execution_status == "COMPLETED"
                        else "FAIL" if nonfinite else "INCONCLUSIVE"
                    ),
                ],
                [
                    "Repeatability",
                    ", ".join(
                        f"{name}={_format_float(value)}" for name, value in repeat_metrics.items()
                    ),
                    "reported thresholds",
                    diagnostic_status(repeat_metrics, repeat_limits),
                ],
                [
                    "dt-halving",
                    ", ".join(
                        f"{name}={_format_float(value)}" for name, value in dt_metrics.items()
                    ),
                    "reported thresholds",
                    diagnostic_status(dt_metrics, dt_limits),
                ],
            ],
        )
    )

    lines.extend(["", "## Scenario results", ""])
    scenario_rows: list[list[object]] = []
    for result in results:
        executions = result.get("executions", [])
        comparison = result.get("comparison", {})
        execution_by_backend = {
            item.get("simulator"): item
            for item in executions
            if isinstance(item, Mapping)
        } if isinstance(executions, list) else {}
        comparison_map = comparison if isinstance(comparison, Mapping) else {}
        metrics = comparison_map.get("metrics", {})
        metrics_map = metrics if isinstance(metrics, Mapping) else {}
        scenario_rows.append(
            [
                result.get("hand"),
                result.get("scenario_id"),
                execution_by_backend.get("mujoco", {}).get("execution_status", "missing"),
                execution_by_backend.get("ovphysx", {}).get("execution_status", "missing"),
                comparison_map.get("comparison_status"),
                _format_float(metrics_map.get("crosssim_joint_max_abs_rad")),
                _format_float(metrics_map.get("crosssim_frame_position_max_m")),
                _format_float(metrics_map.get("crosssim_frame_orientation_max_rad")),
                comparison_map.get("message"),
            ]
        )
    lines.extend(
        _table(
            [
                "Hand",
                "Scenario",
                "MuJoCo execution",
                "OVPhysX execution",
                "Comparison",
                "Joint max (rad)",
                "Frame position max (m)",
                "Frame orientation max (rad)",
                "Observation",
            ],
            scenario_rows,
        )
    )

    lines.extend(["", "## Reported diagnostic thresholds", ""])
    lines.extend(
        _table(
            ["Metric", "Threshold"],
            [[name, _format_float(value)] for name, value in sorted(thresholds.items())],
        )
    )

    lines.extend(["", "## Provenance", ""])
    lines.extend(
        _table(
            ["Field", "Value"],
            [[name, value] for name, value in sorted(metadata.items())],
        )
    )

    lines.extend(["", "## Observations", ""])
    if isinstance(observations, list) and observations:
        lines.extend(f"- {str(item)}" for item in observations)
    else:
        lines.append("- No threshold exceedance or execution blocker was observed.")

    lines.extend(
        [
            "",
            "## Interpretation and scope limits",
            "",
            "- `DIVERGENT` means a reported numerical threshold was exceeded under this pinned setup. It is a reproducible research observation, not by itself an official Sharpa, MuJoCo, NVIDIA, or Isaac Sim bug.",
            "- `ERROR` describes execution or data-integrity failure; affected simulator comparisons are `INCONCLUSIVE`, not `DIVERGENT`.",
            "- OVPhysX here is kit-less and headless: no Kit application, renderer, camera, or RTX rendering path is exercised. This is not a full Isaac Sim comparison.",
            "- The scope is fixed-base position-controlled simulation only. It establishes no hardware behavior, safety, or Sim2Real claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report: Gate0Comparison | Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write stable Gate 0 JSON and Markdown reports."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data = dict(_payload(report))
    json_path = write_json_atomic(destination / "report.json", data)
    markdown_path = destination / "report.md"
    markdown_path.write_text(render_markdown(data), encoding="utf-8", newline="\n")
    return json_path, markdown_path


__all__ = ["render_markdown", "write_reports"]
