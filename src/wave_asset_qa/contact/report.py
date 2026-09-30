"""Sanitized public JSON and Markdown reporting for Contact Gate C0."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from wave_asset_qa.parity.bundle import write_json_atomic

from .compare import ContactGateResult


_PRIVATE_KEY_TOKENS = (
    "native_contact_observation",
    "selected_pair_force_norm",
    "session_id",
    "process_id",
    "worker_pid",
    "launcher_pid",
    "transport_pid",
    "hostname",
    "username",
    "environment",
    "argv",
    "private_plan",
    "camera_prim_paths",
    "created_sensor_types",
    "scene_inventory_verification_phase",
    "forbidden_module_verification_phase",
    "forbidden_module_inventory",
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|ghp|github_pat|glpat)-?[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
)
_ABSOLUTE_PATH_PATTERNS = (
    # Public diagnostics do not need the text following a machine-local path.
    # Consume through the end of the line so a path segment containing spaces
    # cannot leave its basename or suffix behind after redaction.
    re.compile(r"(?i)file://[^\r\n]*"),
    re.compile(r"(?i)(?<![:\\/])(?:\\\\|//)(?![\\/])(?=[^\r\n]*[\\/])[^\r\n]*"),
    re.compile(r"(?i)(?<![A-Z0-9])[A-Z]:[\\/][^\r\n]*"),
    re.compile(r"(?i)(?<![/\\A-Z0-9])/(?!/)(?=\S)[^\r\n]*"),
)


class ContactReportPrivacyError(ValueError):
    """A public report would disclose a private or machine-local value."""


def _scrub_text(value: str) -> str:
    result = value
    for token in _PRIVATE_KEY_TOKENS:
        result = re.sub(re.escape(token), "[redacted-field]", result, flags=re.IGNORECASE)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[redacted-secret]", result)
    for pattern in _ABSOLUTE_PATH_PATTERNS:
        result = pattern.sub("[redacted-path]", result)
    return result


def _finite_public(value: object, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContactReportPrivacyError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContactReportPrivacyError(f"{context} contains a non-string key")
            lowered = key.lower()
            if any(token in lowered for token in _PRIVATE_KEY_TOKENS):
                raise ContactReportPrivacyError(
                    f"{context} contains forbidden public field {key!r}"
                )
            _finite_public(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_public(item, f"{context}[{index}]")
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _ABSOLUTE_PATH_PATTERNS
    ):
        raise ContactReportPrivacyError(
            f"{context} contains a residual absolute path"
        )


def _sanitize_public_value(value: object) -> object:
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, Mapping):
        return {key: _sanitize_public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_value(item) for item in value]
    return value


def public_summary(result: ContactGateResult) -> dict[str, object]:
    """Build the closed, de-identified public C0 result projection."""

    if not isinstance(result, ContactGateResult):
        raise TypeError("result must be ContactGateResult")
    mapping_complete = (
        result.evidence_status.value == "VALID"
        and result.expected_run_count == 32
        and result.completed_run_count == 32
    )
    pass_ready = mapping_complete and result.science_status.value == "WITHIN_TOLERANCE"
    payload: dict[str, object] = {
        "schema_version": 1,
        "gate_id": "contacts-c0",
        "manifest_id": result.manifest_id,
        "manifest_sha256": result.manifest_sha256,
        "evidence_status": result.evidence_status.value,
        "science_status": result.science_status.value,
        "pass_ready": pass_ready,
        "run_summary": {
            "expected_run_count": result.expected_run_count,
            "received_run_count": result.received_run_count,
            "completed_run_count": result.completed_run_count,
        },
        "mapping": {
            "joint_mapping": "44/44" if mapping_complete else "not_verified",
            "distal_frame_mapping": "10/10" if mapping_complete else "not_verified",
            "complete": mapping_complete,
        },
        "claim_boundary": {
            "synthetic_fixture_only": True,
            "hardware_in_scope": False,
            "native_hand_collision_geometry_validated": False,
            "dual_hand_interaction_in_scope": False,
            "floating_base_in_scope": False,
            "full_isaac_sim_in_scope": False,
            "sim2real_claimed": False,
            "raw_pair_force_cross_simulator_compared": False,
        },
        "checks": [check.to_dict() for check in result.checks],
        "case_metrics": [metric.to_dict() for metric in result.metrics],
        "invalid_reasons": [_scrub_text(item) for item in result.invalid_reasons],
        "scientific_reasons": [
            _scrub_text(item) for item in result.scientific_reasons
        ],
        "interpretation": (
            "DIVERGENT is a thresholded observation under the pinned synthetic "
            "simulation protocol, not a hardware, safety, Sim2Real, or vendor-bug claim."
        ),
    }
    sanitized = _sanitize_public_value(payload)
    if not isinstance(sanitized, dict):
        raise AssertionError("public summary sanitizer changed the root type")
    payload = sanitized
    _finite_public(payload, "public_summary")
    # Also exercise the strict JSON encoder here rather than deferring failure
    # until a report-writing side effect.
    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return payload


def _escape_cell(value: object) -> str:
    rendered = "—" if value is None else _scrub_text(str(value))
    return rendered.replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(value) for value in row) + " |"
        for row in rows
    )
    return lines


def render_contact_markdown(
    result: ContactGateResult | Mapping[str, object],
) -> str:
    """Render the public C0 projection without private evidence payloads."""

    data = public_summary(result) if isinstance(result, ContactGateResult) else result
    if not isinstance(data, Mapping):
        raise TypeError("result must be ContactGateResult or a public summary mapping")
    _finite_public(data, "report")
    run_summary = data.get("run_summary")
    mapping = data.get("mapping")
    boundary = data.get("claim_boundary")
    checks = data.get("checks")
    invalid = data.get("invalid_reasons")
    scientific_reasons = data.get("scientific_reasons")
    if (
        not isinstance(run_summary, Mapping)
        or not isinstance(mapping, Mapping)
        or not isinstance(boundary, Mapping)
    ):
        raise ValueError(
            "public report run_summary, mapping, and claim_boundary must be objects"
        )
    if (
        not isinstance(checks, list)
        or not isinstance(invalid, list)
        or not isinstance(scientific_reasons, list)
    ):
        raise ValueError(
            "public report checks, invalid_reasons, and scientific_reasons "
            "must be arrays"
        )

    lines = [
        "# WaveSimParity Contacts Gate C0",
        "",
        "> Synthetic sphere/box contact-versus-sham simulation evidence only.",
        "",
        "## Result",
        "",
        f"**Evidence:** `{_escape_cell(data.get('evidence_status'))}`  ",
        f"**Science:** `{_escape_cell(data.get('science_status'))}`  ",
        f"**C0 pass-ready:** `{str(data.get('pass_ready')).lower()}`",
        "",
    ]
    lines.extend(
        _table(
            ["Run coverage", "Count"],
            [
                ["Expected", run_summary.get("expected_run_count")],
                ["Received", run_summary.get("received_run_count")],
                ["Completed", run_summary.get("completed_run_count")],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"Mapping: **{_escape_cell(mapping.get('joint_mapping'))} joints**, "
            f"**{_escape_cell(mapping.get('distal_frame_mapping'))} distal frames**.",
        ]
    )
    lines.extend(["", "## Gate checks", ""])
    rows = []
    for raw in checks:
        if not isinstance(raw, Mapping):
            raise ValueError("public report contains a malformed check")
        rows.append(
            [
                raw.get("stage"),
                raw.get("check_id"),
                "PASS" if raw.get("passed") is True else "FAIL",
                raw.get("observed"),
                raw.get("criterion"),
            ]
        )
    lines.extend(_table(["Stage", "Check", "Status", "Observed", "Criterion"], rows))
    if invalid:
        lines.extend(["", "## Evidence blockers", ""])
        lines.extend(f"- {_scrub_text(str(item))}" for item in invalid)
    if scientific_reasons:
        lines.extend(["", "## Scientific classification reasons", ""])
        lines.extend(
            f"- `{_scrub_text(str(item))}`" for item in scientific_reasons
        )
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            f"- `hardware_in_scope = {str(boundary.get('hardware_in_scope')).lower()}`",
            "- `native_hand_collision_geometry_validated = "
            f"{str(boundary.get('native_hand_collision_geometry_validated')).lower()}`",
            f"- `floating_base_in_scope = {str(boundary.get('floating_base_in_scope')).lower()}`",
            f"- `sim2real_claimed = {str(boundary.get('sim2real_claimed')).lower()}`",
            "- Native pair-force magnitudes are descriptive private evidence and are not compared across engines.",
            "- A `DIVERGENT` result is a reproducible threshold exceedance under this pinned protocol, not an official simulator or asset defect finding.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    _finite_public({"markdown": rendered}, "rendered_report")
    return rendered


def write_contact_reports(
    result: ContactGateResult, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write deterministic public JSON and Markdown reports."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    payload = public_summary(result)
    json_path = write_json_atomic(destination / "contact-c0-summary.json", payload)
    markdown_path = destination / "contact-c0-report.md"
    markdown_path.write_text(
        render_contact_markdown(payload), encoding="utf-8", newline="\n"
    )
    return json_path, markdown_path


# Short aliases mirror the existing Gate 0 reporting surface.
render_markdown = render_contact_markdown
write_reports = write_contact_reports


__all__ = [
    "ContactReportPrivacyError",
    "public_summary",
    "render_contact_markdown",
    "render_markdown",
    "write_contact_reports",
    "write_reports",
]
