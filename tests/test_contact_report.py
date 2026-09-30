from __future__ import annotations

import json

import pytest

from wave_asset_qa.contact.compare import (
    ContactCheck,
    ContactGateResult,
    EvidenceStatus,
    ScienceStatus,
)
from wave_asset_qa.contact.report import (
    ContactReportPrivacyError,
    public_summary,
    render_contact_markdown,
    write_contact_reports,
)


def _result(*, invalid_reason: str = "") -> ContactGateResult:
    return ContactGateResult(
        manifest_id="wavesimparity-contact-c0",
        manifest_sha256="a" * 64,
        expected_run_count=32,
        received_run_count=32,
        completed_run_count=32,
        evidence_status=(
            EvidenceStatus.INVALID if invalid_reason else EvidenceStatus.VALID
        ),
        science_status=(
            ScienceStatus.INCONCLUSIVE
            if invalid_reason
            else ScienceStatus.WITHIN_TOLERANCE
        ),
        metrics=(),
        checks=(
            ContactCheck(
                stage="admission",
                check_id="contact_hold_duty_min",
                passed=True,
                observed=1.0,
                criterion=">= 0.95",
            ),
        ),
        invalid_reasons=(() if not invalid_reason else (invalid_reason,)),
        scientific_reasons=(
            ("invalid_evidence",) if invalid_reason else ()
        ),
    )


def test_public_summary_states_hardware_and_native_geometry_are_out_of_scope() -> None:
    summary = public_summary(_result())
    boundary = summary["claim_boundary"]

    assert boundary["hardware_in_scope"] is False
    assert boundary["native_hand_collision_geometry_validated"] is False
    assert boundary["floating_base_in_scope"] is False
    assert boundary["sim2real_claimed"] is False
    assert boundary["raw_pair_force_cross_simulator_compared"] is False
    assert "native_contact_observation" not in json.dumps(summary)


def test_invalid_reason_is_sanitized_before_publication() -> None:
    reason = (
        r"native_contact_observation failed at C:\Users\alice\private\run.json "
        "with sk-liveabcdefghijklmnop"
    )
    rendered = json.dumps(public_summary(_result(invalid_reason=reason)))

    assert "alice" not in rendered
    assert "native_contact_observation" not in rendered
    assert "sk-live" not in rendered
    assert "redacted" in rendered


@pytest.mark.parametrize(
    "absolute_path",
    [
        r"C:\Users\Alice Smith\private evidence\run.json",
        "/data/home/exampleuser/My Run/run.json",
        r"\\server\Shared Folder\run.json",
    ],
)
def test_absolute_paths_with_spaces_are_fully_redacted(
    absolute_path: str,
) -> None:
    summary = public_summary(
        _result(invalid_reason=f"failed at {absolute_path} after basename")
    )

    assert summary["invalid_reasons"] == ["failed at [redacted-path]"]


@pytest.mark.parametrize(
    "private_key",
    [
        "camera_prim_paths",
        "created_sensor_types",
        "scene_inventory_verification_phase",
        "forbidden_module_verification_phase",
        "forbidden_module_inventory",
    ],
)
def test_camera_runtime_private_fields_cannot_enter_public_output(
    private_key: str,
) -> None:
    rendered = json.dumps(
        public_summary(
            _result(invalid_reason=f"private runtime field {private_key}=secret")
        )
    )
    assert private_key not in rendered

    with pytest.raises(ContactReportPrivacyError, match="forbidden"):
        render_contact_markdown(
            {
                "run_summary": {},
                "claim_boundary": {},
                "checks": [],
                "invalid_reasons": [],
                "scientific_reasons": [],
                private_key: "private runtime evidence",
            }
        )


def test_public_projection_rejects_forbidden_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(ContactReportPrivacyError, match="forbidden"):
        render_contact_markdown(
            {
                "run_summary": {},
                "claim_boundary": {},
                "checks": [],
                "invalid_reasons": [],
                "scientific_reasons": [],
                "session_id": "private",
            }
        )
    with pytest.raises(ContactReportPrivacyError, match="non-finite"):
        render_contact_markdown(
            {
                "run_summary": {},
                "claim_boundary": {},
                "checks": [],
                "invalid_reasons": [],
                "scientific_reasons": [],
                "metric": float("nan"),
            }
        )
    with pytest.raises(ContactReportPrivacyError, match="absolute path"):
        render_contact_markdown(
            {
                "run_summary": {},
                "claim_boundary": {},
                "checks": [],
                "invalid_reasons": [],
                "scientific_reasons": [],
                "diagnostic": r"failed at C:\Users\Alice Smith\run.json",
            }
        )


def test_markdown_has_unambiguous_status_and_scope_language() -> None:
    markdown = render_contact_markdown(_result())

    assert "Evidence:** `VALID`" in markdown
    assert "Science:** `WITHIN_TOLERANCE`" in markdown
    assert "hardware_in_scope = false" in markdown
    assert "native_hand_collision_geometry_validated = false" in markdown
    assert "not an official simulator or asset defect" in markdown


def test_report_files_are_deterministic_public_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    json_path, markdown_path = write_contact_reports(_result(), tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == public_summary(_result())
    assert markdown_path.read_text(encoding="utf-8").endswith("\n")
    assert json_path.name == "contact-c0-summary.json"
    assert markdown_path.name == "contact-c0-report.md"
