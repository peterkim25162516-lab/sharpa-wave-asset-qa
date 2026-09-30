from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import wave_asset_qa.contact.compare as compare_module
from wave_asset_qa.contact.compare import (
    ContactCheck,
    EvidenceStatus,
    ScienceStatus,
    evaluate_contact_c0,
)
from wave_asset_qa.contact.metrics import ContactRunMetrics, EventInterval
from wave_asset_qa.contact.records import (
    ContactCaseRecord,
    ContactExecutionRecord,
    ContactRun,
)
from wave_asset_qa.contact.scenarios import (
    contact_manifest_sha256,
    expand_contact_cases,
    load_contact_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"


def _matrix_runs(*, duplicate_process_identity: bool = False) -> tuple[ContactRun, ...]:
    manifest = load_contact_manifest(MANIFEST_PATH)
    digest = contact_manifest_sha256(manifest)
    result = []
    for index, case in enumerate(expand_contact_cases(manifest)):
        run = object.__new__(ContactRun)
        steps = round(7.0 / case.dt_s)
        process_digest = "f" * 64 if duplicate_process_identity else f"{index + 1:064x}"
        values = {
            "schema_version": 1,
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": digest,
            "case": ContactCaseRecord.from_dict(case.to_dict()),
            "execution": ContactExecutionRecord("completed", None, steps, steps),
            "mapping": {},
            "fixture_readback": {},
            "contact_observation": {},
            "samples": (),
            "provenance": {
                "backend": case.simulator.value,
                "source_revision": "1" * 40,
                "source_tree": "2" * 40,
                "asset_commit": manifest.provenance.commit,
                "asset_git_tree": manifest.provenance.asset_git_tree,
                "manifest_sha256": digest,
                "fixture_overlay_sha256": "3" * 64,
                "runtime_fingerprint_sha256": "4" * 64,
                "fresh_process_identity_sha256": process_digest,
            },
        }
        for name, value in values.items():
            object.__setattr__(run, name, value)
        result.append(run)
    return tuple(result)


def _metric(run: ContactRun, *, bad_sham_gap: bool = False) -> ContactRunMetrics:
    contact = run.case.condition == "contact"
    event = EventInterval(500, 0.998, 1.0, 0.999) if contact else None
    release = EventInterval(2500, 4.998, 5.0, 4.999) if contact else None
    return ContactRunMetrics(
        case_id=run.case.case_id,
        debounce_interval_count=2 if run.case.dt_s == 0.002 else 4,
        onset_interval=event,
        release_interval=release,
        hold_duty=1.0 if contact else 0.0,
        recovery_duty=0.0,
        initial_gap_m=0.060,
        onset_gap_m=0.0 if contact else None,
        steady_hold_mean_gap_m=(
            0.0 if contact else (-0.001 if bad_sham_gap else -0.010)
        ),
        recovery_min_gap_m=0.060,
        minimum_gap_m=-0.010 if not contact else 0.0,
        debounced_interval_bits=(),
    )


def _stage_check(stage: str, *, passed: bool = True) -> tuple[ContactCheck, ...]:
    return (
        ContactCheck(
            stage=stage,
            check_id=f"{stage}_sentinel",
            passed=passed,
            observed=0.0 if passed else 1.0,
            criterion="<= 0",
        ),
    )


def _patch_science(monkeypatch: pytest.MonkeyPatch, *, cross_pass: bool = True, bad_sham_gap: bool = False) -> None:
    monkeypatch.setattr(
        compare_module,
        "extract_run_metrics",
        lambda run: _metric(run, bad_sham_gap=bad_sham_gap),
    )
    monkeypatch.setattr(
        compare_module, "_repeat_checks", lambda manifest, runs: _stage_check("repeatability")
    )
    monkeypatch.setattr(
        compare_module, "_dt_checks", lambda manifest, runs: _stage_check("dt_halving")
    )
    monkeypatch.setattr(
        compare_module,
        "_crosssim_checks",
        lambda manifest, runs, metrics: _stage_check(
            "cross_simulator", passed=cross_pass
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "_backend_contact_presence",
        lambda runs: {"mujoco": True, "ovphysx": True},
    )


def test_missing_matrix_is_invalid_and_science_is_inconclusive() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    result = evaluate_contact_c0(manifest, ())

    assert result.evidence_status is EvidenceStatus.INVALID
    assert result.science_status is ScienceStatus.INCONCLUSIVE
    assert result.expected_run_count == 32
    assert any("missing cases" in reason for reason in result.invalid_reasons)


def test_valid_prerequisites_and_crosssim_pass_are_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    _patch_science(monkeypatch)

    result = evaluate_contact_c0(manifest, _matrix_runs())

    assert result.evidence_status is EvidenceStatus.VALID
    assert result.science_status is ScienceStatus.WITHIN_TOLERANCE
    assert result.received_run_count == result.completed_run_count == 32
    assert result.scientific_reasons == ()


def test_crosssim_exceedance_is_divergent_only_after_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    _patch_science(monkeypatch, cross_pass=False)
    divergent = evaluate_contact_c0(manifest, _matrix_runs())

    _patch_science(monkeypatch, cross_pass=False, bad_sham_gap=True)
    inconclusive = evaluate_contact_c0(manifest, _matrix_runs())

    assert divergent.evidence_status is EvidenceStatus.VALID
    assert divergent.science_status is ScienceStatus.DIVERGENT
    assert divergent.scientific_reasons == (
        "cross_simulator_threshold_exceeded:cross_simulator_sentinel",
    )
    assert inconclusive.evidence_status is EvidenceStatus.VALID
    assert inconclusive.science_status is ScienceStatus.INCONCLUSIVE
    assert all(check.stage != "cross_simulator" for check in inconclusive.checks)


def test_fresh_process_identity_reuse_invalidates_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    _patch_science(monkeypatch)

    result = evaluate_contact_c0(
        manifest, _matrix_runs(duplicate_process_identity=True)
    )

    assert result.evidence_status is EvidenceStatus.INVALID
    assert result.science_status is ScienceStatus.INCONCLUSIVE
    assert any("fresh process" in reason for reason in result.invalid_reasons)


def test_exactly_one_backend_contact_presence_is_valid_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    _patch_science(monkeypatch)
    monkeypatch.setattr(
        compare_module,
        "_backend_contact_presence",
        lambda runs: {"mujoco": True, "ovphysx": False},
    )

    result = evaluate_contact_c0(manifest, _matrix_runs())

    assert result.evidence_status is EvidenceStatus.VALID
    assert result.science_status is ScienceStatus.DIVERGENT
    assert result.scientific_reasons == ("contact_presence_mismatch",)
    mismatch = next(
        check for check in result.checks if check.check_id == "contact_presence_match"
    )
    assert mismatch.passed is False


def test_neither_backend_contact_presence_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    _patch_science(monkeypatch)
    monkeypatch.setattr(
        compare_module,
        "_backend_contact_presence",
        lambda runs: {"mujoco": False, "ovphysx": False},
    )

    result = evaluate_contact_c0(manifest, _matrix_runs())

    assert result.evidence_status is EvidenceStatus.VALID
    assert result.science_status is ScienceStatus.INCONCLUSIVE
    assert result.scientific_reasons == ("contact_not_exercised",)
    assert all(check.stage != "cross_simulator" for check in result.checks)


def test_contact_presence_deliberately_means_any_raw_active_sample() -> None:
    runs = {
        "mujoco": SimpleNamespace(
            case=SimpleNamespace(simulator="mujoco", condition="contact"),
            samples=(
                SimpleNamespace(pair_active=False),
                SimpleNamespace(pair_active=True),
                SimpleNamespace(pair_active=False),
            ),
        ),
        "ovphysx": SimpleNamespace(
            case=SimpleNamespace(simulator="ovphysx", condition="contact"),
            samples=(SimpleNamespace(pair_active=False),),
        ),
    }

    assert compare_module._backend_contact_presence(runs) == {
        "mujoco": True,
        "ovphysx": False,
    }
