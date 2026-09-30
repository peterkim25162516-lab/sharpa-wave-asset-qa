from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wave_asset_qa.contact import bundle as contact_bundle
from wave_asset_qa.contact import runner as contact_runner
from wave_asset_qa.contact import worker
from wave_asset_qa.contact import adapters as contact_adapters
from wave_asset_qa.contact.bundle import (
    ContactEvidenceError,
    canonical_json_sha256,
    read_json_strict,
    regular_tree_records,
    verify_adapter_private_evidence,
    write_json_exclusive,
)
from wave_asset_qa.contact.records import (
    ContactExecutionRecord,
    ContactRun,
    ContactTraceSample,
)
from wave_asset_qa.contact.runner import (
    ContactRunValidationError,
    case_record_from_case,
    select_contact_case,
    validate_contact_run,
    write_contact_run,
)
from wave_asset_qa.contact.scenarios import (
    analytic_signed_gap_m,
    canonical_contact_position_targets,
    contact_manifest_sha256,
    load_contact_manifest,
)
from wave_asset_qa.parity.contracts import Simulator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "parity" / "contact_c0.json"
SOURCE_REVISION = "1" * 40
SOURCE_TREE = "2" * 40
FRESH_PROCESS = "3" * 64


def _selected_case():  # type: ignore[no-untyped-def]
    manifest = load_contact_manifest(MANIFEST_PATH)
    case = select_contact_case(
        manifest,
        "mujoco.left.press_hold_release.sham.base.r01",
        simulator=Simulator.MUJOCO,
    )
    return manifest, case


def _error_run() -> ContactRun:
    manifest, case = _selected_case()
    return ContactRun.from_dict(
        {
            "schema_version": 1,
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": contact_manifest_sha256(manifest),
            "case": case.to_dict(),
            "execution": {
                "status": "error",
                "message": "unit-test backend unavailable",
                "requested_steps": 3500,
                "completed_steps": 0,
            },
            "mapping": {
                "expected_joint_count": 22,
                "observed_joint_count": 0,
                "expected_distal_frame_count": 5,
                "observed_distal_frame_count": 0,
                "probe_frame_name": "left_index_DP",
                "probe_frame_mapped": False,
            },
            "fixture_readback": {
                "performed": False,
                "profile_id": None,
                "probe_parent_frame_name": None,
                "probe_local_center_m": None,
                "probe_radius_m": None,
                "target_world_center_m": None,
                "target_half_extents_m": None,
                "target_top_surface_z_m": None,
                "static_friction": None,
                "dynamic_friction": None,
                "restitution": None,
                "native_hand_collisions_enabled": None,
                "self_collisions_enabled": None,
                "ccd_enabled": None,
                "allowed_pair_id": None,
                "mujoco_condim": None,
                "native_collision_prim_count": None,
                "native_collision_disabled_count": None,
                "enabled_collision_shape_count": None,
                "collision_inventory_sha256": None,
                "mass_properties_preserved": None,
            },
            "contact_observation": {
                "performed": False,
                "capability": "unavailable",
                "selected_pair_id": None,
                "pair_active_source": None,
                "pair_active_force_threshold_n": None,
                "pair_active_record_count": 0,
                "missing_pair_active_count": 0,
                "filtered_sensor_body_count": 0,
                "filtered_target_count": 0,
            },
            "samples": [],
            "provenance": {
                "backend": "mujoco",
                "source_revision": None,
                "source_tree": None,
                "asset_commit": None,
                "asset_git_tree": None,
                "manifest_sha256": None,
                "fixture_overlay_sha256": None,
                "runtime_fingerprint_sha256": None,
                "fresh_process_identity_sha256": None,
            },
        }
    )


def _runner_run(*, bad_target: bool = False, bad_gap: bool = False) -> ContactRun:
    """Build a small in-memory record that targets runner-only validation.

    The records layer independently tests the full N+1 trace invariant.  This
    helper deliberately bypasses that constructor so this file can isolate
    the runner's canonical-target and analytic-gap checks without allocating
    a 3,501-sample trace.
    """

    manifest, case = _selected_case()
    hand = manifest.hand(case.hand)
    targets = canonical_contact_position_targets(
        manifest.scenario,
        hand.joint_names,
        step_index=1500,
        dt_s=case.dt_s,
    )
    if bad_target:
        targets = dict(targets)
        targets["left_index_PIP"] += 1e-6
    probe_center = (0.0, 0.0, 0.2)
    gap = analytic_signed_gap_m(probe_center, manifest.fixture)
    if bad_gap:
        gap += 1e-6
    zero_joints = {name: 0.0 for name in hand.joint_names}
    frames = {
        name: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        for name in hand.distal_frame_names
    }
    sample = ContactTraceSample(
        step=1500,
        time_s=3.0,
        joint_positions=zero_joints,
        joint_velocities=zero_joints,
        frame_poses=frames,
        position_targets=targets,
        probe_center_world_m=probe_center,
        signed_gap_m=gap,
        pair_active=False,
        raw_contact_count=0,
        native_contact_observation={"selected_pair_force_norm_n": 0.0},
    )
    collision = manifest.fixture.collision_policy
    run = object.__new__(ContactRun)
    fields = {
        "schema_version": 1,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": contact_manifest_sha256(manifest),
        "case": case_record_from_case(case),
        "execution": ContactExecutionRecord(
            status="completed",
            message=None,
            requested_steps=3500,
            completed_steps=3500,
        ),
        "mapping": {
            "expected_joint_count": 22,
            "observed_joint_count": 22,
            "expected_distal_frame_count": 5,
            "observed_distal_frame_count": 5,
            "probe_frame_name": "left_index_DP",
            "probe_frame_mapped": True,
        },
        "fixture_readback": {
            "performed": True,
            "profile_id": manifest.fixture.profile_id,
            "probe_parent_frame_name": "left_index_DP",
            "probe_local_center_m": list(manifest.fixture.probe.local_center_m),
            "probe_radius_m": manifest.fixture.probe.radius_m,
            "target_world_center_m": list(manifest.fixture.target.world_center_m),
            "target_half_extents_m": list(manifest.fixture.target.half_extents_m),
            "target_top_surface_z_m": manifest.fixture.target.top_surface_z_m,
            "static_friction": manifest.fixture.material.static_friction,
            "dynamic_friction": manifest.fixture.material.dynamic_friction,
            "restitution": manifest.fixture.material.restitution,
            "native_hand_collisions_enabled": False,
            "self_collisions_enabled": False,
            "ccd_enabled": False,
            "allowed_pair_id": collision.allowed_pair_id,
            "mujoco_condim": collision.mujoco_condim,
            "native_collision_prim_count": 54,
            "native_collision_disabled_count": 54,
            "enabled_collision_shape_count": 2,
            "collision_inventory_sha256": "4" * 64,
            "mass_properties_preserved": True,
        },
        "contact_observation": {
            "performed": True,
            "capability": "direct_filtered_pair_force_threshold",
            "selected_pair_id": collision.allowed_pair_id,
            "pair_active_source": collision.pair_active_source,
            "pair_active_force_threshold_n": collision.pair_active_force_threshold_n,
            "pair_active_record_count": 1,
            "missing_pair_active_count": 0,
            "filtered_sensor_body_count": 1,
            "filtered_target_count": 1,
        },
        "samples": (sample,),
        "provenance": {
            "backend": "mujoco",
            "source_revision": SOURCE_REVISION,
            "source_tree": SOURCE_TREE,
            "asset_commit": manifest.provenance.commit,
            "asset_git_tree": manifest.provenance.asset_git_tree,
            "manifest_sha256": contact_manifest_sha256(manifest),
            "fixture_overlay_sha256": "5" * 64,
            "runtime_fingerprint_sha256": "6" * 64,
            "fresh_process_identity_sha256": FRESH_PROCESS,
        },
    }
    for name, value in fields.items():
        object.__setattr__(run, name, value)
    return run


def test_case_selection_and_record_keep_the_exact_backend_model_path() -> None:
    manifest, case = _selected_case()

    assert case.model_path == "wave_01/left_sharpa_wave/left_sharpa_wave.xml"
    assert case_record_from_case(case).model_path == case.model_path
    with pytest.raises(ContactRunValidationError, match="not a ovphysx case"):
        select_contact_case(
            manifest,
            case.case_id,
            simulator=Simulator.OVPHYSX,
        )

    # A traversal-free but noncanonical model path is still rejected by the
    # runner's exact case identity comparison.
    bad_case = replace(
        case_record_from_case(case),
        model_path="wave_01/left_sharpa_wave/not_the_frozen_model.xml",
    )
    incomplete = object.__new__(ContactRun)
    object.__setattr__(incomplete, "manifest_sha256", contact_manifest_sha256(manifest))
    object.__setattr__(incomplete, "case", bad_case)
    object.__setattr__(
        incomplete,
        "execution",
        ContactExecutionRecord(
            status="completed",
            message=None,
            requested_steps=3500,
            completed_steps=3500,
        ),
    )
    with pytest.raises(ContactRunValidationError, match="case identity"):
        validate_contact_run(incomplete, manifest=manifest, case=case)


def test_runner_recomputes_canonical_targets_and_backend_independent_gap() -> None:
    manifest, case = _selected_case()
    valid = _runner_run()
    assert validate_contact_run(
        valid,
        manifest=manifest,
        case=case,
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        fresh_process_identity_sha256=FRESH_PROCESS,
    ) is valid

    with pytest.raises(ContactRunValidationError, match="target left_index_PIP"):
        validate_contact_run(_runner_run(bad_target=True), manifest=manifest, case=case)
    with pytest.raises(ContactRunValidationError, match="analytic gap"):
        validate_contact_run(_runner_run(bad_gap=True), manifest=manifest, case=case)


def test_adapter_preimages_must_match_both_provenance_hashes() -> None:
    evidence = {
        "fixture_overlay": {
            "profile": "synthetic_sphere_box_v1",
            "collision_inventory_hash_preimage": ["probe", "target"],
        },
        "runtime_fingerprint": {
            "backend": "mujoco",
            "solver": "unit-test",
        },
        "unexpected_pair_observations": [],
    }
    run = SimpleNamespace(
        provenance={
            "fixture_overlay_sha256": canonical_json_sha256(
                evidence["fixture_overlay"]
            ),
            "runtime_fingerprint_sha256": canonical_json_sha256(
                evidence["runtime_fingerprint"]
            ),
        }
    )

    assert verify_adapter_private_evidence(evidence, run) == evidence
    tampered = dict(evidence)
    tampered["fixture_overlay"] = {"profile": "tampered"}
    with pytest.raises(ContactEvidenceError, match="fixture overlay preimage"):
        verify_adapter_private_evidence(tampered, run)
    with pytest.raises(ContactEvidenceError, match="not finite"):
        canonical_json_sha256({"nonfinite": float("nan")})


def test_evidence_and_run_writes_are_exclusive_and_preserve_first_bytes(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "private-evidence.json"
    write_json_exclusive(evidence_path, {"first": True})
    first_evidence = evidence_path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_json_exclusive(evidence_path, {"first": False})
    assert evidence_path.read_bytes() == first_evidence

    run_path = tmp_path / "case.run.json"
    write_contact_run(run_path, _error_run())
    first_run = run_path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_contact_run(run_path, _error_run())
    assert run_path.read_bytes() == first_run


def test_evidence_writes_and_inventory_reject_linked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked = tmp_path / "linked"
    child = linked / "child"
    child.mkdir(parents=True)

    # Windows CI may not grant SeCreateSymbolicLinkPrivilege.  Model the
    # already-tested is_link_like predicate here so the ancestor walk itself
    # remains a deterministic regression test on every platform.
    def link_like(path: Path) -> bool:
        return Path(path) == linked

    monkeypatch.setattr(contact_bundle, "is_link_like", link_like)
    monkeypatch.setattr(contact_runner, "is_link_like", link_like)
    monkeypatch.setattr(worker, "is_link_like", link_like)

    with pytest.raises(ContactEvidenceError, match="ancestor"):
        write_json_exclusive(child / "evidence.json", {"x": 1})
    with pytest.raises(ContactEvidenceError, match="ancestor"):
        regular_tree_records(child)
    with pytest.raises(ContactRunValidationError, match="linked ancestor"):
        write_contact_run(child / "run.json", _error_run())
    with pytest.raises(worker.ContactWorkerError, match="ancestor"):
        worker._checked_directory(child, "output directory")
    assert list(child.iterdir()) == []


def test_worker_hashes_only_the_manifest_asset_subtree_and_stops_on_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, case = _selected_case()
    asset_root = tmp_path / "asset-checkout"
    frozen_subtree = asset_root / manifest.provenance.asset_root
    frozen_subtree.mkdir(parents=True)
    output = tmp_path / "worker-output"
    output.mkdir()
    observed: list[Path] = []

    def wrong_tree_hash(path: Path) -> str:
        observed.append(Path(path))
        return "f" * 64

    monkeypatch.setattr(worker, "asset_tree_sha256", wrong_tree_hash)
    monkeypatch.setattr(
        worker,
        "current_os_process_identity",
        lambda: {
            "platform": "windows",
            "pid": 123,
            "process_creation_filetime": 456,
        },
    )
    monkeypatch.setattr(worker, "os_process_identity_sha256", lambda value: FRESH_PROCESS)
    args = argparse.Namespace(
        asset_root=asset_root,
        manifest=MANIFEST_PATH,
        case_id=case.case_id,
        output_dir=output,
        session_id="contact-c0-unit-test",
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        asset_tree_sha256=manifest.provenance.canonical_lf_asset_tree_sha256,
    )

    with pytest.raises(worker.ContactWorkerError, match="materialized asset tree"):
        worker.run_contact_worker(args, backend=Simulator.MUJOCO)

    assert observed == [frozen_subtree.resolve()]
    assert (output / "private-process.json").is_file()
    assert not (output / "private-adapter-evidence.json").exists()
    assert not (output / f"{case.case_id}.run.json").exists()
    with pytest.raises(FileExistsError, match="private-process.json"):
        worker.run_contact_worker(args, backend=Simulator.MUJOCO)


@pytest.mark.parametrize(
    ("timestep_variant", "expected_dt_s", "expected_steps"),
    [("base", 0.002, 3500), ("halved", 0.001, 7000)],
)
@pytest.mark.parametrize('candidate',[False,True])
def test_worker_closes_adapter_preimages_and_provenance_without_simulator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timestep_variant: str,
    expected_dt_s: float,
    expected_steps: int,
    candidate: bool,
) -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    case = select_contact_case(
        manifest,
        (
            "mujoco.left.press_hold_release.sham."
            f"{timestep_variant}.r01"
        ),
        simulator=Simulator.MUJOCO,
    )
    asset_root = tmp_path / "asset-checkout"
    (asset_root / manifest.provenance.asset_root).mkdir(parents=True)
    output = tmp_path / "worker-output"
    output.mkdir()
    collision_preimage = {
        "native_collision_items": [
            {"path": "/asset/native-collider", "disabled": True}
        ],
        "synthetic_enabled_shapes": ["synthetic_index_probe", "static_box"],
    }
    private_evidence = {
        "fixture_overlay": {
            "collision_inventory_hash_preimage": collision_preimage,
            "pair_collision_enabled": False,
        },
        "runtime_fingerprint": {
            "backend": "mujoco",
            "solver": "mock-no-physics",
        },
        "unexpected_pair_observations": [],
    }
    fake_run = SimpleNamespace(
        case=case_record_from_case(case),
        execution=SimpleNamespace(completed_steps=expected_steps),
        provenance={
            "fixture_overlay_sha256": canonical_json_sha256(
                private_evidence["fixture_overlay"]
            ),
            "runtime_fingerprint_sha256": canonical_json_sha256(
                private_evidence["runtime_fingerprint"]
            ),
        },
        fixture_readback={
            "collision_inventory_sha256": canonical_json_sha256(
                collision_preimage
            )
        },
    )
    observed: dict[str, object] = {}

    class FakeMuJoCoAdapter:
        def __init__(self, **kwargs: object) -> None:
            observed["constructor"] = kwargs
            self.last_private_evidence = private_evidence

        def run_contact_case(self, hand, scenario, selected, **kwargs):  # type: ignore[no-untyped-def]
            observed["hand"] = hand.side.value
            observed["case"] = selected
            observed["run_kwargs"] = kwargs
            assert round(scenario.duration_s / kwargs["dt_s"]) == expected_steps
            return fake_run

    def validate_spy(run, **kwargs):  # type: ignore[no-untyped-def]
        observed["validate_run"] = run
        observed["validate_kwargs"] = kwargs
        return run

    def write_spy(path: Path, run):  # type: ignore[no-untyped-def]
        observed["write_run"] = run
        return write_json_exclusive(path, {"case_id": run.case.case_id})

    monkeypatch.setattr(
        worker,
        "current_os_process_identity",
        lambda: {
            "platform": "windows",
            "pid": 123,
            "process_creation_filetime": 456,
        },
    )
    monkeypatch.setattr(worker, "os_process_identity_sha256", lambda value: FRESH_PROCESS)
    monkeypatch.setattr(
        worker,
        "asset_tree_sha256",
        lambda path: manifest.provenance.canonical_lf_asset_tree_sha256,
    )
    monkeypatch.setattr(contact_adapters, "MuJoCoContactAdapter", FakeMuJoCoAdapter)
    monkeypatch.setattr(worker, "validate_contact_run", validate_spy)
    monkeypatch.setattr(worker, "write_contact_run", write_spy)
    args = argparse.Namespace(
        asset_root=asset_root,
        manifest=MANIFEST_PATH,
        case_id=case.case_id,
        output_dir=output,
        session_id="contact-c0-unit-test",
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        asset_tree_sha256=manifest.provenance.canonical_lf_asset_tree_sha256,
    )

    campaign = 'contact-c0-async-v1' if candidate else None
    assert worker.run_contact_worker(args, backend=Simulator.MUJOCO,experimental_campaign=campaign) is fake_run
    if candidate:
        envelope = read_json_strict(output/'private-campaign.json')
        assert envelope['experimental_campaign']==campaign
        assert envelope['formal_c0_replacement'] is False
        assert envelope['intervention'] is None
        from wave_asset_qa.contact.bundle import sha256_file
        assert envelope['run_sha256']==sha256_file(output/f'{case.case_id}.run.json')
        assert read_json_strict(output/'private-process.json')['experimental_campaign']==campaign
    else:
        assert not (output/'private-campaign.json').exists()
    assert observed["case"] == case
    assert observed["run_kwargs"] == {
        "dt_s": expected_dt_s,
        "asset_root": asset_root.resolve(),
        "device": None,
    }
    assert observed["constructor"] == {
        "source_revision": SOURCE_REVISION,
        "source_tree": SOURCE_TREE,
        "fresh_process_identity_sha256": FRESH_PROCESS,
    }
    assert observed["validate_kwargs"] == {
        "manifest": manifest,
        "case": case,
        "source_revision": SOURCE_REVISION,
        "source_tree": SOURCE_TREE,
        "fresh_process_identity_sha256": FRESH_PROCESS,
    }
    assert read_json_strict(output / "private-adapter-evidence.json") == private_evidence
    emitted = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sorted(output.iterdir())
    )
    assert "hardware" not in emitted
    assert "sim2real" not in emitted


def test_worker_entrypoints_remain_simulation_only_scope_guards() -> None:
    manifest = load_contact_manifest(MANIFEST_PATH)
    boundary = manifest.claim_boundary
    assert boundary.hardware is False
    assert boundary.sim2real is False

    mujoco_parser = worker.build_worker_parser(backend=Simulator.MUJOCO)
    ovphysx_parser = worker.build_worker_parser(backend=Simulator.OVPHYSX)
    mujoco_options = {
        option for action in mujoco_parser._actions for option in action.option_strings
    }
    ovphysx_options = {
        option for action in ovphysx_parser._actions for option in action.option_strings
    }
    assert "--device" not in mujoco_options
    assert "--device" in ovphysx_options

    scoped_sources = (
        ROOT / "src" / "wave_asset_qa" / "contact" / "bundle.py",
        ROOT / "src" / "wave_asset_qa" / "contact" / "runner.py",
        ROOT / "src" / "wave_asset_qa" / "contact" / "worker.py",
        ROOT / "scripts" / "run_contact_c0_mujoco_worker.py",
        ROOT / "scripts" / "run_contact_c0_ovphysx_worker.py",
    )
    for source in scoped_sources:
        text = source.read_text(encoding="utf-8").lower()
        assert "hardware" not in text
        assert "sim2real" not in text
    assert "Simulator.MUJOCO" in scoped_sources[-2].read_text(encoding="utf-8")
    assert "Simulator.OVPHYSX" in scoped_sources[-1].read_text(encoding="utf-8")
