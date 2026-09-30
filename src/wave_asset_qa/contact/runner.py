"""CPU-safe selection and validation for Contact Gate C0 run payloads.

Simulator imports intentionally stay out of this module. Workers may use it
after a backend run, while the local finalizer can validate OVPhysX evidence
without importing a GPU runtime.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

from wave_asset_qa.parity.contracts import Simulator
from wave_asset_qa.parity.diagnostics import is_link_like

from .contracts import CONTACT_C0_PAIR_ID, ContactManifest
from .records import (
    ContactCaseRecord,
    ContactRecordValidationError,
    ContactRun,
    canonical_contact_run_json,
    load_contact_run,
)
from .scenarios import (
    ContactCase,
    analytic_signed_gap_m,
    canonical_contact_position_targets,
    contact_manifest_sha256,
    expand_contact_cases,
)


class ContactRunValidationError(ValueError):
    """Raised when a run contradicts the frozen manifest or selected case."""


def select_contact_case(
    manifest: ContactManifest,
    case_id: str,
    *,
    simulator: Simulator | None = None,
) -> ContactCase:
    """Return exactly one frozen case, optionally requiring one backend."""

    if not isinstance(manifest, ContactManifest):
        raise TypeError("manifest must be a ContactManifest")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    if simulator is not None and not isinstance(simulator, Simulator):
        raise TypeError("simulator must be a Simulator or None")
    matches = [case for case in expand_contact_cases(manifest) if case.case_id == case_id]
    if len(matches) != 1:
        raise ContactRunValidationError(
            f"manifest does not contain exactly one case {case_id!r}"
        )
    selected = matches[0]
    if simulator is not None and selected.simulator is not simulator:
        raise ContactRunValidationError(
            f"case {case_id!r} is not a {simulator.value} case"
        )
    return selected


def case_record_from_case(case: ContactCase) -> ContactCaseRecord:
    """Convert the manifest case to the closed portable run identity."""

    if not isinstance(case, ContactCase):
        raise TypeError("case must be a ContactCase")
    return ContactCaseRecord(
        case_id=case.case_id,
        simulator=case.simulator.value,
        hand=case.hand.value,
        scenario_id=case.scenario_id,
        condition=case.condition.value,
        timestep_variant=case.timestep_variant.value,
        repeat_index=case.repeat_index,
        dt_s=case.dt_s,
        model_path=case.model_path,
    )


def _require_close(actual: object, expected: float, label: str, *, atol: float) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol)
    ):
        raise ContactRunValidationError(f"{label} differs from the frozen value")


def _require_vector(
    actual: object,
    expected: tuple[float, ...],
    label: str,
    *,
    atol: float,
) -> None:
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise ContactRunValidationError(f"{label} is missing or has the wrong length")
    for index, target in enumerate(expected):
        _require_close(actual[index], target, f"{label}[{index}]", atol=atol)


def validate_contact_run(
    run: ContactRun,
    *,
    manifest: ContactManifest,
    case: ContactCase,
    source_revision: str | None = None,
    source_tree: str | None = None,
    fresh_process_identity_sha256: str | None = None,
) -> ContactRun:
    """Fail closed unless ``run`` is the complete payload for ``case``.

    Optional source/process identities are supplied by formal workers and the
    finalizer. Keeping them optional also lets unit tests validate synthetic
    records without inventing a Git or kernel identity.
    """

    if not isinstance(run, ContactRun):
        raise TypeError("run must be a ContactRun")
    if not isinstance(manifest, ContactManifest):
        raise TypeError("manifest must be a ContactManifest")
    if not isinstance(case, ContactCase):
        raise TypeError("case must be a ContactCase")
    canonical_manifest_sha = contact_manifest_sha256(manifest)
    if run.manifest_sha256 != canonical_manifest_sha:
        raise ContactRunValidationError("run manifest SHA-256 is not canonical")
    if run.case != case_record_from_case(case):
        raise ContactRunValidationError("run case identity differs from the selected case")
    if not run.completed:
        raise ContactRunValidationError("formal Contact C0 case did not complete")

    hand = manifest.hand(case.hand)
    expected_joints = tuple(hand.joint_names)
    expected_joint_set = set(expected_joints)
    expected_frames = tuple(hand.distal_frame_names)
    expected_frame_set = set(expected_frames)
    mapping = run.mapping
    if (
        mapping.get("expected_joint_count") != 22
        or mapping.get("observed_joint_count") != 22
        or mapping.get("expected_distal_frame_count") != 5
        or mapping.get("observed_distal_frame_count") != 5
        or mapping.get("probe_frame_name") != f"{case.hand.value}_index_DP"
        or mapping.get("probe_frame_mapped") is not True
    ):
        raise ContactRunValidationError("run mapping does not prove 22 joints and five DP frames")

    fixture = run.fixture_readback
    collision = manifest.fixture.collision_policy
    if (
        fixture.get("performed") is not True
        or fixture.get("profile_id") != manifest.fixture.profile_id
        or fixture.get("probe_parent_frame_name") != f"{case.hand.value}_index_DP"
        or fixture.get("native_hand_collisions_enabled")
        is not collision.native_hand_collisions_enabled
        or fixture.get("self_collisions_enabled") is not collision.self_collisions_enabled
        or fixture.get("ccd_enabled") is not collision.ccd_enabled
        or fixture.get("allowed_pair_id") != CONTACT_C0_PAIR_ID
        or fixture.get("mujoco_condim") != collision.mujoco_condim
        or fixture.get("enabled_collision_shape_count")
        != collision.expected_enabled_collision_shape_count
        or fixture.get("mass_properties_preserved") is not True
    ):
        raise ContactRunValidationError("fixture readback contradicts the frozen profile")
    native_count = fixture.get("native_collision_prim_count")
    disabled_count = fixture.get("native_collision_disabled_count")
    if (
        isinstance(native_count, bool)
        or not isinstance(native_count, int)
        or native_count < 1
        or disabled_count != native_count
    ):
        raise ContactRunValidationError("native collision inventory is not fully disabled")
    _require_vector(
        fixture.get("probe_local_center_m"),
        manifest.fixture.probe.local_center_m,
        "fixture probe center",
        atol=1e-12,
    )
    _require_close(
        fixture.get("probe_radius_m"),
        manifest.fixture.probe.radius_m,
        "fixture probe radius",
        atol=1e-12,
    )
    _require_vector(
        fixture.get("target_world_center_m"),
        manifest.fixture.target.world_center_m,
        "fixture target center",
        atol=1e-12,
    )
    _require_vector(
        fixture.get("target_half_extents_m"),
        manifest.fixture.target.half_extents_m,
        "fixture target half extents",
        atol=1e-12,
    )
    _require_close(
        fixture.get("target_top_surface_z_m"),
        manifest.fixture.target.top_surface_z_m,
        "fixture target top",
        atol=1e-12,
    )
    for key in ("static_friction", "dynamic_friction", "restitution"):
        _require_close(
            fixture.get(key),
            float(getattr(manifest.fixture.material, key)),
            f"fixture {key}",
            atol=1e-12,
        )

    observation = run.contact_observation
    if (
        observation.get("performed") is not True
        or observation.get("capability") != "direct_filtered_pair_force_threshold"
        or observation.get("selected_pair_id") != CONTACT_C0_PAIR_ID
        or observation.get("pair_active_source") != collision.pair_active_source
        or observation.get("pair_active_force_threshold_n")
        != collision.pair_active_force_threshold_n
        or observation.get("pair_active_record_count") != len(run.samples)
        or observation.get("missing_pair_active_count") != 0
        or observation.get("filtered_sensor_body_count")
        != collision.expected_filtered_sensor_body_count
        or observation.get("filtered_target_count")
        != collision.expected_filtered_target_count
    ):
        raise ContactRunValidationError("selected-pair observation contract is incomplete")

    for sample in run.samples:
        if set(sample.joint_positions) != expected_joint_set:
            raise ContactRunValidationError("sample joint position coverage is not canonical")
        if set(sample.joint_velocities) != expected_joint_set:
            raise ContactRunValidationError("sample joint velocity coverage is not canonical")
        if set(sample.position_targets) != expected_joint_set:
            raise ContactRunValidationError("sample target coverage is not canonical")
        if set(sample.frame_poses) != expected_frame_set:
            raise ContactRunValidationError("sample DP-frame coverage is not canonical")
        expected_targets = canonical_contact_position_targets(
            manifest.scenario,
            expected_joints,
            step_index=sample.step,
            dt_s=case.dt_s,
        )
        for name in expected_joints:
            _require_close(
                sample.position_targets[name],
                expected_targets[name],
                f"sample {sample.step} target {name}",
                atol=1e-12,
            )
        expected_gap = analytic_signed_gap_m(
            sample.probe_center_world_m, manifest.fixture
        )
        _require_close(
            sample.signed_gap_m,
            expected_gap,
            f"sample {sample.step} analytic gap",
            atol=1e-12,
        )
        native = sample.native_contact_observation
        if not isinstance(native, Mapping):
            raise ContactRunValidationError(
                f"sample {sample.step} lacks the direct selected-pair observation"
            )
        force_norm = native.get("selected_pair_force_norm_n")
        if (
            isinstance(force_norm, bool)
            or not isinstance(force_norm, (int, float))
            or not math.isfinite(float(force_norm))
            or float(force_norm) < 0.0
        ):
            raise ContactRunValidationError(
                f"sample {sample.step} selected-pair force is invalid"
            )
        expected_active = float(force_norm) > collision.pair_active_force_threshold_n
        if sample.pair_active is not expected_active:
            raise ContactRunValidationError(
                f"sample {sample.step} pair_active contradicts direct filtered force"
            )
        if case.condition.value == "sham" and sample.pair_active:
            raise ContactRunValidationError("sham run contains an active selected pair")

    provenance = run.provenance
    expected_provenance: dict[str, object] = {
        "backend": case.simulator.value,
        "asset_commit": manifest.provenance.commit,
        "asset_git_tree": manifest.provenance.asset_git_tree,
        "manifest_sha256": canonical_manifest_sha,
    }
    if source_revision is not None:
        expected_provenance["source_revision"] = source_revision
    if source_tree is not None:
        expected_provenance["source_tree"] = source_tree
    if fresh_process_identity_sha256 is not None:
        expected_provenance[
            "fresh_process_identity_sha256"
        ] = fresh_process_identity_sha256
    mismatches = [
        key for key, expected in expected_provenance.items() if provenance.get(key) != expected
    ]
    if mismatches:
        raise ContactRunValidationError(
            "run provenance differs: " + ", ".join(sorted(mismatches))
        )
    for key in (
        "source_revision",
        "source_tree",
        "asset_commit",
        "asset_git_tree",
        "manifest_sha256",
        "fixture_overlay_sha256",
        "runtime_fingerprint_sha256",
        "fresh_process_identity_sha256",
    ):
        if provenance.get(key) is None:
            raise ContactRunValidationError(f"run provenance {key} is missing")
    return run


def load_and_validate_contact_run(
    path: str | Path,
    *,
    manifest: ContactManifest,
    case: ContactCase,
    source_revision: str | None = None,
    source_tree: str | None = None,
    fresh_process_identity_sha256: str | None = None,
) -> ContactRun:
    """Strictly parse and then validate one worker payload."""

    try:
        run = load_contact_run(path)
    except ContactRecordValidationError as error:
        raise ContactRunValidationError(str(error)) from error
    return validate_contact_run(
        run,
        manifest=manifest,
        case=case,
        source_revision=source_revision,
        source_tree=source_tree,
        fresh_process_identity_sha256=fresh_process_identity_sha256,
    )


def write_contact_run(path: str | Path, run: ContactRun) -> Path:
    """Exclusively write one strict canonical run JSON payload."""

    destination = Path(os.path.abspath(Path(path).expanduser()))
    for candidate in (destination, *destination.parents):
        if candidate.exists() and is_link_like(candidate):
            raise ContactRunValidationError(
                f"contact run path has a linked ancestor: {candidate}"
            )
    if destination.exists() or is_link_like(destination):
        raise FileExistsError(f"refusing to overwrite contact run: {destination}")
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ContactRunValidationError("contact run parent must be a regular directory")
    strict = ContactRun.from_dict(run.to_dict())
    resolved = parent / destination.name
    try:
        with resolved.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_contact_run_json(strict) + "\n")
            handle.flush()
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise
    return resolved.resolve(strict=True)


__all__ = [
    "ContactRunValidationError",
    "case_record_from_case",
    "load_and_validate_contact_run",
    "select_contact_case",
    "validate_contact_run",
    "write_contact_run",
]
