from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from wave_asset_qa.parity.bundle import (
    BundleValidationError,
    create_bundle_manifest,
    verify_bundle,
    write_bundle_manifest,
    write_json_atomic,
)


def _payload(bundle: Path) -> list[str]:
    (bundle / "traces").mkdir(parents=True)
    (bundle / "traces" / "left.json").write_text('{"q": [0, 1]}\n', encoding="utf-8")
    (bundle / "provenance.json").write_text('{"backend": "mujoco"}\n', encoding="utf-8")
    return ["traces/left.json", "provenance.json"]


def test_manifest_is_sorted_portable_and_verifiable(tmp_path: Path) -> None:
    relative_paths = _payload(tmp_path)
    manifest = create_bundle_manifest(
        tmp_path,
        relative_paths,
        created_at_utc="2026-08-26T10:00:00Z",
    )

    assert [record["path"] for record in manifest["files"]] == [
        "provenance.json",
        "traces/left.json",
    ]
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    manifest_path = write_json_atomic(tmp_path / "bundle.json", manifest)
    summary = verify_bundle(tmp_path, manifest_path.name)
    assert summary == {
        "valid": True,
        "schema_version": 1,
        "created_at_utc": "2026-08-26T10:00:00Z",
        "file_count": 2,
        "total_size": sum(record["size"] for record in manifest["files"]),
        "root_sha256": manifest["root_sha256"],
    }


def test_root_digest_ignores_creation_time_and_input_order(tmp_path: Path) -> None:
    relative_paths = _payload(tmp_path)
    first = create_bundle_manifest(
        tmp_path,
        relative_paths,
        created_at_utc="2026-08-26T10:00:00Z",
    )
    second = create_bundle_manifest(
        tmp_path,
        reversed(relative_paths),
        created_at_utc="2026-08-27T10:00:00Z",
    )

    assert first["root_sha256"] == second["root_sha256"]
    assert first["files"] == second["files"]


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.json",
        "traces/../../escape.json",
        "/absolute.json",
        r"C:\absolute.json",
        r"\\server\share\payload.json",
    ],
)
def test_bundle_rejects_absolute_and_parent_escape_paths(
    tmp_path: Path, unsafe: str
) -> None:
    _payload(tmp_path)
    with pytest.raises(BundleValidationError, match="relative|normalized"):
        create_bundle_manifest(tmp_path, [unsafe])


def test_verify_detects_payload_tamper(tmp_path: Path) -> None:
    relative_paths = _payload(tmp_path)
    manifest = create_bundle_manifest(
        tmp_path,
        relative_paths,
        created_at_utc="2026-08-26T10:00:00Z",
    )
    (tmp_path / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "traces" / "left.json").write_text('{"q": [99]}\n', encoding="utf-8")

    with pytest.raises(BundleValidationError, match="size mismatch|SHA-256 mismatch"):
        verify_bundle(tmp_path)


def test_verify_detects_manifest_tamper_before_payload(tmp_path: Path) -> None:
    relative_paths = _payload(tmp_path)
    manifest = create_bundle_manifest(
        tmp_path,
        relative_paths,
        created_at_utc="2026-08-26T10:00:00Z",
    )
    tampered = copy.deepcopy(manifest)
    tampered["files"][0]["size"] += 1

    with pytest.raises(BundleValidationError, match="root digest mismatch"):
        verify_bundle(tmp_path, tampered)


def test_atomic_writer_is_stable_and_rejects_nan(tmp_path: Path) -> None:
    destination = tmp_path / "stable.json"
    write_json_atomic(destination, {"z": 1, "a": {"value": "µ"}})
    first = destination.read_bytes()
    write_json_atomic(destination, {"a": {"value": "µ"}, "z": 1})

    assert destination.read_bytes() == first
    assert first.endswith(b"\n")
    assert first.index(b'"a"') < first.index(b'"z"')
    assert list(tmp_path.glob(".stable.json.*.tmp")) == []
    with pytest.raises(BundleValidationError, match="finite"):
        write_json_atomic(destination, {"bad": float("nan")})


def test_write_bundle_manifest_is_not_self_referential(tmp_path: Path) -> None:
    relative_paths = _payload(tmp_path)
    path = write_bundle_manifest(
        tmp_path,
        relative_paths,
        created_at_utc="2026-08-26T10:00:00Z",
    )

    assert path == (tmp_path / "bundle.json").resolve()
    assert verify_bundle(tmp_path)["valid"] is True
    with pytest.raises(BundleValidationError, match="cannot include itself"):
        write_bundle_manifest(tmp_path, [*relative_paths, "bundle.json"])
