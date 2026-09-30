"""Small fail-closed evidence helpers for Contact Gate C0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any

from wave_asset_qa.parity.diagnostics import is_link_like

from .records import ContactRun


class ContactEvidenceError(RuntimeError):
    """Raised when private evidence is incomplete or noncanonical."""


def _reject_link_ancestors(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    for candidate in (raw, *raw.parents):
        if candidate.exists() and is_link_like(candidate):
            raise ContactEvidenceError(
                f"{label} has a symbolic-link or junction ancestor: {candidate}"
            )
    return raw


def canonical_json_bytes(value: object) -> bytes:
    """Serialize finite JSON with the same canonical form used by adapters."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContactEvidenceError("evidence is not finite canonical JSON") from error


def canonical_json_sha256(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = _reject_link_ancestors(Path(path), "hash source")
    if is_link_like(source) or not source.is_file():
        raise ContactEvidenceError(f"hash source is not a regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: str | Path, value: object) -> Path:
    """Write one indented JSON file without ever replacing an existing entry."""

    destination = _reject_link_ancestors(Path(path), "evidence destination")
    if destination.exists() or is_link_like(destination):
        raise FileExistsError(f"refusing to overwrite evidence: {destination}")
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ContactEvidenceError("evidence parent must be a regular directory")
    try:
        payload = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise ContactEvidenceError("evidence is not finite JSON") from error
    resolved = parent / destination.name
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise
    return resolved.resolve(strict=True)


def _object_without_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContactEvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> object:
    raise ContactEvidenceError(f"non-finite JSON constant: {token}")


def read_json_strict(path: str | Path) -> object:
    source = _reject_link_ancestors(Path(path), "JSON source")
    if is_link_like(source) or not source.is_file():
        raise ContactEvidenceError(f"JSON source is not a regular file: {source}")
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContactEvidenceError(f"invalid strict JSON: {source}") from error


def verify_adapter_private_evidence(
    evidence: object,
    run: ContactRun,
) -> dict[str, object]:
    """Recompute the two adapter-preimage hashes carried by a closed run."""

    if not isinstance(evidence, Mapping) or any(
        not isinstance(key, str) for key in evidence
    ):
        raise ContactEvidenceError("adapter evidence must be an object")
    fixture = evidence.get("fixture_overlay")
    runtime = evidence.get("runtime_fingerprint")
    if not isinstance(fixture, Mapping) or not isinstance(runtime, Mapping):
        raise ContactEvidenceError("adapter evidence lacks fixture/runtime preimages")
    fixture_hash = canonical_json_sha256(fixture)
    runtime_hash = canonical_json_sha256(runtime)
    if run.provenance.get("fixture_overlay_sha256") != fixture_hash:
        raise ContactEvidenceError("fixture overlay preimage hash mismatch")
    if run.provenance.get("runtime_fingerprint_sha256") != runtime_hash:
        raise ContactEvidenceError("runtime fingerprint preimage hash mismatch")
    return dict(evidence)


def regular_tree_records(
    root: str | Path,
    *,
    exclude: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    """Inventory a regular, link-free tree in stable relative-path order."""

    raw_base = _reject_link_ancestors(Path(root), "evidence root")
    base = raw_base.resolve(strict=True)
    if not base.is_dir():
        raise ContactEvidenceError("evidence root must be a regular directory")
    excluded = set(exclude)
    records: list[dict[str, object]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or is_link_like(path):
            raise ContactEvidenceError(f"evidence tree contains a link: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ContactEvidenceError(
                f"evidence tree contains a special entry: {relative}"
            )
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(records)


def inventory_root_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash an exact ordered inventory, including path, byte count, and digest."""

    normalized: list[dict[str, object]] = []
    previous = ""
    for record in records:
        if set(record) != {"path", "size", "sha256"}:
            raise ContactEvidenceError("inventory record fields are not exact")
        path = record["path"]
        size = record["size"]
        digest = record["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or path <= previous
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ContactEvidenceError("inventory record is invalid or unsorted")
        previous = path
        normalized.append({"path": path, "size": size, "sha256": digest})
    return canonical_json_sha256(normalized)


__all__ = [
    "ContactEvidenceError",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "inventory_root_sha256",
    "read_json_strict",
    "regular_tree_records",
    "sha256_file",
    "verify_adapter_private_evidence",
    "write_json_exclusive",
]
