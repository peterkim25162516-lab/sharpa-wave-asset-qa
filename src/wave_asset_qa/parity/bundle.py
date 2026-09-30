"""Portable, content-addressed result bundles for cross-simulator runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping


BUNDLE_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = "bundle.json"
_MANIFEST_KEYS = frozenset({"schema_version", "created_at_utc", "files", "root_sha256"})
_FILE_KEYS = frozenset({"path", "size", "sha256"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BundleValidationError(ValueError):
    """Raised when a bundle path, manifest, or payload is unsafe or invalid."""


def _safe_relative_path(value: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise BundleValidationError("bundle path must be a string or path-like value") from exc
    if not isinstance(raw, str):
        raise BundleValidationError("bundle path must resolve to text")
    if not raw or any(ord(character) < 32 for character in raw):
        raise BundleValidationError("bundle path must be non-empty and contain no control characters")

    portable = raw.replace("\\", "/")
    windows = PureWindowsPath(raw)
    if (
        portable.startswith("/")
        or PurePosixPath(portable).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
    ):
        raise BundleValidationError(f"bundle path must be relative: {raw!r}")

    parts = portable.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BundleValidationError(
            f"bundle path must be normalized and must not contain '.' or '..': {raw!r}"
        )
    normalized = PurePosixPath(*parts).as_posix()
    if normalized in {"", "."}:
        raise BundleValidationError(f"bundle path must identify a file: {raw!r}")
    return normalized


def _resolved_root(bundle_root: str | os.PathLike[str]) -> Path:
    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise BundleValidationError(f"bundle root is not a directory: {root}")
    return root


def _resolve_member(root: Path, relative_path: str, *, require_file: bool = True) -> Path:
    normalized = _safe_relative_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    try:
        resolved = candidate.resolve(strict=require_file)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise BundleValidationError(
            f"bundle member escapes the root or does not exist: {normalized!r}"
        ) from exc
    if require_file:
        if candidate.is_symlink():
            raise BundleValidationError(f"bundle member must not be a symbolic link: {normalized!r}")
        if not resolved.is_file():
            raise BundleValidationError(f"bundle member is not a regular file: {normalized!r}")
    return resolved


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a streaming SHA-256 digest for one regular file."""

    source = Path(path)
    if not source.is_file():
        raise BundleValidationError(f"cannot hash a non-file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_root_sha256(files: Iterable[Mapping[str, Any]]) -> str:
    """Hash sorted ``path, size, file_sha256`` records into one root digest.

    The framing mirrors the asset-tree hash style: paths are UTF-8 and
    length-prefixed, sizes use unsigned 64-bit big-endian integers, and file
    SHA-256 values are included as their 32 raw bytes.  Creation time and other
    host-specific provenance are intentionally excluded.
    """

    normalized = [_validate_file_record(record) for record in files]
    paths = [record["path"] for record in normalized]
    if len(paths) != len(set(paths)):
        raise BundleValidationError("bundle manifest contains duplicate file paths")

    digest = hashlib.sha256()
    for record in sorted(normalized, key=lambda item: item["path"]):
        path_bytes = record["path"].encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(record["size"].to_bytes(8, "big"))
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_created_at(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BundleValidationError("created_at_utc must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleValidationError("created_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise BundleValidationError("created_at_utc must include a UTC offset")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise BundleValidationError(f"{field} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _validate_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise BundleValidationError("each bundle file record must be an object")
    keys = set(record)
    if keys != _FILE_KEYS:
        missing = sorted(_FILE_KEYS - keys)
        unknown = sorted(keys - _FILE_KEYS)
        raise BundleValidationError(
            f"invalid bundle file record fields: missing={missing}, unknown={unknown}"
        )
    path = _safe_relative_path(record["path"])
    size = record["size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or size >= 1 << 64
    ):
        raise BundleValidationError(f"invalid size for bundle member {path!r}")
    digest = _validate_sha256(record["sha256"], f"sha256 for {path!r}")
    return {"path": path, "size": size, "sha256": digest}


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise BundleValidationError("bundle manifest must be a JSON object")
    keys = set(manifest)
    if keys != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - keys)
        unknown = sorted(keys - _MANIFEST_KEYS)
        raise BundleValidationError(
            f"invalid bundle manifest fields: missing={missing}, unknown={unknown}"
        )

    schema_version = manifest["schema_version"]
    if isinstance(schema_version, bool) or schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"unsupported bundle schema_version: {schema_version!r}; "
            f"expected {BUNDLE_SCHEMA_VERSION}"
        )
    created_at = _validate_created_at(manifest["created_at_utc"])
    raw_files = manifest["files"]
    if not isinstance(raw_files, list):
        raise BundleValidationError("bundle manifest files must be a list")
    files = [_validate_file_record(record) for record in raw_files]
    paths = [record["path"] for record in files]
    if len(paths) != len(set(paths)):
        raise BundleValidationError("bundle manifest contains duplicate file paths")
    if paths != sorted(paths):
        raise BundleValidationError("bundle manifest files must be sorted by path")

    root_digest = _validate_sha256(manifest["root_sha256"], "root_sha256")
    computed = bundle_root_sha256(files)
    if computed != root_digest:
        raise BundleValidationError(
            f"bundle root digest mismatch: expected {root_digest}, computed {computed}"
        )
    return {
        "schema_version": schema_version,
        "created_at_utc": created_at,
        "files": files,
        "root_sha256": root_digest,
    }


def create_bundle_manifest(
    bundle_root: str | os.PathLike[str],
    relative_paths: Iterable[str | os.PathLike[str]],
    *,
    created_at_utc: str | None = None,
    schema_version: int = BUNDLE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Create a deterministic payload manifest below ``bundle_root``.

    Input and stored paths are always safe relative POSIX paths.  The returned
    mapping is JSON serializable and does not write to disk.
    """

    if isinstance(schema_version, bool) or schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"unsupported bundle schema_version: {schema_version!r}; "
            f"expected {BUNDLE_SCHEMA_VERSION}"
        )
    root = _resolved_root(bundle_root)
    normalized_paths = [_safe_relative_path(path) for path in relative_paths]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise BundleValidationError("bundle payload contains duplicate file paths")

    files: list[dict[str, Any]] = []
    for relative_path in sorted(normalized_paths):
        source = _resolve_member(root, relative_path)
        files.append(
            {
                "path": relative_path,
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    manifest = {
        "schema_version": schema_version,
        "created_at_utc": _validate_created_at(
            _utc_timestamp() if created_at_utc is None else created_at_utc
        ),
        "files": files,
        "root_sha256": bundle_root_sha256(files),
    }
    return _validate_manifest(manifest)


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Atomically write stable UTF-8 JSON and return the resolved destination."""

    destination = Path(path).expanduser().resolve()
    if not destination.parent.is_dir():
        raise BundleValidationError(
            f"JSON destination parent is not a directory: {destination.parent}"
        )
    try:
        serialized = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise BundleValidationError("JSON payload is not finite and serializable") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_bundle_manifest(
    bundle_root: str | os.PathLike[str],
    relative_paths: Iterable[str | os.PathLike[str]],
    *,
    manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST_PATH,
    created_at_utc: str | None = None,
) -> Path:
    """Create and atomically write a bundle manifest below ``bundle_root``."""

    root = _resolved_root(bundle_root)
    normalized_manifest_path = _safe_relative_path(manifest_path)
    normalized_payload_paths = [_safe_relative_path(path) for path in relative_paths]
    if normalized_manifest_path in normalized_payload_paths:
        raise BundleValidationError("bundle manifest cannot include itself as a payload file")
    manifest = create_bundle_manifest(
        root,
        normalized_payload_paths,
        created_at_utc=created_at_utc,
    )
    destination = _resolve_member(root, normalized_manifest_path, require_file=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_json_atomic(destination, manifest)


def _load_manifest(
    root: Path,
    manifest: Mapping[str, Any] | str | os.PathLike[str],
) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return _validate_manifest(manifest)
    source = _resolve_member(root, _safe_relative_path(manifest))
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"cannot read bundle manifest {manifest!r}: {exc}") from exc
    return _validate_manifest(loaded)


def verify_bundle(
    bundle_root: str | os.PathLike[str],
    manifest: Mapping[str, Any] | str | os.PathLike[str] = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Verify manifest integrity and every listed payload file.

    The function raises :class:`BundleValidationError` on the first integrity
    failure and otherwise returns a compact, JSON-safe verification summary.
    """

    root = _resolved_root(bundle_root)
    validated = _load_manifest(root, manifest)
    for record in validated["files"]:
        source = _resolve_member(root, record["path"])
        actual_size = source.stat().st_size
        if actual_size != record["size"]:
            raise BundleValidationError(
                f"bundle member size mismatch for {record['path']!r}: "
                f"expected {record['size']}, found {actual_size}"
            )
        actual_digest = sha256_file(source)
        if actual_digest != record["sha256"]:
            raise BundleValidationError(
                f"bundle member SHA-256 mismatch for {record['path']!r}: "
                f"expected {record['sha256']}, found {actual_digest}"
            )

    return {
        "valid": True,
        "schema_version": validated["schema_version"],
        "created_at_utc": validated["created_at_utc"],
        "file_count": len(validated["files"]),
        "total_size": sum(record["size"] for record in validated["files"]),
        "root_sha256": validated["root_sha256"],
    }


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BundleValidationError",
    "DEFAULT_MANIFEST_PATH",
    "bundle_root_sha256",
    "create_bundle_manifest",
    "sha256_file",
    "verify_bundle",
    "write_bundle_manifest",
    "write_json_atomic",
]
