#!/usr/bin/env python3
"""Safely install one immutable Gate 0 source archive below the approved root."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import BinaryIO, Sequence


APPROVED_ROOT = "/data/home/exampleuser/sharpa-wave-asset-qa-gate0"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_HASH_MARKER = ".snapshot-sha256"
_MARKERS = frozenset(
    {".source-revision", ".source-tree", ".archive-sha256", _SNAPSHOT_HASH_MARKER}
)
_MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
_MAX_MEMBER_COUNT = 20_000
_MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024


class SnapshotInstallError(RuntimeError):
    """Raised when an archive or destination violates the deployment contract."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_tree_sha256(root: Path) -> str:
    """Hash the complete materialized source tree with a portable v1 format.

    Paths, entry kinds, executable bits, file sizes, and file bytes are bound.
    Timestamps, uid/gid, and platform-specific permission bits are deliberately
    excluded.  The hash marker itself is the sole excluded entry so the other
    provenance markers are covered without creating a circular digest.
    """

    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SnapshotInstallError("snapshot root must be a real directory")
    digest = sha256(b"waveqa-source-snapshot-v1\0")
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in entries:
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        if relative == _SNAPSHOT_HASH_MARKER:
            continue
        if path.is_symlink():
            raise SnapshotInstallError(f"snapshot contains a symbolic link: {relative}")
        mode = path.stat(follow_symlinks=False).st_mode
        encoded = relative.encode("utf-8")
        if stat.S_ISDIR(mode):
            kind = b"D"
            executable = 0
        elif stat.S_ISREG(mode):
            kind = b"F"
            executable = int(bool(mode & 0o111))
        else:
            raise SnapshotInstallError(
                f"snapshot contains a non-regular entry: {relative}"
            )
        digest.update(kind)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(executable.to_bytes(1, "big"))
        if kind == b"F":
            size = path.stat(follow_symlinks=False).st_size
            digest.update(size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _freeze_snapshot(root: Path) -> None:
    """Remove every write bit while preserving whether a file is executable."""

    entries = sorted(
        root.rglob("*"),
        key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
        reverse=True,
    )
    for path in entries:
        if path.is_symlink():
            raise SnapshotInstallError(
                f"snapshot contains a symbolic link: {path.relative_to(root).as_posix()}"
            )
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISREG(mode):
            path.chmod(0o555 if mode & 0o111 else 0o444)
        elif stat.S_ISDIR(mode):
            path.chmod(0o555)
        else:
            raise SnapshotInstallError(
                "snapshot contains a non-regular entry: "
                + path.relative_to(root).as_posix()
            )


def _under(path: Path, root: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SnapshotInstallError(f"{label} must not be a symbolic link")
    resolved = expanded.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SnapshotInstallError(f"{label} escapes the approved root") from exc
    return resolved


def _member_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name:
        raise SnapshotInstallError(f"invalid archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotInstallError(f"unsafe archive member path: {name!r}")
    if ":" in path.parts[0]:
        raise SnapshotInstallError(f"unsafe archive member path: {name!r}")
    if path.parts[0] in _MARKERS:
        raise SnapshotInstallError(f"archive attempts to provide reserved marker: {name!r}")
    return path


def _copy_member(source: BinaryIO, destination: Path, size: int) -> None:
    remaining = size
    with destination.open("xb") as handle:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise SnapshotInstallError(
                    f"archive member ended early: {destination.name}"
                )
            handle.write(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise SnapshotInstallError(
                f"archive member exceeded declared size: {destination.name}"
            )
        handle.flush()
        os.fsync(handle.fileno())


def install_snapshot(
    root: Path,
    archive: Path,
    *,
    expected_archive_sha256: str,
    source_revision: str,
    source_tree: str,
) -> dict[str, object]:
    """Validate and atomically materialize one source snapshot.

    The function never overwrites an existing snapshot or partial directory.
    A failed extraction deliberately leaves its partial directory as evidence.
    """

    root = root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SnapshotInstallError("approved root must be a real directory")
    if _HEX64.fullmatch(expected_archive_sha256) is None:
        raise SnapshotInstallError("expected archive SHA-256 must be lowercase hex")
    if _HEX40.fullmatch(source_revision) is None:
        raise SnapshotInstallError("source revision must be a 40-character commit ID")
    if _HEX40.fullmatch(source_tree) is None:
        raise SnapshotInstallError("source tree must be a 40-character Git tree ID")

    downloads_path = root / "downloads"
    project_path = root / "project"
    if downloads_path.is_symlink() or project_path.is_symlink():
        raise SnapshotInstallError("downloads and project directories must not be symlinks")
    downloads = downloads_path.resolve(strict=True)
    project_root = project_path.resolve(strict=True)
    archive = _under(archive, downloads, "archive")
    if not archive.is_file():
        raise SnapshotInstallError("archive must be a regular file")
    if archive.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise SnapshotInstallError("source archive exceeds the 100 MiB limit")
    actual_archive_sha256 = _file_sha256(archive)
    if actual_archive_sha256 != expected_archive_sha256:
        raise SnapshotInstallError(
            "source archive SHA-256 mismatch: "
            f"expected {expected_archive_sha256}, found {actual_archive_sha256}"
        )

    destination = project_root / source_tree
    partial = project_root / f".{source_tree}.partial.{os.getpid()}"
    if destination.exists() or partial.exists():
        raise SnapshotInstallError("snapshot destination already exists")
    partial.mkdir(mode=0o755)

    seen: set[PurePosixPath] = set()
    expanded_bytes = 0
    regular_files = 0
    with tarfile.open(archive, mode="r:*") as bundle:
        members = bundle.getmembers()
        if len(members) > _MAX_MEMBER_COUNT:
            raise SnapshotInstallError("source archive has too many members")
        for member in members:
            relative = _member_path(member.name.rstrip("/"))
            if relative in seen:
                raise SnapshotInstallError(
                    f"duplicate archive member path: {member.name!r}"
                )
            seen.add(relative)
            target = partial.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=False)
                target.chmod(0o755)
                continue
            if not member.isreg():
                raise SnapshotInstallError(
                    f"archive member must be a regular file or directory: {member.name!r}"
                )
            expanded_bytes += member.size
            if expanded_bytes > _MAX_EXPANDED_BYTES:
                raise SnapshotInstallError("expanded source archive exceeds 2 GiB")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise SnapshotInstallError(f"cannot read archive member: {member.name!r}")
            _copy_member(source, target, member.size)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            regular_files += 1

    if regular_files == 0:
        raise SnapshotInstallError("source archive contains no regular files")
    markers = {
        ".source-revision": source_revision,
        ".source-tree": source_tree,
        ".archive-sha256": actual_archive_sha256,
    }
    for name, value in markers.items():
        marker = partial / name
        with marker.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        marker.chmod(0o444)
    snapshot_sha256 = _snapshot_tree_sha256(partial)
    snapshot_marker = partial / _SNAPSHOT_HASH_MARKER
    with snapshot_marker.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(snapshot_sha256 + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Create the final marker while the staging root is still writable.  On
    # POSIX, freezing the root first would make this exclusive create fail.
    # The marker is excluded from the digest to avoid a circular hash, while
    # every other provenance marker is included.
    _freeze_snapshot(partial)
    partial.chmod(0o555)
    os.replace(partial, destination)
    return {
        "status": "installed",
        "source_revision": source_revision,
        "source_tree": source_tree,
        "archive_sha256": actual_archive_sha256,
        "snapshot_sha256": snapshot_sha256,
        "archive_size": archive.stat().st_size,
        "file_count": regular_files,
        "expanded_size": expanded_bytes,
        "destination_name": destination.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.approved_root.expanduser().resolve(strict=True)
    if root.as_posix() != APPROVED_ROOT:
        raise SnapshotInstallError(
            f"approved root must be exactly {APPROVED_ROOT}"
        )
    result = install_snapshot(
        root,
        args.archive,
        expected_archive_sha256=args.archive_sha256,
        source_revision=args.source_revision,
        source_tree=args.source_tree,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
