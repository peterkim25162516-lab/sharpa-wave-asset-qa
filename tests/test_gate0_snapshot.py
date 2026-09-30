from __future__ import annotations

from hashlib import sha256
import io
import os
from pathlib import Path
import stat
import tarfile

import pytest


def _load_installer():
    import importlib.util

    script = Path(__file__).parents[1] / "scripts" / "install_gate0_snapshot.py"
    spec = importlib.util.spec_from_file_location("install_gate0_snapshot", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(path: Path, members: list[tuple[str, bytes, str]]) -> str:
    with tarfile.open(path, "w") as bundle:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(content)
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                bundle.addfile(info, io.BytesIO(content))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                bundle.addfile(info)
            else:
                raise AssertionError(kind)
    return sha256(path.read_bytes()).hexdigest()


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "gate0"
    (root / "downloads").mkdir(parents=True)
    (root / "project").mkdir()
    return root


def test_installs_regular_snapshot_atomically_with_read_only_markers(
    tmp_path: Path,
) -> None:
    module = _load_installer()
    root = _root(tmp_path)
    archive = root / "downloads" / "source.tar"
    digest = _archive(
        archive,
        [
            ("README.md", b"hello\n", "file"),
            ("scripts/run.sh", b"#!/bin/sh\n", "file"),
        ],
    )
    revision = "a" * 40
    tree = "b" * 40

    result = module.install_snapshot(
        root,
        archive,
        expected_archive_sha256=digest,
        source_revision=revision,
        source_tree=tree,
    )

    destination = root / "project" / tree
    assert result["status"] == "installed"
    assert (destination / "README.md").read_bytes() == b"hello\n"
    assert (destination / ".source-revision").read_text().strip() == revision
    assert (destination / ".source-tree").read_text().strip() == tree
    assert (destination / ".archive-sha256").read_text().strip() == digest
    snapshot_digest = (destination / ".snapshot-sha256").read_text().strip()
    assert snapshot_digest == result["snapshot_sha256"]
    assert snapshot_digest == module._snapshot_tree_sha256(destination)
    for path in (destination, *destination.rglob("*")):
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    if os.name != "nt":
        assert stat.S_IMODE((destination / "scripts" / "run.sh").stat().st_mode) & 0o111
    assert not list((root / "project").glob(f".{tree}.partial.*"))


def test_snapshot_hash_covers_source_and_nonhash_markers(tmp_path: Path) -> None:
    module = _load_installer()
    root = _root(tmp_path)
    archive = root / "downloads" / "source.tar"
    digest = _archive(archive, [("source.py", b"value = 1\n", "file")])
    tree = "b" * 40
    module.install_snapshot(
        root,
        archive,
        expected_archive_sha256=digest,
        source_revision="a" * 40,
        source_tree=tree,
    )
    destination = root / "project" / tree
    expected = (destination / ".snapshot-sha256").read_text().strip()

    source = destination / "source.py"
    source.chmod(0o644)
    source.write_text("value = 2\n", encoding="utf-8")
    source.chmod(0o444)
    assert module._snapshot_tree_sha256(destination) != expected


def test_snapshot_marker_is_created_before_posix_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_installer()
    root = _root(tmp_path)
    archive = root / "downloads" / "source.tar"
    digest = _archive(archive, [("source.py", b"value = 1\n", "file")])
    original_freeze = module._freeze_snapshot
    observed_marker: list[str] = []

    def freeze_after_marker(partial: Path) -> None:
        marker = partial / ".snapshot-sha256"
        assert marker.is_file()
        observed_marker.append(marker.read_text(encoding="ascii").strip())
        original_freeze(partial)

    monkeypatch.setattr(module, "_freeze_snapshot", freeze_after_marker)
    result = module.install_snapshot(
        root,
        archive,
        expected_archive_sha256=digest,
        source_revision="a" * 40,
        source_tree="b" * 40,
    )

    assert observed_marker == [result["snapshot_sha256"]]


def test_archive_cannot_supply_snapshot_hash_marker(tmp_path: Path) -> None:
    module = _load_installer()
    root = _root(tmp_path)
    archive = root / "downloads" / "source.tar"
    digest = _archive(archive, [(".snapshot-sha256", b"0" * 64, "file")])

    with pytest.raises(module.SnapshotInstallError, match="reserved marker"):
        module.install_snapshot(
            root,
            archive,
            expected_archive_sha256=digest,
            source_revision="a" * 40,
            source_tree="b" * 40,
        )


@pytest.mark.parametrize("member", ["../escape", "/absolute", "a\\b"])
def test_rejects_unsafe_member_paths(tmp_path: Path, member: str) -> None:
    module = _load_installer()
    root = _root(tmp_path)
    archive = root / "downloads" / "source.tar"
    digest = _archive(archive, [(member, b"bad", "file")])

    with pytest.raises(module.SnapshotInstallError, match="archive member"):
        module.install_snapshot(
            root,
            archive,
            expected_archive_sha256=digest,
            source_revision="a" * 40,
            source_tree="b" * 40,
        )


def test_rejects_links_hash_mismatch_and_existing_destination(tmp_path: Path) -> None:
    module = _load_installer()
    root = _root(tmp_path)
    archive = root / "downloads" / "source.tar"
    digest = _archive(archive, [("link", b"", "symlink")])

    with pytest.raises(module.SnapshotInstallError, match="regular file or directory"):
        module.install_snapshot(
            root,
            archive,
            expected_archive_sha256=digest,
            source_revision="a" * 40,
            source_tree="b" * 40,
        )

    clean = root / "downloads" / "clean.tar"
    clean_digest = _archive(clean, [("file.txt", b"ok", "file")])
    with pytest.raises(module.SnapshotInstallError, match="SHA-256 mismatch"):
        module.install_snapshot(
            root,
            clean,
            expected_archive_sha256="0" * 64,
            source_revision="a" * 40,
            source_tree="c" * 40,
        )

    (root / "project" / ("d" * 40)).mkdir()
    with pytest.raises(module.SnapshotInstallError, match="already exists"):
        module.install_snapshot(
            root,
            clean,
            expected_archive_sha256=clean_digest,
            source_revision="a" * 40,
            source_tree="d" * 40,
        )
