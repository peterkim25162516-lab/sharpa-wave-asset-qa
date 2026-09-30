from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from wave_asset_qa.adapters.base import AdapterProbeResult


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "parity" / "gate0.json"


def _module():
    path = ROOT / "scripts" / "probe_ovphysx_schema.py"
    spec = importlib.util.spec_from_file_location("probe_ovphysx_schema", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_schema_probe_writes_portable_bound_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "gate0"
    asset_root = root / "assets"
    results = root / "results" / "formal" / "launcher" / "schema"
    asset_root.mkdir(parents=True)
    results.mkdir(parents=True)
    manifest_path = root / "project" / "gate0.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(MANIFEST.read_bytes())
    output = results / "left.json"
    monkeypatch.setattr(module, "EXPECTED_APPROVED_ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        module,
        "probe_ovphysx",
        lambda **kwargs: AdapterProbeResult(
            backend="ovphysx",
            mode="resolved_usd_headless",
            status="available",
            message="fixture",
            capabilities={"resolved_usd_open": True, "physics_step": False},
            provenance={
                "imported_modules": {
                    "isaaclab": {"file": "/private/site/isaaclab.py", "version": "x"}
                },
                "usd_inspection": {
                    "source_path": "/data/home/exampleuser/private/left.usd",
                    "source_sha256": "b" * 64,
                },
            },
        ),
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args = module._parser().parse_args(
        [
            "--approved-root",
            str(root),
            "--asset-root",
            str(asset_root),
            "--manifest",
            str(manifest_path),
            "--hand",
            "left",
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--session-id",
            "formal-session",
            "--source-revision",
            "a" * 40,
            "--asset-tree-sha256",
            manifest["provenance"]["canonical_lf_asset_tree_sha256"],
        ]
    )

    payload = module._run(args)

    assert output.is_file()
    provenance = payload["provenance"]
    assert provenance["hand"] == "left"
    assert provenance["session_id"] == "formal-session"
    assert provenance["source_revision"] == "a" * 40
    assert provenance["usd_inspection"]["source_path"].startswith("wave_01/")
    assert "file" not in provenance["imported_modules"]["isaaclab"]
    assert "/data/home/exampleuser" not in json.dumps(payload)
    assert provenance["manifest_file_sha256"] == sha256(manifest_path.read_bytes()).hexdigest()


def test_formal_schema_probe_refuses_overwrite_and_outside_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "gate0"
    (root / "assets").mkdir(parents=True)
    (root / "results").mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(MANIFEST.read_bytes())
    monkeypatch.setattr(module, "EXPECTED_APPROVED_ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    arguments = [
        "--approved-root",
        str(root),
        "--asset-root",
        str(root / "assets"),
        "--manifest",
        str(manifest_path),
        "--hand",
        "left",
        "--device",
        "cuda:0",
        "--session-id",
        "session",
        "--source-revision",
        "a" * 40,
        "--asset-tree-sha256",
        "b" * 64,
    ]
    outside = tmp_path / "outside.json"
    args = module._parser().parse_args([*arguments, "--output", str(outside)])
    with pytest.raises(ValueError, match="escapes"):
        module._run(args)
    assert not outside.exists()
