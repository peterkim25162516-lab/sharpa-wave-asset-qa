from pathlib import Path

from wave_asset_qa.discovery import (
    discover_model_files,
    group_by_canonical_name,
    inventory_counts,
)


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_discovers_and_classifies_supported_models(tmp_path: Path) -> None:
    _touch(tmp_path, "right/right_sharpa_wave_with_wrist.urdf")
    _touch(tmp_path, "right/right_sharpa_wave_with_wrist.xml")
    _touch(tmp_path, "right/right_sharpa_wave_with_wrist.usda")
    _touch(tmp_path, "right/right_sharpa_wave_with_wrist_MITmode.usda")
    _touch(tmp_path, "right/package.xml")
    _touch(tmp_path, "right/mesh.STL")

    models = discover_model_files(tmp_path)

    assert len(models) == 4
    assert inventory_counts(models) == {"mjcf": 1, "urdf": 1, "usda": 2}
    assert {item.control_mode for item in models} == {"position", "mit"}
    assert all(item.side == "right" for item in models)
    assert all(item.mounting == "wrist" for item in models)
    assert set(group_by_canonical_name(models)) == {"right_sharpa_wave_with_wrist"}


def test_detects_float_base_without_misclassifying_mounting(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "float/right_sharpa_wave_with_float_base_with_flange.urdf",
    )
    model = discover_model_files(tmp_path)[0]

    assert model.floating_base is True
    assert model.mounting == "flange"
    assert model.side == "right"


def test_rejects_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    try:
        discover_model_files(missing)
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")
