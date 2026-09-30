"""Deterministic discovery and grouping of robot model entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MODEL_SUFFIXES = {".urdf": "urdf", ".xml": "mjcf", ".usda": "usda"}


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """A model entry point found below an asset root."""

    path: Path
    relative_path: str
    format: str
    canonical_name: str
    control_mode: str
    side: str
    mounting: str
    floating_base: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _classify_stem(stem: str) -> tuple[str, str, str, bool]:
    control_mode = "mit" if stem.endswith("_MITmode") else "position"
    canonical_name = stem.removesuffix("_MITmode")

    if canonical_name.startswith("left_"):
        side = "left"
    elif canonical_name.startswith("right_"):
        side = "right"
    elif canonical_name.startswith("dual_"):
        side = "dual"
    else:
        side = "unknown"

    return canonical_name, control_mode, side, "float_base" in canonical_name


def discover_model_files(asset_root: str | Path) -> list[DiscoveredModel]:
    """Find supported entry points below *asset_root* in stable path order.

    ROS ``package.xml`` manifests are deliberately excluded even though they
    share the ``.xml`` suffix with MuJoCo models.
    """

    root = Path(asset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Asset root is not a directory: {root}")

    discovered: list[DiscoveredModel] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.name == "package.xml":
            continue
        model_format = MODEL_SUFFIXES.get(path.suffix.lower())
        if model_format is None:
            continue

        canonical_name, control_mode, side, floating_base = _classify_stem(path.stem)
        if model_format != "usda" and control_mode != "position":
            # Only USDA overlays currently have a separate MIT-mode entry point.
            control_mode = "position"

        discovered.append(
            DiscoveredModel(
                path=path.resolve(),
                relative_path=path.relative_to(root).as_posix(),
                format=model_format,
                canonical_name=canonical_name,
                control_mode=control_mode,
                side=side,
                mounting=mounting_for_name(canonical_name),
                floating_base=floating_base,
            )
        )
    return discovered


def mounting_for_name(canonical_name: str) -> str:
    if canonical_name.endswith("_with_wrist"):
        return "wrist"
    if canonical_name.endswith("_with_flange"):
        return "flange"
    return "standard"


def group_by_canonical_name(
    models: Iterable[DiscoveredModel],
) -> dict[str, list[DiscoveredModel]]:
    grouped: dict[str, list[DiscoveredModel]] = {}
    for model in models:
        grouped.setdefault(model.canonical_name, []).append(model)
    return {
        key: sorted(value, key=lambda item: (item.format, item.control_mode, item.relative_path))
        for key, value in sorted(grouped.items())
    }


def inventory_counts(models: Iterable[DiscoveredModel]) -> dict[str, int]:
    counts: dict[str, int] = {"urdf": 0, "mjcf": 0, "usda": 0}
    for model in models:
        counts[model.format] = counts.get(model.format, 0) + 1
    return dict(sorted(counts.items()))
