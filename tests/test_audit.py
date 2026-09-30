import json
from pathlib import Path

import pytest

from wave_asset_qa.audit import _Recorder, run_audit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "wave_01"
    package = root / "right_sharpa_wave"
    _write(package / "meshes" / "link.stl", "solid link\nendsolid link\n")
    _write(
        package / "right_sharpa_wave.urdf",
        """
        <robot name="right_sharpa_wave">
          <link name="base">
            <visual><geometry><mesh filename="package://right_sharpa_wave/meshes/link.stl"/></geometry></visual>
          </link>
          <link name="right_index_DP"/>
          <joint name="right_index_PIP" type="revolute">
            <parent link="base"/><child link="right_index_DP"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" effort="2" velocity="3"/>
          </joint>
        </robot>
        """,
    )
    _write(
        package / "right_sharpa_wave.xml",
        """
        <mujoco model="right_sharpa_wave">
          <compiler angle="radian" meshdir="meshes"/>
          <asset><mesh name="link" file="link.stl"/></asset>
          <worldbody>
            <body name="base">
              <body name="right_index_DP">
                <joint name="right_index_PIP" axis="0 0 1" range="-1 1"/>
              </body>
            </body>
          </worldbody>
          <actuator><position name="right_index_PIP_ctrl" joint="right_index_PIP" kp="1"/></actuator>
        </mujoco>
        """,
    )
    usda = """
    #usda 1.0
    (defaultPrim = "right_sharpa_wave")
    def Xform "right_sharpa_wave" {
      over "joints" {
        over "right_index_PIP" {
          float physics:lowerLimit = -57.295779513
          float physics:upperLimit = 57.295779513
        }
      }
    }
    """
    _write(package / "right_sharpa_wave.usda", usda)
    _write(package / "right_sharpa_wave_MITmode.usda", usda)

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "expected_active_joints_per_hand": 1,
                "required_fingers": ["index"],
                "required_formats": ["urdf", "mjcf", "usda"],
                "joint_limit_tolerance_rad": 1e-6,
            }
        ),
        encoding="utf-8",
    )
    return root, config


def test_complete_synthetic_matrix_passes_without_absolute_path_leak(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)

    report = run_audit(
        root,
        config_path=config,
        run_simulation=False,
        fk_samples=8,
        fk_seed=3,
    )

    assert report["summary"]["fail"] == 0
    assert report["summary"]["warn"] == 0
    assert report["inventory"] == {"mjcf": 1, "urdf": 1, "usda": 2}
    assert report["comparisons"][0]["common_joint_count"] == 1
    assert report["fk_comparisons"][0]["qa_status"] == "pass"
    assert str(tmp_path.resolve()) not in json.dumps(report)


def test_missing_mesh_is_unknown_failure_or_explicit_known_issue(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    (root / "right_sharpa_wave" / "meshes" / "link.stl").unlink()

    unknown = run_audit(root, config_path=config, run_simulation=False, fk_samples=2)
    assert unknown["summary"]["fail"] == 2

    known_path = tmp_path / "known.json"
    known_path.write_text(
        json.dumps(
            {
                "upstream_commit": "fixture",
                "asset_tree_sha256": unknown["metadata"]["asset_tree_sha256"],
                "issues": [
                    {
                        "id": "fixture-mesh",
                        "codes": ["MESH_REFERENCE_MISSING"],
                        "path_globs": ["right_sharpa_wave/*"],
                        "message_contains": ["link.stl"],
                        "reference": "https://example.com/known",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    known = run_audit(
        root,
        config_path=config,
        known_issues_path=known_path,
        upstream_commit="fixture",
        expected_asset_tree_sha256=unknown["metadata"]["asset_tree_sha256"],
        run_simulation=False,
        fk_samples=2,
    )

    assert known["summary"]["fail"] == 0
    assert known["summary"]["known"] == 2
    assert {item.get("reference") for item in known["findings"] if item["status"] == "known"} == {
        "https://example.com/known"
    }

    claim_only_path = tmp_path / "claim-only.json"
    claim_only_path.write_text(
        json.dumps(
            {
                "upstream_commit": "fixture",
                "issues": [
                    {
                        "codes": ["MESH_REFERENCE_MISSING"],
                        "path_globs": ["right_sharpa_wave/*"],
                        "message_contains": ["link.stl"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    unverified = run_audit(
        root,
        config_path=config,
        known_issues_path=claim_only_path,
        upstream_commit=None,
        run_simulation=False,
        fk_samples=2,
    )
    assert unverified["summary"]["fail"] == 2
    assert unverified["summary"]["known"] == 0


def test_empty_directory_is_not_a_passing_audit(tmp_path: Path) -> None:
    report = run_audit(tmp_path, run_simulation=False, fk_samples=1)

    assert report["summary"]["status"] == "fail"
    assert any(item["code"] == "NO_MODELS_DISCOVERED" for item in report["findings"])


def test_non_git_tree_cannot_claim_commit_without_expected_hash(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)

    with pytest.raises(ValueError, match="Cannot verify"):
        run_audit(
            root,
            config_path=config,
            upstream_commit="claimed",
            run_simulation=False,
            fk_samples=1,
        )


def test_known_missing_set_does_not_hide_an_additional_reference() -> None:
    recorder = _Recorder(
        (
            {
                "codes": ["MESH_REFERENCE_MISSING"],
                "path_globs": ["dual/model.xml"],
                "all_references_match": ["left/meshes/*", "right/meshes/*"],
                "expected_reference_counts": {"dual/model.xml": 1},
            },
        )
    )

    check = recorder.add(
        "fail",
        "MESH_REFERENCE_MISSING",
        "dual/model.xml",
        "two meshes missing",
        details={"references": ["left/meshes/known.stl", "other/new-breakage.stl"]},
    )

    assert check.status == "fail"
