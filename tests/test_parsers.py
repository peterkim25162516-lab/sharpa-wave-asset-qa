from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wave_asset_qa.models import ModelParseError  # noqa: E402
from wave_asset_qa.parsers import (  # noqa: E402
    parse_mjcf,
    parse_model,
    parse_urdf,
    parse_usda_overlay,
)


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_text(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_urdf_parses_tree_meshes_limits_axes_and_transforms(self) -> None:
        mesh_path = self.write_text("sample_pkg/meshes/finger.stl", "solid fixture")
        urdf_path = self.write_text(
            "sample_pkg/hand.urdf",
            """<?xml version="1.0"?>
<robot name="fixture_hand">
  <link name="base">
    <visual><geometry><mesh filename="package://sample_pkg/meshes/finger.stl" scale="1 1 1"/></geometry></visual>
  </link>
  <link name="finger"/>
  <link name="fingertip_elastomer">
    <collision><geometry><mesh filename="meshes/finger.stl"/></geometry></collision>
  </link>
  <joint name="finger_hinge" type="revolute">
    <parent link="base"/><child link="finger"/>
    <origin xyz="0.1 0.2 0.3" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="1.25" effort="3.2" velocity="4.4"/>
  </joint>
  <joint name="tip_mount" type="fixed">
    <parent link="finger"/><child link="fingertip_elastomer"/>
    <origin xyz="0.04 0 0"/>
  </joint>
</robot>
""",
        )

        model = parse_urdf(urdf_path)

        self.assertEqual(model.format, "urdf")
        self.assertEqual(model.name, "fixture_hand")
        self.assertEqual(model.links, ("base", "finger", "fingertip_elastomer"))
        self.assertEqual(model.root_frames, ("base",))
        self.assertEqual(
            model.tree_edges,
            (
                ("base", "finger", "finger_hinge"),
                ("finger", "fingertip_elastomer", "tip_mount"),
            ),
        )

        hinge = model.get_joint("finger_hinge")
        self.assertIsNotNone(hinge)
        assert hinge is not None
        self.assertEqual(hinge.parent, "base")
        self.assertEqual(hinge.child, "finger")
        self.assertEqual(hinge.axis, (0.0, 1.0, 0.0))
        self.assertEqual(hinge.origin_xyz, (0.1, 0.2, 0.3))
        self.assertEqual(hinge.anchor_xyz, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(hinge.origin_quat_wxyz[0], math.sqrt(0.5), places=7)
        self.assertAlmostEqual(hinge.origin_quat_wxyz[3], math.sqrt(0.5), places=7)
        self.assertEqual(hinge.lower, -0.5)
        self.assertEqual(hinge.upper, 1.25)
        self.assertEqual(hinge.limit_unit, "radian")

        self.assertEqual(len(model.assets), 2)
        self.assertEqual({asset.role for asset in model.assets}, {"visual", "collision"})
        self.assertTrue(all(asset.resolved_path == mesh_path.resolve() for asset in model.assets))
        self.assertTrue(all(asset.exists is True for asset in model.assets))
        self.assertEqual(model.assets[0].metadata["scale"], (1.0, 1.0, 1.0))

    def test_urdf_accepts_explicit_package_root(self) -> None:
        package_root = self.root / "actual_location"
        (package_root / "meshes").mkdir(parents=True)
        (package_root / "meshes" / "part.stl").write_text("solid", encoding="utf-8")
        urdf_path = self.write_text(
            "elsewhere/model.urdf",
            """<robot name="r">
  <link name="base"><visual><geometry><mesh filename="package://robot_assets/meshes/part.stl"/></geometry></visual></link>
</robot>""",
        )

        model = parse_urdf(urdf_path, package_roots={"robot_assets": package_root})
        self.assertEqual(model.assets[0].resolved_path, (package_root / "meshes/part.stl").resolve())
        self.assertTrue(model.assets[0].exists)

    def test_mjcf_parses_nested_bodies_joint_anchor_mesh_and_actuator(self) -> None:
        mesh_path = self.write_text("model/meshes/finger.stl", "solid fixture")
        mjcf_path = self.write_text(
            "model/hand.xml",
            """<mujoco model="fixture_hand">
  <compiler angle="degree" meshdir="meshes"/>
  <asset><mesh name="finger_mesh" file="finger.stl" scale="1 1 1"/></asset>
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="finger" pos="0.1 0 0" quat="0.7071067811865476 0 0 0.7071067811865476">
        <joint name="finger_hinge" type="hinge" pos="0.01 0.02 0.03" axis="0 1 0" range="-90 45"/>
        <body name="fingertip_elastomer" pos="0.04 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="finger_motor" joint="finger_hinge" ctrlrange="-1 1" gear="2"/></actuator>
</mujoco>
""",
        )

        model = parse_mjcf(mjcf_path)

        self.assertEqual(model.format, "mjcf")
        self.assertEqual(model.root_frames, ("base",))
        self.assertEqual(model.get_frame("fingertip_elastomer").parent, "finger")  # type: ignore[union-attr]
        self.assertIsNone(model.get_frame("fingertip_elastomer").joint_name)  # type: ignore[union-attr]
        self.assertIn(("finger", "fingertip_elastomer", None), model.tree_edges)

        hinge = model.get_joint("finger_hinge")
        self.assertIsNotNone(hinge)
        assert hinge is not None
        self.assertEqual(hinge.joint_type, "revolute")
        self.assertEqual(hinge.parent, "base")
        self.assertEqual(hinge.child, "finger")
        self.assertEqual(hinge.origin_xyz, (0.1, 0.0, 0.0))
        self.assertEqual(hinge.anchor_xyz, (0.01, 0.02, 0.03))
        self.assertAlmostEqual(hinge.lower, -math.pi / 2)
        self.assertAlmostEqual(hinge.upper, math.pi / 4)
        self.assertAlmostEqual(hinge.origin_quat_wxyz[0], math.sqrt(0.5), places=7)
        self.assertAlmostEqual(hinge.origin_quat_wxyz[3], math.sqrt(0.5), places=7)

        self.assertEqual(len(model.assets), 1)
        self.assertEqual(model.assets[0].resolved_path, mesh_path.resolve())
        self.assertTrue(model.assets[0].exists)
        self.assertEqual(model.assets[0].metadata["scale"], (1.0, 1.0, 1.0))

        actuator = model.actuator_map["finger_motor"]
        self.assertEqual(actuator.actuator_type, "motor")
        self.assertEqual(actuator.joint, "finger_hinge")
        self.assertEqual((actuator.control_lower, actuator.control_upper), (-1.0, 1.0))
        self.assertEqual(actuator.gear, (2.0,))

    def test_mjcf_radian_limits_are_not_converted(self) -> None:
        mjcf_path = self.write_text(
            "radian.xml",
            """<mujoco><compiler angle="radian"/><worldbody><body name="b"><joint name="j" range="-0.2 0.7"/></body></worldbody></mujoco>""",
        )
        joint = parse_mjcf(mjcf_path).get_joint("j")
        assert joint is not None
        self.assertEqual((joint.lower, joint.upper), (-0.2, 0.7))

    def test_usda_reads_only_immediate_joint_overlays_and_converts_degrees(self) -> None:
        usda_path = self.write_text(
            "hand.usda",
            """#usda 1.0
(
    defaultPrim = "fixture_hand"
)
def Xform "fixture_hand" (
    variants = { string Physics = "PhysX" }
)
{
    over "joints"
    {
        over "joint_a"
        {
            float physics:lowerLimit = -90
            over "nested_diagnostic" { float physics:lowerLimit = -999 }
            float physics:upperLimit = 45
        }
        def PhysicsRevoluteJoint "joint_b"
        {
            double physics:lowerLimit = 0
            double physics:upperLimit = 180
        }
    }
}
""",
        )

        model = parse_usda_overlay(usda_path)

        self.assertEqual(model.name, "fixture_hand")
        self.assertEqual(tuple(joint.name for joint in model.joints), ("joint_a", "joint_b"))
        self.assertNotIn("nested_diagnostic", model.joint_map)
        self.assertAlmostEqual(model.joint_map["joint_a"].lower, -math.pi / 2)
        self.assertAlmostEqual(model.joint_map["joint_a"].upper, math.pi / 4)
        self.assertAlmostEqual(model.joint_map["joint_b"].lower, 0.0)
        self.assertAlmostEqual(model.joint_map["joint_b"].upper, math.pi)
        self.assertEqual(model.joint_map["joint_b"].limit_unit, "radian")
        self.assertEqual(
            model.joint_map["joint_b"].metadata["source_upper_degrees"],
            180.0,
        )

    def test_dispatch_and_parse_errors_are_consistent(self) -> None:
        malformed_urdf = self.write_text("bad.urdf", "<robot><link></robot>")
        malformed_usda = self.write_text(
            "bad.usda",
            '#usda 1.0\nover "joints" { over "j" { float physics:lowerLimit = nope } }',
        )
        unsupported = self.write_text("model.obj", "not a robot model")

        with self.assertRaises(ModelParseError):
            parse_urdf(malformed_urdf)
        with self.assertRaises(ModelParseError):
            parse_usda_overlay(malformed_usda)
        with self.assertRaises(ModelParseError):
            parse_model(unsupported)


if __name__ == "__main__":
    unittest.main()
