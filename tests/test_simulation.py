from pathlib import Path

import pytest

from wave_asset_qa.simulation import run_mujoco_smoke


pytest.importorskip("mujoco")


def test_mujoco_headless_smoke_is_finite_and_repeatable(tmp_path: Path) -> None:
    model = tmp_path / "tiny.xml"
    model.write_text(
        """
        <mujoco model="tiny">
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <body name="link">
              <joint name="hinge" axis="0 0 1" damping="0.1"/>
              <geom type="capsule" size="0.01 0.1" mass="0.1"/>
            </body>
          </worldbody>
          <actuator><position name="servo" joint="hinge" kp="1"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )

    result = run_mujoco_smoke(model, steps=10)

    assert result.passed
    assert result.njnt == 1
    assert result.nu == 1
    assert result.max_state_repeat_delta == 0.0
    assert result.realtime_factor is not None and result.realtime_factor > 0.0


def test_mujoco_smoke_returns_structured_failure(tmp_path: Path) -> None:
    broken = tmp_path / "broken.xml"
    broken.write_text("<mujoco><worldbody>", encoding="utf-8")

    result = run_mujoco_smoke(broken)

    assert result.status == "fail"
    assert "XML" in result.message or "element" in result.message.lower()


def test_mujoco_smoke_rejects_invalid_step_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_mujoco_smoke(tmp_path / "unused.xml", steps=0)


def test_mujoco_smoke_rejects_negative_repeat_tolerance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        run_mujoco_smoke(tmp_path / "unused.xml", repeat_tolerance=-1.0)
