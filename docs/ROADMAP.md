# Roadmap

## v0.1 — CPU-first asset QA

- Discover URDF, MJCF and ASCII USDA entry points.
- Validate references, names, limits, axes, trees and actuator targets.
- Compare canonical joint schemas across formats and model variants.
- Compare fixed-base URDF/MJCF distal-frame kinematics over fixed-seed samples.
- Run headless MuJoCo load, finite-step and repeatability smoke checks.
- Produce deterministic JSON and Markdown reports with strict exit codes.
- Exercise every rule with synthetic and mutation-style unit tests.

## v0.2 — WaveSimParity Gate 0 and MVP

- Pin one canonical manifest for the standard fixed-base left/right hands in position mode.
- Prove the kit-less OVPhysX environment with fully composed USD stages and real finite GPU steps.
- Compare no-contact hold/step and gravity-settling traces against MuJoCo.
- Report repeatability and dt-halving with hashed, locally verifiable result bundles.
- Require 100% completion for the small Gate 0 matrix and at least 99% for the larger MVP.

## v0.3 — Expanded dynamics coverage

- Controlled step, sine and chirp response metrics are complete in Gate 0 and Trajectory T1.
- The synthetic, frictionless fixed-base Contact Gate C0 is complete.
- Add native fingertip contact, wrist/flange, dual-hand and floating-base cases after the
  fixed-base synthetic baseline.
- Keep `DIVERGENT` as a reproducible observation unless separate evidence establishes an upstream bug.

## Future — Full Isaac Sim comparison

- Use an RTX-capable node for a separately scoped official Isaac Sim/Kit comparison.
- Keep renderer/camera evidence separate from the A800 kit-less OVPhysX results.
- Evaluate policy playback only after its official baseline can be reproduced.

Each phase is independently useful. v0.1 real-time factor is only a smoke-test timing; v0.2 will define controlled performance protocols. Later phases must not retroactively overstate what an earlier report validated.
