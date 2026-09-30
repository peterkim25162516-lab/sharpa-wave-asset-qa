# Sharpa Wave Asset QA

> Unofficial, simulation-only tooling for reproducible quality assurance of the public Sharpa Wave robot assets.

`sharpa-wave-asset-qa` is a CPU-first validator for URDF, MuJoCo MJCF and ASCII USDA model entry points. It turns asset review into a repeatable test run: discover every configuration, normalize joint metadata, check file references and trees, compare formats, run headless MuJoCo smoke tests, sample forward kinematics and export both JSON and Markdown evidence.

This is an independent project. It is not affiliated with or endorsed by Sharpa Robotics, and it makes no hardware, Sim2Real or safety claims.

Latest completed follow-up (2026-09-30): a separately identified 32-case async + wait contact
campaign finalized as **VALID / DIVERGENT**, not a pass. OVPhysX release dt sensitivity decreased
from 5.5 ms to 0.5 ms with identical recorded motion, while four cross-backend checks still
exceeded frozen thresholds. See [the diagnosis and limitations](docs/CONTACT_ASYNC_RESULT.md)
and [the generated report](results/contact-async/report.md). Original C0 and Gate 0 reports remain unchanged.

Publication: see [public snapshot provenance and remote setup requirements](docs/PUBLIC_SNAPSHOT.md).

## Why this exists

Robot-description drift is easy to miss when one hand is maintained in several simulators and mounting configurations. A file can parse as XML while still containing a missing mesh, a reversed limit, an actuator bound to the wrong joint or a frame convention that differs from another format. This project makes those checks explicit and regression-testable without requiring Isaac Sim.

The first pinned evaluation targets the public [Sharpa robot asset repository](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) at commit `6eea427eb24189519f32b9f21674cd534d3f973c`.

## Current scope

The v0.1 pipeline checks:

- deterministic discovery of URDF, MJCF and USDA entry points;
- XML/USDA parseability and duplicate names;
- link/body trees, parents, roots and cycles;
- scalar joint counts, axes, limits, effort and velocity metadata;
- mesh resolution, missing files and path-case portability;
- MuJoCo actuator targets and joint coverage;
- URDF ↔ MJCF ↔ position-USDA joint-name and limit parity;
- position ↔ MIT-mode USDA overlay parity;
- fixed-seed URDF ↔ MJCF distal-frame FK comparison;
- optional headless MuJoCo load, finite stepping, repeatability and CPU real-time factor;
- structured `PASS`, `WARN`, `KNOWN`, `FAIL` and `SKIP` outcomes.

USDA support in v0.1 is intentionally text-level. A `.usda` pass is not presented as USD-stage or PhysX validation.

## Pinned v0.1 result

The current local report at upstream commit `6eea427` covers 60 entry points: 15 URDF, 15 MJCF and 30 USDA files.

- All 15 configuration groups have complete cross-format joint sets; among explicitly authored limit pairs, the largest normalized difference is `0.001676°`.
- Nine dual/float-base groups contain USDA joints whose limits are inherited or not authored in the text overlay; those comparisons remain `WARN` until a full USD stage is opened.
- 12 of 15 MJCF models load and complete repeated finite headless rollouts in MuJoCo 3.12; state-repeat delta is zero in this CPU run.
- The three dual-hand MJCF entry points reproduce the mesh-path load failure already addressed by upstream [PR #1](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1), so they are reported as `KNOWN`, not claimed as a new finding.
- The standard left/right hands have sampled distal-frame FK maximum position error below `0.000071 mm` and maximum rotation error below `0.000086°`.
- Wrist and flange variants show consistent base-aligned distal offsets of about `29.0 mm` and `29.5 mm`; these remain `WARN` observations until the intended frame convention is confirmed.
- The v0.1 release was accepted with 57 tests. The current WaveSimParity branch has 1010 passing
  tests, including the manifest, adapters, launchers, immutable snapshot, evidence-integrity,
  comparison and reporting contracts.

See the full [v0.1 Markdown report](results/v0.1/report.md), [JSON evidence](results/v0.1/report.json) and [result interpretation notes](docs/RESULTS_NOTES.md). Timing values are machine-specific and should not be compared across hosts without controlling the environment.

## WaveSimParity Gate 0

WaveSimParity compares the same fixed-base, position-controlled left and right hands in MuJoCo and
NVIDIA's experimental kit-less OVPhysX path. It does not run the full Isaac Sim application: there
is no Kit process, renderer, camera or RTX rendering workload.

The authoritative formal Gate 0 run uses the canonical outer position-mode `.usda` entry points,
an explicit IdealPD effort path, and current-state MuJoCo kinematics when sampling distal-link
frames. It completed all 32 cases with 44/44 joint mappings, 10/10 distal-frame mappings, finite
traces, exact repeatability, and dt-halving metrics within their diagnostic thresholds. The
overall result is `DIVERGENT`, not a pass: left and right `small_step` frame-position maxima were
`2.109 mm` and `2.111 mm`, just above the frozen `2.000 mm` threshold. Their joint and
frame-orientation metrics remained within tolerance, as did every `zero_hold` and
`gravity_settling` comparison.

Earlier runs that opened a nested converter `.usd` directly or sampled MuJoCo Cartesian body
fields before refreshing them to the newly integrated `qpos` are retained only as diagnostic
history. They are not the authoritative Gate 0 result.

This is a pinned simulation observation, not automatically a Sharpa, MuJoCo, NVIDIA or Isaac Sim
bug. See the compact [Gate 0 report](results/gate0/report.md), its machine-readable
[summary](results/gate0/summary.json), the frozen [Gate 0 plan](docs/GATE0_PLAN.md), and the
detailed [Gate 0 status](docs/GATE0_STATUS.md).

A bounded post-formal diagnostic held the left `small_step` OVPhysX trace fixed and generated
same-source MuJoCo traces at `1.0x`, `0.5x`, and `0x` compiled dry friction. The canonical `1.0x`
trace reproduced the formal MuJoCo trace exactly; removing dry friction reduced the global joint
and distal-frame-position maxima by `51.36%` and `59.67%`. This supports dry friction as a
material contributor in that one trajectory, not as a complete explanation or backend bug. See
the [friction diagnostic](docs/MUJOCO_FRICTION_DIAGNOSTIC.md).

The preregistered follow-up then adjusted MuJoCo controller Kv so controller Kv plus passive joint
damping equaled the frozen OV explicit-PD Kd. Its result was `mixed_or_inconclusive`: the named
`left_pinky_CMC` window improved by `82.36%`, but the named `left_thumb_DP` position window improved
by only `4.22%`, while frame-position RMS and the global orientation maximum worsened. This does
not support one common viscous retuning as the explanation for both residuals. See the
[total-viscous retuning diagnostic](docs/MUJOCO_TOTAL_VISCOUS_DIAGNOSTIC.md).

The completed post-hoc common-FK check then replayed the frozen OVPhysX and both MuJoCo
22-joint traces through one fixed-base MuJoCo kinematic model, with no dynamics step or fitted
alignment. For the selected left `thumb_DP` window, the unexplained residual norm was about
`0.0053%` of the recorded gap for both candidates, so both satisfy the frozen `CLOSES` rule.
This supports a narrow kinematic-consistency statement: the frame gap follows the recorded joint-
state gap under that common model. It does not identify a dynamics cause or backend bug. See the
[common-FK attribution](docs/THUMB_DP_COMMON_FK_ATTRIBUTION.md).

The R1 Freeze A effective-parameter readback is also complete. It covered all 44 canonical joints
in four fresh kit-less OVPhysX processes, reproduced the mapping, solver evidence records, and
numeric vectors exactly, and executed no experimental trajectory. Friction and armature records
were available at all three frozen evidence layers, while control and solver evidence remained
partly unavailable. All four semantic families are therefore conservatively `UNMAPPABLE`; the
readback does not establish cross-engine equivalence or change formal Gate 0. See the
[effective-readback note](docs/OVPHYSX_EFFECTIVE_READBACK.md), [compact report](results/effective-readback/report.md),
and [machine-readable summary](results/effective-readback/summary.json).

Freeze B is now complete. Its `24/24` isolated cases passed `44/44` joint and `10/10` fingertip
frame mapping, exact repeatability, dt-halving, and finite/sanity checks. The result is `VALID` with
the label `INCONCLUSIVE`: of the eight frozen sensitivity values, `5/8` are at most `0.10`, `3/8`
are above `0.10`, and `0/8` reach the `0.25` support threshold. The runtime readback observation
that `0/8` zero-treatment cases changed from R1 is descriptive only and does not decide the label.

The trajectories remain bound to campaign G (`d890f8e`, tree `29b6d68`); analysis I (`8e2fd84`,
tree `177e470`) made only a disclosed post-collection time-grid compatibility correction and did
not rerun a simulator or change evidence. A local, unpublished H `INVALID` result is retained as
superseded audit evidence. Formal Gate 0 remains `DIVERGENT` with `pass_ready=false`. See the
[Freeze B note](docs/OVPHYSX_FRICTION_FREEZE_B.md), [sanitized report](results/freeze-b/report.md),
and [machine-readable summary](results/freeze-b/summary.json). This remains an unofficial,
simulation-only result; it is not a backend bug, hardware, Sim2Real, or cross-engine equivalence
claim.

The preregistered WaveSimParity Trajectory T1 extension is complete. All `32/32` fresh-process,
contact-free-intent sine/chirp cases completed with `44/44` joint and `10/10` distal-frame mapping,
exact repeatability, finite traces, and dt-halving within the frozen limits. The evidence is
`VALID`, while all four hand/scenario cells are `DIVERGENT` because their distal-frame-position
maxima are about `2.321–2.324 mm` against the frozen `2.000 mm` threshold. Joint and orientation
maxima remain within their frozen thresholds. Formal Gate 0 therefore remains `DIVERGENT` with
`pass_ready=false`; this is not an official bug finding. See the [Trajectory T1 status](docs/WAVEFORM_T1_STATUS.md),
[correction history](docs/WAVEFORM_T1_CORRIGENDUM.md), [sanitized report](results/waveform-t1/report.md),
and [machine-readable summary](results/waveform-t1/summary.json).

The preregistered Contact Gate C0 is also complete. It compares a single synthetic, frictionless
sphere-box contact pair against a sham condition; it does not validate the native fingertip
collision geometry. All `32/32` fresh-process cases completed with `44/44` joint and `10/10`
distal-frame mapping, finite traces, exact repeatability, and valid private evidence. Its result is
`VALID` / `INCONCLUSIVE`, with `pass_ready=false`: the only failed admission metric was OVPhysX
base-versus-halved release-event timing (`5.5 ms` observed against the frozen `4.0 ms` limit).
The frozen decision order therefore stops before a cross-simulator `DIVERGENT` or
`WITHIN_TOLERANCE` label is assigned. See the [Contact C0 status](docs/CONTACT_C0_STATUS.md),
[frozen plan](docs/CONTACT_C0_PLAN.md), [sanitized report](results/contact-c0/report.md), and
[machine-readable summary](results/contact-c0/summary.json). Hardware, Sim2Real, native fingertip
surfaces, dual-hand interaction, and floating-base behavior are outside C0.

## Quick start

Python 3.10 or newer is required. MuJoCo is optional but recommended for the full report.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,simulation]"
```

Fetch the exact public upstream revision on Windows:

```powershell
.\scripts\fetch_sharpa_assets.ps1
```

Then run the audit:

```bash
waveqa audit external/sharpa-urdf-usd-xml/wave_01 \
  --upstream-commit 6eea427eb24189519f32b9f21674cd534d3f973c \
  --output results/v0.1
```

For a dependency-light static run:

```bash
waveqa audit PATH_TO_ASSETS/wave_01 --no-mujoco --output results/static
```

The command returns exit code `1` only for an unknown failure. Known upstream findings remain visible but do not break the default run. Add `--strict` to make warnings fail a local/CI gate.

## Outputs

Each run produces:

- `report.json`: stable, machine-readable inventory, checks, metrics and provenance;
- `report.md`: compact human-readable tables and scope boundaries.

Absolute asset paths are removed from report records. Reproducibility metadata includes the upstream revision, package versions, platform, fixed seed and test sizes.

## Design

```text
asset tree
   │
   ├── discovery ── configuration matrix
   ├── parsers ──── normalized joints / frames / assets / actuators
   ├── static QA ── names / graph / limits / paths / bindings
   ├── FK parity ── fixed-base alignment + sampled distal frames
   └── MuJoCo ───── load / finite step / repeatability / RTF
                         │
                         └── JSON + Markdown report
```

The validator reads upstream assets in place. It does not copy or modify them. Known findings are registered by upstream commit, path, rule code and public reference rather than silently ignored.

## Testing

```bash
pytest
```

Synthetic and mutation-style tests cover malformed XML, missing meshes, duplicate structures, invalid limits, bad actuator targets, transform math, sampled FK and a tiny headless MuJoCo model. GitHub Actions runs CPU tests on Python 3.10 and 3.12; it does not download the full asset repository.

## Limitations

- Static equivalence does not imply identical contact dynamics.
- Sampled FK uses one zero-pose hand-base alignment and reports distal-phalanx frames available in both formats; it does not fit each sample.
- MuJoCo smoke tests do not evaluate controller quality or physical fidelity.
- v0.1 does not validate binary USD layers. Gate 0 separately performs resolved-stage checks on
  the approved remote machine; those results do not retroactively change v0.1.
- The pinned findings apply only to the recorded upstream commit and software versions.

See [ROADMAP.md](docs/ROADMAP.md) for the control-response and remote Isaac Lab phases, and [PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) before making a public release.

## License and attribution

The validator code is released under Apache License 2.0. Sharpa assets are not bundled; when you fetch or redistribute them, preserve the upstream `LICENSE.txt` and `NOTICE.txt`. See this repository's [NOTICE](NOTICE) for attribution and trademark boundaries.
