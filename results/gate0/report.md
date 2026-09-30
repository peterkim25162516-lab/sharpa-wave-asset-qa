# WaveSimParity Gate 0 report

> Unofficial, simulation-only cross-simulator report for the pinned Sharpa Wave assets.
> Backend scope: MuJoCo ↔ kit-less OVPhysX; this is not a full Isaac Sim rendering workflow.

## Summary

**Execution status:** `COMPLETED`

**Comparison status:** `DIVERGENT`

**Gate 0 acceptance:** `NOT PASS-READY` (`pass_ready=false`)

The authoritative formal run completed all requested work and produced a conclusive comparison. Only
the frame-position metric in the two `small_step` comparisons exceeded its frozen tolerance. The
largest exceedance was about `0.111 mm`; joint and frame-orientation metrics remained within their
thresholds.

| Check | Observed | Target | Status |
| --- | --- | --- | --- |
| Canonical joint mapping | 44/44 | 44/44 | PASS |
| Canonical distal frames | 10/10 | 10/10 | PASS |
| Resolved standard position USD | left and right available | both available | PASS |
| Scenario completion | 32/32 runs, 100% requested steps | 100% | PASS |
| Finite traces | 0 run with NaN/Inf | 0 | PASS |
| Position-target readback | maximum error 0 rad | 0 rad | PASS |
| Explicit PD formula / clipping | maximum validation error 0 Nm; 0 clips | 0 error | PASS |
| Repeatability | joint=0, position=0, orientation=0 | reported thresholds | PASS |
| dt-halving | joint=0.00135195 rad, position=0.000161719 m, orientation=0.00284117 rad | reported thresholds | PASS |
| Cross-simulator acceptance | two frame-position exceedances | all metrics within tolerance | DIVERGENT |

Execution completion and a usable result do not mean the scientific acceptance gate passed.

## Scenario results

| Hand | Scenario | MuJoCo | OVPhysX | Comparison | Joint max (rad) | Frame position max (m) | Frame orientation max (rad) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| left | zero_hold | completed | completed | within tolerance | 0 | 4.7507e-08 | 1.20065e-06 |
| left | small_step | completed | completed | DIVERGENT | 0.0186829 | 0.00210924 | 0.0457305 |
| left | gravity_settling | completed | completed | within tolerance | 0.00277094 | 0.000327972 | 0.00383823 |
| right | zero_hold | completed | completed | within tolerance | 0 | 6.22974e-08 | 8.47544e-07 |
| right | small_step | completed | completed | DIVERGENT | 0.0187022 | 0.00211119 | 0.0458414 |
| right | gravity_settling | completed | completed | within tolerance | 0.00272553 | 0.000319917 | 0.00369807 |

The frozen cross-simulator thresholds are `0.02 rad` for joints, `0.002 m` for frame position,
and `0.05 rad` for frame orientation. The dt-halving thresholds are `0.01 rad`, `0.001 m`, and
`0.02 rad`, respectively. Repeatability thresholds are `1e-9` in the corresponding units.

Left and right `small_step` DP/distal-link frame-position maxima exceeded the `2.000 mm` threshold
by about `0.109 mm` and `0.111 mm`. The compared `*_DP` frames are distal-phalanx link origins,
not physical fingertip contact-surface points. Thresholds were not loosened after observing the
result.

## Corrected control path

The authoritative run uses the upstream standard outer position-mode entry points:

- `wave_01/left_sharpa_wave/left_sharpa_wave.usda`
- `wave_01/right_sharpa_wave/right_sharpa_wave.usda`

Each fully composed stage exposed 22 revolute joints, 22 angular drives, and five distal frames.
OVPhysX used an `IdealPDActuator` with an explicit PD-effort path. The controller gains came from
the resolved actuator tensors; the backend PhysX drive stiffness and damping were independently
read back as zero to prevent a second implicit controller.

| Control evidence | Result |
| --- | --- |
| Controller stiffness range | 0.904127–13.200948 Nm/rad |
| Controller damping range | 0.031513–0.451640 Nm·s/rad |
| Controller parameter identity | one parameter set across all 16 OVPhysX cases |
| Authored USD → runtime gain normalization | maximum K error 3.88484e-7; maximum D error 1.63747e-8 |
| OV runtime K vs compiled MuJoCo K | maximum absolute difference 0.00756512 Nm/rad |
| OV runtime D vs compiled MuJoCo D | maximum absolute difference 0.0490232 Nm·s/rad |
| Position-target readback | maximum error 0 rad in all 16 OVPhysX cases |
| Explicit PD formula and clip checks | maximum error 0 Nm; zero clipping events |
| Maximum computed/applied command | 0.660047 Nm |

These checks establish what control law and parameters were actually exercised. They do not prove
physical equivalence or identify which simulator, if either, is closer to hardware.

## Current-state frame sampling

MuJoCo integrates `qpos` at the end of `mj_step`, while derived Cartesian body fields can still
describe the preceding state. Before recording a frame, the authoritative adapter now calls
`mj_kinematics` so its `xpos` and `xquat` refer to the same current `qpos` stored in that sample.
All 16 MuJoCo runs record this sampling contract in provenance. An independent replay of all
6,016 samples and 30,080 DP poses matched current-`qpos` forward kinematics exactly.

## Evidence identity

| Field | Value |
| --- | --- |
| Validator source commit | `c571d9bb37a619fe9e867447d3be448acc615e3c` |
| Validator source tree | `fe86e1ce279770db1a70ce74609166b282f151ff` |
| Source archive SHA-256 | `7f1f8da89eec446541696544ef182202873bf2a699320da41cd6dd09aa42aecd` |
| Immutable snapshot SHA-256 | `94f1ae53b50e020d0040628868ba8ea38458fa47aecf4a60240f7194935cc6a6` |
| Manifest SHA-256 | `576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e` |
| Asset commit | `6eea427eb24189519f32b9f21674cd534d3f973c` |
| Asset Git tree | `bb00a9d5527b8a76de576ce876ebece67d8ffde1` |
| Canonical LF asset SHA-256 | `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad` |
| Full local evidence bundle | 358 payload files, 84,635,733 bytes |
| Full bundle root SHA-256 | `d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505` |

The compact machine-readable record is [summary.json](summary.json). The full content-hashed
bundle retains host-specific raw logs for audit and is not a publication artifact.

## Superseded diagnostic runs

The `ef4b5fc` bundle opened a nested converter `.usd` layer directly. That bypassed the
standard outer position-mode layer and its tuned dynamics, producing a non-representative
implicit-drive response. It is retained only as diagnostic history and is not an authoritative
Gate 0 result. Source `4162fbb` corrected the entry point and made the IdealPD control path
explicit, but its MuJoCo Cartesian body fields lagged the newly integrated `qpos` by one step.
Source `c571d9b` corrected that sampling alignment and reran both backends from the same immutable
source. Neither correction changed the comparison thresholds.

## Retained runtime warnings

The 16 OVPhysX worker stderr logs total 223,171 bytes. Every log contains a cooking-registry
`[Error]` message, and the logs contain 448 `MaterialBindingAPI` warning lines plus resolver
status output. Worker stdout additionally contains one missing-plugin warning, one UJITSO
cooking-service warning, and one unload call-stack block per case. These messages remain
unexplained and must not be summarized as a warning-free run.

All 16 OVPhysX cases nevertheless exited successfully and completed their requested steps. The
evidence audit found no Python traceback, explicit CUDA error, out-of-memory message, or
non-finite structured sample. Those facts establish execution integrity only within this Gate 0
scope; they do not prove the warnings harmless for other workloads.

## Residual diagnosis

Both remaining maxima occur at the `thumb_DP` distal-link origin around `t=0.222 s`. The curves
are smooth, exactly repeatable, and left/right symmetric. The strongest current hypothesis is a
joint-friction semantics mismatch: across the 22 joints, MuJoCo `frictionloss / Kp` is strongly
correlated with the observed response difference. MuJoCo `frictionloss` uses force units, while
the USD stages author a unitless legacy PhysX joint-friction coefficient. This is a working
hypothesis, not a verified backend bug.

The pre-registered local check is now complete. New same-source MuJoCo traces at `1.0x`, `0.5x`,
and `0x` dry friction held the formal OV trace and all other registered inputs fixed. The `1.0x`
trace had zero drift from the formal MuJoCo trace. Removing dry friction reduced the global joint
and distal-frame-position maxima by `51.36%` and `59.67%`, and reduced the old formal-peak
`thumb_CMC_FE` / `thumb_DP` gaps by `99.49%` / `95.61%`. This supports dry friction as a material
contributor in that one left `small_step` trajectory, but does not explain all residuals or change
this formal result. See the [hashed diagnostic summary](../../docs/MUJOCO_FRICTION_DIAGNOSTIC.md).

The preregistered zero-friction follow-up then changed only MuJoCo controller Kv so controller Kv
plus passive joint damping equaled the frozen OV explicit-PD Kd. The named `left_pinky_CMC` window
improved by `82.36%`, while the named `left_thumb_DP` position window improved by only `4.22%`;
frame-position RMS and the global orientation maximum worsened. The registered result is
`mixed_or_inconclusive`, not evidence for one common viscous retuning or a backend bug. See the
[total-viscous retuning diagnostic](../../docs/MUJOCO_TOTAL_VISCOUS_DIAGNOSTIC.md).

The next post-hoc check replayed the frozen OVPhysX and both MuJoCo 22-joint traces over
`0.11–0.17 s` through one fixed-base MuJoCo FK model, without a dynamics step or fitted alignment.
For both candidates, the common-FK residual norm was about `0.0053%` of the recorded
`left_thumb_DP` gap, satisfying the frozen `CLOSES` rule. This supports the narrow statement that
the selected frame gap is kinematically consistent with the recorded joint-state gap under that
common model. It does not identify why those states diverged or establish a backend bug. See the
[common-FK attribution](../../docs/THUMB_DP_COMMON_FK_ATTRIBUTION.md).

## Interpretation and scope limits

- `DIVERGENT` means a frozen numerical threshold was exceeded under this pinned setup. It is not,
  by itself, an official Sharpa, MuJoCo, NVIDIA, or Isaac Sim bug finding.
- The kit-less path used no Kit application, renderer, camera, display, or RTX rendering path.
- Contacts were neither exercised nor validated; the scenarios used no ground plane or intentional
  contact target.
- This fixed-base, position-control result establishes no hardware behavior, safety, or Sim2Real
  claim.
