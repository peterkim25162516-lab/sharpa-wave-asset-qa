# WaveSimParity Gate 0 status

Formalized on 2026-08-27 (Asia/Shanghai). This is an unofficial, simulation-only comparison of
MuJoCo and kit-less OVPhysX. It is not a full Isaac Sim, hardware, safety, or Sim2Real result.

## Decision

**Gate 0 execution is complete, but the acceptance gate did not pass.**

- Execution: `COMPLETED`
- Comparison: `DIVERGENT`
- Acceptance: `pass_ready=false`

All structural checks and all 32 runs completed. Every structured sample remained finite, exact
repeatability passed, and dt-halving stayed within its diagnostic limits. The only frozen
cross-simulator threshold exceeded was DP/distal-link frame position in the left and right
`small_step` cases: `2.109 mm` and `2.111 mm` against a `2.000 mm` limit. Their joint and
frame-orientation metrics passed. These frames are distal-phalanx link origins, not physical
fingertip contact-surface points.

`DIVERGENT` is a reproducible observation under the pinned setup. It is not, by itself, an
official Sharpa, MuJoCo, NVIDIA, or Isaac Sim bug finding.

## Authoritative run

| Field | Value |
| --- | --- |
| Branch | `feat/wavesim-parity-gate0` |
| Validator source | `c571d9bb37a619fe9e867447d3be448acc615e3c` |
| Validator tree | `fe86e1ce279770db1a70ce74609166b282f151ff` |
| Session / remote run | `formal-c571d9b-20260827-a1` |
| Source archive SHA-256 | `7f1f8da89eec446541696544ef182202873bf2a699320da41cd6dd09aa42aecd` |
| Immutable snapshot SHA-256 | `94f1ae53b50e020d0040628868ba8ea38458fa47aecf4a60240f7194935cc6a6` |
| Manifest SHA-256 | `576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e` |
| Asset commit | `6eea427eb24189519f32b9f21674cd534d3f973c` |
| Asset Git tree | `bb00a9d5527b8a76de576ce876ebece67d8ffde1` |
| Canonical LF asset SHA-256 | `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad` |
| Isaac Lab | `v3.0.0-beta2.patch1` / `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Remote Python | `3.12.14` |
| OVPhysX packages | `isaaclab-ovphysx==3.0.2`, `ovphysx==0.4.13` |
| Formal GPU run | NVIDIA A800-SXM4-40GB, driver `570.158.01`, 240 s |
| Local verification | `239 passed` |

The launcher selected an idle GPU with no compute process, then exposed only that device to the
worker. Monitoring did not record a foreign compute process during the formal run, and the GPU
returned to zero memory use, zero utilization, and no compute process afterwards.

## What was corrected

The first formal bundle at source `ef4b5fc` used a nested converter `.usd` layer as the runtime
entry point. That layer does not contain the standard outer position-mode tuning, so its implicit
drive response did not represent the upstream-recommended configuration.

The authoritative run instead uses these fully composed outer entry points:

- `wave_01/left_sharpa_wave/left_sharpa_wave.usda`
- `wave_01/right_sharpa_wave/right_sharpa_wave.usda`

It also uses an explicit `IdealPDActuator` effort path. The controller gains are read from the
resolved actuator tensors, while backend PhysX drive stiffness and damping are verified as zero.
That control-path correction was formalized at source `4162fbb`, but a later evidence audit found
that its MuJoCo samples read `xpos` and `xquat` immediately after `mj_step`. At that point MuJoCo's
`qpos` had advanced while those derived Cartesian body fields still described the preceding
state. Source `c571d9b` now calls `mj_kinematics` before recording frame poses, making joint and
frame values in each sample refer to the same current `qpos`. A regression test independently
replays the sampled `qpos` and requires the recorded body pose to match.

Both earlier bundles remain private diagnostic evidence but are superseded for all conclusions.
The control path, scenarios, mappings, assets, and thresholds were not changed for this rerun.

## Acceptance evidence

| Check | Result | Status |
| --- | --- | --- |
| Case completion | 32/32 runs and 100% requested steps | PASS |
| Joint mapping | 44/44 | PASS |
| Distal-frame mapping | 10/10 | PASS |
| Resolved USD schema | standard outer left and right stages available | PASS |
| OVPhysX position-target readback | 16/16 cases, maximum error 0 rad | PASS |
| Explicit PD formula / clipping | maximum validation error 0 Nm; 0 clips | PASS |
| Finite values | 0 non-finite runs | PASS |
| Repeatability | joint=0, position=0, orientation=0 | PASS |
| dt-halving | 0.00135195 rad, 0.000161719 m, 0.00284117 rad | PASS |
| Cross-simulator acceptance | both `small_step` frame-position metrics exceed 0.002 m | DIVERGENT |

The dt-halving limits are `0.01 rad`, `0.001 m`, and `0.02 rad`. The cross-simulator limits are
`0.02 rad`, `0.002 m`, and `0.05 rad` for joint, frame-position, and frame-orientation deltas.

## Scenario comparison

| Hand | Scenario | Joint max (rad) | Frame position max (m) | Frame orientation max (rad) | Result |
| --- | --- | ---: | ---: | ---: | --- |
| left | zero_hold | 0 | 4.7507e-08 | 1.20065e-06 | within tolerance |
| left | gravity_settling | 0.00277094 | 0.000327972 | 0.00383823 | within tolerance |
| left | small_step | 0.0186829 | 0.00210924 | 0.0457305 | DIVERGENT |
| right | zero_hold | 0 | 6.22974e-08 | 8.47544e-07 | within tolerance |
| right | gravity_settling | 0.00272553 | 0.000319917 | 0.00369807 | within tolerance |
| right | small_step | 0.0187022 | 0.00211119 | 0.0458414 | DIVERGENT |

## Control-path evidence

- Each resolved stage exposed 22 revolute joints, 22 angular drives, and five distal frames.
- Runtime controller stiffness spans `0.904127–13.200948 Nm/rad`; damping spans
  `0.031513–0.451640 Nm·s/rad`.
- All 16 OVPhysX cases used the same controller parameter set; its canonical JSON SHA-256 is
  `abf8458afbf16b2f4c2a31a18e0471c4f47e9903a3836b4d4838f6eb5ece10b4`.
- After the expected angular-unit normalization, authored USD and runtime controller gains agree
  within `3.88484e-7` for stiffness and `1.63747e-8` for damping.
- Runtime OVPhysX and compiled MuJoCo stiffness agree within `0.00756512 Nm/rad`; the largest
  compiled damping difference is `0.0490232 Nm·s/rad`.
- Position-target readback, explicit PD formula validation, and clipping validation all have zero
  maximum error. No effort clipping occurred; the largest computed/applied command was
  `0.660047 Nm`.

This evidence rules out the previous wrong-entry-point control mismatch. It does not establish
hardware truth or prove that the remaining frame-position difference comes from any one engine.

## Runtime warnings and scope limits

The 16 OVPhysX cases completed successfully, but their logs are not warning-free. Their 223,171
bytes of worker stderr contain one cooking-registry `[Error]` per case, 448 `MaterialBindingAPI`
warning lines, and resolver status output. Worker stdout also contains one missing-plugin warning,
one UJITSO cooking-service warning, and one unload call-stack block per case. The audit found no
Python traceback, explicit CUDA error, out-of-memory message, or non-finite structured sample.
The remaining kit-less runtime messages are retained and unexplained.

Contacts were not checked in any OVPhysX case. The scenarios intentionally used no ground plane
or contact target, so this run says nothing about contact parity or collision-cooking correctness.

The run was headless and kit-less. It loaded no Kit application, renderer, camera, or RTX render
path. The A800 result must not be presented as a full Isaac Sim comparison or as hardware or
Sim2Real validation.

## Evidence and publication boundary

The verified local evidence bundle contains 358 payload files totaling 84,635,733 bytes. Its root SHA-256
is:

```text
d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505
```

The inventory has no unlisted payload. The full bundle remains private local audit evidence
because raw logs contain host-specific paths, a user name, GPU UUID, and worker PIDs. A public raw
bundle would require redaction and a new root hash.

The publication-safe records are the compact [Gate 0 report](../results/gate0/report.md) and
[machine-readable summary](../results/gate0/summary.json). The formal execution remains bound to
source commit `c571d9b`; later documentation commits do not change that identity.

## Reproduction entry points

The validated workflow has three phases:

1. `scripts/run_gate0_local.py` creates the 16-case MuJoCo evidence directory.
2. `scripts/run_gate0_remote.sh` runs two resolved-schema probes and 16 isolated OVPhysX cases.
3. `scripts/finalize_gate0.py` verifies both evidence trees, compares them, and creates the hashed
   bundle and report.

The entry points fail closed on a dirty execution source, identity drift, occupied GPU, evidence
mismatch, or an existing output directory.

## Residual diagnosis and completed local checks

The remaining excess is `0.109 mm` on the left and `0.111 mm` on the right. Both peaks occur at
the `thumb_DP` distal-link origin around `t=0.222 s`. They are smooth, exactly repeatable, and
left/right symmetric; dt-halving leaves them slightly above the same threshold. At the peak,
OVPhysX has moved the upstream `thumb_CMC_FE` joint closer to the `0.05 rad` target than MuJoCo.

The strongest current hypothesis is a joint-friction semantics mismatch, not a proven backend
bug. Across the 22 joints, MuJoCo `frictionloss / Kp` is strongly correlated with the observed
cross-simulator joint response difference. The [MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html)
defines `frictionloss` in force units, whereas the
[PhysX joint schema](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_joint_a_p_i.html)
defines the legacy joint-friction attribute as a unitless coefficient scaled by transmitted
force.

That local-only check is now complete. Three new same-source MuJoCo runs scaled compiled dry
friction to `1.0x`, `0.5x`, and `0x` while keeping the frozen formal OVPhysX trace, assets,
commands, initial state, and timestep unchanged. The `1.0x` trace reproduced the formal MuJoCo
trace exactly. At `0x`, the trajectory-wide joint, distal-frame-position, and frame-orientation
maxima decreased by `51.36%`, `59.67%`, and `75.86%`; at the old formal peak, the selected
`thumb_CMC_FE` and `thumb_DP` gaps decreased by `99.49%` and `95.61%`. This supports MuJoCo dry
friction as a material contributor in the single left `small_step` trajectory, but it neither
fully explains the residual nor establishes equivalent PhysX friction or a backend bug. See the
[friction diagnostic](MUJOCO_FRICTION_DIAGNOSTIC.md) for the registered rule, hashes, raw metrics,
scope boundary, and next falsifiable check.

With dry friction removed, the worst joint gap becomes a short early `left_pinky_CMC` transient
of `0.00908694 rad`, and the remaining `thumb_DP` peak is `0.850667 mm`. The preregistered
total-viscous follow-up is also complete. A new same-source control reproduced the frozen
zero-friction samples exactly. The treatment changed only controller Kv so controller Kv plus
MuJoCo passive joint damping equaled the frozen OV explicit-PD Kd. The pinky window improved by
`82.36%`, but the thumb frame window improved by only `4.22%`; frame-position RMS and the global
orientation maximum worsened. The result is `mixed_or_inconclusive`, so it does not support one
common viscous retuning as the major explanation for both residuals. See the
[total-viscous retuning diagnostic](MUJOCO_TOTAL_VISCOUS_DIAGNOSTIC.md).

That common-kinematics check is now complete. The frozen OVPhysX and both MuJoCo 22-joint traces
were replayed through one fixed-base MuJoCo FK model over `0.11–0.17 s`, without advancing
dynamics or fitting an alignment. The common-FK residual was about `0.0053%` of each recorded
`left_thumb_DP` gap, and both candidates satisfy the frozen `CLOSES` rule. Exact five-thumb-joint
Shapley allocation identified control `left_thumb_CMC_AA` as the only registered dominant
positive closer; the treatment had no player meeting both dominance thresholds. These are
model- and baseline-dependent counterfactual allocations, not causal dynamics results. See the
[common-FK attribution](THUMB_DP_COMMON_FK_ATTRIBUTION.md).

The bounded conclusion is that this selected frame-position gap is kinematically consistent with
the recorded joint-state gap under the common MuJoCo model. It does not explain why the joint
states diverged.

## R1 effective-parameter readback

The bilateral Freeze A follow-up completed on 2026-08-28 with status `READBACK_VALID`. This was a
descriptive, pre-trace kit-less OVPhysX readback, not a formal Gate 0 rerun. It covered all `44/44`
canonical joints in four fresh processes (two per hand), and the mapping, solver evidence records,
and canonical numeric vectors reproduced exactly. It made zero user or experimental trace step
calls and recorded zero experimental commands, targets, or trajectory samples; adapter
initialization/reset writes and the unavailable backend-internal advance count remain disclosed.

All 88 friction records and all 88 armature records expose authored, resolved, and runtime-effective
layers. Some control/drive fields expose only a runtime layer, and several control and solver fields
are not exposed by the pinned kit-less stack. Missing layers were recorded as unavailable rather
than filled with inferred defaults. Consequently `friction`, `armature`, `control_drive`, and
`solver_runtime` all remain conservatively `UNMAPPABLE`. This does not establish MuJoCo equivalence,
identify a trajectory cause, infer realized torque, or support a backend bug claim.

Formal Gate 0 therefore remains `DIVERGENT` with `pass_ready=false`. See the
[R1 readback note](OVPHYSX_EFFECTIVE_READBACK.md),
[compact report](../results/effective-readback/report.md), and
[machine-readable summary](../results/effective-readback/summary.json).

## Freeze B native-input sensitivity

Freeze B is complete with validation status `VALID` and scientific label `INCONCLUSIVE`. All
`24/24` cases completed with `44/44` joint mappings, `10/10` fingertip frame mappings, exact fresh
repeatability, dt-halving within the frozen limits, and finite/sanity checks passing.

The frozen rule assigns `SUPPORTED` only when all eight sensitivity values are at least `0.25`,
and `NOT_SUPPORTED` only when all eight are at most `0.10`. The observed result has `5/8` values at
or below `0.10`, `3/8` above `0.10`, and `0/8` at or above `0.25`, so it is `INCONCLUSIVE`. The
runtime readback observation that zero treatment changed `0/8` cases from R1 is descriptive only;
it does not decide validity or the label.

The trajectories remain campaign-G evidence from revision `d890f8ee83741cf643aadf2d9d1f256bf41c09cc`
and tree `29b6d6846e181897c4be2c3e19361e061524b061`. Final analysis I is revision
`8e2fd84ccd9f9ad89ec2101f1f9e61057e37b71d`, tree
`177e470e9be68f7d014fc9a8bba73d673943540d`. Its private bundle contains 407 payload files and
93,091,294 bytes with root SHA-256
`3aed0876784c4575be20e5f5fbb6879f840829ae0b7885bda49ca133698bf04b`.

H produced one local, unpublished `INVALID`/null analysis bundle. It remains preserved
as superseded private evidence. I changed only the disclosed, analysis-time timestamp compatibility
check; it did not rerun either simulator or alter evidence, the protocol, windows, formulas,
thresholds, targets, case order, or dt.

The sanitized [Freeze B report](../results/freeze-b/report.md) has SHA-256
`914f8e42854de817af4646680e15271000c7d4eb2e787fec3d0b3ca7dad1e507`; its
[machine-readable summary](../results/freeze-b/summary.json) has SHA-256
`6a7cc328b614cc15cba1b3819c90527a371313dad1f70b64bc9e4247ab76d611`. See the
[Freeze B note](OVPHYSX_FRICTION_FREEZE_B.md) for the eight values and correction boundary.

Freeze B does not rewrite formal Gate 0: it remains `DIVERGENT` with `pass_ready=false`. This is an
unofficial, simulation-only result, not a backend bug, hardware, Sim2Real, or cross-engine
equivalence claim. No external publication is claimed.

Keep the frozen thresholds. The additive, synthetic [Contact Gate C0](CONTACT_C0_STATUS.md) is now
complete as `VALID` / `INCONCLUSIVE`; it does not rewrite this Gate 0 result. Native fingertip
surfaces, wrist/flange variants, dual hands, and floating bases remain later phases.
