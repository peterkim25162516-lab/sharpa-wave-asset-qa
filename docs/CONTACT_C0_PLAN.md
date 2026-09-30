# WaveSimParity Contact Gate C0

Contact Gate C0 is a preregistered, simulation-only comparison of one deliberately synthetic,
frictionless normal-contact pair in MuJoCo and the experimental kit-less OVPhysX stack. It extends
the project without changing Gate 0 or Trajectory T1 manifests, evidence, hashes, or conclusions.

C0 is a contact-pipeline gate. It is not a validation of the Sharpa Wave native fingertip meshes or
elastomers. The synthetic fixture makes geometry and contact-pair identity explicit enough for a
falsifiable first comparison before any native-surface experiment is attempted.

The frozen manifest is `configs/parity/contact_c0.json`: file SHA-256
`73cc9dedffc1ea0600052fc1b78696eda292f1f34c7987ec1f433043e65bf215`, canonical semantic
SHA-256 `cb02398665dbed5722b6f9c44fca1c25a78ed495eb63cf5008e875672ceb930e`. Its closed schemas are
`schemas/contact-c0-manifest.schema.json` (SHA-256
`8a0101837f63abe483e1b93f509d41e024f5e137cabe19657e7171752e0c15ca`) and
`schemas/contact-c0-run.schema.json` (SHA-256
`08adc224e03a2dba52ed3b09868e88c60999dff4a10aaa8cf2fbda7e75822a17`). The source revision and
tree must be recorded after the formal source tree is clean and immutable; an engineering pilot or
a prior T1 trace is not C0 scientific evidence.

## Frozen scope

- Standard fixed-base left and right hands are run separately.
- Every hand retains its complete 22-joint canonical position-control mapping and five canonical
  distal-frame mappings: `44/44` joints and `10/10` distal frames across the campaign.
- Only each hand's `index_DP` frame carries the synthetic probe.
- Gravity is zero. All native hand collision shapes and self-collisions are disabled.
- The only enabled collision shapes are the synthetic sphere and static box; the only admitted
  pair is `synthetic_index_probe__static_box`.
- Static friction, dynamic friction, and restitution are all zero. CCD is disabled. MuJoCo uses
  `condim=1`. Engine-native normal-solver parameters remain frozen and disclosed, not translated
  into a claimed equivalent compliance law.
- The compared OVPhysX path is headless and kit-less. No Kit application, renderer, camera, or RTX
  rendering path is part of C0.
- Isaac Lab may transitively import its pure Python camera configuration/data modules while loading
  the sensors namespace. That passive class import is not treated as camera execution. Runtime
  evidence must instead prove that no `UsdGeom.Camera` prim exists, that the only created sensor
  type is `ContactSensor`, and that no Kit, renderer, Replicator, or synthetic-data runtime module
  is loaded.
- Dual-hand models, floating bases, wrist/flange variants, native fingertip collision surfaces,
  complete Isaac Sim, hardware, safety, and Sim2Real are outside C0.

## Synthetic fixture

The probe is a massless collision overlay that must not alter authored or compiled link mass and
inertia:

```text
parent frame:  {side}_index_DP
local center:  [0.025, 0.0, 0.0] m
radius:        0.005 m
```

The target is a static box:

```text
world center:  [0.0, 0.0, 0.120] m
half extents:  [0.150, 0.150, 0.010] m
top surface:   z = 0.130 m
```

The analytic signed surface gap is

```text
g(t) = z_probe_center(t) - 0.130 m - 0.005 m
```

Positive `g` means analytic separation, zero means tangency, and negative `g` means analytic
overlap. This quantity is computed from the sampled world pose and frozen primitive geometry. It
does not use either engine's native contact separation, penetration, or contact-offset field.

The runtime readback must inventory every native collision prim, prove each native prim is
disabled, prove exactly two collision shapes remain enabled, hash the canonical collision
inventory, and prove the original mass properties are preserved. The direct contact filter must
resolve exactly one sensor body and one target. A Boolean flag alone is not sufficient evidence of
the allowlist.

CCD has two independent fail-closed checks. First, the strongest pre-reset session overlay must
compose registered `PhysxSceneAPI` and `PhysxRigidBodyAPI` Boolean `enableCCD=false` opinions.
Second, after the first reset, the worker resolves the live `PxScene` and index `PxArticulationLink`
through OVPhysX's public `get_physx_ptr` interop and reads their actual PhysX 5.9 flags through a
seven-function, read-only C++ helper. The live articulation link resolved by the same canonical
path and `LINK` object type is also the authoritative post-cook mass-property source: the helper
reads `getMass()`, `getMassSpaceInertiaTensor()`, and `getCMassLocalPose()` without mutating or
releasing the object.
The runtime inertia is reconstructed in link axes as `R(q_LC) diag(I_C) R(q_LC)^T` and compared to
the tensor independently reconstructed from the authored USD principal axes and diagonal inertia.
This avoids relying on an ambiguously documented tensor binding or rotating an already transformed
tensor twice. The helper is rebuilt inside every remote run from committed
source and the exact headers extracted from the official OVPhysX `0.4.13` SDK archive (SHA-256
`191dcaff34980f6fdf94bb783c8faacbab058aa4c2671e31ac306fcd89cdb1e7`). Header inventory,
compiler identity, build arguments, helper bytes, environment binding, and per-case helper hash
are private evidence. The helper never changes or releases an engine object.

## Excluded engineering pilot and frozen activity threshold

Before the formal source freeze, an excluded kit-less OVPhysX engineering pilot exercised the
exact synthetic geometry and filter:

- contact condition: filtered selected-pair force peak approximately `0.00554 N`, with analytic
  gap constrained to approximately `7e-8 m`;
- sham condition: filtered selected-pair force peak `0 N`, steady-hold mean analytic gap
  approximately `-0.0107019 m`;
- the direct filter resolved one sensor body and one target.

Based on this positive/negative-control separation, `pair_active` was frozen before formal C0
execution as:

```text
pair_active[k] := filtered_selected_pair_force_norm_n[k] > 1e-4 N
```

The filter must be backend-direct and restricted to the canonical selected pair. Motion, analytic
gap, raw contact count, or an unfiltered net-force tensor may not be substituted. The pilot is
explicitly excluded from C0 scientific evidence, may not be pooled with formal cases, and may not
be used to replace or repair a failed formal case. The raw force magnitude remains descriptive and
is not a cross-simulator metric despite being used to classify the pair-level Boolean.

## Canonical command

Each case lasts `7.0 s` at base timestep `0.002 s` or halved timestep `0.001 s`. Every joint starts
at `0 rad`. All joints remain in position mode. Only the index flexion chain receives a nonzero
peak target:

```text
index_MCP_FE = 0.60 rad
index_MCP_AA = 0.00 rad
index_PIP    = 0.90 rad
index_DIP    = 0.60 rad
all other joints = 0 rad
```

Let

```text
h(u) = 10*u^3 - 15*u^4 + 6*u^5,  u in [0, 1].
```

The scalar command envelope is

```text
s(t) = 0                                      0.0 <= t <= 0.5
s(t) = h((t - 0.5) / 2.5)                    0.5 <  t <  3.0
s(t) = 1                                      3.0 <= t <= 4.0
s(t) = 1 - h((t - 4.0) / 2.5)                4.0 <  t <  6.5
s(t) = 0                                      6.5 <= t <= 7.0
```

The three nonzero peak targets are multiplied by `s(t)`. Commands are functions of physical time,
not rescaled sample indices, so base and halved commands are identical at common physical times.

## Time and sample semantics

`target[k]` is recorded at `t_k` and applies over `[t_k, t_(k+1))`. `state[k]` is the observation
at `t_k` after the preceding interval; the newly recorded target has not yet advanced the physics
state. Every completed run stores `N+1` samples for `N` physics intervals:

- base: `3,500` intervals and `3,501` samples;
- halved: `7,000` intervals and `7,001` samples.

Sample zero has explicit `pair_active=false` and no preceding physics interval. Exact-repeat
bitsets include sample zero, but event detection and duty fractions use interval samples `1..N`.
Base and halved traces are paired by integer step ratio at common physical times; floating-point
timestamps are integrity checks, not primary join keys.

## Conditions and matrix

Both conditions use the same model, fixture transforms, target sequence, state initialization, and
sampling contract:

- `contact`: the single synthetic sphere-box collision pair is enabled;
- `sham`: that pair is disabled while the geometry remains present for analytic-gap measurement.

The frozen matrix is

```text
2 backends * 2 hands * 2 conditions * 2 timestep variants * 2 fresh repeats = 32 cases
```

Case identifiers are deterministic, for example:

```text
mujoco.left.press_hold_release.contact.base.r01
ovphysx.right.press_hold_release.sham.halved.r02
```

Every case runs in a fresh process with independently verifiable creation identity. Completion is
strictly `32/32`; no fractional shortfall is admitted.

## Portable observations

The portable sample contains complete canonical joint position, velocity, target, and five distal
frame poses, plus:

- probe-center world position;
- analytic signed gap `g`;
- direct selected-pair Boolean `pair_active`;
- optional raw backend contact count;
- a private backend-native selected-pair force record.

Frame poses use the project-wide `xyz_m_qwxyz` convention: three world-position components in
metres followed by a scalar-first unit quaternion `(qw, qx, qy, qz)`. Quaternion inversion and
composition for orientation contact effects use that same scalar-first convention.

Raw contact count is never interpreted as the number of physical contacts. Engines may expose
different point, patch, or manifold granularity.

The primary contact effect is paired against the sham run for the same
backend/hand/timestep/repeat:

```text
b(t)   = g_contact(t) - g_sham(t)              blocked normal travel
e_q(t) = q_contact(t) - q_sham(t)              joint contact effect
R_e(t) = R_contact(t) * inverse(R_sham(t))     orientation contact effect
```

Cross-simulator C0 metrics compare the two engines' contact effects, not only their raw contact
trajectories. This difference-in-differences construction prevents a previously observed
contact-free baseline offset from being mislabeled automatically as a contact effect.

## Event semantics

The debounce window is `4 ms`. It corresponds to two consecutive base-dt physics intervals or
four consecutive halved-dt intervals. For `k >= 1`, `pair_active[k]` describes the direct filtered
observation at the end of interval `(t_(k-1), t_k]`.

- onset is the first interval in the first run of at least `4 ms` continuously active intervals;
- release is the first interval in the first post-onset run of at least `4 ms` continuously
  inactive intervals;
- each event retains the interval `[t_(k-1), t_k]`; timing comparisons use its midpoint.

The steady hold window is `[3.25, 3.75) s`. The recovery window is `[6.75, 7.0) s`. Window
membership is based on canonical integer sample indices.

## Validity and admission gates

C0 evidence is `VALID` only when all of the following hold:

1. Source revision/tree/archive, manifest, schema, asset, dependency, overlay, runtime, and fresh
   process identities match their frozen values.
2. The private bundle inventory and root hash verify exactly, with no extra payload.
3. All 32 cases and all requested integration intervals complete.
4. Canonical mapping is exactly `44/44` joints and `10/10` distal frames; both index probe frames
   are mapped.
5. States, velocities, targets, poses, probe positions, gaps, and direct filtered-pair
   observations are finite.
6. Fixture readback differs from the frozen geometry by at most `1e-6 m`; material-zero readback
   differs from zero by at most `1e-9`.
7. Every native collision prim is inventoried and disabled; exactly two synthetic shapes remain
   enabled; the collision inventory hash verifies; MuJoCo compiled mass properties and OVPhysX
   post-first-reset native `PxArticulationLink` mass, center of mass, principal axes, and mass-space
   inertia remain consistent with their pre-overlay source values. The OVPhysX scene and selected
   rigid-body CCD bits are read from the live PhysX objects after reset and are both disabled; the
   mass proof and CCD proof bind to the same run-bound helper, whose SDK build chain verifies
   exactly.
8. The selected-pair filter resolves exactly one sensor body and one target, observation is
   performed directly in both engines, and every completed case contains `N+1` non-null Boolean
   observations with no missing value.
9. Sample zero is inactive. Sham is inactive for every sample. No unintended pair is observed.
10. Initial gap is at least `0.050 m`; sham steady-hold mean gap is at most `-0.005 m`; recovery
    minimum gap is at least `0.050 m` and inactive.
11. If both engines observe the selected pair, every contact case achieves at least `0.95` debounced
    pair-active duty in the steady-hold window before continuous-contact metrics are interpreted.
    If exactly one engine observes the pair, this duty gate is not used to erase the frozen
    `contact_presence_mismatch` divergence classification.
12. Stored target records match the canonical formula and base and halved records match at common
    physical times. This is target-record integrity, not simulator target-readback evidence.
13. OVPhysX runtime evidence contains zero camera prims and exactly one created `ContactSensor`
    after the complete trajectory and before scene cleanup; a separate module inventory contains
    no Kit, renderer, Replicator, or synthetic-data module after runtime cleanup.
14. Worker logs contain no fatal runtime signal, and public/private redaction checks pass.

An unavailable contact capability, a null pair observation, unverified filter cardinality, or a
motion-derived contact guess is a capability/validity failure, not a physics divergence.

## Frozen numerical thresholds

### Fresh-process repeatability

| Quantity | Maximum |
| --- | ---: |
| Joint state or joint effect | `1e-9 rad` |
| Probe position, gap, or blocked travel | `1e-9 m` |
| DP orientation or orientation effect | `1e-9 rad` |
| Complete `N+1` pair-active bitset | exact |
| Debounced onset/release interval | exact |

### Base versus halved timestep

| Quantity | Maximum |
| --- | ---: |
| Joint state or joint effect | `0.01 rad` |
| Probe position, gap, or blocked travel | `0.001 m` |
| DP orientation or orientation effect | `0.02 rad` |
| Onset or release midpoint | `0.004 s` |
| Steady-hold duty fraction | `0.01` |

### MuJoCo versus kit-less OVPhysX

| Quantity | Maximum |
| --- | ---: |
| Onset midpoint delta | `0.010 s` |
| Release midpoint delta | `0.010 s` |
| Steady-hold duty delta | `0.02` |
| Onset analytic-gap delta | `0.001 m` |
| Steady-hold mean-gap delta | `0.001 m` |
| Joint contact-effect delta | `0.02 rad` |
| Blocked-travel delta | `0.002 m` |
| DP orientation-effect geodesic delta | `0.05 rad` |

Cross-simulator maxima cover both timesteps and both fresh repeats. Joint, position, and
orientation budgets retain the prior T1 dimensions; the contact-gap and event-time budgets were
frozen before formal C0 collection.

## Decision semantics

Evidence validity and scientific interpretation are separate:

- `INVALID` evidence always forces science to `INCONCLUSIVE`.
- A repeatability or dt-halving failure forces `INCONCLUSIVE`; it is not called a simulator
  divergence.
- If either sham does not analytically cross the top surface by the frozen margin, excitation is
  inadequate and the result is `INCONCLUSIVE`.
- If neither backend ever observes the selected pair, C0 was not exercised and is `INCONCLUSIVE`.
- If the same admitted sham crossing is established in both engines but exactly one engine ever
  observes the selected pair, the valid result is `DIVERGENT` with reason
  `contact_presence_mismatch`.
- If both engines observe the pair but either lacks the frozen steady-hold duty, continuous-contact
  comparison is `INCONCLUSIVE`.
- After validity, excitation, repeatability, and dt-halving gates pass, any primary
  cross-simulator maximum above its frozen threshold gives `DIVERGENT`; otherwise the result is
  `WITHIN_TOLERANCE`.
- `pass_ready=true` only for `VALID` plus `WITHIN_TOLERANCE` with both hands fully exercised.

`DIVERGENT` means a reproducible difference under this exact pinned synthetic experiment. It does
not identify which simulator is closer to reality and is not automatically an asset, MuJoCo,
NVIDIA, or Isaac Sim bug.

## Backend-native quantities are descriptive only

The following may be preserved for diagnosis but must not enter a cross-simulator pass/fail
metric:

- raw point, patch, manifold, or contact counts;
- native separation, penetration, contact offset, or rest offset;
- MuJoCo constraint/contact force and PhysX force, impulse, or net-force tensor magnitude;
- solver iterations, residuals, compliance, stiffness, or stabilization fields;
- native actuator or constraint forces.

The two native position-control paths, their parameters, and their runtime identities are pinned
and may be described. The closed C0 run record does not carry immediate target readback,
computed/applied controller effort, formula-error, clipping-error, or clip-count evidence.
Therefore none is a C0 admission gate or cross-simulator metric, and C0 makes no controller-parity,
controller-load, saturation, or controller-quality claim.

## Failure and rerun policy

Capability probes and engineering pilots are excluded evidence. After formal integration begins,
a failed scientific case is retained without selective replacement. A code, schema, fixture,
threshold, or protocol correction requires a new immutable source revision, new run identity, and
complete 32-case rerun. Formal results must never be combined across revisions.

## Claim boundary

C0 can establish only a reproducible MuJoCo-to-kit-less-OVPhysX result for the pinned fixed-base
hands, synthetic sphere-box fixture, frictionless normal-contact condition, position controller,
command, and versions. It does not validate the physical fingertip contact surface, elastomer
deformation, native collision-mesh cooking, full Isaac Sim, dual-hand or floating-base behavior,
hardware behavior, safety, controller quality, or Sim2Real. It does not modify the frozen
Trajectory T1 result.
