# WaveSimParity Trajectory T1

Trajectory T1 is a preregistered, simulation-only extension of WaveSimParity. It adds two
contact-free-intent position-command waveforms to the fixed-base left and right Wave hands. It is
independent of formal Gate 0 and the later friction diagnostics: their manifests, evidence, hashes,
and conclusions are not changed by T1.

The compared backends are MuJoCo and the experimental kit-less OVPhysX stack. T1 is not a full
Isaac Sim run and does not use Kit, a renderer, cameras, hardware, or Sim2Real evidence.

The frozen manifest is `configs/parity/waveform_t1.json`: file SHA-256
`74e85e133f511944a9a9feeae3f29656db477e0983fa741995ce2c65ce1141a9`, canonical semantic
SHA-256 `68b40b1a0f7bbbdd9860750f92d58445bb8aa16cb6682d39eb5f41e4ac7df85c`. Its closed v2 schema is
`schemas/parity-manifest-v2.schema.json`, SHA-256
`242134b8413936b8747c849db2e494c8a0234241cfa4a48d7d3e99c6b815a71f`.

## Frozen scope

- Standard fixed-base left and right hands are run separately; this is not a dual-hand model.
- Each hand uses its canonical 22-joint position-control mapping and five canonical `*_DP` frame
  origins. A `*_DP` origin is not claimed to be a physical fingertip contact surface.
- Gravity is zero and no ground or deliberate contact target is present.
- All 22 joints receive the same canonical command.
- Initial position is `0 rad` and every command remains within `[0, 0.05] rad`, the command envelope
  already used by the formal small-step case.
- Native assets, controller parameters, friction, solver settings, and backend versions are not
  tuned using T1 outcomes.

## Canonical commands

Both cases last `4.0 s`. The base timestep is `0.002 s`; the halved variant is `0.001 s`.
Commands are functions of physical time `t`, not independently rescaled sample indices.

### `offset_sine_1hz`

```text
q_d(t) = 0.025 * [1 - cos(2*pi*t)]
```

This is a 1 Hz offset sine containing four complete periods.

### `offset_linear_chirp_0p5_4hz`

```text
phase_cycles(t) = 0.5*t + 0.4375*t^2
q_d(t) = 0.025 * [1 - cos(2*pi*phase_cycles(t))]
```

The instantaneous frequency rises linearly from `0.5 Hz` to `4 Hz` at `0.875 Hz/s`. The command
contains nine complete periods. Both frozen commands start and end at target position and target
velocity zero. At 4 Hz, the base grid still provides 125 integration steps per period.

## Time semantics

The target recorded as `target[k]` becomes active at `t_k` and applies over
`[t_k, t_(k+1))`. `state[k]` is the observation at `t_k` after the preceding interval; the newly
recorded target has not yet advanced the physics state.

- Cross-simulator and fresh-repeat state traces are compared at the same physical time.
- Base and halved traces are paired by integer step ratios at common physical times. Floating-point
  timestamps are integrity checks, not primary join keys.
- Tracking uses `target[k] -> state[k+1]`.
- `target[N]` has no following integration interval. It remains part of formula and runtime-readback
  validation, but is excluded from response metrics.

## Frozen matrix

The active matrix is:

```text
2 backends * 2 hands * 2 waveforms * 2 timestep variants * 2 fresh repeats = 32 cases
```

MuJoCo contributes 16 local cases and kit-less OVPhysX contributes 16 Server 6 cases. Every case
runs in a fresh process whose kernel-derived creation identity is unique within the campaign; PID
alone is not accepted as freshness evidence. The matrix requests 48,000 integration steps per
backend and 96,000 total.
Because 31/32 is only 96.875%, the preregistered `>=99%` completion requirement implies all 32
cases must complete.

Formal Gate 0 and Freeze B traces are excluded from T1 active evidence. A separately identified
small-step source bridge may be used as an engineering regression check, but it is not a T1
scientific case.

## Frozen local runtime identity

The MuJoCo half of T1 is frozen to CPython `3.12.6`, MuJoCo `3.12.0`, and NumPy `2.5.2`,
on `Windows-11-10.0.26200-SP0` / `AMD64`. The frozen CPython executable SHA-256 is
`3470f7919170d235d7e6079691462c4b217745ec67ee612e745730e46d98f238`.
The local launcher accepts only its own `sys.executable`; an alternate Python path is rejected even
if it reports the same version. The launcher hashes that interpreter file and records the SHA-256
plus implementation, platform, machine, and all three versions in launcher evidence and in every
one of the 16 MuJoCo run provenances. Each run's adapter-reported platform must match that frozen
runtime identity.
The local finalizer must run under the same interpreter bytes and frozen versions: it recomputes
the interpreter SHA-256 and independently checks the launcher record and all 16 runs.

Formal launcher and finalizer entrypoints set `PYTHONDONTWRITEBYTECODE=1`. Before importing any
project, MuJoCo, or NumPy package, and again during each source-identity check, they reject any
`.pyc`, `.pyo`, or `__pycache__` artifact anywhere below `src/` or `scripts/`. These ignored files
are never silently deleted; their presence blocks the formal run or finalization until the operator
cleans the checkout explicitly.

## Validity gates

T1 is `VALID` only when all of the following hold:

1. Immutable code commit/tree/archive/snapshot, manifest, asset, and dependency identities match.
2. The private bundle has an exact inventory and verified root hash with no extra payload.
3. All 32 fresh-process cases and every requested integration step complete.
4. Canonical coverage is exactly `44/44` joints and `10/10` `*_DP` frames.
5. All states, velocities, targets, poses, effort observations, and readbacks are finite.
6. Existing integrity bounds hold: `|q| <= 2*pi`, `|qvel| <= 1000 rad/s`, frame-origin distance
   `<=10 m`, and quaternion-norm error `<=1e-3`.
7. Every stored target matches the canonical formula; base and halved commands agree at common
   physical times.
8. OVPhysX immediate target readback error is `<=1e-6 rad`. Canonical scheduled, requested,
   immediate-readback, and pre-step-applied sequences are independently hashed in canonical joint
   order. The first three cover `target[0..N]`; the pre-step digest covers the targets actually
   applied to intervals, `target[0..N-1]`. Digest values use an explicit IEEE-754 binary32
   round-trip projection before hexadecimal encoding so the float32 OVPhysX target tensor can be
   checked exactly; the separate float64-to-readback error retains the observed quantization.
9. OVPhysX explicit-PD formula error is `<=1e-5 Nm` and effort-clip validation error is
   `<=1e-6 Nm`. Actual clipping is disclosed; a clipped result is described as nonlinear/saturated,
   not as a linear transfer function.
10. MuJoCo reports zero contacts. OVPhysX contact observation remains unavailable and is recorded as
    `null/not observed`; absence of measurement is not reported as zero contacts.
11. Worker logs contain no traceback, CUDA error, or out-of-memory error. Known kit-less warnings
    remain preserved and counted.
12. Private/public redaction checks pass; public artifacts contain no host paths, process IDs, GPU
    UUIDs, or private input plans.

If a provenance, execution, finite-value, readback, repeatability, or dt-halving gate fails, the
cross-simulator conclusion is `INCONCLUSIVE`. A failed scientific case is retained; it is never
silently rerun or selectively replaced.

## Frozen primary thresholds

Thresholds are inherited unchanged from formal Gate 0 and are not tuned after observing T1:

| Comparison | Joint max | DP position max | DP orientation max |
|---|---:|---:|---:|
| MuJoCo vs kit-less OVPhysX | `0.02 rad` | `0.002 m` | `0.05 rad` |
| Fresh repeatability | `1e-9 rad` | `1e-9 m` | `1e-9 rad` |
| Base vs halved timestep | `0.01 rad` | `0.001 m` | `0.02 rad` |

Only after the validity, repeatability, and dt-halving gates pass is cross-simulator status assigned:
the cross-simulator maxima cover both timestep variants and both fresh repeats. Any such maximum
above its frozen threshold gives `DIVERGENT`; otherwise the result is `WITHIN_TOLERANCE`.
`DIVERGENT` is a reproducible observation, not automatically an official asset or simulator bug.
Formal Gate 0 remains `DIVERGENT` with `pass_ready=false` regardless of T1.

## Secondary descriptive metrics

These metrics are preregistered for interpretation and do not add pass/fail thresholds:

- joint, DP-position, and DP-orientation RMSE, P95, per-channel maximum, and peak step;
- per-backend one-step target-tracking RMSE and maximum;
- 1 Hz sine fits over `[1.0, 4.0) s` using a constant plus sine/cosine ordinary least-squares model,
  reporting fundamental amplitude, gain relative to `0.025 rad`, phase lag, and complex coefficient;
- chirp tracking and cross-simulator RMSE/maximum in fixed instantaneous-frequency bins
  `[0.5,1)`, `[1,2)`, `[2,3)`, and `[3,4] Hz`.

If a fitted sine output amplitude is below `1e-6 rad`, phase is unavailable, while the complex
coefficient remains reported. No confidence intervals or population claims are made from two
deterministic repeats.

## Resource and failure policy

- Remote execution uses one strictly idle A800 sequentially; it never reserves or occupies multiple
  GPUs for this matrix.
- Expected use is `0.25-0.75 GPU hours`; the planned allowance is one GPU hour and the hard campaign
  ceiling is two GPU hours.
- Expected private evidence is `0.6-1.2 GB`; the hard artifact ceiling is `2 GB`.
- If no GPU satisfies the strict idle rule, no remote run directory is created.
- Once scientific integration begins, a failure is retained without automatic retry. A code or
  protocol correction requires a new immutable revision, new run ID, and complete 32-case rerun.

## Claim boundary

T1 can establish only a reproducible MuJoCo-to-kit-less-OVPhysX result for the pinned assets,
versions, commands, and fixed-base contact-free-intent setup. It does not establish which simulator
is closer to reality, full Isaac Sim parity, contact parity, hardware behavior, Sim2Real, safety,
controller quality, dual-hand behavior, wrist/flange behavior, or floating-base behavior.
