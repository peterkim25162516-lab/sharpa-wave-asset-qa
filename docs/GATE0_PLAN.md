# WaveSimParity Gate 0

Gate 0 is a feasibility check for the experimental, kit-less OVPhysX path. It covers a minimal
MuJoCo-to-OVPhysX comparison, not the full MVP matrix, and it does not make hardware, Sim2Real,
safety, or official Sharpa claims.

## Frozen inputs

- Validator baseline: `916df2a63e369766f8c56bfd11f1cd9e719363c4`.
- Sharpa asset commit: `6eea427eb24189519f32b9f21674cd534d3f973c`.
- Cross-platform `HEAD:wave_01` Git tree: `bb00a9d5527b8a76de576ce876ebece67d8ffde1`.
- Canonical LF checkout SHA-256: `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad`.
- v0.1 Windows CRLF checkout SHA-256, retained only as historical evidence: `9c2d71aec9f8fe77aeb03660f735eb55bc53b93ba595f6cbfe5cc4e8fbb67de4`.
- Isaac Lab tag `v3.0.0-beta2.patch1`, commit `ffff603eafc6b74264a5261cc0183d6a65390d78`.
- Vendored `uv.lock` SHA-256: `cc77b3f9862bd561224aef0cf56084f329a0c722af71e5b5851bd23541813522`.
- Python `3.12.14`, `isaaclab-ovphysx==3.0.2`, and `ovphysx==0.4.13` in an isolated user environment.
- Backend wording: `kit-less OVPhysX`, not full Isaac Sim.

The Git tree is the authoritative cross-platform asset identity. The two worktree SHA-256 values
differ only because Git materialized text files with different line endings; they must not be
treated as contradictory asset revisions.

## Scope

Gate 0 covers only the standard fixed-base left and right Wave hands in position-control mode.
Each hand must expose 22 scalar joints and five distal `*_DP` frames. Contacts, wrist/flange
variants, dual-hand models, floating bases, cameras, renderers, and Kit are out of scope.

The micro-suite contains three short scenarios per hand:

1. `zero_hold`: hold the initial position without contacts.
2. `small_step`: apply a bounded position-target step.
3. `gravity_settling`: let the driven fingers settle under gravity without a ground plane.

Each primary run is repeated in a fresh process. One bounded scenario is also repeated at half the
time step. The manifest, not adapter-specific code, owns the initial state, target trajectory,
duration, time step, gravity setting, and no-contact expectation.

v0.1 inspected top-level ASCII USDA overlay text. Gate 0 opens that same standard position-mode
`.usda` entry, loads its referenced and payload layers, and inspects the fully composed stage. The Python dependency "runtime
overlay" in `configs/parity/ovphysx-runtime-overlay.json` is an environment workaround and is not
a USD layer or asset edit.

## Remote boundary

The approved Server 6 root is:

```text
/data/home/exampleuser/sharpa-wave-asset-qa-gate0
```

Every project checkout, asset checkout, Python runtime, virtual environment, dependency cache,
temporary file, log, and result must remain below that root. The run must not use `sudo`, Docker,
Conda, system package changes, driver changes, Kit, a renderer, a camera, or `/mnt/ceph2`.

Only one GPU may be used. The launcher must re-check the selected device immediately before the
run, refuse a GPU with a compute process or material memory use, and expose only that device via
`CUDA_VISIBLE_DEVICES`. OVPhysX uses `cuda:0` inside the isolated process because the selected
physical GPU is remapped to the only visible device.

On the pinned driver, an otherwise idle A800 can report a small driver allocation. The launcher
therefore permits at most 32 MiB only when utilization is zero and the compute-process list is
empty; any larger allocation, utilization, or compute PID remains a hard stop.

## Required evidence

The result bundle must contain:

- the exact manifest and mapping used;
- validator, asset, Isaac Lab, and adapter revisions;
- platform-independent Git tree, actual worktree, configuration, and source hashes;
- Python/package versions, driver, CUDA runtime, GPU index and UUID;
- resolved USD-stage/schema observations;
- canonical and backend joint/frame names and indices;
- per-sample time, joint positions, velocities, targets, and distal-frame poses when available;
- whether contact observation was performed and, if evaluated, whether it is supported; contact
  validation itself is deferred;
- execution status, comparison status, finite-state diagnostics, repeatability, and dt-halving;
- stdout/stderr and a SHA-256 inventory for every bundle file.

Absolute private paths and credentials must not be written into the portable records.

## Pass/fail rule

Gate 0 passes only if both resolved USD assets open without Kit or a renderer, all 44 joints and 10
distal frames map explicitly, the authored position-drive and PhysX properties are observable,
OVPhysX performs real finite physics steps, every micro-scenario completes, all recorded numeric
values remain finite, repeatability and dt-halving are reported, and the returned bundle verifies
locally. These scenarios contain no ground plane or intentional contact target. Contact observation
is deferred and must not be reported as validated when the backend does not expose it.

Because Gate 0 contains only a small number of runs, it requires 100% completion. The `>=99%`
completion target belongs to the larger MVP matrix.

Execution and scientific interpretation are separate:

- `COMPLETED` / `ERROR` describe whether a run executed.
- `WITHIN_TOLERANCE` / `DIVERGENT` / `INCONCLUSIVE` describe comparison evidence.

`DIVERGENT` is a reproducible observation, not automatically an upstream defect.

## Stop conditions

Stop without broad retries if any of the following occurs:

- host, user, approved root, Git tree, canonical LF hash, dependency revision, or selected GPU differs;
- the GPU becomes occupied;
- installation needs privileged or system-wide changes;
- the supposedly kit-less path requires Kit, a renderer, a camera, a display, or RT hardware;
- the resolved USD cannot be opened or required articulation/drive schemas are absent;
- a no-contact run produces unexplained contacts;
- any state becomes non-finite or numerically explosive;
- GPU use reaches two hours, the remote tree reaches 30 GiB, or the result bundle reaches 500 MiB.
