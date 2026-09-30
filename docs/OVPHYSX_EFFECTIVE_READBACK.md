# OVPhysX effective-parameter readback

Freeze A was completed on 2026-08-28 (Asia/Shanghai) as an unofficial, simulation-only,
pre-trajectory readback of the pinned kit-less OVPhysX stack. It used neither Kit nor a renderer,
camera, or RTX rendering path. It is not a full Isaac Sim, hardware, safety, or Sim2Real result.

## Result

- Status: `READBACK_VALID`
- Classification: `descriptive_effective_parameter_readback_not_formal_gate0`
- Joint coverage: `44/44`
- Fresh processes: `4` (two per hand)
- Exact repeatability: joint mapping, solver evidence records, and canonical numeric vectors
- Experimental trace steps / commands / targets / samples: `0 / 0 / 0 / 0`
- Formal Gate 0: unchanged at `DIVERGENT`, `pass_ready=false`
- Freeze B follow-up: completed as `VALID` / `INCONCLUSIVE`

`READBACK_VALID` means the evidence bundle is complete, internally consistent, and exactly
repeatable under this frozen stack. It does not mean that an OVPhysX parameter has the same physical
meaning or realized effect as a similarly named MuJoCo parameter.

## What the readback established

| Family | Frozen evidence availability | Conservative classification |
| --- | --- | --- |
| Friction | 88/88 records have authored, resolved, and runtime-effective layers | `UNMAPPABLE` |
| Armature | 88/88 records have authored, resolved, and runtime-effective layers | `UNMAPPABLE` |
| Control/drive | Runtime evidence exists for the controller model, Kp/Kd, backend drive stiffness/damping, effort limit, and target interpretation; velocity limit, separate passive damping, and gear/transmission are unavailable | `UNMAPPABLE` |
| Solver/runtime | Execution-pipeline and configured-dt evidence are partial; integrator, iteration counts, solver type, stabilization, sleep threshold, substeps, and tolerance composites are unavailable | `UNMAPPABLE` |

The friction label exposed by the pinned wrapper conflicts with the authoritative API description
for its non-viscous slots, so equal or similar numbers cannot establish MuJoCo friction equivalence.
The armature record is complete on the OVPhysX side, but this one-engine readback does not verify the
cross-engine action and physical-law requirements needed for direct transfer. Incomplete control and
solver layers take precedence under Freeze A. The validator records those layers as unavailable and
does not invent defaults or describe a configuration accessor as a compiled runtime getter.

The four exact solver repetitions mean the same evidence records and availability states were
observed in each fresh process. They do not mean every requested solver value was exposed.

## Execution boundary

The probe ran only after adapter initialization and reset. Those zero-state and zero-target writes
are disclosed in the private evidence. The probe itself issued no experimental control command and
made no user or experimental `simulation.step` call. Any backend-internal GPU warmup or initialization
advance is reported separately where exposed and is never inferred when the pinned API does not
provide it.

This readback is descriptive evidence only. It cannot identify the cause of an existing trajectory
difference, infer measured or realized torque, establish cross-engine parameter equivalence, or
support an upstream/backend bug claim.

## Reproducibility identity

| Field | Value |
| --- | --- |
| Source revision | `44571ea60201f77ca93d49c9c14cb2d28152c30f` |
| Source tree | `d2a03e8ecceb46971e5c75debdb738baea63a4a3` |
| Source archive SHA-256 | `a4e925c2ce0cd58463738d2472bb77980c20077d42bce7717528dff693a35006` |
| Immutable remote snapshot SHA-256 | `839ec464e600ce87d622b640b65e5119db05aa4e96f0fe2678da91e21d2af988` |
| Freeze A config SHA-256 | `12c89fd8e0cd3bfa849a96a2e133b22e9191fc0ca6188f18fde4960d4221ab4c` |
| Gate 0 manifest semantic SHA-256 | `576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e` |
| Canonical LF asset SHA-256 | `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad` |
| Private bundle inventory | 56 payload files, 2,331,969 bytes, exact |
| Private bundle root SHA-256 | `858eee53e0ece9956a06b7cbd72a3d0a8348156e3c236ebf31ffd72fceb4fd4a` |
| Public report SHA-256 | `5cbc1f33b974e519bc91b864d859f8904d806baf5ab266014496871225330898` |
| Public summary SHA-256 | `d359ffd9622907658edc3c412b1870aab8ae717d1a5805f77bc7b5f530b2d055` |

The full content-hashed bundle remains private audit evidence because it contains raw parameter
records and machine/process identifiers. Only the generated [compact report](../results/effective-readback/report.md)
and [machine-readable summary](../results/effective-readback/summary.json) are publication-safe.
Earlier failed attempts are retained privately and are not included in this valid result.

## Freeze B completion

Freeze B is complete: all `24/24` cases passed `44/44` joint and `10/10` fingertip frame mapping,
fresh repeatability, dt-halving, and finite/sanity checks. Its final status is `VALID` with the
scientific label `INCONCLUSIVE`. The runtime observation that zero treatment changed `0/8` cases
from this R1 readback is descriptive only and does not decide the label.

The active trajectories remain campaign-G evidence (`d890f8e`, tree `29b6d68`). Analysis I
(`8e2fd84`, tree `177e470`) applied only a disclosed post-collection timestamp-compatibility
correction; it did not rerun either simulator or change evidence. The earlier local H
`INVALID`/null analysis bundle was not published externally and is preserved as superseded private
audit evidence.

Formal Gate 0 remains `DIVERGENT` with `pass_ready=false`. See the
[Freeze B note](OVPHYSX_FRICTION_FREEZE_B.md),
[sanitized report](../results/freeze-b/report.md), and
[machine-readable summary](../results/freeze-b/summary.json). Freeze B remains unofficial and
simulation-only; it is not a backend bug, hardware, Sim2Real, or cross-engine equivalence claim.
