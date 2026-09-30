# WaveSimParity Contact Gate C0 status

Contact Gate C0 completed on 2026-08-31 (Asia/Shanghai). The evidence is `VALID`, the scientific
classification is `INCONCLUSIVE`, and `pass_ready=false`.

2026-09-30 follow-up: the separately identified async + wait candidate completed 32/32 cases
and finalized as `VALID / DIVERGENT`. It reduced OVPhysX release dt sensitivity from 5.5 ms to
0.5 ms without changing its recorded motion, but four cross-backend checks exceeded the original
thresholds. [Candidate results and limitations](CONTACT_ASYNC_RESULT.md). This does not replace
or relabel the original C0 result below.

In plain language: both simulators completed the same synthetic contact-versus-sham experiment,
the evidence and repeat runs are sound, but one timestep-sensitivity check missed its preregistered
limit. The result is therefore not promoted to either `DIVERGENT` or `WITHIN_TOLERANCE`.

## Scope

C0 compares the standard fixed-base left and right Wave hands separately in MuJoCo and the
experimental kit-less OVPhysX path. Each run uses one deliberately synthetic, frictionless sphere
attached to `index_DP`, one static box, zero gravity, and position control. Native hand collisions
are disabled so that exactly one selected sphere-box pair can be observed directly.

C0 does not cover native fingertip collision meshes or elastomers, wrist/flange variants,
dual-hand interaction, floating bases, full Isaac Sim, hardware, safety, or Sim2Real.

## Result

| Item | Result |
| --- | --- |
| Evidence status | `VALID` |
| Scientific status | `INCONCLUSIVE` |
| Pass-ready | `false` |
| Completed cases | `32/32` in `32` fresh processes |
| Joint mapping | `44/44` |
| Distal-frame mapping | `10/10` |
| Finite-state and runtime-log gates | passed |
| Fresh-process repeatability | exact within every frozen threshold |
| Admission/excitation checks | passed |
| Timestep-halving checks | four passed, one failed |

The only failed check was contact-event midpoint stability under timestep halving:

| Backend | Event | Maximum base/halved delta | Frozen maximum | Result |
| --- | --- | ---: | ---: | --- |
| MuJoCo | onset or release | `1.5 ms` | `4.0 ms` | pass |
| kit-less OVPhysX | onset | `0.5 ms` | `4.0 ms` | pass |
| kit-less OVPhysX | release | `5.5 ms` | `4.0 ms` | fail |

The same OVPhysX release value was reproduced for both hands and both fresh repeats. Other
aggregate timestep-halving maxima passed: joint/contact-effect `0.0012479424 rad` against
`0.01 rad`, gap/blocked-travel `0.0000111756 m` against `0.001 m`, orientation/contact-effect
`0.0005679039 rad` against `0.02 rad`, and steady-hold duty delta `0` against `0.01`.

The frozen decision order requires repeatability and timestep-halving admission to pass before
cross-simulator thresholds are used for a scientific label. Consequently the valid result stops at
`INCONCLUSIVE` with reason `admission_repeatability_or_dt_failed`. Any descriptive cross-engine
differences in this run are not promoted to a formal C0 divergence finding.

## Reproducibility identities

| Artifact | SHA-256 or identity |
| --- | --- |
| Formal source revision | `17eccdb672d89cb1dca78efe41181097e26c4e12` |
| Formal source tree | `bb0dfccffeb0b2a1e50d60862507fc4e9411ee3b` |
| Source archive | `5d446fb138c3c9aebcfb6008351ab645981db055bd71ebf9dee60c77fa3dc95f` |
| Remote snapshot | `f744bc0f24fa1e7fbe448b44d5e25a1b00df9900bf6510e6132a8bcd37e592f1` |
| Frozen manifest file | `73cc9dedffc1ea0600052fc1b78696eda292f1f34c7987ec1f433043e65bf215` |
| Frozen manifest semantics | `cb02398665dbed5722b6f9c44fca1c25a78ed495eb63cf5008e875672ceb930e` |
| Manifest schema | `8a0101837f63abe483e1b93f509d41e024f5e137cabe19657e7171752e0c15ca` |
| Run schema | `08adc224e03a2dba52ed3b09868e88c60999dff4a10aaa8cf2fbda7e75822a17` |

The local MuJoCo evidence inventory contains 98 payload files and 236,038,953 bytes with root
SHA-256 `6eb41643c064c33c5f2acd7909ace4c931d1abdeea254ceecd92df699da25729`.
The remote OVPhysX inventory contains 585 payload files and 239,817,683 bytes with root SHA-256
`4f5f9a4555548b7ebff3bc295e0e93ed00cdb386b6ccef404ff2815c9b8237cd`.
The sealed private bundle contains 689 payload files and 476,054,170 bytes with root SHA-256
`7f69ff9416c755401a7b2915684c457930feb13cb0803effc2955c90468dd69d`.
All three inventories were independently re-enumerated byte-for-byte and their record lists and
root hashes matched exactly.

The public projection is exactly two files. The sanitized
[summary](../results/contact-c0/summary.json) has SHA-256
`168070d53be26c0a092b3069713128c3e98625839e41d80bb16ae4c1a63b6801`; the
[report](../results/contact-c0/report.md) has SHA-256
`1078d311745d195eadb6fd588a4362f63a63ac83fb595b9c9cac0be3e7c13728`.
Their bytes match the public copies sealed inside the private bundle.

One presentation-only limitation is recorded explicitly. Re-parsing the canonical JSON and then
calling only the generic Markdown renderer swaps the display order of the two Boolean keys in the
`condition_signal_semantics` cell and does not add the finalizer-owned existing-Gate-0 paragraph.
The two sealed public artifacts themselves are deterministic, byte-identical to their bundle
copies, semantically consistent, and independently hash-verified; JSON-to-Markdown byte round-trip
through that lower-level renderer was not a frozen C0 acceptance requirement. The sealed report is
therefore left unchanged rather than retroactively rewritten.

## Runtime evidence boundary

The private OVPhysX evidence verifies the frozen asset and dependency identities, direct
one-body/one-target filter, native-collision disablement, two enabled synthetic shapes, material
zeros, complete target/sample grids, and camera-free kit-less execution. A run-bound read-only
native helper verifies post-reset PhysX CCD flags and preserves authored mass, center-of-mass, and
inertia properties. The helper, process, GPU, host, and machine-local records remain private and
are not published in the two-file result.

## Excluded correction history

Engineering pilots and failed formal attempts are retained as excluded private evidence; none is
pooled with the final campaign:

- The early pilots exposed an over-strict passive camera-module guard, late PhysX schema
  registration, and a double-rotation error in the runtime inertia audit. The final excluded pilot
  then exercised one complete OVPhysX case before the formal run.
- Formal attempt a1 stopped when halved-dt cases incorrectly retained the base-dt step count. A
  new source revision fixed duration-to-step derivation and required a complete rerun.
- Formal attempt a2 stopped when floating-point accumulation placed the last halved-dt sample just
  above the exact seven-second public time boundary. A new revision used a canonical integer time
  grid while retaining a separate runtime-time integrity check.
- Formal attempt a3 completed the local matrix, but pre-finalization audit found that an unstable
  `acos` quaternion formula reported a tiny nonzero angle for byte-identical orientations. The
  owned remote run was stopped safely at a case boundary. A new revision replaced the near-zero
  calculation with the equivalent stable `atan2` geodesic and reran all 32 cases as a4.

No threshold, contact fixture, command, decision rule, or collected final trace was changed after
observing the a4 result.

## Interpretation boundary

`VALID` means the frozen evidence, identity, completeness, finite-value, mapping, runtime, and
privacy contracts passed. `INCONCLUSIVE` means C0 did not meet every prerequisite for a scientific
cross-simulator label; it does not mean the evidence is corrupt. The `5.5 ms` observation is a
pinned simulation result, not automatically a Sharpa, MuJoCo, NVIDIA, OVPhysX, or Isaac Sim bug.

Formal Gate 0 and Trajectory T1 remain `DIVERGENT` with `pass_ready=false`; C0 neither rewrites nor
repairs them. This project remains unofficial and simulation-only, with no hardware or Sim2Real
claim and no external publication, push, or pull request.

## Post-hoc release-event diagnosis (2026-09-29)

A read-only reanalysis of all 32 sealed cases reproduced the 5.5 ms event delta
and found a timestep-dependent scale in the OVPhysX reported contact signal.
Its halved/base steady-hold ratio is about 0.498, whereas the MuJoCo ratio is
about 0.99998. An exploratory dt-scaled detection threshold reduces the OVPhysX
event delta to 0.5 ms; geometric separation crossings at positive 1/10/100 micrometer
gaps differ by less than 0.17 ms using descriptive linear interpolation.
This localizes a reporting/threshold mechanism, without establishing the native
cause or changing the frozen verdict. See [the diagnostic](CONTACT_RELEASE_DIAGNOSIS.md).

The subsequent [known-load CPU calibration](CONTACT_LOAD_CALIBRATION.md) reproduced
a native step_sync contact-report scaling behavior consistent with reuse of the
last asynchronous timestep: all 24 load cases and 14 crossover phases matched the
frozen predictions. Subsequently, normal Slurm allocation restored GPU access;
the [native DirectGPU crossover](CONTACT_GPU_CROSSOVER_DIAGNOSIS.md) reproduced
the same history-dependent ratios in 12/12 phases across two fresh processes,
with byte-identical raw phase samples. The GPU load matrix and C0 ContactSensor
wrapper comparison remain outstanding. This narrows the diagnosis but does not
change formal C0 acceptance or establish corrected C0 parity.

The later [24-case native GPU matrix](CONTACT_GPU_MATRIX_DIAGNOSIS.md) also completed:
all static-load checks passed and paired raw samples were byte-identical.
Pure-sync reads exhibited mg*dt scaling; async+wait reads matched mg. The actual
ContactSensor comparison attempt failed before trajectory collection because the
new diagnostic fixture omitted the target root-layer /World parent. Failure evidence
was retained; no wrapper runtime conclusion or corrected C0 result is claimed.

In a separately frozen a2 attempt, the missing parent was created and a CPU-only
USD-copy preflight passed. The [actual ContactSensor comparison](CONTACT_SENSOR_PASSTHROUGH_DIAGNOSIS.md)
then verified exact native-before / public-sensor / native-after equality in all
1,500 samples, with byte-identical fresh repeats and expected 1 / 0.5 / 1 ratios.
This closes the primitive sensor-passthrough diagnostic, not the full Sharpa C0
controller/adapter rerun. Formal acceptance and the original evidence remain unchanged.

An opt-in [async candidate through SimulationContext](CONTACT_ASYNC_CANDIDATE.md)
subsequently restored the primitive sensor ratio from 0.5 to 1 in the controlled
middle phase, with exact fresh repeats and one native step/wait per context call.
The separate full 32-case corrected campaign is not yet implemented or executed;
this is preparation evidence, not a new C0 verdict.
