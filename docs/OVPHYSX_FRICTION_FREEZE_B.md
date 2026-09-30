# OVPhysX Freeze B friction sensitivity

Freeze B completed on 2026-08-30 (Asia/Shanghai). The final result is `VALID` with the scientific
label `INCONCLUSIVE`.

This is an unofficial, simulation-only study of native inputs in the pinned kit-less OVPhysX
stack. It is not evidence of a backend bug, hardware behavior, Sim2Real behavior, or cross-engine
parameter equivalence.

## Validity result

- Case completion: `24/24` (`8` MuJoCo and `16` OVPhysX).
- Canonical mapping: `44/44` joints and `10/10` fingertip frames.
- Fresh-process isolation: `24/24` cases.
- Formal same-source bridge and fresh repeatability: exact zero deltas.
- dt-halving: passed the frozen limits for joint, frame-position, and frame-orientation deltas.
- Finite-value and preregistered sanity checks: passed for every trace.
- Position-target readback: `16/16`, with an aggregate maximum error of `0 rad`.

## Scientific decision

The frozen rule evaluates eight normalized sensitivity values, `S_Dq` and `S_Dp` for each
hand/timestep cell:

| Hand | Timestep | `S_Dq` | `S_Dp` |
| --- | --- | ---: | ---: |
| left | base | 0.09305827319415706 | 0.10210679734906178 |
| left | halved | 0.09702777058089153 | 0.10679712580803115 |
| right | base | 0.09000382487930997 | 0.09914972602988767 |
| right | halved | 0.09384820757062738 | 0.10367760461408876 |

- `SUPPORTED` requires all eight values to be at least `0.25`.
- `NOT_SUPPORTED` requires all eight values to be at most `0.10`.
- Every other valid result is `INCONCLUSIVE`.

Here, `5/8` values are at most `0.10`, `3/8` are above `0.10`, and `0/8` are at least `0.25`.
The only result allowed by the frozen rule is therefore `INCONCLUSIVE`.

The runtime-friction readback also found that `0/8` zero-treatment cases changed from frozen R1.
That count is descriptive only. It does not decide validity or the scientific label.

## Post-collection corrections

The active trajectories were collected once under campaign G and were not rerun. H added an
analysis-only canonical joint-order correction. H then produced a local `INVALID`/null bundle when
its metric builder required bit-exact cross-trace timestamps. That H bundle was never published
externally; it is preserved as superseded private audit evidence.

I is a disclosure-complete, post-collection analysis-only correction. It accepts only same-step,
same-length cross-trace timestamps within the already-frozen absolute tolerance of `1e-12 s`.
It does not round or modify timestamps, raw trajectories, evidence, the protocol, windows,
formulas, thresholds, targets, case order, or dt. It required no simulator rerun.

## Reproducibility identity

| Field | Value |
| --- | --- |
| Campaign G revision | `d890f8ee83741cf643aadf2d9d1f256bf41c09cc` |
| Campaign G tree | `29b6d6846e181897c4be2c3e19361e061524b061` |
| Analysis I revision | `8e2fd84ccd9f9ad89ec2101f1f9e61057e37b71d` |
| Analysis I tree | `177e470e9be68f7d014fc9a8bba73d673943540d` |
| Private bundle inventory | 407 payload files, 93,091,294 bytes, exact |
| Private bundle root SHA-256 | `3aed0876784c4575be20e5f5fbb6879f840829ae0b7885bda49ca133698bf04b` |
| Safe summary SHA-256 | `6a7cc328b614cc15cba1b3819c90527a371313dad1f70b64bc9e4247ab76d611` |
| Safe report SHA-256 | `914f8e42854de817af4646680e15271000c7d4eb2e787fec3d0b3ca7dad1e507` |

The locally prepared, sanitized outputs are the [compact report](../results/freeze-b/report.md) and
[machine-readable summary](../results/freeze-b/summary.json). This repository update does not claim
that either file has been published to an external service.

Formal Gate 0 remains `DIVERGENT` with `pass_ready=false`; Freeze B does not rewrite it.
