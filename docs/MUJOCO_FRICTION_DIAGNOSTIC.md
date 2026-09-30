# MuJoCo dry-friction diagnostic

Formalized on 2026-08-27 (Asia/Shanghai). This is an unofficial, simulation-only diagnostic of
one frozen WaveSimParity trajectory. It is not a new Gate 0 acceptance run, a full Isaac Sim
result, a hardware or Sim2Real claim, or a backend bug finding.

## Question and method

The formal left-hand `small_step` result showed a strong correlation between the joint response
gap and MuJoCo `frictionloss / Kp`. To test whether that relationship was material rather than
merely descriptive, the diagnostic held the formal OVPhysX trace fixed and generated three new
MuJoCo traces from the same clean source commit and session. The only intentional intervention
was an in-memory scale applied to all 22 compiled `MjModel.dof_frictionloss` values:

- `1.0`: canonical compiled friction, used to prove zero drift from the old formal MuJoCo trace;
- `0.5`: half the canonical compiled friction;
- `0.0`: zero compiled dry friction.

Canonical assets were not edited. Each run completed 250/250 steps with 251 finite samples, zero
contacts, the same commands, and the same initial state. The registered positive result required
the selected and trajectory-wide joint and distal-frame-position gaps to decrease by minimum
effect sizes at both scale transitions. The new `1.0` trace also had to reproduce the frozen
formal MuJoCo trace within `1e-9` for all three formal metrics.

## Evidence identity

| Field | Value |
| --- | --- |
| Diagnostic source | `e238b29934f2e341703ba1aa7decdab7cae53688` |
| Diagnostic source tree | `65d5cc536ec221e10652bdf03cacb045675c2531` |
| Session | `friction-scan-e238b29-20260827-a1` |
| Diagnostic bundle root SHA-256 | `0804626ef0b7009a1e6c223284c210550e568593852046bc9b2b31d5bf3e4803` |
| Diagnostic bundle inventory | 22 files / 6,487,855 bytes / exact inventory |
| Frozen formal bundle root SHA-256 | `d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505` |
| Frozen OVPhysX run SHA-256 | `008ed804406ee32e58a5b701bc9feb9f4123a23bc13d521392a2525d0de2dc30` |
| Canonical asset tree SHA-256 | `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad` |

The `1.0` trace reproduced the old formal MuJoCo trace exactly: joint, frame-position, and
frame-orientation drift were all zero. An independent audit recomputed the bundle root, every
payload hash, and all reported trajectory metrics from the raw samples.

## Result

The registered local hypothesis was supported for this one trajectory.

| MuJoCo friction scale | `thumb_CMC_FE` friction (Nm) | Joint max (rad) | DP position max (mm) | Orientation max (rad) | `thumb_CMC_FE` gap at formal peak (rad) | `thumb_DP` gap at formal peak (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.132 | 0.0186829 | 2.10924 | 0.0457305 | 0.0186709 | 2.10924 |
| 0.5 | 0.066 | 0.00949583 | 1.04679 | 0.0230670 | 0.00948250 | 1.03373 |
| 0.0 | 0 | 0.00908694 | 0.850667 | 0.0110405 | 0.0000953933 | 0.0925607 |

From scale `1.0` to `0.0`, the trajectory-wide joint maximum decreased by `51.36%`, the distal
frame-position maximum by `59.67%`, and the frame-orientation maximum by `75.86%`. Metrics that
use the whole trace rather than only its worst sample changed more strongly: joint RMSE decreased
from `0.00889769` to `0.000936137 rad` (`89.48%`), and distal-frame position RMS decreased from
`1.38483` to `0.129393 mm` (`90.66%`). At the old formal peak time, the selected joint and frame
gaps decreased by `99.49%` and `95.61%`.

This establishes that MuJoCo dry friction is a material contributor to the observed joint and
distal-frame-position differences in this left-hand `small_step` trajectory. It does not show
that MuJoCo and PhysX friction parameters are equivalent, isolate a PhysX defect, explain every
remaining difference, generalize to the right hand or other scenarios, or change the formal
Gate 0 result from `DIVERGENT / pass_ready=false`.

## Remaining transient

With dry friction removed, the long-lived `thumb_CMC_FE` lag nearly disappears. The identity of
the worst joint sample changes to `left_pinky_CMC` at `t=0.130 s`: MuJoCo is ahead of OVPhysX by
`0.00908694 rad` during a short early transient. The `thumb_DP` residual position peak is
`0.850667 mm` at `t=0.132 s`, when several MuJoCo thumb joints are also ahead. These gaps decay;
by `t=0.5 s`, the maximum joint and frame-position differences are approximately `7.39e-7 rad`
and `1.57e-7 m`. The residual is therefore a short-time response-shape difference, not the old
steady lag or evidence of a mapping swap.

The compiled `left_pinky_CMC` stiffness values are nearly equal, while MuJoCo's compiled actuator
velocity gain is lower than the frozen OVPhysX explicit-PD damping value. That direction is
consistent with a faster, lightly overshooting MuJoCo transient, but it is only a mechanism clue;
armature, integration, solver, and backend semantics have not been aligned.

## Completed follow-up

The smallest residual experiment was completed with a new same-source control and treatment. Dry
friction stayed at zero; the treatment changed controller Kv so controller Kv plus MuJoCo passive
joint damping equaled the frozen OV explicit-PD Kd. The two named-window metrics were:

- `left_pinky_CMC` joint gap in `0.11–0.16 s`, baseline `0.00908694 rad`;
- `left_thumb_DP` position gap in `0.11–0.17 s`, baseline `0.850667 mm`.

The pinky window decreased by `82.36%`, but the thumb frame window decreased by only `4.22%`.
The preregistered result is therefore `mixed_or_inconclusive`, consistent with different dominant
mechanisms for the two transients. See the
[total-viscous retuning diagnostic](MUJOCO_TOTAL_VISCOUS_DIAGNOSTIC.md) for the exact intervention,
hashes, decision rule, secondary metrics, and next falsifiable check. A broader follow-up must
still add effective OVPhysX friction, armature, and solver readback and test both hands before any
general cross-simulator claim.

## Reproduction

Run from a clean checkout whose `HEAD` is the diagnostic source commit above, with the pinned LF
asset tree materialized locally:

```powershell
.\.venv\Scripts\python.exe -B -P scripts\run_mujoco_friction_scan.py `
  --asset-root external\sharpa-assets-6eea427-lf `
  --manifest configs\parity\gate0.json `
  --formal-bundle results\gate0-bundle-c571d9b-20260827-a1 `
  --expected-formal-root-sha256 d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505 `
  --output-dir results\mujoco-friction-scan-NEW-RUN `
  --session-id friction-scan-NEW-RUN `
  --source-revision e238b29934f2e341703ba1aa7decdab7cae53688
```

The authoritative internal bundle is immutable and retained locally. It includes copied formal
raw evidence with a stable GPU UUID and short-lived GPU/process identifiers, so it is not the
publication artifact. This document contains the publication-safe scientific result and evidence
hashes without those infrastructure identifiers.
