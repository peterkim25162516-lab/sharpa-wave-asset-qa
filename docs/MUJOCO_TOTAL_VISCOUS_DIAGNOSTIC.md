# MuJoCo total-viscous retuning diagnostic

Formalized on 2026-08-27 (Asia/Shanghai). This is an unofficial, simulation-only endpoint
diagnostic for one frozen WaveSimParity trajectory. It is not a new Gate 0 acceptance run, a
full Isaac Sim result, a hardware or Sim2Real claim, or a backend bug finding.

## Question and intervention

After MuJoCo dry friction was set to zero, two early residuals remained in the left-hand
`small_step` trajectory:

- a `left_pinky_CMC` joint gap in `0.11–0.16 s`;
- a `left_thumb_DP` distal-frame-position gap in `0.11–0.17 s`.

The diagnostic asked whether one controller-side viscous retuning would materially reduce both
registered gaps. It generated two new MuJoCo runs from the same clean source and session:

1. a zero-dry-friction control using the canonical MuJoCo controller gains;
2. a zero-dry-friction treatment that changed only each position actuator's velocity-gain term so
   that `effective controller Kv + MuJoCo passive joint damping = frozen OV explicit-PD Kd`.

The numeric equality above is an intervention definition, not a claim that MuJoCo and OVPhysX
implement equivalent continuous- or discrete-time damping. OV backend drive damping and stiffness
were zero in the frozen evidence; passive PhysX damping was not aligned or inferred.

All 22 treatment targets were read back exactly. The adapter changed only
`actuator_biasprm[2]`; full actuator gain arrays were unchanged. Both runs used the same assets,
commands, initial state, timestep, controller stiffness, passive joint damping, armature, gear,
force limits, solver, and gravity. Each completed 250/250 steps with 251 finite samples and zero
contacts. The new control reproduced the frozen zero-friction MuJoCo samples exactly.

## Frozen evidence identity

| Field | Value |
| --- | --- |
| Diagnostic source | `85ca5680eec5be9197af0f2437e6eb913d0262a8` |
| Diagnostic source tree | `dba00025c35014257880bf18c4e751bf949c4c29` |
| Session | `total-viscous-85ca568-20260827-a1` |
| Preregistration SHA-256 | `5acff5d5ce9b34a133032830e62cc831f5c583087c5e18e244367edd8464d5ba` |
| Diagnostic bundle root SHA-256 | `a27826a0909afbf550f2bda198d7c7aab564e019a06a40aa63e829de92c0018c` |
| Diagnostic `bundle.json` SHA-256 | `2022d931615b678f8669c5847efe292dec18c08b6945c1476b034b49390bda81` |
| Diagnostic bundle inventory | 39 files / 5,750,736 bytes / exact inventory |
| Frozen friction bundle root SHA-256 | `0804626ef0b7009a1e6c223284c210550e568593852046bc9b2b31d5bf3e4803` |
| Frozen formal bundle root SHA-256 | `d2e028d884f5484f7a7d5ecc0506201ff751c0847ed87b7a63d3a3b8cd056505` |
| New control run SHA-256 | `60004cd34f368019453e2b2c5f012b359f6d5bef47679823347673d4688b7e60` |
| New treatment run SHA-256 | `4a0413bcdf93d6e8304dc0b0a91afdf5548696c4eaefa943e7c5a8efb3766d56` |

The raw bundle contains copied GPU UUID/index and CUDA-device/worker-process identifiers and
remains private. The numbers and hashes in this document are the sanitized publication record.

## Preregistered result

Scientific status: `mixed_or_inconclusive`. The registered joint window improved substantially,
but the registered distal-frame window did not.

| Registered metric | Zero-friction control | Retuned treatment | Reduction |
| --- | ---: | ---: | ---: |
| `left_pinky_CMC`, peak joint gap in `0.11–0.16 s` | 0.00908694 rad | 0.00160271 rad | 82.3625% |
| `left_thumb_DP`, peak position gap in `0.11–0.17 s` | 0.850667 mm | 0.814749 mm | 4.2222% |

Support required both reductions to be at least `25%`. Rejection required both to be at most
`10%`. Because only one registered metric cleared the support threshold, neither rule fired.
The experiment therefore does not support one common total-viscous retuning as a major explanation
for both local residuals.

## Secondary observations

| Whole-trace metric | Zero-friction control | Retuned treatment | Relative change |
| --- | ---: | ---: | ---: |
| Global joint maximum | 0.00908694 rad | 0.00776686 rad | 14.5272% lower |
| Global distal-frame-position maximum | 0.850667 mm | 0.814749 mm | 4.2222% lower |
| Global distal-frame-orientation maximum | 0.0110405 rad | 0.0122121 rad | 10.6120% higher |
| Joint RMSE | 0.000936137 rad | 0.000819889 rad | 12.4178% lower |
| Distal-frame-position RMS | 0.129393 mm | 0.140539 mm | 8.6141% higher |

These metrics were registered as secondary and are not part of the decision rule. The treatment
moves the worst joint sample from `left_pinky_CMC` to `left_thumb_CMC_FE` at `t=0.136 s`, with a
remaining `0.00776686 rad` gap. The worst position sample remains `left_thumb_DP` at `t=0.132 s`.
The worsened orientation maximum and frame-position RMS are further reasons not to describe the
treatment as a general trajectory improvement.

The result is consistent with the pinky and thumb transients having different dominant
mechanisms. It does not prove a cross-backend parameter mismatch: this endpoint retuning may
compensate for other integration, armature, solver, or backend-semantic differences.

The authoritative formal result remains `DIVERGENT / pass_ready=false`.

## Completed next check

The proposed common-kinematics replay is complete. Across `0.11–0.17 s`, replaying the frozen OV
and both MuJoCo 22-joint vectors through one fixed-base MuJoCo FK model left about `0.0053%` of
each recorded `left_thumb_DP` gap unexplained. Both candidates satisfy the frozen `CLOSES` rule.
The bounded interpretation is that the selected frame gap follows the recorded joint-state gap
under this common model, not that any dynamics cause or backend bug has been identified. See the
[common-FK attribution](THUMB_DP_COMMON_FK_ATTRIBUTION.md). Any broader dynamic conclusion still
requires effective OVPhysX friction, armature, and solver readback plus both hands.

## Reproduction

Run from a clean checkout whose `HEAD` is the diagnostic source commit above, with the pinned LF
asset tree and frozen friction bundle materialized locally:

```powershell
.\.venv\Scripts\python.exe -B -P scripts\run_mujoco_controller_kv_diagnostic.py `
  --asset-root external\sharpa-assets-6eea427-lf `
  --manifest configs\parity\gate0.json `
  --preregistration configs\parity\controller_kv_diagnostic.json `
  --friction-bundle results\mujoco-friction-scan-e238b29-20260827-a1 `
  --expected-friction-root-sha256 0804626ef0b7009a1e6c223284c210550e568593852046bc9b2b31d5bf3e4803 `
  --output-dir results\mujoco-total-viscous-diagnostic-NEW-RUN `
  --session-id total-viscous-NEW-RUN `
  --source-revision 85ca5680eec5be9197af0f2437e6eb913d0262a8
```

The command refuses a dirty tracked source, identity drift, a non-authoritative parent bundle, an
existing output directory, copied-evidence drift, non-canonical run filenames, non-Kv actuator
changes, or a non-exact final inventory.
