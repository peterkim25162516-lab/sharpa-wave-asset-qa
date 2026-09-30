# Left thumb common-FK attribution

Formalized on 2026-08-27 (Asia/Shanghai). This is an offline, simulation-only, post-hoc
diagnostic of one already-observed WaveSimParity trajectory. It is not a formal Gate 0 rerun,
an independent confirmation, a dynamics intervention, a hardware or Sim2Real result, or a
backend bug finding.

## Question and method

The frozen formal OVPhysX trace and two frozen MuJoCo traces disagree at the
`left_thumb_DP` distal-link origin during steps 55–85 (`0.110–0.170 s`) of the left
`small_step` case. The analysis asks a narrow question: if all three recorded 22-joint vectors
are replayed through the same fixed-base MuJoCo forward-kinematics model, does the resulting
thumb-frame displacement reproduce the recorded cross-simulator position gap?

No simulator dynamics were advanced. The analysis used only canonical
`TraceSample.joint_positions`, set all 22 named joints for every replay, zeroed velocity, and
applied no translation, rotation, scale, offset, alignment, or time shift. The plan was frozen
after a disclosed one-step feasibility inspection at step 66 (`0.132 s`) and before computing
the complete 31-step window, so the result is explicitly post-hoc rather than blind.

For attribution, the five thumb joints were treated as players. All 32 coalitions were replayed
at every sample, and exact Shapley values were computed with factorial weights. These values are
counterfactual allocations under this particular model, window, baseline, and player set; they
are not physical parameter estimates or causal effects.

## Result

Both candidate gaps close under common MuJoCo FK:

| Candidate | Window residual/raw | Raw-peak residual/raw | Raw peak | Residual at raw peak | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Same-source zero-friction control | 0.005274% | 0.003892% | 0.850667 mm | 0.0000331 mm | `CLOSES` |
| Total-viscous treatment | 0.005273% | 0.004063% | 0.814749 mm | 0.0000331 mm | `CLOSES` |

Both raw peaks occur at step 66 (`0.132 s`). Under the frozen decision rule, `CLOSES` requires
both ratios to be at most `10%`. The much smaller observed ratios support the bounded statement
that the recorded `left_thumb_DP` position gap is kinematically consistent with the recorded
canonical joint-state gap under this common MuJoCo model.

This does not identify why the simulators produced different joint states. In particular, it
does not establish a controller, friction, armature, integration, solver, or backend cause, and
it does not say which simulator is closer to hardware.

## Exact five-joint allocation

Signed values below are fractions of the raw window-gap energy. Negative values mean that, under
the fixed counterfactual construction, adding that player slightly increases rather than closes
the residual energy.

| Candidate | Joint | Signed energy fraction | Positive share | Registered label |
| --- | --- | ---: | ---: | --- |
| Control | `left_thumb_CMC_AA` | 66.624% | 66.434% | `DOMINANT_POSITIVE_CLOSER` |
| Control | `left_thumb_CMC_FE` | 24.060% | 23.991% | `NON_DOMINANT` |
| Control | `left_thumb_MCP_FE` | 9.602% | 9.575% | `NON_DOMINANT` |
| Control | `left_thumb_IP` | 0.000% | 0.000% | `NON_DOMINANT` |
| Control | `left_thumb_MCP_AA` | -0.286% | 0.000% | `NON_DOMINANT` |
| Treatment | `left_thumb_CMC_FE` | 50.779% | 49.257% | `NON_DOMINANT` |
| Treatment | `left_thumb_CMC_AA` | 34.775% | 33.733% | `NON_DOMINANT` |
| Treatment | `left_thumb_MCP_FE` | 17.536% | 17.010% | `NON_DOMINANT` |
| Treatment | `left_thumb_IP` | 0.000% | 0.000% | `NON_DOMINANT` |
| Treatment | `left_thumb_MCP_AA` | -3.089% | 0.000% | `NON_DOMINANT` |

Only control `left_thumb_CMC_AA` satisfies both frozen dominance conditions. Treatment
`left_thumb_CMC_FE` has the largest signed allocation but misses the required `50%` positive-share
threshold, so it remains `NON_DOMINANT`; the label was not relaxed after seeing the values.

## Sanity and integrity checks

- The MuJoCo topology was exactly 22 scalar hinge joints, with the five thumb ancestors in the
  registered canonical order.
- At `t=0`, identity-only replay of all five OV distal frames differed by at most
  `4.75070e-8 m` and `1.20065e-6 rad`, within the frozen coordinate gates.
- Both candidates had 31 samples, 32 coalitions per sample, and exact empty/full-coalition and
  non-thumb-invariance checks at recorded precision.
- MuJoCo self-replay errors were zero. Exact energy-Shapley efficiency error was zero; maximum
  vector-Shapley efficiency error was below `1.72e-19 m`.
- All published numeric records were finite. No fit was performed.

## Evidence identity and publication boundary

| Field | Value |
| --- | --- |
| Analysis plan SHA-256 | `da63d86f48f9b40e7bfff034582f62c96d3787400ef46184a50fe90a429b30e2` |
| Analysis source commit | `cb76e54ecfa3e321020d3a52bce6684937283a24` |
| Analysis source tree | `6745c64c8e202e9800b5afd056de2220879f3412` |
| Parent diagnostic bundle root | `a27826a0909afbf550f2bda198d7c7aab564e019a06a40aa63e829de92c0018c` |
| MuJoCo model SHA-256 | `3cbeb46259d4ba63cbdb83085255d1a8f8031c51e0101a6622f6e7e81a64dc11` |
| Common-FK bundle root | `d2fbad78e704f7cc673e2ab751473ebe625c86e52b5a6082854b98e1112a1624` |
| Common-FK `bundle.json` SHA-256 | `0695f52847e044f81ea1210d3ea61edaf6f1b9513fd3f7fb4f8eadd7ef622721` |
| Common-FK bundle inventory | 30 files / 6,622,792 bytes / exact inventory |
| Common-FK `summary.json` SHA-256 | `dd9c6c0385730808951def4b8886889aa31e11b879b4e93a3f0fc0ecc3824838` |
| Common-FK `replay.json` SHA-256 | `0b33eb00d0ea4bdbc378cbc4dfce8e5df92da7ff130485079002a70a87f2ea74` |

The full bundle remains private because it copies frozen OVPhysX evidence containing
machine-specific metadata. This document and the compact Gate 0 records contain only sanitized
aggregate metrics and hashes. Publishing the raw bundle would require redaction and a new root
hash.

Formal Gate 0 remains `DIVERGENT / pass_ready=false`.

## Reproduction

Run from a clean checkout at source commit `cb76e54ecfa3e321020d3a52bce6684937283a24`, with the
pinned LF asset tree and parent diagnostic bundle materialized locally:

```powershell
.\.venv\Scripts\python.exe -B -P scripts\run_thumb_dp_common_fk_analysis.py `
  --source-revision cb76e54ecfa3e321020d3a52bce6684937283a24 `
  --asset-root external\sharpa-assets-6eea427-lf `
  --parent-bundle results\mujoco-total-viscous-diagnostic-85ca568-20260827-a1 `
  --output-dir results\thumb-dp-common-fk-NEW-RUN `
  --session-id common-fk-NEW-RUN
```

The command fails closed on source or asset drift, an invalid or non-exact parent bundle, unsafe
paths, missing canonical joints or frames, non-finite values, coordinate/sanity-gate failure,
persisted-output drift, an unexpected payload, or an existing output directory.
