# v0.1 result notes

These notes separate reproduced observations from interpretation. They apply only to upstream commit `6eea427eb24189519f32b9f21674cd534d3f973c` and the software versions recorded in the report.

## Dual-hand MuJoCo paths

All three dual-hand MJCF entry points fail before simulation because referenced meshes resolve below the dual-hand directory rather than the sibling left/right asset directories. The static path rule and the independent MuJoCo loader reproduce the same condition.

This is registered as `KNOWN` because upstream [PR #1](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1) already proposes the corresponding `meshdir` changes. v0.1 does not modify the upstream files and does not present this as a new discovery.

## Fixed-base kinematic parity

For the standard left and right hands, 256 fixed-seed joint configurations were sampled from the intersection of URDF and MJCF limits. After one zero-pose hand-base alignment, all five common distal-phalanx (`*_DP`) frames remain within:

- maximum position error: approximately `0.000071 mm`;
- maximum orientation error: approximately `0.000086°`.

This supports a narrow claim: the parser-normalized kinematic chains agree numerically for the sampled configurations. It does not establish contact, inertial or controller equivalence.

## Wrist/flange frame offsets

The wrist and flange variants retain very small orientation error but show nearly constant position offsets:

- wrist variants: approximately `29.0 mm`;
- flange variants: approximately `29.5 mm`.

The authored transforms explain why the validator reports this:

- in URDF, the palm frame `*_hand_C_MC` is attached to the wrist/mount chain, while the proximal finger origins retain the standard-hand offsets;
- in MJCF, the root body remains named `*_hand_C_MC`, while proximal finger origins include an additional mount-dependent Z offset.

Because those roots may intentionally encode different mounting conventions, v0.1 labels the result `WARN`, not `FAIL`. The validator performs one fixed base alignment and does not fit a separate transform per pose or per finger.

Before interpreting this as an asset defect, the next phase should confirm the intended frame contract, inspect binary USD stages through official USD/Isaac APIs, and compare visual/collision transforms in addition to link frames.

## Performance numbers

The recorded MuJoCo real-time factors are local smoke-test timings, not a simulator benchmark. They use short zero-control rollouts and are useful only as a regression signal on a controlled host.
