# Sharpa Wave Asset QA report

> Unofficial, simulation-only report. Results are specific to the pinned asset revision and software versions below.

## Summary

**Overall status:** `KNOWN`

| PASS | KNOWN | WARN | FAIL | SKIP |
| --- | --- | --- | --- | --- |
| 497 | 6 | 13 | 0 | 0 |

## Reproducibility

| Field | Value |
| --- | --- |
| asset_root | wave_01 |
| asset_tree_hash_algorithm | sha256(path_length \|\| relative_path \|\| file_size \|\| file_bytes) |
| asset_tree_matches_expected | True |
| asset_tree_sha256 | 9c2d71aec9f8fe77aeb03660f735eb55bc53b93ba595f6cbfe5cc4e8fbb67de4 |
| cpu_logical_count | 16 |
| cpu_model | AMD Ryzen 7 6800HS Creator Edition |
| fk_samples | 256 |
| fk_seed | 20260826 |
| mujoco | 3.12.0 |
| numpy | 2.5.2 |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.12.6 |
| run_at_utc | 2026-08-26T02:14:25+00:00 |
| simulation_steps | 50 |
| upstream_commit | 6eea427eb24189519f32b9f21674cd534d3f973c |
| upstream_commit_verified_from_git | True |
| upstream_git_dirty | False |
| upstream_repository | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml |
| validator_commit | 8d2d2282d208ff84e58abed6911ea5720eb7e69a |
| validator_dirty_excluding_results | False |
| validator_version | 0.1.0 |

## Inventory

| Format | Entry points |
| --- | --- |
| MJCF | 15 |
| URDF | 15 |
| USDA | 30 |

## Cross-format schema comparison

| Variant | Formats | Common joints | Missing | Limit gaps | Max limit delta (deg) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| dual_sharpa_wave | mjcf, urdf, usda | 62 | 0 | 36 | 0.001676 | warn |
| dual_sharpa_wave_with_flange | mjcf, urdf, usda | 62 | 0 | 36 | 0.001676 | warn |
| dual_sharpa_wave_with_wrist | mjcf, urdf, usda | 62 | 0 | 36 | 0.001676 | warn |
| left_sharpa_wave | mjcf, urdf, usda | 22 | 0 | 0 | 0.001676 | pass |
| left_sharpa_wave_with_flange | mjcf, urdf, usda | 22 | 0 | 0 | 0.001676 | pass |
| left_sharpa_wave_with_float_base | mjcf, urdf, usda | 28 | 0 | 12 | 0.001676 | warn |
| left_sharpa_wave_with_float_base_with_flange | mjcf, urdf, usda | 28 | 0 | 12 | 0.001676 | warn |
| left_sharpa_wave_with_float_base_with_wrist | mjcf, urdf, usda | 28 | 0 | 12 | 0.001676 | warn |
| left_sharpa_wave_with_wrist | mjcf, urdf, usda | 22 | 0 | 0 | 0.001676 | pass |
| right_sharpa_wave | mjcf, urdf, usda | 22 | 0 | 0 | 0.001676 | pass |
| right_sharpa_wave_with_flange | mjcf, urdf, usda | 22 | 0 | 0 | 0.001676 | pass |
| right_sharpa_wave_with_float_base | mjcf, urdf, usda | 28 | 0 | 12 | 0.001676 | warn |
| right_sharpa_wave_with_float_base_with_flange | mjcf, urdf, usda | 28 | 0 | 12 | 0.001676 | warn |
| right_sharpa_wave_with_float_base_with_wrist | mjcf, urdf, usda | 28 | 0 | 12 | 0.001676 | warn |
| right_sharpa_wave_with_wrist | mjcf, urdf, usda | 22 | 0 | 0 | 0.001676 | pass |

## Sampled URDF ↔ MJCF forward kinematics

| Variant | Samples | Joints | Position p95 (mm) | Position max (mm) | Rotation max (deg) | QA status |
| --- | --- | --- | --- | --- | --- | --- |
| left_sharpa_wave | 256 | 22 | 5.183e-05 | 7.073e-05 | 8.585e-05 | pass |
| left_sharpa_wave_with_flange | 256 | 22 | 29.5 | 29.5 | 8.585e-05 | warn |
| left_sharpa_wave_with_wrist | 256 | 22 | 29 | 29 | 8.585e-05 | warn |
| right_sharpa_wave | 256 | 22 | 5.183e-05 | 7.073e-05 | 8.585e-05 | pass |
| right_sharpa_wave_with_flange | 256 | 22 | 29.5 | 29.5 | 8.585e-05 | warn |
| right_sharpa_wave_with_wrist | 256 | 22 | 29 | 29 | 8.585e-05 | warn |

## MuJoCo headless smoke checks

| Model | Status | joints | actuators | RTF | Repeat delta | Message |
| --- | --- | --- | --- | --- | --- | --- |
| dual_sharpa_wave/dual_sharpa_wave.xml | known | — | — | — | — | ValueError: Error: Error opening file 'left_sharpa_wave/meshes/left_hand_C_MC_visual_.STL' |
| dual_sharpa_wave/dual_sharpa_wave_with_flange.xml | known | — | — | — | — | ValueError: Error: Error opening file 'left_sharpa_wave/meshes/flange_B.STL' |
| dual_sharpa_wave/dual_sharpa_wave_with_wrist.xml | known | — | — | — | — | ValueError: Error: Error opening file 'left_sharpa_wave/meshes/wrist_collision.STL' |
| left_sharpa_wave/left_sharpa_wave.xml | pass | 22 | 22 | 40.58 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| left_sharpa_wave/left_sharpa_wave_with_flange.xml | pass | 22 | 22 | 47.9 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| left_sharpa_wave/left_sharpa_wave_with_wrist.xml | pass | 22 | 22 | 39.1 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| right_sharpa_wave/right_sharpa_wave.xml | pass | 22 | 22 | 47.26 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| right_sharpa_wave/right_sharpa_wave_with_flange.xml | pass | 22 | 22 | 39.65 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| right_sharpa_wave/right_sharpa_wave_with_wrist.xml | pass | 22 | 22 | 40.3 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| sharpa_wave_float_base_urdf_usd/left_sharpa_wave_with_float_base/left_sharpa_wave_with_float_base.xml | pass | 28 | 28 | 35.36 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| sharpa_wave_float_base_urdf_usd/left_sharpa_wave_with_float_base/left_sharpa_wave_with_float_base_with_flange.xml | pass | 28 | 28 | 32.88 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| sharpa_wave_float_base_urdf_usd/left_sharpa_wave_with_float_base/left_sharpa_wave_with_float_base_with_wrist.xml | pass | 28 | 28 | 36.03 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| sharpa_wave_float_base_urdf_usd/right_sharpa_wave_with_float_base/right_sharpa_wave_with_float_base.xml | pass | 28 | 28 | 29.36 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| sharpa_wave_float_base_urdf_usd/right_sharpa_wave_with_float_base/right_sharpa_wave_with_float_base_with_flange.xml | pass | 28 | 28 | 24.83 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |
| sharpa_wave_float_base_urdf_usd/right_sharpa_wave_with_float_base/right_sharpa_wave_with_float_base_with_wrist.xml | pass | 28 | 28 | 26.12 | 0 | Model loaded and repeated headless rollouts remained finite and within the repeat tolerance. |

## Findings

| Status | Code | Model | Message | Reference |
| --- | --- | --- | --- | --- |
| known | MESH_REFERENCE_MISSING | dual_sharpa_wave/dual_sharpa_wave.xml | 43 unique mesh reference(s) are missing; examples: left_sharpa_wave/meshes/DP_HB1_4F.STL, left_sharpa_wave/meshes/DP_HB1_TH.STL, left_sharpa_wave/meshes/DP_Visual_HB1_TH.STL | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1 |
| known | MESH_REFERENCE_MISSING | dual_sharpa_wave/dual_sharpa_wave_with_flange.xml | 49 unique mesh reference(s) are missing; examples: left_sharpa_wave/meshes/DP_HB1_4F.STL, left_sharpa_wave/meshes/DP_HB1_TH.STL, left_sharpa_wave/meshes/DP_Visual_HB1_TH.STL | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1 |
| known | MESH_REFERENCE_MISSING | dual_sharpa_wave/dual_sharpa_wave_with_wrist.xml | 47 unique mesh reference(s) are missing; examples: left_sharpa_wave/meshes/DP_HB1_4F.STL, left_sharpa_wave/meshes/DP_HB1_TH.STL, left_sharpa_wave/meshes/DP_Visual_HB1_TH.STL | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1 |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | dual_sharpa_wave | Authored limits match within tolerance, but 36 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | dual_sharpa_wave_with_flange | Authored limits match within tolerance, but 36 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | dual_sharpa_wave_with_wrist | Authored limits match within tolerance, but 36 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | left_sharpa_wave_with_float_base | Authored limits match within tolerance, but 12 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | left_sharpa_wave_with_float_base_with_flange | Authored limits match within tolerance, but 12 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | left_sharpa_wave_with_float_base_with_wrist | Authored limits match within tolerance, but 12 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | right_sharpa_wave_with_float_base | Authored limits match within tolerance, but 12 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | right_sharpa_wave_with_float_base_with_flange | Authored limits match within tolerance, but 12 format/joint pair(s) inherit or omit limits. | — |
| warn | JOINT_LIMIT_PARITY_INCOMPLETE | right_sharpa_wave_with_float_base_with_wrist | Authored limits match within tolerance, but 12 format/joint pair(s) inherit or omit limits. | — |
| warn | FK_PARITY_DELTA | left_sharpa_wave_with_flange | Sampled distal-frame FK exceeds the reporting threshold after one fixed base alignment. | — |
| warn | FK_PARITY_DELTA | left_sharpa_wave_with_wrist | Sampled distal-frame FK exceeds the reporting threshold after one fixed base alignment. | — |
| warn | FK_PARITY_DELTA | right_sharpa_wave_with_flange | Sampled distal-frame FK exceeds the reporting threshold after one fixed base alignment. | — |
| warn | FK_PARITY_DELTA | right_sharpa_wave_with_wrist | Sampled distal-frame FK exceeds the reporting threshold after one fixed base alignment. | — |
| known | MUJOCO_LOAD_FAILED | dual_sharpa_wave/dual_sharpa_wave.xml | ValueError: Error: Error opening file 'left_sharpa_wave/meshes/left_hand_C_MC_visual_.STL' | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1 |
| known | MUJOCO_LOAD_FAILED | dual_sharpa_wave/dual_sharpa_wave_with_flange.xml | ValueError: Error: Error opening file 'left_sharpa_wave/meshes/flange_B.STL' | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1 |
| known | MUJOCO_LOAD_FAILED | dual_sharpa_wave/dual_sharpa_wave_with_wrist.xml | ValueError: Error: Error opening file 'left_sharpa_wave/meshes/wrist_collision.STL' | https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/pull/1 |

## Scope limits

- USDA inspection is text-level overlay validation, not USD stage or PhysX validation.
- MuJoCo smoke checks validate loading, finite stepping and repeatability only.
- This report does not establish controller quality, hardware behavior, Sim2Real or safety.
