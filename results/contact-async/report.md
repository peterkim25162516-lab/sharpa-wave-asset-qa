# Experimental async + wait C0 comparison

This is a separately collected intervention campaign, not the original C0. The original C0 VALID / INCONCLUSIVE result remains unchanged.

# WaveSimParity Contacts Gate C0

> Synthetic sphere/box contact-versus-sham simulation evidence only.

## Result

**Evidence:** `VALID`  
**Science:** `DIVERGENT`  
**C0 pass-ready:** `false`

| Run coverage | Count |
| --- | --- |
| Expected | 32 |
| Received | 32 |
| Completed | 32 |

Mapping: **44/44 joints**, **10/10 distal frames**.

## Gate checks

| Stage | Check | Status | Observed | Criterion |
| --- | --- | --- | --- | --- |
| admission | initial_gap_min_m | PASS | 0.06439997196181785 | >= 0.05 in every run |
| admission | recovery_gap_min_m | PASS | 0.06438678731888897 | >= 0.05 in every run |
| admission | sham_hold_mean_gap_max_m | PASS | -0.009811847623301494 | <= -0.005 in every sham run |
| admission | condition_signal_semantics | PASS | {'sham_never_active': True, 'recovery_inactive': True} | sham and recovery stay inactive |
| repeatability | joint_or_effect_max_abs_rad | PASS | 0.0 | <= 1e-09 |
| repeatability | probe_gap_or_blocked_max_abs_m | PASS | 0.0 | <= 1e-09 |
| repeatability | orientation_or_effect_max_abs_rad | PASS | 0.0 | <= 1e-09 |
| repeatability | pair_active_exact | PASS | True | must be exactly true for complete N+1 bitsets |
| repeatability | event_interval_exact | PASS | True | must be exactly true |
| dt_halving | joint_or_effect_max_abs_rad | PASS | 0.0012479424476623535 | <= 0.01 |
| dt_halving | probe_gap_or_blocked_max_abs_m | PASS | 1.1175583435552028e-05 | <= 0.001 |
| dt_halving | orientation_or_effect_max_abs_rad | PASS | 0.0005679039167269905 | <= 0.02 |
| dt_halving | hold_duty_max_abs | PASS | 0.0 | <= 0.01 |
| dt_halving | event_midpoint_max_abs_s | PASS | 0.0015000000000000568 | <= 0.004 and must be available |
| admission | selected_pair_exercised | PASS | {'mujoco': True, 'ovphysx': True} | at least one backend must observe the selected pair |
| admission | contact_hold_duty_min | PASS | 1.0 | >= 0.95 in every contact run |
| admission | contact_events_complete | PASS | True | every contact run has debounced onset and release |
| cross_simulator | event_midpoint_max_abs_s | FAIL | 0.05600000000000005 | <= 0.01 |
| cross_simulator | hold_duty_max_abs | PASS | 0.0 | <= 0.02 |
| cross_simulator | onset_gap_max_abs_m | PASS | 0.00013751424955601999 | <= 0.001 |
| cross_simulator | steady_hold_mean_gap_max_abs_m | PASS | 0.0002571112189793253 | <= 0.001 |
| cross_simulator | joint_effect_max_abs_rad | FAIL | 0.03430622877733036 | <= 0.02 |
| cross_simulator | blocked_travel_max_abs_m | FAIL | 0.002580752207712478 | <= 0.002 |
| cross_simulator | orientation_effect_max_abs_rad | FAIL | 0.050860444620017334 | <= 0.05 |

## Scientific classification reasons

- `cross_simulator_threshold_exceeded:event_midpoint_max_abs_s`
- `cross_simulator_threshold_exceeded:joint_effect_max_abs_rad`
- `cross_simulator_threshold_exceeded:blocked_travel_max_abs_m`
- `cross_simulator_threshold_exceeded:orientation_effect_max_abs_rad`

## Scope boundary

- `hardware_in_scope = false`
- `native_hand_collision_geometry_validated = false`
- `floating_base_in_scope = false`
- `sim2real_claimed = false`
- Native pair-force magnitudes are descriptive private evidence and are not compared across engines.
- A `DIVERGENT` result is a reproducible threshold exceedance under this pinned protocol, not an official simulator or asset defect finding.

## Relationship to the existing Gate 0

Contacts C0 is additive. The existing Trajectory T1 result remains `DIVERGENT` with `pass_ready=false`; C0 does not reinterpret it.
