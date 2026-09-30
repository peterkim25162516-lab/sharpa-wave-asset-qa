# WaveSimParity Freeze B

Validation status: **VALID**
Scientific label: **INCONCLUSIVE**

This is an unofficial, simulation-only, kit-less OVPhysX native-input sensitivity study. It does not establish cross-engine parameter equivalence, an upstream bug, hardware behavior, or Sim2Real evidence.

## Corrigenda

- One earlier local collection attempt produced one raw MuJoCo run file but admitted 0/24 evidence cases because launcher and actual-worker process identities did not bind.
- That excluded attempt ran 0 OVPhysX cases and used no remote GPU.
- Its trajectory file was deserialized by the failed launcher for structural validation, but trajectory sample values were not reviewed or used for metrics, correction choices, or a scientific label.
- A later a2 campaign completed and automatically validated 8/8 local MuJoCo cases, then executed one completed OVPhysX worker case on a remote GPU; the post-worker payload admission subprocess failed before admitting any remote case because `-S` hid pinned NumPy.
- The a2 trajectory payloads may have been reviewed during diagnosis, but they were not used to select or design the one-flag admission fix, compute final metrics, or assign a scientific label.
- A later local-a3 operator invocation disclosed a noncanonical asset materialization. One worker entered the runner's canonical-LF asset-tree preflight and exited on the recorded hash-mismatch error before adapter construction, model creation, trajectory output, or any simulator step; it admitted 0 cases and used no remote GPU.
- All four prior evidence trees are immutable and excluded. The three earlier scientific-evidence trees and the operator-preflight-only local-a3 tree remain private; the corrected G execution reruns the full 24-case matrix under one source identity.
- After G collection, the finalizer deserialized the R1, formal, local, and all remote traces and completed structural, finite-value, and sanity checks before exposing an analysis-only backend-order assumption. Combined formal/repeatability/dt/sensitivity metrics had not run, no scientific label was assigned, and no final bundle was published.
- Runtime readback metadata and values were reviewed during diagnosis. The H correction follows the existing canonical-ID/backend-index mapping contract, not a trajectory effect or threshold: it validates the backend order and mapping inverse, then projects records and target readback into the frozen plan order without changing values or evidence bytes.
- Active local-a4 and remote-a4 evidence remains campaign-G evidence. H and I only analyze that immutable 24-case evidence; neither reruns either simulator.
- H produced one local, unpublished INVALID/null bundle after every evidence check passed but primary metric construction rejected sub-picosecond floating-accumulation timestamp differences. Its public summary and report therefore contradicted the private decision; that bundle is verified, retained as a private superseded subset, and is not reused as a final result.
- The I correction applies the already-frozen 1e-12 single-trace time semantics only as a same-step, same-length cross-trace compatibility check. It does not alter or round timestamps; window selection still uses the reference trace's original recorded times. Baseline and zero-vs-MuJoCo use 200 base/400 halved samples, treatment-vs-sham uses 201 base/401 halved samples, and frozen baselines are revalidated exactly.
- Runtime readback, trajectory values, primary metrics, and the diagnostic final S values and label were reviewed before I. The compatibility rule comes from the frozen preparer and existing timestamp contract, not from effects, thresholds, or a desired label.
- The hypothesis, intervention, scenario, case order, metrics, thresholds, claim boundary, and formal Gate 0 status remain unchanged.

## Evidence gates

- Corrected execution completed cases: 24/24
- Process isolation: 24/24 cases ran under distinct, scheme-bound process identities; identifiers remain private.
- Canonical mapping: 44/44 joints and 10/10 fingertip frames
- OVPhysX contacts were not measured; `contact_count=null` records that observation boundary and is not evidence of zero contacts or contact absence.
- OVPhysX timestep evidence verifies requested/configured dt and same-step, same-length recorded timestamp compatibility within 1e-12 s for comparisons only; the pinned kit-less stack exposes no compiled-runtime effective-dt getter.
- OVPhysX position-target binding: 16/16 cases verified; aggregate maximum absolute readback error was 0.0 rad against a 1e-06 rad limit.
- Every trace passed finite-value and preregistered sanity bounds for joints, targets, velocity, frame origins, and unit quaternions.
- Runtime friction readback: the sham matched frozen R1 exactly; 0/8 zero-treatment cases differed from R1. That count is descriptive and does not itself decide validity or the scientific label.
- Preregistered evidence checks passed: true
- Primary metric construction and evaluation passed: true
- Overall checks-and-metrics disposition passed: true

## Interpretation

The eight preregistered axes did not satisfy either all-axis decision boundary, so the result is INCONCLUSIVE.

The original formal Gate 0 result remains **DIVERGENT** with `pass_ready=false`; Freeze B does not rewrite it.
