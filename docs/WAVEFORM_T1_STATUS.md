# WaveSimParity Trajectory T1 status

Trajectory T1 completed on 2026-08-30 (Asia/Shanghai). The evidence validation status is `VALID`
and the scientific comparison is `DIVERGENT`. Formal Gate 0 remains `DIVERGENT` with
`pass_ready=false`.

This is an unofficial, simulation-only comparison of MuJoCo and kit-less OVPhysX. It is not full
Isaac Sim, hardware validation, Sim2Real evidence, or an official bug finding.

## Scope

- Separate fixed-base left and right Sharpa Wave hands, 22 canonical joints per hand.
- Position control with the frozen explicit-PD contract.
- Offset sine and offset linear-chirp scenarios at base and halved timesteps.
- Two fresh-process repeats for every simulator, hand, scenario, and timestep cell.
- No ground plane or contact target; contacts, wrist/flange, dual-hand, and floating-base models
  remain outside T1.
- Headless kit-less OVPhysX only: no Kit application, renderer, camera, or RTX render path.

## Validity result

| Check | Result |
| --- | --- |
| Case completion | `32/32`, completion fraction `1.0` |
| Requested-step completion | `1.0` |
| Fresh-process isolation | `32/32` |
| Joint mapping | `44/44` |
| Distal-frame mapping | `10/10` |
| Non-finite runs | `0` |
| Repeatability maxima | joint `0 rad`, position `0 m`, orientation `0 rad` |
| dt-halving maxima | joint `0.0003000945 rad`, position `0.00004576161 m`, orientation `0.0008436642 rad` |
| OVPhysX effort clipping | `0` clips in `48,016` controller-state observations |
| Runtime-log gate | `0` fatal matches and `0` unclassified warning lines |

The frozen dt-halving limits are `0.01 rad`, `0.001 m`, and `0.02 rad` for joint,
distal-frame-position, and distal-frame-orientation deltas. All observed maxima are below those
limits.

## Scientific result

| Hand | Scenario | Joint max (rad) | Distal-frame position max (m) | Distal-frame orientation max (rad) | Result |
| --- | --- | ---: | ---: | ---: | --- |
| left | offset sine | 0.0198873629 | 0.00232437888 | 0.0483022033 | DIVERGENT |
| left | offset linear chirp | 0.0197837961 | 0.00232210176 | 0.0481766067 | DIVERGENT |
| right | offset sine | 0.0198749746 | 0.00232343702 | 0.0482773667 | DIVERGENT |
| right | offset linear chirp | 0.0197731586 | 0.00232091111 | 0.0481630828 | DIVERGENT |

The frozen cross-simulator limits are `0.02 rad`, `0.002 m`, and `0.05 rad`. Every joint and
orientation maximum stays within its corresponding limit. Every distal-frame-position maximum is
about `0.321–0.324 mm` above its limit, so all four cells are `DIVERGENT`.

These frames are distal-phalanx link origins, not physical fingertip contact-surface points. The
result establishes a reproducible difference under the pinned simulation setup; it does not
identify a dynamics cause or prove a defect in Sharpa, MuJoCo, NVIDIA software, or Isaac Sim.

## Reproducibility identity

| Field | Value |
| --- | --- |
| Source revision | `8af132ca02b425a923ea94200f5bc9533864806b` |
| Source tree | `f10542710dfa07dc767f36da6b25283881bb0876` |
| Source archive SHA-256 | `b972de9a2d33e66a96544e2bcb49de63e8d8b15ab783eae76aa3c79c626e0579` |
| Immutable remote snapshot SHA-256 | `9d622401a4aa2f1f8bc218a75cccc1b6b11fe1f7c4f7f1fe9aa92c128ffcf870` |
| T1 manifest semantic SHA-256 | `68b40b1a0f7bbbdd9860750f92d58445bb8aa16cb6682d39eb5f41e4ac7df85c` |
| Upstream asset revision | `6eea427eb24189519f32b9f21674cd534d3f973c` |
| Upstream asset Git tree | `bb00a9d5527b8a76de576ce876ebece67d8ffde1` |
| Canonical LF asset SHA-256 | `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad` |
| Private bundle inventory | `404` payload files, `755,944,170` bytes |
| Private bundle root SHA-256 | `cf00924cf075cd4813750280aabfe58de6a119e21c4b3b1cb9fe9f84357eb464` |
| Sanitized summary SHA-256 | `49b3834e6b30a82a82a398e8498add13395b999d9ad64ff4efa60fc7a5d8ba44` |
| Sanitized report SHA-256 | `d6591dd1c76f50dbb0a56c0ff79e7d9802c922e706b8996e3ba36a7f1ba70b27` |
| Local test suite | `721 passed` |

The remote matrix used one strictly idle NVIDIA A800-SXM4-40GB under driver `570.158.01` and
completed its 16 OVPhysX cases in 984 GPU-elapsed seconds. A100/A800 GPUs have no RT cores, and
this run did not attempt a full rendered Isaac Sim workflow.

## Evidence and publication boundary

Raw local and remote launcher evidence, logs, process identities, and the full verified bundle
remain in ignored local result directories. They are private audit evidence and are not eligible
for direct publication.

The publication-safe output is exactly two files:

- [machine-readable summary](../results/waveform-t1/summary.json)
- [human-readable report](../results/waveform-t1/report.md)

The checked copies are byte-identical to the two public members in the verified private bundle.
An independent sensitive-pattern scan found no local/remote path, account name, host, GPU UUID,
session identifier, or process identifier in these two files.

The full rerun and its correction history are recorded in
[WAVEFORM_T1_CORRIGENDUM.md](WAVEFORM_T1_CORRIGENDUM.md). Nothing has been pushed or submitted as
a pull request.
