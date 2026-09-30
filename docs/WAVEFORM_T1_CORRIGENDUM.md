# WaveSimParity Trajectory T1 correction history

This note records the excluded T1 attempts and prevents their evidence from being mixed with the
authoritative result. It is a provenance correction record, not a claim that either simulator or
the upstream assets contain a bug.

## Attempt a1: monitor-envelope incompatibility

The first full T1 collection used source revision
`ddec6aadbe4afba9fd73255a1e27dd6f0974755d`. Both the local and remote 16-case matrices completed,
but finalization stopped while parsing an empty GPU compute-process envelope.

The remote writer represented an empty process list with a blank line between begin/end markers,
while the online monitor ignored that blank line and the finalizer treated it as unclassifiable.
Source revision `7db7e64e2fe9eb0a7613f36f179c4629e960baa0` canonicalized the empty envelope and retained narrow
compatibility for the exact legacy representation.

No private/public T1 bundle was finalized from a1. Its scientific label is not assigned; the raw
evidence remains private and excluded.

## Attempt a2: session-consistency scope

The second full collection used source revision
`7db7e64e2fe9eb0a7613f36f179c4629e960baa0`. All 32 cases completed and the fixed GPU-monitor audit
passed, but the comparison code applied the legacy global-session rule to T1.

T1 deliberately launches MuJoCo locally and OVPhysX remotely, so the two backends have different
launcher sessions. The old rule rejected those two valid backend sessions, marked all comparison
cells inconclusive, and emitted empty metric maps. A downstream coverage audit then misreported
the empty maps as missing base/halved-dt coverage.

An independent read-only recomputation established that the cross-dt aggregation itself was
correct. Nevertheless, a2 was not retroactively finalized: its runs are bound to their original
source revision, so repaired analysis code cannot be presented as that revision's formal result.
No private/public T1 bundle was finalized from a2. Its scientific label is not assigned; the raw
evidence remains private and excluded.

## Authoritative a3

Source revision `8af132ca02b425a923ea94200f5bc9533864806b` scopes session consistency by backend for manifest
schema v2, while preserving the global single-session rule for schema v1 and the global
source-revision consistency check. It also makes the coverage audit handle an intentionally empty
`INCONCLUSIVE` metric map without disguising it as a numerical dt mismatch; non-inconclusive empty
or partial metric maps still fail closed.

The full matrix was rerun from scratch under this source identity: 16 fresh MuJoCo processes and
16 fresh kit-less OVPhysX processes. Only this a3 pair is active evidence. It finalized as
`VALID` / `DIVERGENT`, with formal Gate 0 unchanged at `pass_ready=false`.

## Evidence-selection rule

- Active evidence: a3 local and a3 remote only.
- Excluded evidence: a1 local/remote and a2 local/remote.
- No trajectory, run payload, log, or process record is copied from an excluded attempt into the
  authoritative bundle.
- Excluded evidence is retained unchanged in ignored private result directories for auditability.
- Only the sanitized a3 [summary](../results/waveform-t1/summary.json) and
  [report](../results/waveform-t1/report.md) are tracked publication candidates.

`DIVERGENT` is a bounded cross-simulator observation under the pinned protocol. It does not
automatically imply an official bug, hardware behavior, or Sim2Real validity.
