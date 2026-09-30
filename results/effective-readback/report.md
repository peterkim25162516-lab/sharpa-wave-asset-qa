# OVPhysX Effective-Parameter Readback (Freeze A)

- Status: `READBACK_VALID`
- Scope: descriptive, kit-less OVPhysX, fixed-base Wave hands, pre-trace readback only
- Coverage: `44/44` canonical left/right joints
- Fresh processes: `4` (two per hand)
- Exact repeatability: mapping=`true`, solver evidence records=`true`, vectors=`true`
- Experimental trace steps/commands/targets/samples: `0 / 0 / 0 / 0`
- Adapter initialization/reset zero-state and zero-target writes: disclosed and performed
- Backend-internal warmup/advance: disclosed separately; exact internal advance is not exposed or inferred

## Conservative semantic boundary

- `friction`: **UNMAPPABLE** — Pinned wrapper and authoritative API labels conflict for non-viscous slots; numeric equality or closeness cannot establish MuJoCo equivalence.
- `armature`: **UNMAPPABLE** — This descriptive OVPhysX-only readback does not by itself verify every cross-backend physical/action-law requirement for DIRECT_TRANSFERABLE.
- `control_drive`: **UNMAPPABLE** — Same names or numbers do not prove equal controller, transmission, limit, or passive-force laws across engines.
- `solver_runtime`: **UNMAPPABLE** — Incomplete or ambiguous solver evidence takes precedence and is UNMAPPABLE; only fully available solver evidence could remain SENSITIVITY_ONLY, never an equivalent parameter.

## Frozen identities

- Source revision/tree: `44571ea60201f77ca93d49c9c14cb2d28152c30f` / `d2a03e8ecceb46971e5c75debdb738baea63a4a3`
- Freeze A config SHA-256: `12c89fd8e0cd3bfa849a96a2e133b22e9191fc0ca6188f18fde4960d4221ab4c`
- Gate 0 manifest semantic SHA-256: `576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e`
- Canonical asset tree SHA-256: `b198e39b1030f279bf531c01d7e2321339dccf4482e662e6e8c671c9499d54ad`

## Claim boundary

This is descriptive configuration evidence only. It does not identify the cause of a trajectory difference, establish cross-engine parameter equivalence, infer realized torque, or support a backend/upstream bug claim.

Formal Gate 0 remains **DIVERGENT** with `pass_ready=false`. Freeze B is still required before any trajectory intervention.

The exact raw records and private machine/process identifiers remain in the private content-hashed bundle; this report intentionally omits them.
