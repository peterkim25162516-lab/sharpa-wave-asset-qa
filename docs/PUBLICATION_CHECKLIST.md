# Publication checklist

This is the historical v0.1 planning checklist, retained for context. Unchecked items
are not claims that those checks have been performed. The public snapshot's actual
scope, provenance and reproducibility requirements are recorded in
[PUBLIC_SNAPSHOT.md](PUBLIC_SNAPSHOT.md); initial publication does not declare a new
tagged release or a passing cross-simulator result.

## Evidence and reproducibility

- [ ] Add the final public repository URL and actual release date to `CITATION.cff`.
- [ ] Run the full audit against the pinned upstream commit on Windows and Ubuntu.
- [ ] Save the generated JSON and Markdown reports under `results/v0.1/`.
- [ ] Record Python, NumPy, MuJoCo, OS, CPU and upstream commit versions.
- [ ] Run CPU unit tests in a clean environment and retain the test log.
- [ ] If MuJoCo checks are reported, verify every claimed model actually loaded.
- [ ] If Isaac/USD physics checks are reported, run them on a compatible remote Isaac Lab machine.

## Claim boundaries

- [ ] Use the labels `unofficial` and `simulation-only` prominently.
- [ ] Describe findings as version-specific validator observations, not hardware defects.
- [ ] Do not claim Sim2Real, physical-hardware validation or Sharpa endorsement.
- [ ] Do not call static USDA text checks a USD stage or PhysX validation.
- [ ] Keep all private-repository code, configuration, data, screenshots and metrics out of this repository.

## Licensing and privacy

- [ ] Keep this repository's `LICENSE` and `NOTICE` files.
- [ ] Do not vendor Sharpa assets unless their `LICENSE.txt` and `NOTICE.txt` are preserved.
- [ ] Do not use a Sharpa logo as the repository artwork.
- [ ] Scan the Git history for tokens, local paths, names of private repositories and large binaries.
- [ ] Confirm generated reports contain no username or absolute local filesystem path.

## Release quality

- [ ] README quick start works from a fresh clone.
- [ ] CI passes on Python 3.10 and 3.12.
- [ ] Known failures are documented and are not silently excluded.
- [ ] Add a 60–90 second terminal/demo video only after the output is reproducible.
- [ ] Tag `v0.1.0` only after the report and commit hashes are frozen.
