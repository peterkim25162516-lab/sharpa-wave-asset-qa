# Public snapshot provenance

This public repository starts from a privacy-sanitized snapshot of the completed local project.
The original experiment source commits and hashes recorded in the reports refer to the
retained private history, not to this publication commit. They are not rewritten or relabeled.

Actual remote account names, node names, endpoint addresses and GPU UUIDs have been replaced
consistently with example values in scripts, tests and historical documentation. These
substitutions change source hashes. Remote launchers are site-specific examples: configure
and review their account, root, node, GPU and scheduling contracts before using another site.
Public report artifacts and numerical conclusions remain byte-for-byte unchanged.

Raw traces, private manifests, SSH material, remote logs and the original Git history are
retained locally and are not included. The public reports retain their original provenance
hashes; access to the private evidence is required for independently verifying those bundles.
CPU installation and tests are reproducible from this checkout; remote campaigns require
the pinned dependencies and an explicitly configured compatible GPU environment.

The full test suite also requires the pinned public assets at `external/lfstd`.
On Linux, run `bash scripts/fetch_sharpa_assets.sh external/lfstd` before `pytest`.
On Windows, run `git config --local core.autocrlf false`, then
`./scripts/fetch_sharpa_assets.ps1 -Destination external/lfstd`.
Assets are fetched separately and remain ignored; CI performs this fetch automatically.

The original source snapshot was `6de3f6b84c6f808bca44e57381265c6861601f23`.
Initial GitHub publication date: 2026-09-30 (Asia/Shanghai).
This is an unofficial, simulation-only project.

## Publication verification

The sanitized checkout was tested on Windows with Python 3.12 on 2026-09-30:
`1007 passed, 3 skipped` (124.29 seconds). The three skipped tests require ignored
superseded analysis bundles or private Freeze B inputs. The retained original project
passed all 1010 tests with those inputs present. Public checkout verification does not
claim that those private-bundle tests were executed from a fresh public clone.

All 14 public JSON/Markdown reports match the canonical files from the original
commit byte-for-byte. The publication contains only tracked source, tests, documentation,
dependency metadata and those public reports. Remote identifiers were replaced consistently,
and the new public Git history uses the GitHub noreply author address.
