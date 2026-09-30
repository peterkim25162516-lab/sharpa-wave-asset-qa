"""Supplementary read-only diagnostics, separate from the frozen C0 finalizer.

Run after strict finalization. JSON output contains hashes and scalar summaries,
never raw force samples or private host/process identifiers.
"""
import argparse
import json
import statistics
from pathlib import Path

from wave_asset_qa.contact.bundle import read_json_strict, sha256_file, write_json_exclusive

GEOMETRY = ('frame_poses', 'joint_positions', 'joint_velocities', 'position_targets',
            'probe_center_world_m', 'signed_gap_m', 'step', 'time_s')


def max_delta(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError('geometry keys differ')
        return max((max_delta(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('geometry lengths differ')
        return max((max_delta(x, y) for x, y in zip(a, b)), default=0.0)
    return abs(a - b)


def compare_case(before, after, *, force_diagnostic=False):
    if before['case'] != after['case'] or len(before['samples']) != len(after['samples']):
        raise ValueError('case or sample grid differs')
    deltas = dict.fromkeys(GEOMETRY, 0.0)
    ratios = []
    changed = 0
    for a, b in zip(before['samples'], after['samples']):
        for key in GEOMETRY:
            deltas[key] = max(deltas[key], max_delta(a[key], b[key]))
        changed += a['pair_active'] != b['pair_active']
        if force_diagnostic:
            x = a['native_contact_observation']['selected_pair_force_norm_n']
            y = b['native_contact_observation']['selected_pair_force_norm_n']
            if x > 1e-7:
                ratios.append(y / x)
    return {'sample_count': len(after['samples']), 'maximum_absolute_differences': deltas,
            'all_samples_exact': before['samples'] == after['samples'],
            'pair_active_changed_samples': changed,
            'force_ratio_sample_count': len(ratios),
            'candidate_over_original_force_median': statistics.median(ratios) if ratios else None}


def analyze(original, candidate, *, backend):
    original, candidate = Path(original), Path(candidate)
    files = sorted(candidate.glob('*/*.run.json'))
    if len(files) != 16 or {p.parent.name for p in files} != {p.parent.name for p in original.glob('*/*.run.json')}:
        raise ValueError('expected exactly 16 matching original/candidate cases')
    rows, repeats = [], []
    for path in files:
        prior = original / path.parent.name / path.name
        a, b = read_json_strict(prior), read_json_strict(path)
        if a['case']['simulator'] != backend or b['case']['simulator'] != backend:
            raise ValueError('backend mismatch')
        row = compare_case(a, b, force_diagnostic=backend == 'ovphysx')
        row.update(case_id=path.parent.name, original_run_sha256=sha256_file(prior),
                   candidate_run_sha256=sha256_file(path))
        rows.append(row)
        if path.parent.name.endswith('.r01'):
            repeat_id = path.parent.name[:-4] + '.r02'
            other = candidate / repeat_id / (repeat_id + '.run.json')
            repeats.append({'pair': path.parent.name[:-4],
                'all_samples_exact': b['samples'] == read_json_strict(other)['samples'],
                'first_sha256': sha256_file(path), 'second_sha256': sha256_file(other)})
    if len(repeats) != 8:
        raise ValueError('expected 8 fresh-repeat pairs')
    return {'cases': rows, 'fresh_repeat_pairs': repeats}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-mujoco', 'candidate-mujoco', 'original-ovphysx', 'candidate-ovphysx', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = {'schema_version': 1, 'scope': 'supplementary_diagnostic_not_a_gate',
        'formal_c0_replacement': False, 'analysis_script_sha256': sha256_file(__file__),
        'force_ratio_filter': 'original magnitude > 1e-7; diagnostic only, no threshold retuning',
        'mujoco': analyze(args.original_mujoco, args.candidate_mujoco, backend='mujoco'),
        'ovphysx': analyze(args.original_ovphysx, args.candidate_ovphysx, backend='ovphysx')}
    write_json_exclusive(args.output, result)
    print(json.dumps({'status': 'analyzed', 'sha256': sha256_file(args.output)}, sort_keys=True))
