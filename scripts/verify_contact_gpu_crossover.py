"""Read-only exact-inventory and raw-sample verification of DirectGPU crossover."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

PHASES = [('async', .002, .5, 1), ('sync', .001, .25, .5),
          ('sync', .0005, .25, .25), ('async', .0005, .25, 1),
          ('sync', .002, .25, 4), ('async', .002, .25, 1)]
SOURCE_HASH = '7c3811b4eeba085a87e98f3e20abfab5f07349043fd48d33b0ec54903b8e072f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(root):
    inventory = read(root/'inventory.json')
    expected = {'protocol.json', 'r1.log', 'r2.log'}
    for repeat in (1,2):
        expected.update(f'r{repeat}/{name}' for name in
                        ('fixture.usda','provenance.json','gpu_processes_private.json','summary.json'))
        expected.update(f'r{repeat}/phase{i}.json' for i in range(6))
    assert len(inventory) == len(expected)
    assert {e['path'] for e in inventory} == expected
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == expected|{'inventory.json'}
    for entry in inventory:
        path = root/entry['path']
        assert not path.is_symlink()
        assert digest(path) == entry['sha256'] and path.stat().st_size == entry['size']
    protocol = read(root/'protocol.json')
    assert protocol['script_sha256'] == SOURCE_HASH
    assert protocol['phases'] == [list(v) for v in PHASES]
    rows = []
    pids = []
    for repeat in (1,2):
        folder = root/f'r{repeat}'
        provenance = read(folder/'provenance.json')
        assert provenance['script_sha256'] == SOURCE_HASH
        assert provenance['api_sha256'] == 'cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960'
        pids.append(provenance['pid'])
        gpu = read(folder/'gpu_processes_private.json')
        assert gpu['pid'] == provenance['pid']
        assert any(line.split(',')[1].strip()==str(gpu['pid']) for line in gpu['query'].splitlines())
        summaries = read(folder/'summary.json')
        assert len(summaries) == 6
        for phase,(mode,dt,duration,predicted) in enumerate(PHASES):
            samples = read(folder/f'phase{phase}.json')
            assert len(samples) == round(duration/dt)
            for sample in samples:
                for key,value in sample.items():
                    assert all(math.isfinite(v) for v in (value if isinstance(value,list) else [value]))
            steady = samples[len(samples)//2:]
            z = [s['pose'][2] for s in steady]
            speed = max(math.sqrt(sum(v*v for v in s['velocity'][:3])) for s in steady)
            valid = all(abs(s['mass'][0]-1)<1e-5 for s in steady) and max(z)-min(z)<1e-4 and speed<1e-3 and .045<min(z)<=max(z)<.055
            ratios = {key:statistics.median(math.sqrt(sum(v*v for v in s[key])) for s in steady)/(steady[-1]['mass'][0]*9.81) for key in ('matrix','net')}
            supported = valid and all(abs(r/predicted-1)<.02 for r in ratios.values())
            summary = summaries[phase]
            assert summary['steady_valid'] == valid and summary['prediction_supported'] == supported
            assert all(math.isclose(summary['ratios'][key],value,rel_tol=1e-12,abs_tol=1e-12) for key,value in ratios.items())
            rows.append({'repeat':repeat,'phase':phase,'mode':mode,'dt_s':dt,'predicted_ratio':predicted,
                         'ratios':ratios,'steady_valid':valid,'prediction_supported':supported})
    assert len(set(pids)) == 2
    exact = all((root/f'r1/phase{i}.json').read_bytes()==(root/f'r2/phase{i}.json').read_bytes() for i in range(6))
    return {'inventory_sha256':digest(root/'inventory.json'),'payload_files':len(expected),
            'fresh_process_count':2,'raw_samples_exact_repeatability':exact,
            'all_predictions_supported':all(r['prediction_supported'] for r in rows),'rows':rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2,allow_nan=False))
