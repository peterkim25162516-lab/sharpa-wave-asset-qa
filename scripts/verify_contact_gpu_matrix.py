"""Recompute the native GPU load matrix from hashed raw samples (read-only)."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

SOURCE_HASH='eb9a485969be51e02bbaf7923cdde2fdc68d8ed08a80e2bc103f8eaaaa85f051'
CASES=list(itertools.product((1,2),(.002,.001,.0005),('sync','async'),(1,2)))


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def name(case):
    mass,dt,mode,repeat=case
    return f'm{mass}-dt{dt}-{mode}-r{repeat}'


def verify(root):
    expected={'protocol.json'}
    for case in CASES:
        expected.add(name(case)+'.log')
        expected.update(name(case)+'/'+p for p in ('fixture.usda','provenance.json','samples.json','gpu_private.json','summary.json'))
    inventory=read(root/'inventory.json')
    assert len(inventory)==len(expected) and {e['path'] for e in inventory}==expected
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected|{'inventory.json'}
    for entry in inventory:
        path=root/entry['path']
        assert not path.is_symlink() and path.stat().st_size==entry['size'] and digest(path)==entry['sha256']
    protocol=read(root/'protocol.json')
    assert protocol['script_sha256']==SOURCE_HASH and protocol['cases']==[list(c) for c in CASES]
    rows=[]
    pids=[]
    for case in CASES:
        mass,dt,mode,repeat=case
        folder=root/name(case)
        prov=read(folder/'provenance.json')
        assert prov['script_sha256']==SOURCE_HASH and prov['api_sha256']=='cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960'
        assert (prov['mass_kg'],prov['dt_s'],prov['mode'])==(mass,dt,mode)
        pids.append(prov['pid'])
        gpu=read(folder/'gpu_private.json')
        assert gpu['pid']==prov['pid'] and any(line.split(',')[1].strip()==str(gpu['pid']) for line in gpu['query'].splitlines())
        samples=read(folder/'samples.json')
        assert len(samples)==round(1/dt)
        for i,s in enumerate(samples):
            assert math.isclose(s['time_s'],(i+1)*dt,abs_tol=1e-12)
            for key,length in [('pose',7),('velocity',6),('mass',1),('matrix',3),('net',3)]:
                assert len(s[key])==length and all(math.isfinite(x) for x in s[key])
        steady=[s for s in samples if s['time_s']>=.5]
        z=[s['pose'][2] for s in steady]
        speed=max(math.sqrt(sum(x*x for x in s['velocity'][:3])) for s in steady)
        valid=all(abs(s['mass'][0]-mass)<1e-5 for s in steady) and speed<1e-3 and max(z)-min(z)<1e-4 and .045<min(z)<=max(z)<.055
        ratios={key:statistics.median(math.sqrt(sum(x*x for x in s[key])) for s in steady)/(mass*9.81) for key in ('matrix','net')}
        summary=read(folder/'summary.json')
        assert summary['steady_valid']==valid
        assert all(math.isclose(summary['ratios'][key],r,rel_tol=1e-12,abs_tol=1e-12) for key,r in ratios.items())
        force_supported=valid and all(abs(r-1)<.02 for r in ratios.values())
        assert summary['async_force_supported']==(force_supported if mode=='async' else None)
        rows.append({'case':name(case),'mass_kg':mass,'dt_s':dt,'mode':mode,'repeat':repeat,
            'steady_valid':valid,'ratios':ratios,'async_force_supported':force_supported if mode=='async' else None})
    assert len(set(pids))==24
    exact=all((root/name((m,d,mode,1))/'samples.json').read_bytes()==(root/name((m,d,mode,2))/'samples.json').read_bytes()
              for m,d,mode in itertools.product((1,2),(.002,.001,.0005),('sync','async')))
    return {'inventory_sha256':digest(root/'inventory.json'),'payload_files':len(expected),'case_count':len(rows),
        'all_steady_valid':all(r['steady_valid'] for r in rows),'raw_samples_exact_repeatability':exact,
        'all_async_force_supported':all(r['async_force_supported'] for r in rows if r['mode']=='async'),'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2,allow_nan=False))
