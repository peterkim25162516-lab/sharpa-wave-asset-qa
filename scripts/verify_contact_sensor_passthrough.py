"""Independent, read-only verification of actual ContactSensor passthrough."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SOURCE_HASH='d027e3b326312767965f8669700f912edf6738e6e5fcfa8a6ebc6f47741c04dc'
PHASES=[('async',.002,.5,1),('sync',.001,.25,.5),('async',.001,.25,1)]


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, *, context_candidate=False):
    source_hash = ('3ad0534f8c0cdf4a6ee4e460f61dc2ac484a207678a38486d4cd149ee12daa67'
                   if context_candidate else SOURCE_HASH)
    intervention_hash = '8a4f8b091eb1112550f4b2a4b252644f5be5e18efe277b2e0fc6205943a34995'
    expected={'protocol.json','r1.log','r2.log'}
    for repeat in (1,2):
        expected.update(f'r{repeat}/{n}' for n in ('fixture.usda','provenance.json','gpu_private.json',
                                                   'phase0.json','phase1.json','phase2.json'))
    entries=read(root/'inventory.json')
    assert len(entries)==len(expected) and {e['path'] for e in entries}==expected
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected|{'inventory.json'}
    for entry in entries:
        p=root/entry['path']
        assert not p.is_symlink() and p.stat().st_size==entry['size'] and digest(p)==entry['sha256']
    protocol=read(root/'protocol.json')
    assert protocol['script_sha256']==source_hash and protocol['phases']==[list(p) for p in PHASES]
    if context_candidate:
        assert protocol['context_async_candidate'] is True and protocol['intervention_sha256']==intervention_hash
    rows=[]
    pids=[]
    for repeat in (1,2):
        folder=root/f'r{repeat}'
        provenance=read(folder/'provenance.json')
        assert provenance['script_sha256']==source_hash
        if context_candidate:
            assert provenance['context_async_candidate'] is True and provenance['intervention_sha256']==intervention_hash
        assert provenance['sensor_sha256']=='7fbc14a9f5d37aff3c267a51c14e1e8ad50aee8b03eecd2485c27d8f7f2b26bf'
        assert provenance['kernels_sha256']=='8d5d9625fe674df26d9111b4d2f6f287aba9e9eeccb0f6b9d98fa3517a214894'
        pids.append(provenance['pid'])
        gpu=read(folder/'gpu_private.json')
        assert gpu['pid']==provenance['pid'] and any(line.split(',')[1].strip()==str(gpu['pid']) for line in gpu['query'].splitlines())
        time=0.
        for phase,(mode,dt,duration,predicted) in enumerate(PHASES):
            if context_candidate:
                predicted = 1
            samples=read(folder/f'phase{phase}.json')
            assert len(samples)==round(duration/dt)
            for s in samples:
                time+=dt
                assert math.isfinite(s['time_s']) and abs(s['time_s']-time)<1e-10
                for key,length in [('pose',7),('velocity',6),('mass',1),('before',3),('sensor',3),('after',3)]:
                    assert len(s[key])==length and all(math.isfinite(v) for v in s[key])
                if context_candidate and mode == 'sync':
                    audit = s['step_audit']
                    assert audit['mode']=='diagnostic_async_wait_v1'
                    assert audit['calls']==audit['waits']==1 and audit['dt_s']==dt
                    assert math.isfinite(audit['sim_time_s'])
            exact=all(s['before']==s['sensor']==s['after'] for s in samples)
            steady=samples[len(samples)//2:]
            z=[s['pose'][2] for s in steady]
            speed=max(math.sqrt(sum(v*v for v in s['velocity'][:3])) for s in steady)
            valid=all(abs(s['mass'][0]-1)<1e-5 for s in steady) and speed<1e-3 and max(z)-min(z)<1e-4 and .045<min(z)<=max(z)<.055
            ratios={k:statistics.median(math.sqrt(sum(v*v for v in s[k])) for s in steady)/9.81 for k in ('before','sensor','after')}
            rows.append({'repeat':repeat,'phase':phase,'mode':mode,'dt_s':dt,'predicted_ratio':predicted,
                         'ratios':ratios,'exact_passthrough':exact,'steady_valid':valid,
                         'prediction_supported':valid and all(abs(r/predicted-1)<.02 for r in ratios.values())})
    assert len(set(pids))==2
    repeats=all((root/f'r1/phase{i}.json').read_bytes()==(root/f'r2/phase{i}.json').read_bytes() for i in range(3))
    return {'inventory_sha256':digest(root/'inventory.json'),'payload_files':len(entries),
            'exact_fresh_repeatability':repeats,'all_exact_passthrough':all(r['exact_passthrough'] for r in rows),
            'all_predictions_supported':all(r['prediction_supported'] for r in rows),'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--context-candidate',action='store_true')
    args = p.parse_args()
    print(json.dumps(verify(args.root,context_candidate=args.context_candidate),indent=2,allow_nan=False))
