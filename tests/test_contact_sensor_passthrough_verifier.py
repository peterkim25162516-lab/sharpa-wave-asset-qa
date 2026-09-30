import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('sensor_verifier',Path(__file__).parents[1]/'scripts/verify_contact_sensor_passthrough.py')
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write(path,value): path.write_text(json.dumps(value),encoding='utf-8')
def seal(root):
    write(root/'inventory.json',[{'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'sha256':v.digest(p)}
                                for p in sorted(root.rglob('*')) if p.is_file() and p.name!='inventory.json'])


@pytest.fixture
def evidence(tmp_path):
    write(tmp_path/'protocol.json',{'script_sha256':v.SOURCE_HASH,'phases':v.PHASES})
    for repeat in (1,2):
        folder=tmp_path/f'r{repeat}'
        folder.mkdir()
        (tmp_path/f'r{repeat}.log').write_text('fixture')
        (folder/'fixture.usda').write_text('fixture')
        write(folder/'provenance.json',{'script_sha256':v.SOURCE_HASH,'pid':repeat,
            'sensor_sha256':'7fbc14a9f5d37aff3c267a51c14e1e8ad50aee8b03eecd2485c27d8f7f2b26bf',
            'kernels_sha256':'8d5d9625fe674df26d9111b4d2f6f287aba9e9eeccb0f6b9d98fa3517a214894'})
        write(folder/'gpu_private.json',{'pid':repeat,'query':f'GPU-test, {repeat}, 1 MiB'})
        time=0.
        for phase,(_,dt,duration,predicted) in enumerate(v.PHASES):
            rows=[]
            for i in range(round(duration/dt)):
                time+=dt
                rows.append({'time_s':time,'mass':[1],'pose':[0,0,.05,0,0,0,1],'velocity':[0]*6,
                    'before':[0,0,9.81*predicted],'sensor':[0,0,9.81*predicted],'after':[0,0,9.81*predicted]})
            write(folder/f'phase{phase}.json',rows)
    seal(tmp_path)
    return tmp_path


def test_valid_passthrough(evidence):
    result=v.verify(evidence)
    assert result['payload_files']==15
    assert result['exact_fresh_repeatability'] and result['all_exact_passthrough']
    assert result['all_predictions_supported']


def test_tamper_rejected(evidence):
    (evidence/'r1/phase0.json').write_text('[]')
    with pytest.raises(AssertionError): v.verify(evidence)


def test_resealed_sensor_difference_not_hidden(evidence):
    path=evidence/'r1/phase1.json'
    rows=v.read(path)
    for row in rows: row['sensor'][2]*=2
    write(path,rows)
    seal(evidence)
    result=v.verify(evidence)
    assert not result['all_exact_passthrough']
    assert not result['all_predictions_supported']
    assert not result['exact_fresh_repeatability']


def test_extra_file_rejected(evidence):
    (evidence/'extra').write_text('extra')
    with pytest.raises(AssertionError): v.verify(evidence)


def make_candidate(root):
    source='3ad0534f8c0cdf4a6ee4e460f61dc2ac484a207678a38486d4cd149ee12daa67'
    helper='8a4f8b091eb1112550f4b2a4b252644f5be5e18efe277b2e0fc6205943a34995'
    for path in (root/'protocol.json',root/'r1/provenance.json',root/'r2/provenance.json'):
        value=v.read(path)
        value.update(script_sha256=source,context_async_candidate=True,intervention_sha256=helper)
        write(path,value)
    for repeat in (1,2):
        path=root/f'r{repeat}/phase1.json'
        rows=v.read(path)
        for index,row in enumerate(rows):
            for key in ('before','sensor','after'): row[key]=[0,0,9.81]
            row['step_audit']={'mode':'diagnostic_async_wait_v1','calls':1,'waits':1,'dt_s':.001,'sim_time_s':index*.001}
        write(path,rows)
    seal(root)


def test_candidate_requires_explicit_mode(evidence):
    make_candidate(evidence)
    with pytest.raises(AssertionError): v.verify(evidence)
    result=v.verify(evidence,context_candidate=True)
    assert result['all_predictions_supported'] and result['all_exact_passthrough']


def test_candidate_missing_wait_rejected(evidence):
    make_candidate(evidence)
    path=evidence/'r1/phase1.json'
    rows=v.read(path)
    rows[0]['step_audit']['waits']=0
    write(path,rows)
    seal(evidence)
    with pytest.raises(AssertionError): v.verify(evidence,context_candidate=True)
