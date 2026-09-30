import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('matrix_verifier',Path(__file__).parents[1]/'scripts/verify_contact_gpu_matrix.py')
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write(path,value): path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def evidence(tmp_path):
    write(tmp_path/'protocol.json',{'script_sha256':v.SOURCE_HASH,'cases':v.CASES})
    for pid,case in enumerate(v.CASES,1):
        mass,dt,mode,repeat=case
        folder=tmp_path/v.name(case)
        folder.mkdir()
        (tmp_path/(v.name(case)+'.log')).write_text('fixture')
        (folder/'fixture.usda').write_text('fixture')
        write(folder/'provenance.json',{'script_sha256':v.SOURCE_HASH,'pid':pid,'mass_kg':mass,'dt_s':dt,'mode':mode,
             'api_sha256':'cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960'})
        write(folder/'gpu_private.json',{'pid':pid,'query':f'GPU-test, {pid}, 1 MiB'})
        ratio=1 if mode=='async' else dt
        write(folder/'samples.json',[{'mass':[mass],'pose':[0,0,.05,0,0,0,1],'velocity':[0]*6,
             'matrix':[0,0,mass*9.81*ratio],'net':[0,0,mass*9.81*ratio],'time_s':(i+1)*dt} for i in range(round(1/dt))])
        write(folder/'summary.json',{'steady_valid':True,'ratios':{'matrix':ratio,'net':ratio},
             'async_force_supported':True if mode=='async' else None})
    write(tmp_path/'inventory.json',[{'path':p.relative_to(tmp_path).as_posix(),'size':p.stat().st_size,'sha256':v.digest(p)}
                                    for p in sorted(tmp_path.rglob('*')) if p.is_file()])
    return tmp_path


def test_matrix_recompute(evidence):
    result=v.verify(evidence)
    assert result['case_count']==24 and result['payload_files']==145
    assert result['all_steady_valid'] and result['all_async_force_supported']
    assert result['raw_samples_exact_repeatability']


def test_matrix_tampering(evidence):
    (evidence/'m1-dt0.002-sync-r1/samples.json').write_text('[]')
    with pytest.raises(AssertionError): v.verify(evidence)
