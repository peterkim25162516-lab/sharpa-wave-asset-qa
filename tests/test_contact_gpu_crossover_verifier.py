import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('gpu_verifier',Path(__file__).parents[1]/'scripts/verify_contact_gpu_crossover.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def seal(root):
    write(root/'inventory.json',[{'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'sha256':v.digest(p)}
                                for p in sorted(root.rglob('*')) if p.is_file() and p.name!='inventory.json'])


@pytest.fixture
def evidence(tmp_path):
    write(tmp_path/'protocol.json',{'script_sha256':v.SOURCE_HASH,'phases':v.PHASES})
    for repeat in (1,2):
        folder = tmp_path/f'r{repeat}'
        folder.mkdir()
        (tmp_path/f'r{repeat}.log').write_text('test fixture')
        (folder/'fixture.usda').write_text('test fixture')
        write(folder/'provenance.json',{'pid':repeat,'script_sha256':v.SOURCE_HASH,
              'api_sha256':'cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960'})
        write(folder/'gpu_processes_private.json',{'pid':repeat,'query':f'GPU-test, {repeat}, 1 MiB'})
        summaries=[]
        for phase,(mode,dt,duration,predicted) in enumerate(v.PHASES):
            sample={'mass':[1.],'pose':[0.,0.,.05,0.,0.,0.,1.],'velocity':[0.]*6,
                    'matrix':[0.,0.,9.81*predicted],'net':[0.,0.,9.81*predicted],'time_s':0.}
            write(folder/f'phase{phase}.json',[sample]*round(duration/dt))
            summaries.append({'steady_valid':True,'prediction_supported':True,
                              'ratios':{'matrix':predicted,'net':predicted}})
        write(folder/'summary.json',summaries)
    seal(tmp_path)
    return tmp_path


def test_exact_valid_inventory(evidence):
    result=v.verify(evidence)
    assert result['payload_files']==23
    assert result['raw_samples_exact_repeatability']
    assert result['all_predictions_supported']


def test_tampering_rejected(evidence):
    (evidence/'r1/phase0.json').write_text('[]')
    with pytest.raises(AssertionError):
        v.verify(evidence)


def test_unexpected_file_rejected(evidence):
    (evidence/'extra').write_text('unexpected')
    with pytest.raises(AssertionError):
        v.verify(evidence)


def test_forged_summary_rejected_even_with_new_hash(evidence):
    path=evidence/'r1/summary.json'
    data=v.read(path)
    data[0]['ratios']['matrix']=2.
    write(path,data)
    seal(evidence)
    with pytest.raises(AssertionError):
        v.verify(evidence)
