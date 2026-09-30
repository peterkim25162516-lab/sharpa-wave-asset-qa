import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('async_analysis', Path(__file__).parents[1]/'scripts/analyze_contact_async_result.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    sample = {key: 0.0 for key in analysis.GEOMETRY}
    sample['frame_poses'] = {'frame': [0.0, 1.0]}
    sample['pair_active'] = False
    sample['native_contact_observation'] = {'selected_pair_force_norm_n': 0.002}
    return {'case': {'id': 'test'}, 'samples': [sample]}


def test_observation_change_is_not_geometry_change():
    a=fixture(); b=copy.deepcopy(a)
    b['samples'][0]['pair_active']=True
    b['samples'][0]['native_contact_observation']['selected_pair_force_norm_n']=1.0
    result=analysis.compare_case(a,b,force_diagnostic=True)
    assert all(v==0 for v in result['maximum_absolute_differences'].values())
    assert result['pair_active_changed_samples']==1
    assert result['candidate_over_original_force_median']==500
    assert not result['all_samples_exact']


def test_geometry_difference_is_detected():
    a=fixture(); b=copy.deepcopy(a)
    b['samples'][0]['frame_poses']['frame'][0]=0.5
    assert analysis.compare_case(a,b)['maximum_absolute_differences']['frame_poses']==0.5


def test_mismatched_cases_are_rejected():
    a=fixture(); b=copy.deepcopy(a); b['case']['id']='other'
    with pytest.raises(ValueError):
        analysis.compare_case(a,b)
