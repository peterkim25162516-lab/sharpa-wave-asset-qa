import copy
import pytest
from wave_asset_qa.contact.slurm_binding import UUID, validate_binding


def binding():
    return {'schema_version': 1, 'hostname': 'example-gpu-node-6', 'job_id': '38696',
        'job_record': 'JobId=38696 UserId=exampleuser(1013) JobState=RUNNING NodeList=example-gpu-node-6 '
                      'Partition=a800 NumNodes=1 NumTasks=1 TresPerNode=gres/shard:idx7:1',
        'physical_index': 7, 'query_index': 0, 'gpu_uuid': UUID,
        'inherited_cuda_visible_devices': '0', 'worker_cuda_visible_devices': '0'}


def test_binding_keeps_physical_and_query_indices_distinct():
    assert validate_binding(binding()) == binding()


@pytest.mark.parametrize('key,value', [
    ('query_index', 7), ('physical_index', 0), ('physical_index', True),
    ('gpu_uuid', 'GPU-other'), ('worker_cuda_visible_devices', '7'),
    ('inherited_cuda_visible_devices', '0,1'), ('hostname', 'other'), ('job_id', '0'),
])
def test_binding_rejects_identity_drift(key, value):
    data = binding()
    data[key] = value
    with pytest.raises(ValueError):
        validate_binding(data)


@pytest.mark.parametrize('old,new', [
    ('idx7', 'idx6'), ('RUNNING', 'PENDING'), ('exampleuser(1013)', 'other(1000)'),
    ('NumNodes=1', 'NumNodes=2'), ('JobId=38696', 'JobId=38697'),
    ('shard:idx7:1', 'shard:idx7:2'),
])
def test_binding_rejects_scheduler_mismatch(old, new):
    data = binding()
    data['job_record'] = data['job_record'].replace(old, new)
    with pytest.raises(ValueError):
        validate_binding(data)


def test_binding_rejects_missing_and_extra_fields():
    for key in binding():
        data = copy.deepcopy(binding())
        del data[key]
        with pytest.raises(ValueError):
            validate_binding(data)
    data = binding()
    data['unexpected'] = True
    with pytest.raises(ValueError):
        validate_binding(data)
