from copy import deepcopy
from types import SimpleNamespace

import pytest

from wave_asset_qa.contact import async_campaign as c
from wave_asset_qa.contact.adapters import ContactAdapterError, OVPhysXContactAdapter
from wave_asset_qa.contact.bundle import ContactEvidenceError


def rows():
    return [{'calls':1,'waits':1,'dt_s':.002,'sim_time_s':i*.002,'mode':c.MODE} for i in range(3)]


def private():
    audit=rows()
    return {'async_step_audit':audit,'runtime_fingerprint':{'stepping_intervention':{
        'campaign_id':c.CAMPAIGN_ID,'mode':c.MODE,'helper_sha256':c.helper_sha256(),
        'completed_steps':3,'audit_sha256':c.validate_step_audit(audit,expected_steps=3,dt_s=.002)}}}


def test_candidate_evidence():
    run=SimpleNamespace(execution=SimpleNamespace(completed_steps=3))
    assert c.validate_candidate_evidence(private(),run,dt_s=.002)['completed_steps']==3


@pytest.mark.parametrize('mutation',['count','wait','dt','time','extra','bool','nan'])
def test_bad_audits(mutation):
    audit=rows()
    if mutation=='count': audit.pop()
    if mutation=='wait': audit[0]['waits']=0
    if mutation=='dt': audit[0]['dt_s']=.001
    if mutation=='time': audit[1]['sim_time_s']=.03
    if mutation=='extra': audit[0]['extra']=1
    if mutation=='bool': audit[0]['calls']=True
    if mutation=='nan': audit[0]['sim_time_s']=float('nan')
    with pytest.raises(ContactAdapterError): c.validate_step_audit(audit,expected_steps=3,dt_s=.002)


@pytest.mark.parametrize('field,value',[('campaign_id','old-c0'),('helper_sha256','0'*64),('audit_sha256','0'*64),('completed_steps',4)])
def test_identity_mismatch(field,value):
    evidence=private()
    evidence['runtime_fingerprint']['stepping_intervention'][field]=value
    with pytest.raises(ContactAdapterError):
        c.validate_candidate_evidence(evidence,SimpleNamespace(execution=SimpleNamespace(completed_steps=3)),dt_s=.002)


def test_bridge_preserves_write_context_update():
    bridge=c.AsyncCandidateBridge()
    events=[]
    class SDK:
        def step_sync(self,**kwargs): raise AssertionError('sync not allowed')
        def step(self,**kwargs): events.append('async'); return 10
        def wait_op(self,op): assert op==10; events.append('wait')
    sdk=SDK()
    bridge._native_sdk=lambda:sdk
    bridge._dt_s=.002
    bridge._articulation=SimpleNamespace(write_data_to_sim=lambda:events.append('write'),
                                         update=lambda dt:events.append(('update',dt)))
    def context_step(*,render):
        assert render is False
        events.append('context-before')
        sdk.step_sync(dt=.002,sim_time=0)
        events.append('context-after')
    bridge._simulation=SimpleNamespace(step=context_step)
    bridge.step(dt_s=.002)
    assert events==['write','context-before','async','wait','context-after',('update',.002)]
    assert len(bridge.step_audit)==1 and 'step_sync' not in vars(sdk)


@pytest.mark.parametrize('field,value',[('kitless',False),('renderer',True),('camera',True),('camera_prim_paths',['/Camera']),('created_sensor_types',[])])
def test_candidate_keeps_runtime_safety(monkeypatch,field,value):
    fingerprint={'kitless':True,'renderer':False,'camera':False,'camera_prim_paths':[],'created_sensor_types':['ContactSensor']}
    fingerprint[field]=value
    monkeypatch.setattr(OVPhysXContactAdapter,'_runtime_fingerprint_snapshot',lambda *a,**kw:deepcopy(fingerprint))
    with pytest.raises(ContactAdapterError):
        c.AsyncCandidateContactAdapter()._runtime_fingerprint_snapshot(bridge=None,phase='test')


def test_default_not_opted_in():
    assert OVPhysXContactAdapter()._bridge_factory is None
    with pytest.raises(ContactAdapterError):
        c.AsyncCandidateContactAdapter().open_contact_case(device='cpu')


@pytest.mark.parametrize('kind',['missing','legacy','extra'])
def test_candidate_case_rejects_missing_or_wrong_envelope(tmp_path,kind):
    import json
    if kind != 'missing':
        value={'schema_version':1,'experimental_campaign':'old-c0'}
        if kind=='extra': value['unrecognized']=True
        (tmp_path/'private-campaign.json').write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises((ContactAdapterError, ContactEvidenceError)):
        c.load_candidate_case(tmp_path,manifest=None,case=None,source_revision='a'*40,source_tree='b'*40)
