import pytest

from wave_asset_qa.contact.async_step_diagnostic import async_wait_intervention


class SDK:
    def __init__(self): self.events=[]
    def step_sync(self, *, dt, sim_time):
        self.events.append(('original',dt,sim_time))
    def step(self, *, dt, sim_time):
        self.events.append(('async',dt,sim_time))
        return 42
    def wait_op(self, operation): self.events.append(('wait',operation))


def test_context_order_and_restore():
    sdk=SDK()
    with async_wait_intervention(sdk,expected_dt=.002) as audit:
        sdk.events.append('context_before')
        sdk.step_sync(dt=.002,sim_time=.3)
        sdk.events.append('kinematics_and_context_after')
    assert sdk.events==['context_before',('async',.002,.3),('wait',42),'kinematics_and_context_after']
    assert audit['calls']==audit['waits']==1
    assert 'step_sync' not in vars(sdk)
    sdk.step_sync(dt=.002,sim_time=.4)
    assert sdk.events[-1]==('original',.002,.4)


@pytest.mark.parametrize('dt',[0,-1,float('nan'),float('inf')])
def test_invalid_dt(dt):
    with pytest.raises(ValueError):
        with async_wait_intervention(SDK(),expected_dt=dt): pass


def test_wrong_step_dt_restores():
    sdk=SDK()
    with pytest.raises(RuntimeError):
        with async_wait_intervention(sdk,expected_dt=.002): sdk.step_sync(dt=.001,sim_time=0)
    assert sdk.events==[] and 'step_sync' not in vars(sdk)


def test_double_step_refused():
    sdk=SDK()
    with pytest.raises(RuntimeError):
        with async_wait_intervention(sdk,expected_dt=.002):
            sdk.step_sync(dt=.002,sim_time=0)
            sdk.step_sync(dt=.002,sim_time=.002)
    assert len(sdk.events)==2 and 'step_sync' not in vars(sdk)


def test_missing_step_refused():
    sdk=SDK()
    with pytest.raises(RuntimeError):
        with async_wait_intervention(sdk,expected_dt=.002): pass
    assert 'step_sync' not in vars(sdk)


def test_wait_error_restores():
    sdk=SDK()
    def fail(_): raise ValueError('wait failed')
    sdk.wait_op=fail
    with pytest.raises(ValueError,match='wait failed'):
        with async_wait_intervention(sdk,expected_dt=.002): sdk.step_sync(dt=.002,sim_time=0)
    assert 'step_sync' not in vars(sdk)


def test_nested_or_existing_override_refused():
    sdk=SDK()
    with async_wait_intervention(sdk,expected_dt=.002):
        with pytest.raises(RuntimeError):
            with async_wait_intervention(sdk,expected_dt=.002): pass
        sdk.step_sync(dt=.002,sim_time=0)
    assert 'step_sync' not in vars(sdk)
