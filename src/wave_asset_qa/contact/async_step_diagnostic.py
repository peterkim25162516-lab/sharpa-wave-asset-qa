"""Opt-in stepping intervention for a separate C0 diagnostic, never the default.

Preserve SimulationContext.step bookkeeping and the manager's kinematic update.
Only a single native step_sync call is redirected to step followed by wait_op.
This is a single-threaded fresh-worker experiment, not a general SDK patch.
"""
from contextlib import contextmanager
import math


@contextmanager
def async_wait_intervention(sdk, *, expected_dt):
    if not math.isfinite(expected_dt) or expected_dt <= 0:
        raise ValueError('expected_dt must be finite and positive')
    namespace = vars(sdk)
    if 'step_sync' in namespace:
        raise RuntimeError('refuse an existing instance-level step override')
    original = sdk.step_sync
    audit = {'calls': 0, 'waits': 0, 'dt_s': expected_dt, 'sim_time_s': None,
             'mode': 'diagnostic_async_wait_v1'}

    def step_sync(*, dt, sim_time):
        if audit['calls'] or dt != expected_dt or not math.isfinite(sim_time):
            raise RuntimeError('unexpected step count, dt, or simulation time')
        audit['calls'] += 1
        audit['sim_time_s'] = sim_time
        operation = sdk.step(dt=dt, sim_time=sim_time)
        sdk.wait_op(operation)
        audit['waits'] += 1

    sdk.step_sync = step_sync
    try:
        yield audit
        if audit['calls'] != 1 or audit['waits'] != 1:
            raise RuntimeError('SimulationContext must make exactly one completed native step')
    finally:
        del sdk.step_sync
        if sdk.step_sync != original:
            raise RuntimeError('native step_sync restoration failed')
