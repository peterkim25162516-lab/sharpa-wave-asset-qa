"""Explicit, separately identified C0 async candidate; default C0 is unchanged."""
from copy import deepcopy
import math
from pathlib import Path

from .adapters import ContactAdapterError, OVPhysXContactAdapter, _DefaultOVPhysXBridge
from .async_step_diagnostic import async_wait_intervention
from .bundle import canonical_json_sha256, sha256_file, read_json_strict, verify_adapter_private_evidence

CAMPAIGN_ID = 'contact-c0-async-v1'
MODE = 'diagnostic_async_wait_v1'


def helper_sha256():
    return sha256_file(Path(__file__).with_name('async_step_diagnostic.py'))


def validate_step_audit(records, *, expected_steps, dt_s):
    if type(expected_steps) is not int or expected_steps < 0:
        raise ContactAdapterError('invalid expected async step count')
    if not isinstance(records, list) or len(records) != expected_steps:
        raise ContactAdapterError('async audit count differs from completed trajectory')
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ContactAdapterError('invalid async audit dt')
    previous = None
    for row in records:
        if not isinstance(row, dict) or set(row) != {'calls','waits','dt_s','sim_time_s','mode'}:
            raise ContactAdapterError('async audit is not closed')
        if (row['mode'] != MODE or type(row['calls']) is not int or type(row['waits']) is not int
                or row['calls'] != 1 or row['waits'] != 1 or row['dt_s'] != dt_s):
            raise ContactAdapterError('async audit step/wait/dt mismatch')
        time = row['sim_time_s']
        if isinstance(time, bool) or not isinstance(time, (int,float)) or not math.isfinite(time) or time < 0:
            raise ContactAdapterError('invalid native simulation time')
        if previous is not None and not math.isclose(time-previous, dt_s, rel_tol=1e-7, abs_tol=1e-9):
            raise ContactAdapterError('native simulation time did not advance by dt')
        previous = time
    return canonical_json_sha256(records)


class AsyncCandidateBridge(_DefaultOVPhysXBridge):
    def __init__(self):
        super().__init__()
        self.step_audit = []

    def _native_sdk(self):
        return self._runtime['ov_physics'].OvPhysxManager.get_physx_instance()

    def step(self, *, dt_s):
        if dt_s != self._dt_s:
            raise ContactAdapterError('OV candidate timestep drifted')
        sdk = self._native_sdk()
        if sdk is None:
            raise ContactAdapterError('OV candidate native runtime unavailable')
        self._articulation.write_data_to_sim()
        with async_wait_intervention(sdk, expected_dt=dt_s) as audit:
            self._simulation.step(render=False)
        self._articulation.update(dt_s)
        self.step_audit.append(dict(audit))

    def runtime_fingerprint(self):
        result = dict(super().runtime_fingerprint())
        result['stepping_intervention'] = {
            'campaign_id': CAMPAIGN_ID, 'mode': MODE, 'helper_sha256': helper_sha256(),
            'completed_steps': len(self.step_audit),
            'audit_sha256': validate_step_audit(self.step_audit, expected_steps=len(self.step_audit),dt_s=self._dt_s),
        }
        return result


class AsyncCandidateContactAdapter(OVPhysXContactAdapter):
    def __init__(self, **kwargs):
        self.candidate_bridge = AsyncCandidateBridge()
        super().__init__(runtime_bridge_factory=lambda: self.candidate_bridge, **kwargs)

    def open_contact_case(self, *args, device, **kwargs):
        if device != 'cuda:0':
            raise ContactAdapterError('async candidate requires the sole logical CUDA device')
        return super().open_contact_case(*args, device=device, **kwargs)

    def _runtime_fingerprint_snapshot(self, **kwargs):
        result = super()._runtime_fingerprint_snapshot(**kwargs)
        # Custom bridge factories bypass these checks in test-oriented base code.
        # The real candidate must keep every original runtime safety condition.
        if (result.get('kitless') is not True or result.get('renderer') is not False
                or result.get('camera') is not False or result.get('camera_prim_paths') != []
                or result.get('created_sensor_types') != ['ContactSensor']):
            raise ContactAdapterError('async candidate violates kit-less sensor-only boundary')
        return result

    @property
    def last_private_evidence(self):
        result = dict(super().last_private_evidence)
        result['async_step_audit'] = deepcopy(self.candidate_bridge.step_audit)
        return result


def validate_candidate_evidence(private, run, *, dt_s):
    runtime = private['runtime_fingerprint']
    identity = runtime.get('stepping_intervention')
    required = {'campaign_id','mode','helper_sha256','completed_steps','audit_sha256'}
    if not isinstance(identity, dict) or set(identity) != required:
        raise ContactAdapterError('candidate identity is missing or not closed')
    expected = run.execution.completed_steps
    if (identity['campaign_id'] != CAMPAIGN_ID or identity['mode'] != MODE
            or identity['helper_sha256'] != helper_sha256() or identity['completed_steps'] != expected):
        raise ContactAdapterError('candidate identity differs from source/trajectory')
    digest = validate_step_audit(private.get('async_step_audit'),expected_steps=expected,dt_s=dt_s)
    if digest != identity['audit_sha256']:
        raise ContactAdapterError('candidate step audit hash mismatch')
    return dict(identity)


def load_candidate_case(root, *, manifest, case, source_revision, source_tree):
    """Close candidate identity and trace linkage; not a launcher/runtime-log gate."""
    from .runner import load_and_validate_contact_run
    from .scenarios import contact_manifest_sha256
    from wave_asset_qa.parity.process_identity import os_process_identity_sha256
    root = Path(root)
    envelope = read_json_strict(root/'private-campaign.json')
    keys = {'schema_version','experimental_campaign','formal_c0_replacement','backend','case_id',
            'source_revision','source_tree','manifest_sha256','intervention','run_sha256',
            'private_adapter_sha256','private_process_sha256'}
    if not isinstance(envelope, dict) or set(envelope) != keys:
        raise ContactAdapterError('candidate envelope is missing or not closed')
    expected = {'schema_version':1,'experimental_campaign':CAMPAIGN_ID,'formal_c0_replacement':False,
                'backend':case.simulator.value,'case_id':case.case_id,'source_revision':source_revision,
                'source_tree':source_tree,'manifest_sha256':contact_manifest_sha256(manifest)}
    if any(envelope[k] != value for k,value in expected.items()) or envelope['formal_c0_replacement'] is not False:
        raise ContactAdapterError('candidate envelope identity mismatch')
    for key,path in [('run_sha256',f'{case.case_id}.run.json'),
                     ('private_adapter_sha256','private-adapter-evidence.json'),
                     ('private_process_sha256','private-process.json')]:
        if envelope[key] != sha256_file(root/path):
            raise ContactAdapterError('candidate file binding mismatch')
    process = read_json_strict(root/'private-process.json')
    for key,value in {'experimental_campaign':CAMPAIGN_ID,'case_id':case.case_id,'backend':case.simulator.value,
                      'source_revision':source_revision,'source_tree':source_tree,
                      'manifest_semantic_sha256':contact_manifest_sha256(manifest),
                      'asset_tree_sha256':manifest.provenance.canonical_lf_asset_tree_sha256,
                      'worker_module_sha256':sha256_file(Path(__file__).with_name('worker.py'))}.items():
        if process.get(key) != value:
            raise ContactAdapterError('candidate process identity mismatch')
    fresh_hash = os_process_identity_sha256(process['os_process_identity'])
    if process.get('fresh_process_identity_sha256') != fresh_hash:
        raise ContactAdapterError('candidate fresh-process hash mismatch')
    run = load_and_validate_contact_run(root/f'{case.case_id}.run.json',manifest=manifest,case=case,
        source_revision=source_revision,source_tree=source_tree,fresh_process_identity_sha256=fresh_hash)
    private = read_json_strict(root/'private-adapter-evidence.json')
    verify_adapter_private_evidence(private,run)
    if case.simulator.value == 'ovphysx':
        identity = validate_candidate_evidence(private,run,dt_s=case.dt_s)
        if envelope['intervention'] != identity:
            raise ContactAdapterError('candidate envelope intervention mismatch')
    elif envelope['intervention'] is not None or 'async_step_audit' in private:
        raise ContactAdapterError('MuJoCo must not carry the OV stepping intervention')
    return run
