"""Actual kit-less ContactSensor versus its native binding, not a C0 rerun."""
import argparse
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys

PHASES=[('async',.002,.5,1),('sync',.001,.25,.5),('async',.001,.25,1)]


def copy_fixture_prims(source, target):
    from pxr import Sdf
    layer = target.GetRootLayer()
    if layer.GetPrimAtPath('/World') is None:
        parent = Sdf.CreatePrimInLayer(layer, '/World')
        parent.specifier = Sdf.SpecifierDef
        parent.typeName = 'Xform'
    for name in ('Ball', 'Ground'):
        assert Sdf.CopySpec(source.GetRootLayer(), f'/World/{name}', layer, f'/World/{name}')
        assert target.GetPrimAtPath(f'/World/{name}').IsValid()


def preflight(args, f):
    from pxr import Usd
    source = Usd.Stage.CreateInMemory()
    assert source.GetRootLayer().ImportFromString(f.scene(1))
    target = Usd.Stage.CreateInMemory()
    assert target.GetRootLayer().GetPrimAtPath('/World') is None
    copy_fixture_prims(source, target)
    assert target.GetPrimAtPath('/World/Ball').GetAttribute('physics:mass').Get() == 1
    assert target.GetPrimAtPath('/World/Ground').GetTypeName() == 'Cube'
    assert not target.GetPrimAtPath('/World/Scene').IsValid()
    f.write(args.output/'preflight.json', {'passed': True, 'scope': 'USD_copy_only_no_GPU',
        'script_sha256': f.digest(__file__), 'root_parent_created': True})


def worker(args,f):
    intervention = None
    if args.context_async:
        spec = importlib.util.spec_from_file_location('candidate_step', args.intervention_script)
        intervention = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(intervention)
    from wave_asset_qa.adapters.ovphysx import _runtime_api
    rt=_runtime_api('cuda:0')
    import numpy as np
    import warp as wp
    from pxr import Sdf,Usd,UsdGeom
    from isaaclab_ovphysx.sensors import ContactSensor,ContactSensorCfg
    from isaaclab_ovphysx.physics import OvPhysxManager
    from ovphysx.types import TensorType
    sensor_source=Path(inspect.getfile(ContactSensor))
    assert f.digest(sensor_source)=='7fbc14a9f5d37aff3c267a51c14e1e8ad50aee8b03eecd2485c27d8f7f2b26bf'
    assert f.digest(sensor_source.with_name('kernels.py'))=='8d5d9625fe674df26d9111b4d2f6f287aba9e9eeccb0f6b9d98fa3517a214894'
    f.write(args.output/'provenance.json',{'script_sha256':f.digest(__file__),'sensor_sha256':f.digest(sensor_source),
        'kernels_sha256':f.digest(sensor_source.with_name('kernels.py')),'pid':os.getpid(),'job_id':os.environ['SLURM_JOB_ID'],
        'scope':'actual_sensor_primitive_fixture_not_full_C0','read_order':'native_before_sensor_then_native_after',
        'phases':PHASES, 'context_async_candidate': args.context_async,
        'intervention_sha256': f.digest(args.intervention_script) if args.context_async else None})
    fixture=args.output/'fixture.usda'
    with fixture.open('x') as stream: stream.write(f.scene(1))
    sim=rt['sim']
    cfg=sim.SimulationCfg(physics=rt['ov_physics'].OvPhysxCfg(),device='cuda:0',dt=.002,
                         gravity=(0,0,-9.81),create_stage_in_memory=True,render_interval=1000)
    with sim.build_simulation_context(create_new_stage=True,device='cuda:0',sim_cfg=cfg,
            add_ground_plane=False,add_lighting=False,auto_add_lighting=False) as context:
        if hasattr(context,'_app_control_on_stop_handle'): context._app_control_on_stop_handle=None
        OvPhysxManager._ensure_physx_schemas_registered()
        source=Usd.Stage.Open(str(fixture))
        copy_fixture_prims(source, context.stage)
        sensor=ContactSensor(ContactSensorCfg(prim_path='/World/Ball',update_period=0.,history_length=0,
            debug_vis=False,filter_prim_paths_expr=['/World/Ground']))
        context.reset()
        sdk=OvPhysxManager.get_physx_instance()
        binding=sensor.contact_view
        assert binding.sensor_count==binding.filter_count==1
        bindings={n:sdk.create_tensor_binding(pattern='/World/Ball',tensor_type=t) for n,t in
            (('pose',TensorType.RIGID_BODY_POSE),('velocity',TensorType.RIGID_BODY_VELOCITY),('mass',TensorType.RIGID_BODY_MASS))}
        arrays={n:wp.zeros(tuple(b.shape),dtype=wp.float32,device='cuda:0') for n,b in bindings.items()}
        before=wp.zeros((1,1,3),dtype=wp.float32,device='cuda:0')
        after=wp.zeros((1,1,3),dtype=wp.float32,device='cuda:0')
        for n,b in bindings.items(): b.read(arrays[n])
        wp.synchronize()
        time=0.
        for phase,(mode,dt,duration,predicted) in enumerate(PHASES):
            rows=[]
            for i in range(round(duration/dt)):
                if mode=='sync' and args.context_async:
                    context.cfg.dt = dt
                    with intervention.async_wait_intervention(sdk, expected_dt=dt) as audit:
                        context.step(render=False)
                elif mode=='sync': sdk.step_sync(dt=dt,sim_time=time)
                else: sdk.wait_op(sdk.step(dt=dt,sim_time=time))
                time+=dt
                binding.read_force_matrix(before)
                sensor.update(dt,force_recompute=True)
                output=sensor.data.force_matrix_w.warp
                binding.read_force_matrix(after)
                for n,b in bindings.items(): b.read(arrays[n])
                wp.synchronize()
                row={n:a.numpy().flatten().tolist() for n,a in arrays.items()}
                row.update(before=before.numpy().flatten().tolist(),sensor=output.numpy().flatten().tolist(),
                           after=after.numpy().flatten().tolist(),time_s=time)
                assert all(np.isfinite(v).all() for v in row.values())
                if mode=='sync' and args.context_async:
                    row['step_audit'] = dict(audit)
                rows.append(row)
            f.write(args.output/f'phase{phase}.json',rows)
        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        f.write(args.output/'gpu_private.json',{'pid':os.getpid(),'query':query})
        assert any(line.split(',')[1].strip()==str(os.getpid()) for line in query.splitlines())
        assert not any(p.IsA(UsdGeom.Camera) for p in context.stage.TraverseAll())
        assert not any(n=='isaacsim' or n.startswith(('omni.kit','omni.renderer','omni.replicator')) for n in sys.modules)
        for b in bindings.values(): b.destroy()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fixture-script',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--context-async',action='store_true')
    p.add_argument('--intervention-script',type=Path)
    args=p.parse_args()
    if args.context_async:
        if args.intervention_script is None or not args.intervention_script.is_file():
            p.error('--context-async requires --intervention-script')
    assert hashlib.sha256(args.fixture_script.read_bytes()).hexdigest()=='4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25'
    spec=importlib.util.spec_from_file_location('fixture',args.fixture_script)
    f=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    args.output.mkdir(exist_ok=False)
    if args.preflight:
        preflight(args,f)
        return
    if args.worker:
        worker(args,f)
        return
    f.write(args.output/'protocol.json',{'phases':PHASES,'script_sha256':f.digest(__file__),
        'expectation':('native_before == public_sensor == native_after exactly; ratios 1,1,1 within 2%'
            if args.context_async else 'native_before == public_sensor == native_after exactly; ratios 1,0.5,1 within 2%'),
        'context_async_candidate':args.context_async,
        'intervention_sha256':f.digest(args.intervention_script) if args.context_async else None,
        'repeats':2,'scope':'primitive_sensor_passthrough_not_C0'})
    for repeat in (1,2):
        try:
            with (args.output/f'r{repeat}.log').open('x') as log:
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--fixture-script',
                    str(args.fixture_script.resolve()),'--output',str(args.output/f'r{repeat}')]
                    + (['--context-async','--intervention-script',str(args.intervention_script.resolve())]
                       if args.context_async else []),
                    stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
            f.write(args.output/'error.json',{'repeat':repeat,'error':str(exc)})
            raise
    f.write(args.output/'inventory.json',[{'path':x.relative_to(args.output).as_posix(),'size':x.stat().st_size,'sha256':f.digest(x)}
        for x in sorted(args.output.rglob('*')) if x.is_file()])


if __name__=='__main__': main()
