"""24 fresh-process native DirectGPU load cases. Diagnostic, not C0."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys


def worker(args,f):
    import numpy as np
    import warp as wp
    import ovphysx
    from ovphysx import PhysX,PhysXConfig
    from ovphysx.types import TensorType
    assert os.environ.get('SLURM_JOB_ID')
    assert importlib.metadata.version('ovphysx')=='0.4.13'
    wp.init()
    assert wp.is_cuda_available()
    root=args.output
    usd=root/'fixture.usda'
    with usd.open('x') as stream:
        stream.write(f.scene(args.mass))
    f.write(root/'provenance.json',{'script_sha256':f.digest(__file__),'pid':os.getpid(),
        'job_id':os.environ['SLURM_JOB_ID'],'api_sha256':f.digest(Path(ovphysx.__file__).parent/'api.py'),
        'packages':{n:importlib.metadata.version(n) for n in ('ovphysx','numpy','warp-lang','packaging')},
        'mass_kg':args.mass,'dt_s':args.dt,'mode':args.mode,
        'order':['load','bindings','first_tensor_reads_auto_warmup','requested_mode_1_second'],
        'initial_sync_scale':'descriptive_only_no_assumed_default_denominator'})
    sdk=PhysX(device='gpu',config=PhysXConfig(carbonite_overrides={
        '/physics/suppressReadback':True,'/physics/suppressFabricUpdate':True}))
    sdk.add_usd(str(usd.resolve()))
    sdk.wait_all()
    cb=sdk.create_contact_binding(sensor_patterns=['/World/Ball'],filter_patterns=['/World/Ground'],
        filters_per_sensor=1,max_contact_data_count=64)
    assert cb.sensor_count==cb.filter_count==1
    bindings={n:sdk.create_tensor_binding(pattern='/World/Ball',tensor_type=t) for n,t in
        (('pose',TensorType.RIGID_BODY_POSE),('velocity',TensorType.RIGID_BODY_VELOCITY),('mass',TensorType.RIGID_BODY_MASS))}
    arrays={n:wp.zeros(tuple(b.shape),dtype=wp.float32,device='cuda:0') for n,b in bindings.items()}
    matrix=wp.zeros((1,1,3),dtype=wp.float32,device='cuda:0')
    net=wp.zeros((1,3),dtype=wp.float32,device='cuda:0')
    for n,b in bindings.items(): b.read(arrays[n])
    wp.synchronize()
    samples=[]
    for i in range(round(1/args.dt)):
        if args.mode=='sync': sdk.step_sync(dt=args.dt,sim_time=i*args.dt)
        else: sdk.wait_op(sdk.step(dt=args.dt,sim_time=i*args.dt))
        cb.read_force_matrix(matrix)
        cb.read_net_forces(net)
        for n,b in bindings.items(): b.read(arrays[n])
        wp.synchronize()
        value={n:a.numpy().flatten().tolist() for n,a in arrays.items()}
        value.update(matrix=matrix.numpy().flatten().tolist(),net=net.numpy().flatten().tolist(),time_s=(i+1)*args.dt)
        assert all(np.isfinite(v).all() for v in value.values())
        samples.append(value)
    f.write(root/'samples.json',samples)
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    f.write(root/'gpu_private.json',{'pid':os.getpid(),'query':query})
    assert any(line.split(',')[1].strip()==str(os.getpid()) for line in query.splitlines())
    steady=[s for s in samples if s['time_s']>=.5]
    z=[s['pose'][2] for s in steady]
    speed=max(float(np.linalg.norm(s['velocity'][:3])) for s in steady)
    valid=all(abs(s['mass'][0]-args.mass)<1e-5 for s in steady) and speed<1e-3 and max(z)-min(z)<1e-4 and .045<min(z)<=max(z)<.055
    ratios={n:float(np.median([np.linalg.norm(s[n]) for s in steady]))/(args.mass*9.81) for n in ('matrix','net')}
    f.write(root/'summary.json',{'steady_valid':bool(valid),'ratios':ratios,'max_speed':speed,'z_range':max(z)-min(z),
        'async_force_supported':bool(valid and all(abs(r-1)<.02 for r in ratios.values())) if args.mode=='async' else None})
    for b in bindings.values(): b.destroy()
    cb.destroy()
    sdk.release()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fixture-script',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--mass',type=float,default=1)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--mode',choices=['sync','async'],default='sync')
    args=p.parse_args()
    assert hashlib.sha256(args.fixture_script.read_bytes()).hexdigest()=='4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25'
    spec=importlib.util.spec_from_file_location('fixture',args.fixture_script)
    f=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    root=args.output
    root.mkdir(exist_ok=False)
    if args.worker:
        worker(args,f)
        return
    cases=list(itertools.product((1,2),(.002,.001,.0005),('sync','async'),(1,2)))
    f.write(root/'protocol.json',{'cases':cases,'script_sha256':f.digest(__file__),
        'async_expected_ratio':1,'relative_tolerance':.02,'sync_scale':'descriptive_only_after_auto_warmup',
        'scope':'native_GPU_diagnostic_not_formal_C0','steady_window_s':[.5,1.]})
    for mass,dt,mode,repeat in cases:
        name=f'm{mass}-dt{dt}-{mode}-r{repeat}'
        try:
            with (root/f'{name}.log').open('x') as log:
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
                    '--fixture-script',str(args.fixture_script.resolve()),'--output',str(root/name),
                    '--mass',str(mass),'--dt',str(dt),'--mode',mode],stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
            f.write(root/'error.json',{'case':name,'error':str(exc)})
            raise
        print(name,'collected',flush=True)
    f.write(root/'inventory.json',[
        {'path':x.relative_to(root).as_posix(),'size':x.stat().st_size,'sha256':f.digest(x)}
        for x in sorted(root.rglob('*')) if x.is_file()])


if __name__=='__main__': main()
