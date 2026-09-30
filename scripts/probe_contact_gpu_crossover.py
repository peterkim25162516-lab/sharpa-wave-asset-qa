"""Predeclared DirectGPU dt crossover, two fresh processes, not formal C0."""
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

# Explicit async seed after automatic warmup; no assumed default denominator.
PHASES = [('async', .002, .5, 1), ('sync', .001, .25, .5),
          ('sync', .0005, .25, .25), ('async', .0005, .25, 1),
          ('sync', .002, .25, 4), ('async', .002, .25, 1)]


def worker(args, f):
    import numpy as np
    import warp as wp
    import ovphysx
    from ovphysx import PhysX, PhysXConfig
    from ovphysx.types import TensorType
    assert os.environ.get('SLURM_JOB_ID')
    assert importlib.metadata.version('ovphysx') == '0.4.13'
    wp.init()
    assert wp.is_cuda_available(), 'CPU fallback forbidden'
    device = wp.get_device('cuda:0')
    root = args.output
    usd = root/'fixture.usda'
    with usd.open('x') as stream:
        stream.write(f.scene(1))
    f.write(root/'provenance.json', {
        'job_id': os.environ['SLURM_JOB_ID'], 'pid': os.getpid(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'script_sha256': f.digest(__file__),
        'api_sha256': f.digest(Path(ovphysx.__file__).parent/'api.py'),
        'packages': {n: importlib.metadata.version(n) for n in ('ovphysx','numpy','warp-lang','packaging')},
        'order': ['load', 'bindings', 'first_tensor_reads_auto_warmup', 'explicit_async_seed_phase', 'remaining_phases']})
    sdk = PhysX(device='gpu', config=PhysXConfig(carbonite_overrides={
        '/physics/suppressReadback': True, '/physics/suppressFabricUpdate': True}))
    sdk.add_usd(str(usd.resolve()))
    sdk.wait_all()
    cb = sdk.create_contact_binding(sensor_patterns=['/World/Ball'], filter_patterns=['/World/Ground'],
                                   filters_per_sensor=1, max_contact_data_count=64)
    assert cb.sensor_count == cb.filter_count == 1
    bindings = {n: sdk.create_tensor_binding(pattern='/World/Ball', tensor_type=t) for n,t in
                (('pose',TensorType.RIGID_BODY_POSE),('velocity',TensorType.RIGID_BODY_VELOCITY),('mass',TensorType.RIGID_BODY_MASS))}
    arrays = {n: wp.zeros(tuple(b.shape),dtype=wp.float32,device=device) for n,b in bindings.items()}
    matrix = wp.zeros((1,1,3),dtype=wp.float32,device=device)
    net = wp.zeros((1,3),dtype=wp.float32,device=device)
    for n,b in bindings.items():
        b.read(arrays[n])
    wp.synchronize()
    time = 0.
    summaries = []
    for phase,(mode,dt,duration,predicted) in enumerate(PHASES):
        samples = []
        for i in range(round(duration/dt)):
            if mode == 'sync':
                sdk.step_sync(dt=dt,sim_time=time)
            else:
                sdk.wait_op(sdk.step(dt=dt,sim_time=time))
            time += dt
            cb.read_force_matrix(matrix)
            cb.read_net_forces(net)
            for n,b in bindings.items():
                b.read(arrays[n])
            wp.synchronize()
            value = {n:a.numpy().flatten().tolist() for n,a in arrays.items()}
            value.update(matrix=matrix.numpy().flatten().tolist(),net=net.numpy().flatten().tolist(),time_s=time)
            assert all(np.isfinite(v).all() for v in value.values())
            samples.append(value)
        f.write(root/f'phase{phase}.json',samples)
        steady = samples[len(samples)//2:]
        mass = steady[-1]['mass'][0]
        z = [s['pose'][2] for s in steady]
        speed = max(float(np.linalg.norm(s['velocity'][:3])) for s in steady)
        valid = all(abs(s['mass'][0]-1)<1e-5 for s in steady) and max(z)-min(z)<1e-4 and speed<1e-3 and .045<min(z)<=max(z)<.055
        ratios = {n:float(np.median([np.linalg.norm(s[n]) for s in steady]))/(mass*9.81) for n in ('matrix','net')}
        summaries.append({'phase':phase,'mode':mode,'dt_s':dt,'predicted_ratio':predicted,
                          'ratios':ratios,'steady_valid':bool(valid),'max_speed':speed,'z_range':max(z)-min(z),
                          'prediction_supported':bool(valid and all(abs(r/predicted-1)<.02 for r in ratios.values()))})
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    f.write(root/'gpu_processes_private.json',{'pid':os.getpid(),'query':processes})
    assert any(str(os.getpid())==line.split(',')[1].strip() for line in processes.splitlines())
    f.write(root/'summary.json',summaries)
    for b in bindings.values():
        b.destroy()
    cb.destroy()
    sdk.release()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--fixture-script',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    args = p.parse_args()
    import hashlib
    assert hashlib.sha256(args.fixture_script.read_bytes()).hexdigest() == '4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25'
    spec = importlib.util.spec_from_file_location('fixture',args.fixture_script)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    root = args.output
    root.mkdir(exist_ok=False)
    if args.worker:
        worker(args,f)
        return
    f.write(root/'protocol.json',{'phases':PHASES,'script_sha256':f.digest(__file__),
        'hypothesis':'reported contact force scales with current dt / last async dt',
        'relative_tolerance':.02,'scope':'native_directgpu_diagnostic_not_C0','fresh_process_repeats':2})
    for repeat in (1,2):
        try:
            with (root/f'r{repeat}.log').open('x') as log:
                result = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
                    '--fixture-script',str(args.fixture_script.resolve()),'--output',str(root/f'r{repeat}')],
                    stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
            f.write(root/'error.json',{'repeat':repeat,'error':str(exc)})
            raise
    f.write(root/'inventory.json',[{'path':x.relative_to(root).as_posix(),'size':x.stat().st_size,'sha256':f.digest(x)}
                                  for x in sorted(root.rglob('*')) if x.is_file()])
    print((root/'r1'/'summary.json').read_text(),flush=True)


if __name__ == '__main__':
    main()
