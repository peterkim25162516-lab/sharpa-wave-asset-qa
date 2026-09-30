"""Native DirectGPU engineering smoke; separate from frozen C0 evidence."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--fixture-script', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import hashlib
    assert hashlib.sha256(args.fixture_script.read_bytes()).hexdigest() == '4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25'
    assert os.environ.get('SLURM_JOB_ID'), 'scheduler allocation required'
    spec = importlib.util.spec_from_file_location('fixture', args.fixture_script)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    args.output.mkdir(exist_ok=False)
    import numpy as np
    import warp as wp
    import ovphysx
    from ovphysx import PhysX, PhysXConfig
    from ovphysx.types import TensorType
    assert importlib.metadata.version('ovphysx') == '0.4.13'
    wp.init()
    assert wp.is_cuda_available(), 'CPU fallback forbidden'
    device = wp.get_device('cuda:0')
    usd = args.output/'fixture.usda'
    with usd.open('x') as stream:
        stream.write(f.scene(1))
    f.write(args.output/'protocol.json', {
        'scope': 'engineering_smoke_not_formal_C0', 'script_sha256': f.digest(__file__),
        'api_sha256': f.digest(Path(ovphysx.__file__).parent/'api.py'),
        'job_id': os.environ['SLURM_JOB_ID'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device': str(device), 'mass_kg': 1, 'dt_s': .002,
        'order': ['load', 'bindings', 'first_tensor_reads_auto_warmup', 'async_wait_500_steps'],
        'packages': {n: importlib.metadata.version(n) for n in ('ovphysx','numpy','warp-lang','packaging')},
    })
    sdk = PhysX(device='gpu', config=PhysXConfig(carbonite_overrides={
        '/physics/suppressReadback': True, '/physics/suppressFabricUpdate': True}))
    sdk.add_usd(str(usd.resolve()))
    sdk.wait_all()
    cb = sdk.create_contact_binding(sensor_patterns=['/World/Ball'], filter_patterns=['/World/Ground'],
                                   filters_per_sensor=1, max_contact_data_count=64)
    assert cb.sensor_count == cb.filter_count == 1
    bindings = {n: sdk.create_tensor_binding(pattern='/World/Ball', tensor_type=t) for n,t in
                (('pose',TensorType.RIGID_BODY_POSE),('velocity',TensorType.RIGID_BODY_VELOCITY),('mass',TensorType.RIGID_BODY_MASS))}
    arrays = {n: wp.zeros(tuple(b.shape), dtype=wp.float32, device=device) for n,b in bindings.items()}
    matrix = wp.zeros((1,1,3),dtype=wp.float32,device=device)
    net = wp.zeros((1,3),dtype=wp.float32,device=device)
    for n,b in bindings.items():
        b.read(arrays[n])
    wp.synchronize()
    samples = []
    for i in range(500):
        sdk.wait_op(sdk.step(dt=.002, sim_time=i*.002))
        cb.read_force_matrix(matrix)
        cb.read_net_forces(net)
        for n,b in bindings.items():
            b.read(arrays[n])
        wp.synchronize()
        values = {n:a.numpy().flatten().tolist() for n,a in arrays.items()}
        values.update(matrix=matrix.numpy().flatten().tolist(), net=net.numpy().flatten().tolist(), time_s=(i+1)*.002)
        assert all(np.isfinite(v).all() for v in values.values())
        samples.append(values)
    f.write(args.output/'samples.json', samples)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    f.write(args.output/'gpu_processes_private.json', {'pid':os.getpid(),'query':processes})
    assert any(str(os.getpid()) == line.split(',')[1].strip() for line in processes.splitlines()), 'native process GPU evidence missing'
    steady = samples[249:]
    mass = steady[-1]['mass'][0]
    z = [s['pose'][2] for s in steady]
    speed = max(float(np.linalg.norm(s['velocity'][:3])) for s in steady)
    ratio = float(np.median([np.linalg.norm(s['matrix']) for s in steady]))/(mass*9.81)
    valid = abs(mass-1)<1e-5 and max(z)-min(z)<1e-4 and speed<1e-3 and .045<min(z)<.055 and abs(ratio-1)<.02
    result = {'steady_valid':bool(valid),'raw_over_mg':ratio,'mass_kg':mass,'max_speed':speed,'z_range':max(z)-min(z),'scope':'native_gpu_smoke_only'}
    f.write(args.output/'result.json',result)
    for b in bindings.values():
        b.destroy()
    cb.destroy()
    sdk.release()
    f.write(args.output/'inventory.json',[{'path':x.relative_to(args.output).as_posix(),'sha256':f.digest(x),'size':x.stat().st_size} for x in sorted(args.output.rglob('*')) if x.is_file()])
    print(json.dumps(result),flush=True)
    assert valid, 'known-load smoke failed; retain evidence and stop'


if __name__ == '__main__':
    main()
