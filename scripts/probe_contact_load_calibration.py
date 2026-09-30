"""Isolated CPU-only OVPhysX known-load calibration, not formal C0 evidence.

24 fresh workers: 1/2 kg x 2/1/0.5 ms x sync/async-wait x 2 repeats.
Frozen decision: steady velocity <1e-3 m/s, vertical range <1e-4 m,
mass readback within 1e-5 kg; force-like if raw/mg is within 2% of 1,
impulse-like if raw/(mg*dt) is within 2% of 1. Anything else is unresolved.
No GPU, Kit, renderer, camera, IsaacLab or changes to installed packages.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def scene(mass):
    return '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    def PhysicsScene "Scene" {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }
    def Cube "Ground" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double size = 1
        double3 xformOp:translate = (0, 0, -0.05)
        double3 xformOp:scale = (2, 2, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Sphere "Ball" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysxRigidBodyAPI", "PhysxContactReportAPI"]) {
        double radius = 0.05
        float physics:mass = MASS
        point3f physics:centerOfMass = (0, 0, 0)
        float3 physics:diagonalInertia = (INERTIA, INERTIA, INERTIA)
        float physxRigidBody:sleepThreshold = 0
        float physxContactReport:threshold = 0
        double3 xformOp:translate = (0, 0, 0.05)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
'''.replace("MASS", str(mass)).replace("INERTIA", str(.4*mass*.05**2))


def worker(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU isolation required"
    import numpy as np
    import ovphysx
    from ovphysx import PhysX
    from ovphysx.types import TensorType

    assert importlib.metadata.version("ovphysx") == "0.4.13"
    root = args.output
    root.mkdir(exist_ok=False)
    usd = root / "fixture.usda"
    with usd.open("x", encoding="utf-8") as f:
        f.write(scene(args.mass))
    sdk = PhysX(device="cpu")
    sdk.add_usd(str(usd.resolve()))
    sdk.wait_all()
    cb = sdk.create_contact_binding(sensor_patterns=["/World/Ball"],
                                   filter_patterns=["/World/Ground"],
                                   filters_per_sensor=1, max_contact_data_count=64)
    assert cb.sensor_count == cb.filter_count == 1
    bindings = {name: sdk.create_tensor_binding(pattern="/World/Ball", tensor_type=kind)
                for name, kind in (("pose", TensorType.RIGID_BODY_POSE),
                                   ("velocity", TensorType.RIGID_BODY_VELOCITY),
                                   ("mass", TensorType.RIGID_BODY_MASS))}
    arrays = {k: np.zeros(v.shape, dtype=np.float32) for k,v in bindings.items()}
    matrix = np.zeros((1,1,3), dtype=np.float32)
    net = np.zeros((1,3), dtype=np.float32)
    samples = []
    dt = args.dt
    for i in range(round(1.0/dt)):
        if args.mode == "sync":
            sdk.step_sync(dt=dt, sim_time=i*dt)
        else:
            op = sdk.step(dt=dt, sim_time=i*dt)
            sdk.wait_op(op)
        if (i+1)*dt >= .5:
            cb.read_force_matrix(matrix)
            cb.read_net_forces(net)
            for name, binding in bindings.items():
                binding.read(arrays[name])
            assert all(np.isfinite(v).all() for v in (*arrays.values(), matrix, net))
            samples.append({"time_s": (i+1)*dt, "matrix": matrix.flatten().tolist(),
                            "net": net.flatten().tolist(),
                            **{k:v.flatten().tolist() for k,v in arrays.items()}})
    actual_mass = float(arrays["mass"].reshape(-1)[0])
    median = float(np.median([np.linalg.norm(x["matrix"]) for x in samples]))
    z = [x["pose"][2] for x in samples]
    speed = max(float(np.linalg.norm(x["velocity"][:3])) for x in samples)
    valid = abs(actual_mass-args.mass)<1e-5 and max(z)-min(z)<1e-4 and speed<1e-3 and .045<min(z)<.055
    force_ratio = median/(actual_mass*9.81)
    impulse_ratio = force_ratio/dt
    label = "unresolved"
    if valid:
        if abs(force_ratio-1)<.02:
            label = "force_like"
        elif abs(impulse_ratio-1)<.02:
            label = "impulse_like"
    forbidden = [name for name in sys.modules if name == "isaacsim" or name.startswith(("omni.kit", "omni.renderer", "omni.replicator"))]
    assert not forbidden
    write(root / "result.json", {"device":"cpu", "mode":args.mode, "dt_s":dt,
          "mass_kg": args.mass, "mass_readback_kg":actual_mass,
          "steady_valid":valid, "steady_max_speed_m_s":speed, "steady_z_range_m":max(z)-min(z),
          "raw_median":median, "raw_over_mg":force_ratio, "raw_over_mg_dt":impulse_ratio,
          "classification":label, "samples":samples, "script_sha256":digest(__file__),
          "fixture_sha256":digest(usd), "ovphysx_version":importlib.metadata.version("ovphysx"),
          "api_sha256":digest(Path(ovphysx.__file__).parent / "api.py"),
          "scope":"native_cpu_calibration_not_C0_or_GPU"})
    for binding in bindings.values():
        binding.destroy()
    cb.destroy()
    sdk.release()


def campaign(args):
    root = args.output
    root.mkdir(exist_ok=False)
    write(root / "protocol.json", {"script_sha256":digest(__file__), "scope":"native_cpu_only",
          "masses_kg":[1,2],"dt_s":[.002,.001,.0005],"modes":["sync","async"],"repeats":2,
          "steady_window_s":[.5,1], "force_or_impulse_relative_tolerance":.02,
          "velocity_limit_m_s":.001,"z_range_limit_m":.0001})
    summaries=[]
    for mass,dt,mode,repeat in itertools.product((1,2),(.002,.001,.0005),("sync","async"),(1,2)):
        name=f"mass{mass}-dt{dt}-{mode}-r{repeat}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (root/f"{name}.log").open("x", encoding="utf-8") as log:
            outcome=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",
                    "--output",str(root/name),"--mass",str(mass),"--dt",str(dt),"--mode",mode],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        if outcome.returncode:
            write(root/"error.json",{"case":name,"returncode":outcome.returncode})
            raise RuntimeError(f"failed case {name}; stop, preserve evidence")
        value=json.loads((root/name/"result.json").read_text())
        summaries.append({k:v for k,v in value.items() if k!="samples"} | {"case":name})
        print(name, value["classification"], value["raw_median"],flush=True)
    write(root/"summary.json",summaries)
    write(root/"inventory.json", [{"path":p.relative_to(root).as_posix(),"size":p.stat().st_size,
                                   "sha256":digest(p)} for p in sorted(root.rglob("*")) if p.is_file()])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--mass",type=float,default=1)
    parser.add_argument("--dt",type=float,default=.002)
    parser.add_argument("--mode",choices=("sync","async"),default="sync")
    args=parser.parse_args()
    if args.worker:
        worker(args)
    else:
        campaign(args)
