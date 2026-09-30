"""CPU-only crossover testing whether contact conversion retains the last async dt.

Each phase holds a 1kg sphere at rest. Predicted raw/mg under stale-dt hypothesis:
0.002, 1, 0.5, 0.25, 1, 4, 1. This is diagnostic, not a C0 correction.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

PHASES = [("sync", .002, .5, .002), ("async", .002, .25, 1),
          ("sync", .001, .25, .5), ("sync", .0005, .25, .25),
          ("async", .0005, .25, 1), ("sync", .002, .25, 4),
          ("async", .002, .25, 1)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fixture-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    assert hashlib.sha256(args.fixture_script.read_bytes()).hexdigest() == "4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25"
    spec = importlib.util.spec_from_file_location("calibration_fixture", args.fixture_script)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = args.output
    root.mkdir(exist_ok=False)
    if not args.worker:
        fixture.write(root/"protocol.json", {"phases": PHASES, "script_sha256":fixture.digest(__file__),
                      "hypothesis":"contact conversion uses last async dt; default denominator 1 second",
                      "relative_ratio_tolerance":.02,"scope":"CPU only, fresh process twice"})
        for repeat in (1,2):
            with (root/f"r{repeat}.log").open("x") as log:
                result=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",
                    "--fixture-script",str(args.fixture_script.resolve()),"--output",str(root/f"r{repeat}")],
                    stdout=log,stderr=subprocess.STDOUT,timeout=90,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
            if result.returncode:
                fixture.write(root/"error.json",{"repeat":repeat,"returncode":result.returncode})
                raise RuntimeError("worker failed; evidence retained")
        fixture.write(root/"inventory.json",[{"path":p.relative_to(root).as_posix(),"size":p.stat().st_size,
                       "sha256":fixture.digest(p)} for p in sorted(root.rglob("*")) if p.is_file()])
        print((root/"r1"/"summary.json").read_text())
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    import numpy as np
    from ovphysx import PhysX
    from ovphysx.types import TensorType
    usd=root/"fixture.usda"
    with usd.open("x") as f:
        f.write(fixture.scene(1))
    sdk=PhysX(device="cpu")
    sdk.add_usd(str(usd.resolve()))
    sdk.wait_all()
    cb=sdk.create_contact_binding(sensor_patterns=["/World/Ball"],filter_patterns=["/World/Ground"],
                                 filters_per_sensor=1,max_contact_data_count=64)
    bindings={n:sdk.create_tensor_binding(pattern="/World/Ball",tensor_type=k) for n,k in
              (("pose",TensorType.RIGID_BODY_POSE),("velocity",TensorType.RIGID_BODY_VELOCITY),("mass",TensorType.RIGID_BODY_MASS))}
    arrays={k:np.zeros(b.shape,dtype=np.float32) for k,b in bindings.items()}
    matrix=np.zeros((1,1,3),dtype=np.float32)
    net=np.zeros((1,3),dtype=np.float32)
    time=0.
    summaries=[]
    for phase,(mode,dt,duration,predicted) in enumerate(PHASES):
        samples=[]
        for i in range(round(duration/dt)):
            if mode=="sync":
                sdk.step_sync(dt=dt,sim_time=time)
            else:
                sdk.wait_op(sdk.step(dt=dt,sim_time=time))
            time+=dt
            if (i+1)*dt >= duration/2:
                cb.read_force_matrix(matrix)
                cb.read_net_forces(net)
                for k,b in bindings.items(): b.read(arrays[k])
                assert all(np.isfinite(a).all() for a in (*arrays.values(),matrix,net))
                samples.append({"time_s":time,"matrix":matrix.flatten().tolist(),"net":net.flatten().tolist(),
                                **{k:a.flatten().tolist() for k,a in arrays.items()}})
        raw=float(np.median([np.linalg.norm(s["matrix"]) for s in samples]))
        z=[s["pose"][2] for s in samples]
        velocity=max(float(np.linalg.norm(s["velocity"][:3])) for s in samples)
        mass=float(arrays["mass"].flatten()[0])
        valid=abs(mass-1)<1e-5 and max(z)-min(z)<1e-4 and velocity<1e-3 and .045<min(z)<.055
        summaries.append({"phase":phase,"mode":mode,"dt_s":dt,"raw_median":raw,"raw_over_mg":raw/(mass*9.81),
                          "predicted_ratio":predicted,"steady_valid":valid,"max_speed_m_s":velocity,
                          "prediction_supported":valid and abs(raw/(mass*9.81)/predicted-1)<.02})
        fixture.write(root/f"phase{phase}.json",samples)
    fixture.write(root/"summary.json",summaries)
    for b in bindings.values(): b.destroy()
    cb.destroy()
    sdk.release()


if __name__=="__main__":
    main()
