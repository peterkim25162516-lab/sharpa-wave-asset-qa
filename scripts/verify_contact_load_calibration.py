"""Verify retrieved calibration inventories and recompute conclusions from samples."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from statistics import median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inventory(root, expected):
    manifest=root/"inventory.json"
    assert sha(manifest)==expected
    records=json.loads(manifest.read_text())
    expected_paths={r["path"] for r in records}
    assert len(expected_paths)==len(records)
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual==expected_paths|{"inventory.json"}
    for r in records:
        p=root/r["path"]
        assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
        assert p.stat().st_size==r["size"] and sha(p)==r["sha256"]
    return len(records)


def norm(values):
    return sum(x*x for x in values)**.5


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    load=args.root/"evidence"
    cross=args.root/"crossover-evidence"
    counts=[verify_inventory(load,"67091c120ae98cd7fd1ef77c79f09f9f4879fe72178a9716b4f6c59d82367133"),
            verify_inventory(cross,"0f4c517c373094770d974a8bf019329ccfa2b5fbb9c35f19e3ad174c086c1c41")]
    cases={}
    ratio_error=[]
    samples_count=0
    expected_cases={f"mass{m}-dt{dt}-{mode}-r{repeat}" for m,dt,mode,repeat in
                    itertools.product((1,2),(.002,.001,.0005),("sync","async"),(1,2))}
    paths=list(load.glob("*/result.json"))
    assert {path.parent.name for path in paths}==expected_cases
    for path in paths:
        data=json.loads(path.read_text())
        assert data["steady_valid"] and data["ovphysx_version"]=="0.4.13"
        assert data["script_sha256"]=="4277d8041a336d67fda1ebe4dea390e513ab9af420f21e833f453ef714870a25"
        assert data["api_sha256"]=="cd0dd9890998571a97df9fe5f04610e4d654082e14d7ebab93a238a3b8777960"
        samples=data["samples"]
        assert len(samples)==round(.5/data["dt_s"])+1
        samples_count+=len(samples)
        for s in samples:
            assert abs(s["mass"][0]-data["mass_kg"])<1e-5
            assert norm(s["velocity"][:3])<.001
            assert .045<s["pose"][2]<.055
            assert norm([a-b for a,b in zip(s["net"],s["matrix"])])<1e-6
            assert abs(s["matrix"][0])+abs(s["matrix"][1])<1e-6
        assert max(s["pose"][2] for s in samples)-min(s["pose"][2] for s in samples)<1e-4
        raw=median(norm(s["matrix"]) for s in samples)
        target=data["mass_kg"]*9.81*(data["dt_s"] if data["mode"]=="sync" else 1)
        assert abs(raw/target-1)<.02
        assert abs(raw-data["raw_median"])<1e-10
        ratio_error.append(abs(raw/target-1))
        cases[path.parent.name]=data
    for name,data in cases.items():
        if name.endswith("r1"):
            assert data["samples"]==cases[name[:-1]+"2"]["samples"]
    predicted=[.002,1,.5,.25,1,4,1]
    for repeat in (1,2):
        summary=json.loads((cross/f"r{repeat}"/"summary.json").read_text())
        assert len(summary)==7
        for i,target in enumerate(predicted):
            data=json.loads((cross/f"r{repeat}"/f"phase{i}.json").read_text())
            for s in data:
                assert s["mass"]==[1.0] and norm(s["velocity"][:3])<.001
                assert .045<s["pose"][2]<.055
                assert norm([a-b for a,b in zip(s["net"],s["matrix"])])<1e-6
            assert max(s["pose"][2] for s in data)-min(s["pose"][2] for s in data)<1e-4
            observed=median(norm(s["matrix"]) for s in data)/9.81
            assert abs(observed/target-1)<.02
            assert abs(observed-summary[i]["raw_over_mg"])<1e-10
            assert data==json.loads((cross/"r1"/f"phase{i}.json").read_text())
    print(json.dumps({"payload_files_verified":counts,"load_cases":len(cases),"load_samples":samples_count,
                      "fresh_repeat_exact":True,"crossover_phases":14,
                      "max_load_relative_prediction_error":max(ratio_error),
                      "native_cpu_stale_dt_behavior_supported":True,
                      "gpu_reproduction":"not_run_NVML_unavailable"},indent=2))


if __name__=="__main__":
    main()
