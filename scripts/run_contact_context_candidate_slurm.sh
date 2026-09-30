#!/usr/bin/env bash
set -euo pipefail
base=/data/home/exampleuser/sharpa-wave-asset-qa-gate0
root="$base/contact-sensor-diagnosis-20260929/context-candidate-a1"
test -n "${SLURM_JOB_ID:-}"
test ! -e "$root/evidence"
for check in 1 2; do
  metrics=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  "$base/env/bin/python" -c 'import sys; rows=sys.argv[1].splitlines(); assert len(rows)==1; uuid,mem,util=[v.strip() for v in rows[0].split(",")]; assert uuid=="GPU-9379a83b-330d-58a4-9036-bf7485af93ef"; assert int(mem)<=32 and int(util)==0; assert not sys.argv[2].strip()' "$metrics" "$processes"
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$base/project/bb0dfccffeb0b2a1e50d60862507fc4e9411ee3b/src"
timeout --signal=TERM --kill-after=10s 210s "$base/env/bin/python" -B "$root/probe_contact_sensor_passthrough.py" --fixture-script "$root/probe_contact_load_calibration.py" --output "$root/evidence" --context-async --intervention-script "$root/async_step_diagnostic.py"
