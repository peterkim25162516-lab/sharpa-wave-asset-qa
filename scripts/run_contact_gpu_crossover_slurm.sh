#!/usr/bin/env bash
set -euo pipefail
root=/data/home/exampleuser/sharpa-contact-gpu-diagnosis-20260929
test -n "${SLURM_JOB_ID:-}"
test ! -e "$root/crossover-a1"
for check in 1 2; do
  metrics=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  "$root/env/bin/python" -c 'import sys; rows=sys.argv[1].splitlines(); assert len(rows)==1; uuid,mem,util=[v.strip() for v in rows[0].split(",")]; assert uuid=="GPU-7d1adf8d-3004-54d0-aace-3b68bc444363"; assert int(mem)<=32 and int(util)==0; assert not sys.argv[2].strip()' "$metrics" "$processes"
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
timeout --signal=TERM --kill-after=10s 210s "$root/env/bin/python" -B "$root/probe_contact_gpu_crossover.py" --fixture-script "$root/probe_contact_load_calibration.py" --output "$root/crossover-a1"
