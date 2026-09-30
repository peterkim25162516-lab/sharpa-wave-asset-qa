"""Closed, site-specific single-GPU binding for the experimental C0 campaign."""
from __future__ import annotations

import os
import re
import socket
import subprocess

UUID = "GPU-9379a83b-330d-58a4-9036-bf7485af93ef"
KEYS = {"schema_version", "hostname", "job_id", "job_record", "physical_index",
        "query_index", "gpu_uuid", "inherited_cuda_visible_devices", "worker_cuda_visible_devices"}


def validate_binding(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError("Slurm binding fields are not exact")
    fixed = {"schema_version": 1, "hostname": "example-gpu-node-6", "physical_index": 7,
             "query_index": 0, "gpu_uuid": UUID, "inherited_cuda_visible_devices": "0",
             "worker_cuda_visible_devices": "0"}
    if any(type(value[k]) is not type(v) or value[k] != v for k, v in fixed.items()):
        raise ValueError("Slurm GPU binding differs from the pinned Server 6 device")
    job = value["job_id"]
    if not isinstance(job, str) or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("invalid Slurm job ID")
    raw = value["job_record"]
    if not isinstance(raw, str) or len(raw) > 65536:
        raise ValueError("invalid Slurm job record")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9/:]*)=([^\s]+)", raw))
    expected = {"JobId": job, "JobState": "RUNNING", "NodeList": "example-gpu-node-6",
                "Partition": "a800", "NumNodes": "1", "NumTasks": "1",
                "TresPerNode": "gres/shard:idx7:1"}
    if any(fields.get(k) != v for k, v in expected.items()):
        raise ValueError("Slurm allocation does not bind the pinned GPU shard")
    if not re.fullmatch(r"exampleuser\([0-9]+\)", fields.get("UserId", "")):
        raise ValueError("Slurm allocation belongs to another user")
    return dict(value)


def capture_binding() -> dict:
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("a live Slurm allocation is required")
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    result = validate_binding({"schema_version": 1, "hostname": socket.gethostname(),
        "job_id": job, "job_record": raw, "physical_index": 7, "query_index": 0,
        "gpu_uuid": UUID, "inherited_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker_cuda_visible_devices": "0"})
    observed = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits"], text=True).strip()
    if observed != "0, " + UUID:
        raise ValueError("allocated device inventory is not exactly the pinned UUID")
    return result
