"""Post-hoc, read-only C0 event diagnosis; never changes formal acceptance."""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
from statistics import median

from wave_asset_qa.contact.bundle import (
    inventory_root_sha256, regular_tree_records, read_json_strict,
    sha256_file, write_json_exclusive,
)
from wave_asset_qa.contact.records import load_contact_run
from wave_asset_qa.contact.metrics import debounce_pair_activity

EXPECTED_ROOT = "7f69ff9416c755401a7b2915684c457930feb13cb0803effc2955c90468dd69d"


def event(samples, dt, window=0.004):
    result = debounce_pair_activity(samples, dt_s=dt, debounce_window_s=window)
    return {name: None if value is None else value.to_dict() for name, value in
            (("onset", result.onset_interval), ("release", result.release_interval))}


def crossing(samples, value):
    """First upward gap crossing after return begins; interpolation is descriptive."""
    for a, b in zip(samples, samples[1:]):
        if a.time_s >= 4 and a.signed_gap_m <= value < b.signed_gap_m:
            return a.time_s + (b.time_s-a.time_s)*(value-a.signed_gap_m)/(b.signed_gap_m-a.signed_gap_m)
    return None


def threshold_event(samples, dt, threshold):
    # Samples were validated on input. Shallow analysis copies retain immutable
    # state mappings; only the exploratory bit and native payload reference change.
    derived = []
    for sample in samples:
        item = copy(sample)
        object.__setattr__(item, "pair_active", sample.native_contact_observation["selected_pair_force_norm_n"] > threshold)
        object.__setattr__(item, "native_contact_observation", None)
        derived.append(item)
    return event(tuple(derived), dt)


def analyze(run):
    s = run.samples
    dt = run.case.dt_s
    forces = [x.native_contact_observation["selected_pair_force_norm_n"] for x in s]
    assert all(x.pair_active == (f > 0.0001) for x, f in zip(s, forces))
    official = event(s, dt)
    release = official["release"]
    around = []
    if release:
        for x, f in zip(s, forces):
            if abs(x.time_s-release["midpoint_s"]) <= 0.015:
                around.append({"time_s": x.time_s, "force_n": f,
                               "gap_m": x.signed_gap_m, "active": x.pair_active})
    thresholds = {}
    for threshold in (0, 1e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2):
        thresholds[str(threshold)] = threshold_event(s, dt, threshold)
    return {
        "events": official,
        "debounce_sweep": {str(w): event(s, dt, w) for w in (0.001, 0.002, 0.004, 0.008, 0.016)},
        "force_threshold_sweep_n": thresholds,
        "gap_crossing_s": {str(g): crossing(s, g) for g in (0, 1e-6, 1e-5, 1e-4)},
        "release_neighborhood": around,
        "pair_active_count": sum(x.pair_active for x in s),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json_strict(args.bundle / "bundle-manifest.json")
    records = regular_tree_records(args.bundle, exclude=("bundle-manifest.json",))
    assert list(records) == manifest["records"], "inventory mismatch"
    assert inventory_root_sha256(records) == manifest["root_sha256"] == EXPECTED_ROOT
    runs = {}
    identities = {}
    for backend in ("mujoco", "ovphysx"):
        for path in sorted((args.bundle / "raw" / backend / "cases").glob("*/*.run.json")):
            run = load_contact_run(path)
            assert run.completed
            runs[run.case.case_id] = run
            identities[run.case.case_id] = sha256_file(path)
    assert len(runs) == 32
    details = {k: analyze(r) for k, r in runs.items()}
    comparisons = {}
    for backend in ("mujoco", "ovphysx"):
        for hand in ("left", "right"):
            for repeat in ("r01", "r02"):
                key = f"{backend}.{hand}.press_hold_release.contact"
                b = runs[f"{key}.base.{repeat}"]
                h = runs[f"{key}.halved.{repeat}"]
                assert len(h.samples) == 2*len(b.samples)-1
                window = [(x, h.samples[2*i]) for i, x in enumerate(b.samples) if 4 <= x.time_s <= 6.5]
                bs = runs[f"{backend}.{hand}.press_hold_release.sham.base.{repeat}"]
                hs = runs[f"{backend}.{hand}.press_hold_release.sham.halved.{repeat}"]
                gap_delta = [y.signed_gap_m-x.signed_gap_m for x, y in window]
                joint_delta = [max(abs(y.joint_positions[n]-x.joint_positions[n]) for n in x.joint_positions) for x,y in window]
                hold = [(x,h.samples[2*i]) for i,x in enumerate(b.samples) if 3.25 <= x.time_s < 3.75]
                force = lambda s: s.native_contact_observation["selected_pair_force_norm_n"]
                ratios = [force(y)/force(x) for x,y in hold if force(x)>0]
                scaled_event = threshold_event(h.samples, h.case.dt_s, 1e-4*h.case.dt_s/b.case.dt_s)
                comparisons[f"{backend}.{hand}.{repeat}"] = {
                    "release_midpoint_delta_ms": 1000*(details[h.case.case_id]["events"]["release"]["midpoint_s"]-details[b.case.case_id]["events"]["release"]["midpoint_s"]),
                    "return_window_gap_max_delta_um": 1e6*max(map(abs, gap_delta)),
                    "return_window_joint_max_delta_rad": max(joint_delta),
                    "sham_return_gap_max_delta_um": 1e6*max(abs(hs.samples[2*i].signed_gap_m-x.signed_gap_m) for i,x in enumerate(bs.samples) if 4 <= x.time_s <= 6.5),
                    "hold_force_halved_over_base_median": median(ratios),
                    "hold_force_base_median": median(force(x) for x,y in hold),
                    "hold_force_halved_median": median(force(y) for x,y in hold),
                    "hold_joint_max_delta_rad": max(abs(x.joint_positions[n]-y.joint_positions[n]) for x,y in hold for n in x.joint_positions),
                    "hold_target_max_delta_rad": max(abs(x.position_targets[n]-y.position_targets[n]) for x,y in hold for n in x.position_targets),
                    "exploratory_dt_scaled_threshold_release_delta_ms": None if scaled_event["release"] is None else 1000*(scaled_event["release"]["midpoint_s"]-details[b.case.case_id]["events"]["release"]["midpoint_s"]),
                    "gap_crossing_delta_ms": {str(g): None if crossing(b.samples,g) is None or crossing(h.samples,g) is None else 1000*(crossing(h.samples,g)-crossing(b.samples,g)) for g in (0,1e-6,1e-5,1e-4)},
                }
    for key, run in runs.items():
        if key.endswith("r01"):
            assert run.samples == runs[key[:-3]+"r02"].samples, "fresh repeat mismatch"
    # Re-hash after reading: neither input payloads nor their inventory may change.
    assert regular_tree_records(args.bundle, exclude=("bundle-manifest.json",)) == records
    result = {"kind": "post_hoc_offline_diagnostic_not_acceptance", "input_bundle_root": EXPECTED_ROOT,
              "script_sha256": sha256_file(Path(__file__)), "input_run_sha256": identities,
              "comparisons": comparisons, "cases": details,
              "formal_status_unchanged": "VALID / INCONCLUSIVE; pass_ready=false"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.output, result)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
