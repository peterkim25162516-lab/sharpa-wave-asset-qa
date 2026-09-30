"""Synthetic counterexamples for the post-hoc event diagnostic."""
import importlib.util
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from wave_asset_qa.contact.records import ContactTraceSample

spec = importlib.util.spec_from_file_location(
    "release_diagnosis", Path(__file__).resolve().parents[1] / "scripts/diagnose_contact_release.py"
)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def sample(step, dt, force):
    item = object.__new__(ContactTraceSample)
    for field in fields(ContactTraceSample):
        object.__setattr__(item, field.name, None)
    for name, value in {"step": step, "time_s": step*dt, "pair_active": force>1e-4,
                        "native_contact_observation": {"selected_pair_force_norm_n": force}}.items():
        object.__setattr__(item, name, value)
    return item


def test_threshold_reanalysis_does_not_mutate_frozen_bits():
    samples = tuple(sample(i, .002, .0002 if 1 <= i <= 4 else 0) for i in range(10))
    before = [(s.pair_active, dict(s.native_contact_observation)) for s in samples]
    assert diagnosis.threshold_event(samples, .002, .001)["release"] is None
    assert diagnosis.threshold_event(samples, .002, .0001)["release"]["midpoint_s"] == pytest.approx(.009)
    assert before == [(s.pair_active, dict(s.native_contact_observation)) for s in samples]


def test_debounce_backdates_and_rejects_short_dropouts():
    # One inactive interval cannot end a contact under a two-interval window.
    forces = [0, 1, 1, 0, 1, 1, 0, 0, 0]
    samples = tuple(sample(i, .002, f) for i,f in enumerate(forces))
    assert diagnosis.event(samples, .002)["release"]["first_sample_index"] == 6
    assert diagnosis.event(samples, .002, .001)["release"]["first_sample_index"] == 3


def test_geometric_crossing_is_interpolated_and_missing_is_none():
    samples = [SimpleNamespace(time_s=t, signed_gap_m=g) for t,g in
               [(3, -1), (3.5, 1), (4, -1), (4.002, 3)]]
    assert diagnosis.crossing(samples, 0) == pytest.approx(4.0005)
    assert diagnosis.crossing(samples, 5) is None
