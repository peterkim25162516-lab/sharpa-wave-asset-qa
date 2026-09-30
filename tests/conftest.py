"""Portable inputs for launcher unit tests; production freezes stay unchanged."""
from pathlib import Path
import sys

import pytest


@pytest.fixture(autouse=True)
def portable_launcher_unit_inputs(request, monkeypatch):
    name = request.module.__name__
    launchers = {
        "test_gate0_local_launcher",
        "test_contact_c0_local_launcher",
        "test_waveform_t1_local_launcher",
    }
    if name in launchers:
        # Hosted Linux exposes python through a symlink. These tests exercise
        # other launcher contracts; dedicated link-rejection tests use their
        # own explicit paths and continue to exercise the real guard.
        monkeypatch.setattr(sys, "executable", str(Path(sys.executable).resolve()))
    if name not in {"test_waveform_t1_local_launcher", "test_waveform_t1_finalizer"}:
        return
    loader_name = "_load_launcher" if name.endswith("local_launcher") else "_load_finalizer"
    original_loader = getattr(request.module, loader_name)

    def load_with_frozen_input_fixture():
        module = original_loader()
        # Supply the original frozen observations to the real validation code.
        # Version/hash drift tests subsequently alter one input or expected
        # value and must still fail. No production expected constant is changed.
        for function, constant in (
            ("python_version", "EXPECTED_LOCAL_PYTHON_VERSION"),
            ("python_implementation", "EXPECTED_LOCAL_PYTHON_IMPLEMENTATION"),
            ("platform", "EXPECTED_LOCAL_PLATFORM"),
            ("machine", "EXPECTED_LOCAL_MACHINE"),
        ):
            value = getattr(module, constant)
            monkeypatch.setattr(module.platform, function, lambda value=value: value)
        monkeypatch.setattr(module.numpy, "__version__", module.EXPECTED_LOCAL_NUMPY_VERSION)
        monkeypatch.setattr(module.mujoco, "__version__", module.EXPECTED_LOCAL_MUJOCO_VERSION)
        original_hash = module._sha256_file
        executable = Path(sys.executable).resolve()
        frozen_hash = module.EXPECTED_LOCAL_PYTHON_EXECUTABLE_SHA256

        def hash_with_interpreter_fixture(path):
            return frozen_hash if Path(path).resolve() == executable else original_hash(path)

        monkeypatch.setattr(module, "_sha256_file", hash_with_interpreter_fixture)
        return module

    monkeypatch.setattr(request.module, loader_name, load_with_frozen_input_fixture)
