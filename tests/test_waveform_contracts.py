from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

import pytest

from wave_asset_qa.parity.contracts import (
    ParityManifest,
    RESULT_SCHEMA_VERSION,
    RunPolicy,
    SCHEMA_VERSION,
    SUPPORTED_MANIFEST_SCHEMA_VERSIONS,
    ScenarioKind,
    WaveformScenarioSpec,
)
from wave_asset_qa.parity.scenarios import (
    CanonicalTargetSequenceDigest,
    TARGET_SEQUENCE_DIGEST_SCHEMA_VERSION,
    TimestepVariant,
    canonical_manifest_json,
    canonical_position_targets,
    canonical_target_sequence_sha256,
    expand_scenario_cases,
    load_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
GATE0_PATH = ROOT / "configs" / "parity" / "gate0.json"
V1_SCHEMA_PATH = ROOT / "schemas" / "parity-manifest.schema.json"
WAVEFORM_PATH = ROOT / "configs" / "parity" / "waveform_t1.json"
V2_SCHEMA_PATH = ROOT / "schemas" / "parity-manifest-v2.schema.json"

GATE0_FILE_SHA256 = "a95a1da0d38abac8b3eccf18ea0b618b4898e35a6102017a2f0e024a9a4d5244"
GATE0_SEMANTIC_SHA256 = "576d8663a71692d6f5bd148ad719c9074ce0faceca6e4169bd8fcd14dc09899e"
V1_SCHEMA_FILE_SHA256 = "228accdf80b2fc4685c56996f2da885b79b0446a4302e0a109667e776b066697"
WAVEFORM_FILE_SHA256 = "74e85e133f511944a9a9feeae3f29656db477e0983fa741995ce2c65ce1141a9"
WAVEFORM_SEMANTIC_SHA256 = "68b40b1a0f7bbbdd9860750f92d58445bb8aa16cb6682d39eb5f41e4ac7df85c"
V2_SCHEMA_FILE_SHA256 = "242134b8413936b8747c849db2e494c8a0234241cfa4a48d7d3e99c6b815a71f"

LEFT_TARGET_HASHES = {
    "offset_sine": {
        "base_full": "7244673afcee194b5e7fc1be3baded11bdce347dd168100fee7410c06de98472",
        "base_prestep": "cfc89ebb6741a57c1c2484305b42fe31494f715b9298f3d1498d08a9f726c536",
        "halved_full": "6a97aafbd4ffaf181b1e659045a543a228f88665b352e73c24a63158efa94c9c",
        "halved_prestep": "f4dce4163d17a314143245ad78dc8f531c55986158ef7959983d361498a0113a",
    },
    "offset_linear_chirp": {
        "base_full": "e994d1c344a93432991a421fb86f0107b751f183a71f83ffe2477b4142ab8592",
        "base_prestep": "903c6c852eabd0f43209a64211c3cc9727e42a2f226623ba4598ce22b81e4a31",
        "halved_full": "60a1b4d7ed8c2e5d3b36c2c586ed2ec41bde0b53dc8ccd871d5574b46a17e1e7",
        "halved_prestep": "47b260dba1ba1dfdd1ec9b80de4a34fde01d99d6bc7ba05fcac8f3d16c755e25",
    },
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v1_gate0_bytes_semantics_and_schema_are_frozen() -> None:
    source = GATE0_PATH.read_bytes()
    schema_source = V1_SCHEMA_PATH.read_bytes()
    manifest = load_manifest(GATE0_PATH)

    assert sha256(source).hexdigest() == GATE0_FILE_SHA256
    assert sha256(schema_source).hexdigest() == V1_SCHEMA_FILE_SHA256
    assert manifest_sha256(manifest) == GATE0_SEMANTIC_SHA256
    assert manifest.to_dict() == json.loads(source)
    assert json.loads(canonical_manifest_json(manifest)) == json.loads(source)
    assert manifest.schema_version == 1
    assert SCHEMA_VERSION == RESULT_SCHEMA_VERSION == 1
    assert SUPPORTED_MANIFEST_SCHEMA_VERSIONS == (1, 2)


def test_v2_manifest_is_strict_complete_and_round_trips() -> None:
    source_bytes = WAVEFORM_PATH.read_bytes()
    source = json.loads(source_bytes)
    manifest = load_manifest(WAVEFORM_PATH)

    assert sha256(source_bytes).hexdigest() == WAVEFORM_FILE_SHA256
    assert manifest_sha256(manifest) == WAVEFORM_SEMANTIC_SHA256
    assert manifest.schema_version == 2
    assert manifest.manifest_id == "wavesimparity-waveform-t1"
    assert manifest.to_dict() == source
    assert ParityManifest.from_dict(manifest.to_dict()) == manifest
    assert tuple(type(item) for item in manifest.scenarios) == (
        WaveformScenarioSpec,
        WaveformScenarioSpec,
    )
    assert tuple(item.kind for item in manifest.scenarios) == (
        ScenarioKind.OFFSET_SINE,
        ScenarioKind.OFFSET_LINEAR_CHIRP,
    )
    assert manifest.run_policy == RunPolicy(
        repeat_count=2,
        dt_halving_scenario_ids=("offset_sine", "offset_linear_chirp"),
        minimum_completion_fraction=0.99,
    )

    cases = expand_scenario_cases(manifest)
    assert len(cases) == 32
    assert len({case.case_id for case in cases}) == 32
    assert sum(case.timestep_variant is TimestepVariant.BASE for case in cases) == 16
    assert sum(case.timestep_variant is TimestepVariant.HALVED for case in cases) == 16
    assert cases[0].case_id == "mujoco.left.offset_sine.base.r01"
    assert cases[-1].case_id == "ovphysx.right.offset_linear_chirp.halved.r02"


def test_v2_published_schema_is_closed_and_locks_t1_matrix() -> None:
    schema_bytes = V2_SCHEMA_PATH.read_bytes()
    schema = json.loads(schema_bytes)

    assert sha256(schema_bytes).hexdigest() == V2_SCHEMA_FILE_SHA256
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"] == {"const": 2}
    scenarios = schema["properties"]["scenarios"]
    assert scenarios["minItems"] == scenarios["maxItems"] == 2
    policy = schema["$defs"]["runPolicy"]["properties"]
    assert policy["repeat_count"] == {"const": 2}
    assert policy["dt_halving_scenario_ids"] == {
        "const": ["offset_sine", "offset_linear_chirp"]
    }
    assert policy["minimum_completion_fraction"] == {"const": 0.99}


@pytest.mark.parametrize("scenario_id", ["offset_sine", "offset_linear_chirp"])
def test_waveform_targets_are_bounded_smooth_and_dt_invariant(
    scenario_id: str,
) -> None:
    manifest = load_manifest(WAVEFORM_PATH)
    scenario = manifest.scenario(scenario_id)
    joint_names = manifest.hands[0].joint_names
    base_steps = scenario.steps

    base_values: list[float] = []
    for step in range(base_steps + 1):
        base = canonical_position_targets(
            scenario, joint_names, step_index=step, dt_s=scenario.dt_s
        )
        halved = canonical_position_targets(
            scenario, joint_names, step_index=2 * step, dt_s=scenario.dt_s / 2.0
        )
        assert base == halved
        assert len(set(base.values())) == 1
        base_values.append(base[joint_names[0]])

    assert base_values[0] == pytest.approx(0.0, abs=1e-15)
    assert base_values[-1] == pytest.approx(0.0, abs=1e-15)
    assert min(base_values) >= -1e-15
    assert max(base_values) <= 0.05 + 1e-15
    assert max(base_values) > 0.04999
    if scenario_id == "offset_sine":
        assert base_values[250] == pytest.approx(0.05, abs=1e-15)
        assert base_values[500] == pytest.approx(0.0, abs=1e-15)
    else:
        cycles = (
            scenario.frequency_start_hz * scenario.duration_s
            + 0.5
            * (scenario.frequency_end_hz - scenario.frequency_start_hz)
            * scenario.duration_s
        )
        assert cycles == 9.0


def test_target_sequence_digest_encoding_is_incremental_and_binary32_projected() -> None:
    digest = CanonicalTargetSequenceDigest(("j0", "j1"))
    digest.append({"j1": 0.0, "j0": 0.05})

    projected = struct.unpack("!f", struct.pack("!f", 0.05))[0]
    expected_stream = (
        '{"joint_names":["j0","j1"],"projection":"ieee754_binary32_roundtrip",'
        '"schema_version":1}\n'
        f'{{"step_index":0,"values_hex":["{projected.hex()}","0x0.0p+0"]}}\n'
        '{"record_count":1}\n'
    ).encode("utf-8")
    expected = sha256(expected_stream).hexdigest()

    assert TARGET_SEQUENCE_DIGEST_SCHEMA_VERSION == 1
    assert digest.count == 1
    assert digest.joint_names == ("j0", "j1")
    assert digest.hexdigest() == expected
    assert digest.hexdigest() == expected
    assert digest.digest().hex() == expected
    clone = digest.copy()
    clone.append({"j0": 0.0, "j1": 0.0})
    assert clone.count == 2
    assert clone.hexdigest() != expected
    assert digest.count == 1
    assert digest.hexdigest() == expected


@pytest.mark.parametrize("scenario_id", ["offset_sine", "offset_linear_chirp"])
def test_target_sequence_hashes_lock_full_and_prestep_streams(
    scenario_id: str,
) -> None:
    manifest = load_manifest(WAVEFORM_PATH)
    scenario = manifest.scenario(scenario_id)
    joint_names = manifest.hands[0].joint_names
    expected = LEFT_TARGET_HASHES[scenario_id]

    assert canonical_target_sequence_sha256(scenario, joint_names) == expected["base_full"]
    assert (
        canonical_target_sequence_sha256(
            scenario, joint_names, include_terminal=False
        )
        == expected["base_prestep"]
    )
    assert (
        canonical_target_sequence_sha256(
            scenario, joint_names, dt_s=scenario.dt_s / 2.0
        )
        == expected["halved_full"]
    )
    assert (
        canonical_target_sequence_sha256(
            scenario,
            joint_names,
            dt_s=scenario.dt_s / 2.0,
            include_terminal=False,
        )
        == expected["halved_prestep"]
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda data: data["run_policy"].update(
                {"minimum_completion_fraction": 1.0}
            ),
            "must be 0.99",
        ),
        (
            lambda data: data["run_policy"].update({"repeat_count": 3}),
            "repeat_count must be 2",
        ),
        (
            lambda data: data["run_policy"].update(
                {"dt_halving_scenario_ids": ["offset_sine"]}
            ),
            "both waveform scenarios",
        ),
        (lambda data: data["scenarios"].reverse(), "offset_sine then"),
        (
            lambda data: data["scenarios"][0].update({"duration_s": 3.0}),
            "frozen T1 profile",
        ),
        (
            lambda data: data["scenarios"][0].update({"surprise": True}),
            "unknown field",
        ),
    ],
)
def test_v2_manifest_rejects_protocol_drift(mutate, match: str) -> None:
    payload = deepcopy(_json(WAVEFORM_PATH))
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        ParityManifest.from_dict(payload)


def test_v1_completion_rule_remains_exactly_one() -> None:
    payload = deepcopy(_json(GATE0_PATH))
    payload["run_policy"]["minimum_completion_fraction"] = 0.99

    with pytest.raises(ValueError, match="must be 1.0 for Gate 0"):
        ParityManifest.from_dict(payload)

    assert RunPolicy(2, ("small_step",), 0.99).minimum_completion_fraction == 0.99
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        RunPolicy(2, ("small_step",), 0.0)


def test_waveform_scenario_rejects_invalid_envelope_and_phase() -> None:
    manifest = load_manifest(WAVEFORM_PATH)
    sine = manifest.scenario("offset_sine")
    assert isinstance(sine, WaveformScenarioSpec)

    with pytest.raises(ValueError, match=r"within \[0, 0.05\]"):
        replace(sine, command_offset_rad=0.03, command_amplitude_rad=0.03)
    with pytest.raises(ValueError, match="must equal"):
        replace(sine, command_offset_rad=0.02)
    with pytest.raises(ValueError, match="constant frequency"):
        replace(sine, frequency_end_hz=1.5)
    with pytest.raises(ValueError, match="integer number of cycles"):
        replace(sine, frequency_start_hz=1.1, frequency_end_hz=1.1)


def test_target_digest_rejects_ambiguous_or_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        CanonicalTargetSequenceDigest(("j0", "j0"))
    digest = CanonicalTargetSequenceDigest(("j0", "j1"))
    with pytest.raises(ValueError, match="exactly cover"):
        digest.append({"j0": 0.0})
    with pytest.raises(ValueError, match="finite"):
        digest.append({"j0": math.nan, "j1": 0.0})
    with pytest.raises(ValueError, match="binary32"):
        digest.append({"j0": 1e100, "j1": 0.0})
