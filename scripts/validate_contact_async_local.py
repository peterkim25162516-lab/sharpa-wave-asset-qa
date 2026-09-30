"""Read-only validation of the local 16-case candidate half, not a C0 verdict."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import finalize_contact_c0 as original
from wave_asset_qa.contact.async_campaign import CAMPAIGN_ID
from wave_asset_qa.contact.bundle import canonical_json_sha256
from wave_asset_qa.contact.scenarios import load_contact_manifest
from wave_asset_qa.parity.contracts import Simulator


def validate(root, *, source_revision):
    project = Path(__file__).resolve().parents[1]
    tree = original._validate_source(project,source_revision)
    manifest_path = project/'configs/parity/contact_c0.json'
    manifest = load_contact_manifest(manifest_path)
    root = original._resolved_directory(root,'local candidate evidence')
    inventory = original._validate_evidence_manifest(root,CAMPAIGN_ID+'-mujoco')
    runs,meta,processes = original._load_backend_evidence(root,backend=Simulator.MUJOCO,
        manifest=manifest,source_revision=source_revision,source_tree=tree,
        manifest_file_sha256=original._file_sha256(manifest_path),
        manifest_semantic_sha256=canonical_json_sha256(manifest.to_dict()),
        source_archive_sha256=sha256(original._git_archive_bytes(project,source_revision)).hexdigest(),
        remote_snapshot_sha256=None,project_root=project,
        experimental_campaign=CAMPAIGN_ID)
    return {'experimental_campaign':CAMPAIGN_ID,'scope':'local_half_only',
            'case_count':len(runs),'fresh_process_count':len(processes),
            'source_revision':source_revision,'source_tree':tree,'inventory':inventory,
            'runtime_log_gate_passed':True,'scientific_verdict':'NOT_EVALUATED',
            'formal_c0_replacement':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--source-revision',required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.evidence_root,source_revision=args.source_revision),indent=2,allow_nan=False))
