"""Explicit experimental entrypoint; never relabel the original C0 campaign."""
import json

import finalize_contact_c0 as original
from wave_asset_qa.contact.async_campaign import CAMPAIGN_ID


if __name__ == '__main__':
    args = original._parser().parse_args()
    result = original.finalize_contact_c0(
        manifest_path=args.manifest, local_evidence_root=args.local_evidence_root,
        remote_evidence_root=args.remote_evidence_root, output_dir=args.output_dir,
        public_output_dir=args.public_output_dir, source_revision=args.source_revision,
        source_archive_sha256=args.source_archive_sha256,
        remote_snapshot_sha256=args.remote_snapshot_sha256, experimental_campaign=CAMPAIGN_ID)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
