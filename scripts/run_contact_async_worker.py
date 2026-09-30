"""Separate candidate entrypoint; never invoke as the original formal C0 worker."""
import argparse
import os
import sys


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from wave_asset_qa.contact.async_campaign import CAMPAIGN_ID
    from wave_asset_qa.contact.worker import worker_main
    from wave_asset_qa.parity.contracts import Simulator
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend',choices=['mujoco','ovphysx'],required=True)
    args, remaining = parser.parse_known_args()
    raise SystemExit(worker_main(remaining,backend=Simulator(args.backend),experimental_campaign=CAMPAIGN_ID))
