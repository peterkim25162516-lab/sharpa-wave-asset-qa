#!/usr/bin/env python3
"""Execute one frozen Contact Gate C0 OVPhysX case."""

from __future__ import annotations

import os
import sys


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from wave_asset_qa.contact.worker import worker_main
    from wave_asset_qa.parity.contracts import Simulator

    raise SystemExit(worker_main(None, backend=Simulator.OVPHYSX))
