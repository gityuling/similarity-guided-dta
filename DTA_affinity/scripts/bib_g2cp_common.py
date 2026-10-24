"""Paths and integrity helpers for the independent Results 5 experiment."""

import json
from datetime import datetime, timezone

from bib_common import PROJECT, ROOT, record, sha256, write_once

CONFIG = PROJECT / "configs/bib_g2cp_dta_20261019.json"
OUT = PROJECT / "reports/bib_g2cp_dta_20261019"
RUN = PROJECT / "results/bib_g2cp_dta_20261019"
DATA = PROJECT / "data/processed/bib_g2cp_dta_20261019"
SOURCE = ROOT / "UniPert-G2CP_reproduce/G2CP"
COVERAGE = PROJECT / "reports/bib_perturbation_bridge_20261019"
TOKENS = DATA / "tokens"
KIBA = PROJECT / "data/processed/kiba_dtiam_protocol"


def plan():
    return json.loads(CONFIG.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data, immutable=False):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if immutable:
        write_once(path, text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def check_records(records):
    for item in records:
        path = ROOT / item["path"]
        if not path.exists() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input changed: {path}")


def verify_data():
    audit = json.loads((OUT / "data_audit.json").read_text())
    check_records(audit["files"])
    check_records(audit["sources"])
    return audit
