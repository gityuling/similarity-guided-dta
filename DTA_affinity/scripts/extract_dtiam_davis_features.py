#!/usr/bin/env python3
"""Extract Davis features through unmodified official DTIAM feature functions.

This narrow wrapper intentionally processes only Davis, unlike DTIAM's
``extract_feature.py`` entry point, which also computes all five unrelated
benchmark datasets.  Compound BerMol and ESM-2 feature extraction are called
directly from that official module; no model or feature algorithm is replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import dill
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DTIAM_ROOT = PROJECT / "baselines" / "DTIAM"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtiam-root", type=Path, default=DTIAM_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DTIAM_ROOT / "models" / "BerMolModel_base.pkl",
        help="Official BerMol weight linked from the DTIAM README.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.dtiam_root.resolve()
    data_dir = root / "data" / "dta" / "davis"
    feature_dir = data_dir / "features"
    code_dir = root / "code"
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing official BerMol checkpoint: {model_path}")
    sys.path.insert(0, str(code_dir))
    from data_process import extract_feature as official_extract  # noqa: PLC0415

    with (data_dir / "ligands_can.txt").open() as handle:
        ligands = json.load(handle, object_pairs_hook=OrderedDict)
    with (data_dir / "proteins.txt").open() as handle:
        proteins = json.load(handle, object_pairs_hook=OrderedDict)
    drugs = pd.DataFrame(ligands.items(), columns=["cid", "smi"])
    targets = pd.DataFrame(proteins.items(), columns=["pid", "seq"])

    feature_dir.mkdir(parents=True, exist_ok=True)
    compound_path = feature_dir / "compound_features.pkl"
    protein_path = feature_dir / "protein_features.pkl"
    print(f"Extracting official DTIAM BerMol features for {len(drugs)} Davis drugs", flush=True)
    compound_features = official_extract.cal_comp_feat(drugs, str(model_path), device=args.device)
    with compound_path.open("wb") as handle:
        dill.dump(compound_features, handle)
    print(f"Extracting official DTIAM ESM-2 features for {len(targets)} Davis proteins", flush=True)
    protein_features = official_extract.cal_prot_feat(targets)
    with protein_path.open("wb") as handle:
        dill.dump(protein_features, handle)
    audit = {
        "implementation": "DTIAM code/data_process/extract_feature.py cal_comp_feat + cal_prot_feat",
        "bermol_checkpoint": str(model_path),
        "compound_features": str(compound_path),
        "protein_features": str(protein_path),
        "drug_count": len(compound_features),
        "protein_count": len(protein_features),
    }
    (feature_dir / "official_extraction_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
