#!/usr/bin/env python3
"""Convert downloaded Davis/KIBA files into explicit pairwise tables.

The raw Davis file is a whitespace-delimited 68 x 442 matrix in nM. Values
equal to 10,000 are the dataset's missing/uncensored sentinel and are not
emitted as observations. pKd is computed as 9 - log10(Kd_nM).

The raw KIBA workbook is kept untouched. This script extracts only the
non-empty cells from its KIBA sheet, preserving the original KIBA score and
the compound/protein identifiers.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def prepare_davis(raw_dir: Path, processed_dir: Path) -> None:
    matrix_path = raw_dir / "drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt"
    drug_path = raw_dir / "drug_PubChem_CIDs.txt"
    target_path = raw_dir / "target_gene_names.txt"
    matrix = np.loadtxt(matrix_path)
    drug_ids = read_lines(drug_path)
    target_ids = read_lines(target_path)

    if matrix.shape != (len(drug_ids), len(target_ids)):
        raise ValueError(
            f"Davis shape mismatch: matrix={matrix.shape}, "
            f"drugs={len(drug_ids)}, targets={len(target_ids)}"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "davis_kd_nM_matrix.npy", matrix)

    long_path = processed_dir / "davis_affinity_long.csv"
    n_obs = 0
    with long_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["drug_index", "drug_pubchem_cid", "target_index", "target_gene", "kd_nM", "pKd"]
        )
        for drug_index, drug_id in enumerate(drug_ids):
            for target_index, target_id in enumerate(target_ids):
                kd_nM = float(matrix[drug_index, target_index])
                if not math.isfinite(kd_nM) or kd_nM >= 10000.0:
                    continue
                pKd = 9.0 - math.log10(kd_nM)
                writer.writerow(
                    [drug_index, drug_id, target_index, target_id, f"{kd_nM:.8g}", f"{pKd:.8f}"]
                )
                n_obs += 1

    summary = processed_dir / "summary.txt"
    summary.write_text(
        "dataset=davis\n"
        f"matrix_shape={matrix.shape[0]}x{matrix.shape[1]}\n"
        f"observed_pairs={n_obs}\n"
        "missing_sentinel=10000_nM\n"
        "label=pKd=9-log10(Kd_nM)\n"
    )
    print(f"Davis: {matrix.shape[0]} drugs x {matrix.shape[1]} targets; {n_obs} observed pairs")
    print(f"Wrote {long_path}")

    # The public entity table supplies reproducible SMILES and protein
    # sequences. Its affinity file uses the conventional pKd=5 floor for
    # the raw 10,000 nM sentinel. Keep both versions explicit: the clean
    # table is the primary regression set, while the floor table is useful
    # for reproducing common DeepDTA-style benchmarks.
    entity_drugs = raw_dir / "dta_entities_drugs.csv"
    entity_proteins = raw_dir / "dta_entities_proteins.csv"
    entity_affinity = raw_dir / "dta_entities_affinity.csv"
    if entity_drugs.exists() and entity_proteins.exists() and entity_affinity.exists():
        import pandas as pd

        drugs = pd.read_csv(entity_drugs)
        proteins = pd.read_csv(entity_proteins)
        affinity = pd.read_csv(entity_affinity)
        merged = affinity.merge(
            drugs[["Drug_Index", "CID", "Canonical_SMILES"]],
            left_on="Drug_Index",
            right_on="Drug_Index",
            how="left",
        ).merge(
            proteins[["Protein_Index", "Accession_Number", "Gene_Name", "Sequence"]],
            left_on="Protein_Index",
            right_on="Protein_Index",
            how="left",
        )
        merged["is_censored_floor"] = merged["Affinity"].le(5.0 + 1e-8)
        merged = merged.rename(columns={"Affinity": "pKd_floor"})
        merged.to_csv(processed_dir / "davis_entity_affinity_floor.csv", index=False)
        merged.loc[~merged["is_censored_floor"]].to_csv(
            processed_dir / "davis_entity_affinity_clean.csv", index=False
        )
        (processed_dir / "entity_summary.txt").write_text(
            f"entity_drugs={len(drugs)}\n"
            f"entity_proteins={len(proteins)}\n"
            f"floor_pairs={len(merged)}\n"
            f"clean_pairs={(~merged['is_censored_floor']).sum()}\n"
            "floor_definition=pKd=5 corresponds to raw Kd=10000 nM sentinel\n"
        )
        print(
            f"Davis entity table: {len(drugs)} drugs, {len(proteins)} proteins, "
            f"{len(merged)} floor pairs, {(~merged['is_censored_floor']).sum()} clean pairs"
        )


def prepare_kiba(raw_dir: Path, processed_dir: Path) -> None:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit("KIBA extraction requires openpyxl") from exc

    workbook_path = raw_dir / "extracted" / "KiBA.xlsx"
    processed_dir.mkdir(parents=True, exist_ok=True)
    long_path = processed_dir / "kiba_affinity_long.csv"

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    if "KIBA" not in workbook.sheetnames:
        raise ValueError(f"KIBA sheet not found; sheets={workbook.sheetnames}")
    sheet = workbook["KIBA"]

    n_rows = 0
    n_obs = 0
    with long_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["compound_id", "protein_id", "kiba_score"])
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        protein_ids = [str(value).strip() if value is not None else "" for value in header[1:]]
        for row in rows:
            n_rows += 1
            if not row or row[0] is None:
                continue
            compound_id = str(row[0]).strip()
            for protein_id, value in zip(protein_ids, row[1:]):
                if not protein_id or value is None or value == "":
                    continue
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score):
                    continue
                writer.writerow([compound_id, protein_id, f"{score:.8g}"])
                n_obs += 1
    workbook.close()

    (processed_dir / "summary.txt").write_text(
        "dataset=kiba\n"
        f"compound_rows={n_rows}\n"
        f"observed_pairs={n_obs}\n"
        "label=KIBA_score (integrated score; not a single physical affinity unit)\n"
    )
    print(f"KIBA: {n_rows} compound rows; {n_obs} observed pairs")
    print(f"Wrote {long_path}")

    # Also materialize the standard 2,111-drug x 229-target DeepDTA
    # benchmark. It is the compact benchmark used for most published DTA
    # comparisons and is independent of the larger raw workbook above.
    deepdta_y = raw_dir / "deepdta_Y"
    deepdta_drugs = raw_dir / "deepdta_ligands_can.txt"
    deepdta_proteins = raw_dir / "deepdta_proteins.txt"
    if deepdta_y.exists() and deepdta_drugs.exists() and deepdta_proteins.exists():
        import json
        import pickle
        import pandas as pd

        with deepdta_y.open("rb") as handle:
            y = np.asarray(pickle.load(handle, encoding="latin1"), dtype=float)
        drugs = json.loads(deepdta_drugs.read_text())
        proteins = json.loads(deepdta_proteins.read_text())
        drug_ids = list(drugs)
        protein_ids = list(proteins)
        if y.shape != (len(drug_ids), len(protein_ids)):
            raise ValueError(
                f"DeepDTA KIBA shape mismatch: Y={y.shape}, "
                f"drugs={len(drug_ids)}, proteins={len(protein_ids)}"
            )
        rows, cols = np.where(np.isfinite(y))
        deepdta = pd.DataFrame(
            {
                "drug_index": rows,
                "drug_id": [drug_ids[i] for i in rows],
                "smiles": [drugs[drug_ids[i]] for i in rows],
                "target_index": cols,
                "protein_id": [protein_ids[i] for i in cols],
                "protein_sequence": [proteins[protein_ids[i]] for i in cols],
                "kiba_score": y[rows, cols],
            }
        )
        deepdta_path = processed_dir / "kiba_deepdta_affinity_long.csv"
        deepdta.to_csv(deepdta_path, index=False)
        (processed_dir / "deepdta_summary.txt").write_text(
            "dataset=kiba_deepdta_standard\n"
            f"drugs={len(drug_ids)}\n"
            f"proteins={len(protein_ids)}\n"
            f"observed_pairs={len(deepdta)}\n"
            "label=KIBA_score\n"
        )
        print(
            f"KIBA DeepDTA benchmark: {len(drug_ids)} drugs x {len(protein_ids)} proteins; "
            f"{len(deepdta)} observed pairs"
        )
        print(f"Wrote {deepdta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["davis", "kiba", "all"])
    args = parser.parse_args()
    if args.dataset in {"davis", "all"}:
        prepare_davis(ROOT / "data/raw/davis", ROOT / "data/processed/davis")
    if args.dataset in {"kiba", "all"}:
        prepare_kiba(ROOT / "data/raw/kiba", ROOT / "data/processed/kiba")


if __name__ == "__main__":
    main()
