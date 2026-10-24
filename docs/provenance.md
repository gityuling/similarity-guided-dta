# Feature provenance

The release preserves the exact arrays used by the affinity models. Each file
has a SHA-256 hash in `manifests/files.json`. The original benchmark inventory
also records array shape and dataset membership in
`DTA_affinity/reports/bib_protocol_20261017/feature_inventory.csv`.

## KIBA

The KIBA cache was generated in a separate directory, without copying legacy
features. It contains 2,068 molecular arrays and 229 protein arrays. Molecular
features have width 1,536; protein features have width 1,280, with a maximum of
1,024 protein tokens. All 118,254 benchmark pairs are retained.

The extraction record is
`DTA_affinity/data/processed/kiba_dtiam_protocol/tokens/build_audit.json`.
It records the extractor hash, settings, device and encoder hashes:

- Uni-Mol2 checkpoint: `48e1ba230119a5f286b81dd71bec4889aebebffa03e66d9241f3ceb7bede9a5e`.
- ESM-2 model: `c82661e60305b0c3d747e3c4ba308f4762f6df99ef0c1911712871cdcb6d6d15`.
- ESM-2 configuration: `16d490981c34b2d73e6b967a53d4dd7b9d6100ddb933382dc5256446935970e4`.

## Davis

The benchmark uses 68 molecular arrays and 379 protein arrays for 442 target
IDs. Target IDs assigned the same input sequence share a feature array and a
training-reference pool. Some such IDs describe distinct mutant or experimental
constructs in the source benchmark.

The historical Davis extraction checkpoint lineage is incomplete. The original
arrays are therefore distributed directly, with their recorded hashes, to
support replay and retraining on the same inputs. Matching shapes or current
encoder filenames do not establish which checkpoint generated an old array.

## External panels

Extraction and cache-copy records accompany the NanoBRET and ABL1/analog-series
features:

- `DTA_affinity/data/processed/bib_nanobret_20261019/tokens/build_audit.json`
- `DTA_affinity/data/processed/bib_downstream_20261022/tokens/build_audit.json`

These records retain the original machine paths as provenance. Runtime scripts
resolve their inputs from the repository layout; the original home directory is
not required for the portable reproduction checks.

## Versioned records

Training settings are in `configs/bib_validation_20261017.json`. The later
`configs/bib_analysis_20261017.json` specifies the implemented global
chemical-identity bootstrap, superseding the earlier within-fold description.
The original freeze and source snapshot are retained unchanged. The release
manifest records the files distributed in this version.
