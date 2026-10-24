# Frozen BIB evidence, 17 October 2026

`protocol.json` records the original drug-ID-cold data tables, outer and inner
manifests, checkpoint arguments, component predictions, neighbor parameters,
source-data archive and current source hashes. All 30,056 Davis and 118,254
KIBA pairs occur exactly once in outer tests.

`feature_inventory.csv` inventories all 2,744 used token-cache files with
SHA-256, shape and finite-value checks. KIBA has 2,111 drug IDs and 2,068
canonical structures; its original drug-ID protocol is not identity-disjoint.

`source_snapshot.zip` records current code. The project has no Git history;
this does not reconstruct historical training source. The baseline DTIAM Git
revision is recorded separately. Davis legacy feature files lack a fully
recoverable historical checkpoint lineage. Hashes establish the inputs now.

Verify without regenerating data or predictions, from the DTA project:

```bash
/home/lsy/miniconda3/envs/psichic_fp/bin/python scripts/freeze_bib_protocol.py --verify
```

Paths are relative to the parent Drug_repurpose workspace. Keep this freeze
intact; new experiments use separate outputs.
